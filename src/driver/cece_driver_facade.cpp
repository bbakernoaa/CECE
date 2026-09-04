#include "cece/cece_driver_facade.hpp"

#include <amio/amio.h>
#include <yaml-cpp/yaml.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/axis.hpp>
#include <cmath>
#include <dagr/logging.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tick/tick.hpp>
#include <vector>

#include "cece/cece_helm_graph.hpp"
#include "cece/cece_internal.hpp"
#include "cece/cece_logger.hpp"
#include "cece/cece_regridder_utils.hpp"
#include "cece/cece_standalone_writer.hpp"

namespace fs = std::filesystem;

extern "C" {
void cece_ingestor_set_field(void* data_ptr, const char* field_name, int name_len, const double* field_data, int n_lev, int n_elem, int* rc);
void amio_set_parent_communicator(MPI_Fint comm);
}

namespace cece {

namespace {

/**
 * @brief Simulation datetime fields derived from an ISO-8601 timestamp.
 *
 * Used by the per-stream temporal-cadence mechanism to map the current
 * simulation time onto a record index within an input file.
 */
struct SimDateTime {
    int year = 0;
    int month = 0;        ///< 1-12
    int day = 0;          ///< 1-31
    int hour = 0;         ///< 0-23
    int day_of_week = 0;  ///< 0=Sunday .. 6=Saturday
    bool valid = false;
};

/**
 * @brief Parse an ISO-8601 timestamp ("YYYY-MM-DDThh:mm:ss") into calendar fields.
 *
 * Parsing and calendar arithmetic use the HELM TICK library (tick::parse_iso8601
 * and tick::Gregorian_Calendar) rather than std::chrono, keeping time handling
 * consistent with the rest of CECE. The day-of-week is derived from TICK's
 * proleptic-Gregorian day count (TICK's epoch 2026-01-01 is a Thursday), so it
 * is correct for any date.
 */
SimDateTime parse_sim_datetime(const std::string& iso8601) {
    SimDateTime dt;
    try {
        const tick::Date_Time tdt = tick::parse_iso8601(iso8601);
        dt.year = tdt.year;
        dt.month = tdt.month;
        dt.day = tdt.day;
        dt.hour = tdt.hour;

        // Whole days since TICK's epoch (2026-01-01T00:00:00), floored so dates
        // before the epoch map correctly. 2026-01-01 is a Thursday, i.e. index 4
        // in a 0=Sunday..6=Saturday week; offset by that to anchor the cycle.
        const std::int64_t nanos = tick::Gregorian_Calendar::to_time_point(tdt).nanos();
        std::int64_t days = nanos / tick::nanos_per_day;
        if (nanos < 0 && nanos % tick::nanos_per_day != 0) --days;  // floor toward -inf
        dt.day_of_week = static_cast<int>(((days + 4) % 7 + 7) % 7);
        dt.valid = true;
    } catch (const std::exception&) {
        // Malformed timestamp: use explicit default values so callers fall back
        // to legacy step-index cycling.
        dt = SimDateTime{};
    }
    return dt;
}

// RecordBracket is defined in cece/cece_driver_facade.hpp so it can be used as
// SliceCacheEntry::last_bracket; the temporal-cadence helpers below use that
// shared definition (resolved as cece::RecordBracket).

/**
 * @brief Map a simulation datetime onto a record bracket for a given cadence.
 *
 * @param cadence  One of "hourly", "weekly", "monthly" (case-insensitive).
 *                 Any other value (including empty) returns an invalid bracket,
 *                 signalling the caller to fall back to legacy step-index cycling.
 * @param tintalgo Time-interpolation algorithm: "linear" enables interpolation
 *                 for the (continuous) monthly cadence; anything else -> nearest.
 * @param dt       Parsed simulation datetime.
 * @param file_nt  Number of records available in the file (for clamping).
 *
 * Hourly and weekly cadences select discrete profile records (hour-of-day,
 * day-of-week) and are always nearest-neighbour: interpolating between, say,
 * two day-type weights is not physically meaningful. Only the monthly cadence
 * honours @c tintalgo, using the mid-month convention so that, e.g., Jan 1 is
 * interpolated between the December and January climatological records.
 */
RecordBracket cadence_record_bracket(const std::string& cadence, const std::string& tintalgo, const SimDateTime& dt, int file_nt) {
    RecordBracket br;
    if (cadence.empty() || !dt.valid) return br;

    std::string c = cadence;
    std::transform(c.begin(), c.end(), c.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::string algo = tintalgo;
    std::transform(algo.begin(), algo.end(), algo.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    const bool linear = (algo == "linear");

    auto clamp_idx = [&](int idx) {
        if (file_nt > 0 && idx >= file_nt) idx = file_nt - 1;
        if (idx < 0) idx = 0;
        return idx;
    };

    if (c == "hourly") {
        br.i0 = br.i1 = clamp_idx(dt.hour);  // 0-23, discrete of-day profile
        br.valid = true;
    } else if (c == "weekly") {
        br.i0 = br.i1 = clamp_idx(dt.day_of_week);  // 0=Sunday..6=Saturday, discrete day-type
        br.valid = true;
    } else if (c == "monthly") {
        const int m = dt.month - 1;  // 0-11
        if (!linear) {
            br.i0 = br.i1 = clamp_idx(m);
            br.valid = true;
            return br;
        }
        // Mid-month convention: each monthly record is valid at the midpoint of
        // its month. Interpolate between the two records whose anchors bracket
        // the current instant, cycling across the Dec<->Jan boundary.
        const int dim = tick::Gregorian_Calendar::days_in_month(dt.year, dt.month);
        const double frac = (static_cast<double>(dt.day - 1) + dt.hour / 24.0) / static_cast<double>(dim);  // [0,1)
        const int nrec = (file_nt > 0) ? file_nt : 12;
        if (frac >= 0.5) {
            br.i0 = m % nrec;
            br.i1 = (m + 1) % nrec;
            br.weight = frac - 0.5;  // 0 at mid-month, ->0.5 approaching next anchor
        } else {
            br.i0 = (m - 1 + nrec) % nrec;
            br.i1 = m % nrec;
            br.weight = frac + 0.5;  // ->1 at mid-month, 0.5 just after previous anchor
        }
        br.valid = true;
    }
    return br;
}

bool collective_all_ready(MPI_Comm comm, bool local_ready, const std::string& context, std::string& failure_detail) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized || comm == MPI_COMM_NULL) {
        if (!local_ready && failure_detail.empty()) failure_detail = context;
        return local_ready;
    }

    int mpi_size = 1;
    MPI_Comm_size(comm, &mpi_size);
    if (mpi_size <= 1) {
        if (!local_ready && failure_detail.empty()) failure_detail = context;
        return local_ready;
    }

    const int local_value = local_ready ? 1 : 0;
    int global_value = 0;
    const int rc = MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN, comm);
    if (rc != MPI_SUCCESS) {
        failure_detail = "MPI_Allreduce failed while synchronizing " + context + " (error code " + std::to_string(rc) + ")";
        return false;
    }
    if (global_value != 1) {
        if (failure_detail.empty()) failure_detail = context + " failed on one or more ranks";
        return false;
    }
    return true;
}

bool collective_int_matches(MPI_Comm comm, int local_value, const std::string& name, std::string& failure_detail) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized || comm == MPI_COMM_NULL) return true;

    int mpi_size = 1;
    MPI_Comm_size(comm, &mpi_size);
    if (mpi_size <= 1) return true;

    int minimum = 0;
    int maximum = 0;
    const int min_rc = MPI_Allreduce(&local_value, &minimum, 1, MPI_INT, MPI_MIN, comm);
    const int max_rc = MPI_Allreduce(&local_value, &maximum, 1, MPI_INT, MPI_MAX, comm);
    if (min_rc != MPI_SUCCESS || max_rc != MPI_SUCCESS) {
        failure_detail = "MPI_Allreduce failed while comparing " + name + " across ranks";
        return false;
    }
    if (minimum != maximum) {
        failure_detail = name + " differs across ranks (minimum " + std::to_string(minimum) + ", maximum " + std::to_string(maximum) + ")";
        return false;
    }
    return true;
}

}  // namespace

void CeceDriverOrchestrator::ResolveStreamConfigs() {
    // Thin wrapper: run the pure YAML->StreamConfig resolution against
    // config_file_ and store the results into this orchestrator's members.
    // The actual resolution lives in the static ResolveStreamConfigsFromFile so
    // it can be exercised in isolation by tests (Property 4) via the SAME code
    // path without constructing a full orchestrator (which requires MPI/DAGR/
    // CeceIO setup). Production behavior is unchanged.
    ResolveStreamConfigsFromFile(config_file_, stream_configs_, gridspec_file_);
}

void CeceDriverOrchestrator::ResolveStreamConfigsFromFile(const std::string& config_file,
                                                          std::unordered_map<std::string, StreamConfig>& out_configs,
                                                          std::string& out_gridspec_file) {
    // Parse config_file exactly once. On a YAML error keep the orchestrator
    // robust (out_gridspec_file = "", out_configs left empty) so the existing
    // collective gates in AdvanceTime surface any downstream problem, matching
    // the legacy inline parse which simply found no matching variable.
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_file);
    } catch (const YAML::Exception& e) {
        out_gridspec_file = "";
        return;
    }

    // Driver-level values shared by every StreamConfig. Preserve the existing
    // constructor validation: a < 1 value is rejected with std::invalid_argument
    // and the same message.
    int amio_worker_threads = 1;
    int amio_staging_buffer_count = 8;
    if (config["driver"]) {
        if (config["driver"]["gridspec_file"]) {
            out_gridspec_file = config["driver"]["gridspec_file"].as<std::string>();
        }
        if (config["driver"]["amio_worker_threads"]) {
            amio_worker_threads = config["driver"]["amio_worker_threads"].as<int>();
            if (amio_worker_threads < 1) {
                throw std::invalid_argument("driver.amio_worker_threads must be >= 1; got " + std::to_string(amio_worker_threads) + ".");
            }
        }
        if (config["driver"]["amio_staging_buffer_count"]) {
            amio_staging_buffer_count = config["driver"]["amio_staging_buffer_count"].as<int>();
            if (amio_staging_buffer_count < 1) {
                throw std::invalid_argument("driver.amio_staging_buffer_count must be >= 1; got " + std::to_string(amio_staging_buffer_count) + ".");
            }
        }
    }

    if (!config["cece_data"] || !config["cece_data"]["streams"]) {
        return;
    }

    // Walk every stream and populate a StreamConfig for each model variable,
    // keyed by model name. Field resolution + defaults mirror the legacy inline
    // AdvanceTime parse exactly.
    for (const auto& stream : config["cece_data"]["streams"]) {
        for (const auto& var : stream["variables"]) {
            std::string model_name;
            std::string file_name;
            if (var.IsScalar()) {
                model_name = var.as<std::string>();
                file_name = model_name;
            } else if (var.IsMap() && var["model"]) {
                model_name = var["model"].as<std::string>();
                file_name = var["file"] ? var["file"].as<std::string>() : model_name;
            } else {
                continue;
            }

            StreamConfig cfg;
            // Missing file path is recorded as empty (not thrown); the existing
            // collective gate in AdvanceTime surfaces it later (Req 1.4).
            if (stream["file"]) {
                cfg.input_file_path = stream["file"].as<std::string>();
            }
            cfg.input_var_name = file_name;
            if (stream["mapalgo"]) {
                cfg.mapalgo = stream["mapalgo"].as<std::string>();
            }
            if (stream["cadence"]) {
                cfg.cadence = stream["cadence"].as<std::string>();
            }
            if (stream["tintalgo"]) {
                cfg.tintalgo = stream["tintalgo"].as<std::string>();
            }
            if (stream["data_model"]) {
                std::string requested_model = stream["data_model"].as<std::string>();
                std::transform(requested_model.begin(), requested_model.end(), requested_model.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (requested_model == "classic" || requested_model == "enhanced") {
                    cfg.data_model = requested_model;
                    cfg.data_model_explicit = true;
                } else if (requested_model == "auto") {
                    cfg.data_model = "enhanced";
                    cfg.data_model_explicit = false;
                } else {
                    CECE_LOG_WARNING("[DRIVER] Invalid stream data_model='" + requested_model + "' for stream variable '" + model_name +
                                     "'; using default auto behavior (enhanced then classic fallback).");
                    cfg.data_model = "enhanced";
                    cfg.data_model_explicit = false;
                }
            }
            cfg.amio_worker_threads = amio_worker_threads;
            cfg.amio_staging_buffer_count = amio_staging_buffer_count;

            // Match the legacy inline parse's first-match-wins behavior: it
            // stopped at the first stream/variable matching the model name.
            out_configs.emplace(model_name, std::move(cfg));
        }
    }
}

std::string CeceDriverOrchestrator::BuildManifestContent(const StreamConfig& cfg, const std::string& data_model) const {
    // Mirror the exact keys, order, values, newlines, and indentation the
    // legacy inline AdvanceTime writer produced (the `m_file << "backend: ..."`
    // block), just to a string instead of a file, so the resulting manifest is
    // byte-for-byte identical (Req 2.3, 2.5, 2.6).
    std::ostringstream m_content;
    m_content << "backend: netcdf4\n"
              << "path: " << cfg.input_file_path << "\n"
              << "data_model: " << data_model << "\n"
              << "staging_pool:\n"
              << "  buffer_count: " << cfg.amio_staging_buffer_count << "\n"
              << "  buffer_capacity_bytes: 268435456\n"
              << "worker_pool:\n"
              << "  threads: " << cfg.amio_worker_threads << "\n"
              << "prefetch:\n"
              << "  depth: 2\n"
              << "  read_timeout_s: 120\n"
              << "staging_timeout_ms: 30000\n";
    return m_content.str();
}

std::string CeceDriverOrchestrator::StreamKey(const StreamConfig& cfg) {
    // Stream identity key = HandleKey extended by mapalgo. It keys the regrid
    // plan cache (regrid_plans_): variables sharing a HandleKey but requesting
    // a different mapalgo get distinct StreamKeys and therefore distinct plans,
    // while everything else is shared at the coarser HandleKey level (Req 11.5).
    return HandleKey(cfg) + "|" + cfg.mapalgo;
}

std::string CeceDriverOrchestrator::HandleKey(const StreamConfig& cfg) {
    // File/manifest-scoped identity key: it concatenates exactly the
    // StreamConfig fields that BuildManifestContent consumes
    // (input_file_path, data_model, amio_worker_threads,
    // amio_staging_buffer_count), so two configs share a HandleKey iff they
    // would produce a byte-identical AMIO manifest. data_model here is the
    // resolved pre-open value (identical across ranks). The "|" separator
    // cannot appear in NetCDF file paths, so the concatenation is unambiguous
    // (Req 11.1). Variables reading the same file/manifest share one open AMIO
    // handle set and one file record count even when mapalgo differs.
    return cfg.input_file_path + "|" + cfg.data_model + "|" +
           std::to_string(cfg.amio_worker_threads) + "|" +
           std::to_string(cfg.amio_staging_buffer_count);
}

bool CeceDriverOrchestrator::bracket_equal(const RecordBracket& a, const RecordBracket& b) {
    // Two resolved brackets are equal when they select the same record indices
    // and blend weight (within tolerance). The `valid` field is intentionally
    // NOT compared: bracket_equal answers "does b select the same slice as a"
    // for two already-resolved brackets (Req 3.5, 6.3).
    return a.i0 == b.i0 && a.i1 == b.i1 && std::fabs(a.weight - b.weight) <= kBracketWeightTol;
}

AmioHandleSet* CeceDriverOrchestrator::GetOrOpenHandleSet(const std::string& handle_key, const StreamConfig& cfg, std::string& failure_detail) {
    // Lazy-open-once: if the handle set already exists for this
    // Handle_Identity_Key, reuse it without any re-open. Variables that read the
    // same file/manifest share this single open handle set even when their
    // mapalgo differs (Req 2.2, 3.1, 3.2, 7.1, 11.2, 11.7).
    auto existing = amio_handles_.find(handle_key);
    if (existing != amio_handles_.end()) {
        return &existing->second;
    }

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    // Candidate data models, mirroring the legacy AdvanceTime fallback ordering
    // (Req 2.5, 2.6).
    std::vector<std::string> data_models_to_try;
    if (cfg.data_model_explicit) {
        data_models_to_try.push_back(cfg.data_model);
    } else {
        data_models_to_try.push_back("enhanced");
        data_models_to_try.push_back("classic");
    }

    for (const auto& candidate_model : data_models_to_try) {
        // Build the manifest in memory once per candidate; no file is written
        // to disk (Req 9.2).
        const std::string manifest_content = BuildManifestContent(cfg, candidate_model);

        amio_core_handle read_core = nullptr;
        amio_dataset_handle read_dataset = nullptr;

        // Force serial I/O fallback for reading offline datasets to prevent MPI
        // multithreading deadlocks. Only the open is wrapped in the swap.
        if (mpi_initialized) {
            amio_set_parent_communicator(MPI_Comm_c2f(MPI_COMM_SELF));
        }

        amio_status_t amio_rc = amio_init_from_string(manifest_content.c_str(), "yaml", &read_core);
        if (amio_rc != AMIO_OK) {
            failure_detail = std::string("amio_init_from_string failed for handle '") + handle_key + "': rc=" + std::to_string(amio_rc) + " (" +
                             amio_strerror(amio_rc) + ")";
        } else {
            amio_rc = amio_open_dataset_from_string(read_core, manifest_content.c_str(), "yaml", AMIO_MODE_READ, &read_dataset);
            if (amio_rc != AMIO_OK) {
                failure_detail = std::string("amio_open_dataset_from_string failed for '") + cfg.input_file_path + "': rc=" + std::to_string(amio_rc) +
                                 " (" + amio_strerror(amio_rc) + ")";
            }
        }

        // Restore parent communicator for downstream operations (match the
        // legacy guards exactly).
        if (mpi_initialized && comm_c_ != MPI_COMM_NULL) {
            amio_set_parent_communicator(MPI_Comm_c2f(comm_c_));
        }

        if (amio_rc == AMIO_OK && read_core != nullptr && read_dataset != nullptr) {
            // Emit the same auto-fallback INFO log the legacy code emitted when
            // a non-explicit stream fell back to a non-"enhanced" model.
            if (!cfg.data_model_explicit && candidate_model != "enhanced") {
                CECE_LOG_INFO("[DRIVER] AMIO read manifest auto-fell back to data_model='" + candidate_model + "' for " + cfg.input_file_path);
            }

            failure_detail.clear();
            AmioHandleSet set;
            set.core = read_core;
            set.dataset = read_dataset;
            set.active_data_model = candidate_model;
            set.manifest_content = manifest_content;
            auto inserted = amio_handles_.emplace(handle_key, std::move(set));
            return &inserted.first->second;
        }

        CECE_LOG_DEBUG("[DRIVER] AMIO open attempt failed (data_model='" + candidate_model + "') with rc = " + std::to_string(amio_rc) + " (" +
                       amio_strerror(amio_rc) + ")");

        // Close/finalize any partially-opened handle for this attempt before
        // trying the next candidate. Nothing is cached on failure (Req 8.1).
        if (read_dataset) {
            amio_close(read_dataset);
            read_dataset = nullptr;
        }
        if (read_core) {
            amio_finalize(read_core);
            read_core = nullptr;
        }
    }

    // All candidates failed: leave failure_detail set, cache nothing.
    CECE_LOG_DEBUG("[DRIVER] amio open failed for " + cfg.input_file_path + " after trying all candidate data models");
    return nullptr;
}

CeceDriverOrchestrator::CeceDriverOrchestrator(const std::string& config_file, int nx, int ny, int nz, const double* lon_coords, int lon_len,
                                               const double* lat_coords, int lat_len, MPI_Comm comm_c)
    : config_file_(config_file),
      nx_(nx),
      ny_(ny),
      nz_(nz),
      target_lons_(lon_coords, lon_coords + lon_len),
      target_lats_(lat_coords, lat_coords + lat_len),
      comm_c_(comm_c) {
    // Parse the YAML once and resolve the StreamConfig for every stream
    // variable, plus driver-level values and gridspec_file_ (Req 1.1). This
    // preserves the previous constructor's driver-block validation (a < 1
    // amio_worker_threads / amio_staging_buffer_count throws) and its
    // YAML::Exception robustness (gridspec_file_ = "" on a bad/missing file).
    ResolveStreamConfigs();

    cece_io_ = std::make_unique<io::CeceIO>();
    cece_io_->Initialize(config_file_, nx_, ny_, nz_);
    CompileHelmGraph(config_file_, dagr_, *cece_io_, comm_c_);

    // Route DAGR's diagnostics through its shared LOGS logger with the same
    // MPI communicator CECE uses, and quiet non-root ranks (they still emit
    // FATAL). Without this, DAGR's logger is unconfigured and every rank prints
    // identical "GraphOrchestrator: shutdown initiated" lines with a [RANK:----]
    // sentinel stamp.
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        int rank = 0;
        if (mpi_initialized && comm_c_ != MPI_COMM_NULL) {
            MPI_Comm_rank(comm_c_, &rank);
        }
        dagr::configure_logging(comm_c_ != MPI_COMM_NULL ? comm_c_ : MPI_COMM_WORLD, rank == 0 ? dagr::Log_Level::info : dagr::Log_Level::error);
    }
}

void CeceDriverOrchestrator::TeardownHandles() {
    // Release every retained AMIO handle set. Close the dataset first, then
    // finalize the core, each wrapped in its own best-effort try/catch so one
    // failing handle does not prevent the rest from tearing down (Req 7.2, 7.4).
    // amio_handles_ is now keyed by Handle_Identity_Key, so this generic loop
    // closes each shared handle set exactly once regardless of how many
    // variables shared it (Req 3.6, 7.4).
    for (auto& entry : amio_handles_) {
        AmioHandleSet& set = entry.second;
        if (set.dataset != nullptr) {
            try {
                amio_close(set.dataset);
            } catch (const std::exception& e) {
                CECE_LOG_DEBUG("[DRIVER] amio_close threw during teardown for '" + entry.first + "': " + e.what());
            } catch (...) {
                CECE_LOG_DEBUG("[DRIVER] amio_close threw an unknown exception during teardown for '" + entry.first + "'");
            }
            set.dataset = nullptr;
        }
        if (set.core != nullptr) {
            try {
                amio_finalize(set.core);
            } catch (const std::exception& e) {
                CECE_LOG_DEBUG("[DRIVER] amio_finalize threw during teardown for '" + entry.first + "': " + e.what());
            } catch (...) {
                CECE_LOG_DEBUG("[DRIVER] amio_finalize threw an unknown exception during teardown for '" + entry.first + "'");
            }
            set.core = nullptr;
        }
    }

    // No manifest files to delete: manifests are in-memory strings, never
    // written to disk (Req 7.3). Drop the loop-invariant caches (Req 7.5).
    amio_handles_.clear();
    stream_configs_.clear();
    slice_caches_.clear();
}

CeceDriverOrchestrator::~CeceDriverOrchestrator() {
    // Release retained AMIO handle sets first, before tearing down the pipeline
    // (Req 7.2-7.5).
    TeardownHandles();

    // Cleanly drain any in-flight pipeline tasks and release hijacked ranks
    // before destroying the graph. Without this, tearing down the DAGR
    // GraphOrchestrator while a task is still in flight races with the
    // Event_Loop worker(s) and can segfault at teardown. shutdown() is
    // idempotent and safe to call here.
    if (dagr_) {
        dagr_->shutdown();
    }
    dagr_.reset();
    cece_io_.reset();
}

bool CeceDriverOrchestrator::AssembleReplicatedField(const std::string& var_name, const io::RegridPlan& plan, const std::vector<double>& source,
                                                     int file_nx, int file_ny, int field_nlev, DeviceView3D stream_view, void* cece_core_data_ptr,
                                                     std::vector<double>& ingest_buffer, std::string& failure_detail) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    int mpi_size = 1;
    int mpi_rank = 0;
    if (mpi_initialized && comm_c_ != MPI_COMM_NULL) {
        MPI_Comm_size(comm_c_, &mpi_size);
        MPI_Comm_rank(comm_c_, &mpi_rank);
    }
    const bool distributed_regrid = mpi_initialized && mpi_size > 1 && comm_c_ != MPI_COMM_NULL;
    const int band_base = ny_ / mpi_size;
    const int band_rem = ny_ % mpi_size;
    auto band_start = [&](int rank) { return rank * band_base + std::min(rank, band_rem); };
    const int expected_j0 = band_start(mpi_rank);
    const int expected_j1 = band_start(mpi_rank + 1);

    // This must be the helper's first distributed gate. Every caller reaches
    // it before any rank-local return, so a bad source buffer or inconsistent
    // AMIO metadata cannot leave peer ranks waiting in the layer collectives.
    const bool positive_dimensions = file_nx > 0 && file_ny > 0 && field_nlev > 0 && nx_ > 0 && ny_ > 0;
    const size_t source_spatial = positive_dimensions ? static_cast<size_t>(file_nx) * file_ny : 0;
    const size_t expected_source_size = positive_dimensions ? static_cast<size_t>(field_nlev) * source_spatial : 0;
    const bool local_source_ready = positive_dimensions && cece_core_data_ptr != nullptr && plan.built && plan.file_nx == file_nx &&
                                    plan.file_ny == file_ny && plan.j0 == expected_j0 && plan.j1 == expected_j1 &&
                                    source.size() == expected_source_size && stream_view.extent(0) == static_cast<size_t>(nx_) &&
                                    stream_view.extent(1) == static_cast<size_t>(ny_) && stream_view.extent(2) == static_cast<size_t>(field_nlev);
    if (!local_source_ready && failure_detail.empty()) {
        failure_detail = "source buffer or regrid metadata changed after AMIO validation";
    }
    if (!collective_all_ready(comm_c_, local_source_ready, "source and regrid metadata readiness", failure_detail)) return false;
    if (!collective_int_matches(comm_c_, file_nx, "source longitude count", failure_detail) ||
        !collective_int_matches(comm_c_, file_ny, "source latitude count", failure_detail) ||
        !collective_int_matches(comm_c_, field_nlev, "source level count", failure_detail) ||
        !collective_int_matches(comm_c_, plan.identity ? 1 : 0, "regrid-plan identity mode", failure_detail)) {
        return false;
    }

    const size_t target_spatial = static_cast<size_t>(nx_) * ny_;

    std::vector<int> counts;
    std::vector<int> displs;
    if (distributed_regrid) {
        counts.resize(mpi_size);
        displs.resize(mpi_size);
        for (int rank = 0; rank < mpi_size; ++rank) {
            counts[rank] = (band_start(rank + 1) - band_start(rank)) * nx_;
            displs[rank] = band_start(rank) * nx_;
        }
    }

    // Regridding is decomposed into rank-local latitude bands, but CECE's
    // current downstream contract is replicated: stream_view,
    // CeceImportState, the ingestor cache, physics, and output all consume
    // nx_ x ny_ x levels fields on every rank. Assemble each layer globally
    // before populating those views. Removing this collective requires a
    // coordinated distributed-state/output redesign rather than a local
    // driver change.
    std::vector<double> full_destination(static_cast<size_t>(field_nlev) * target_spatial, 0.0);
    for (int level = 0; level < field_nlev; ++level) {
        std::vector<double> local_destination;
        const double* source_layer = source.data() + static_cast<size_t>(level) * source_spatial;
        bool local_regrid_succeeded = false;
        try {
            local_regrid_succeeded =
                cece::io::apply_regrid_plan(plan, /*time_offset=*/0, /*is_float=*/false, source_layer, file_nx, file_ny, nx_, local_destination);
        } catch (const std::exception& error) {
            failure_detail = "regrid weight application threw an exception: " + std::string(error.what());
        } catch (...) {
            failure_detail = "regrid weight application threw an unknown exception";
        }
        const size_t expected_local_size = static_cast<size_t>(plan.j1 - plan.j0) * nx_;
        const bool gather_count_matches = !distributed_regrid || expected_local_size == static_cast<size_t>(counts[mpi_rank]);
        const bool local_layer_ready = local_regrid_succeeded && local_destination.size() == expected_local_size && gather_count_matches;
        bool all_ranks_ready = local_layer_ready;

        // All ranks must make the same decision before entering the gather. A
        // rank-local early exit here would strand peers in MPI_Allgatherv.
        if (distributed_regrid) {
            const int local_ready = local_layer_ready ? 1 : 0;
            int global_ready = 0;
            const int ready_rc = MPI_Allreduce(&local_ready, &global_ready, 1, MPI_INT, MPI_MIN, comm_c_);
            if (ready_rc != MPI_SUCCESS) {
                failure_detail = "MPI_Allreduce failed while validating rank-local regrid bands (error code " + std::to_string(ready_rc) + ")";
                CECE_LOG_DEBUG("[DRIVER] rank-local regrid or replicated-field assembly failed!");
                return false;
            }
            all_ranks_ready = global_ready == 1;
        }

        if (!all_ranks_ready) {
            if (failure_detail.empty()) {
                failure_detail = "rank-local regrid failed or produced an unexpected destination-band size";
            }
            CECE_LOG_DEBUG("[DRIVER] rank-local regrid or replicated-field assembly failed!");
            return false;
        }

        double* destination_layer = full_destination.data() + static_cast<size_t>(level) * target_spatial;
        if (distributed_regrid) {
            const int gather_rc = MPI_Allgatherv(local_destination.data(), counts[mpi_rank], MPI_DOUBLE, destination_layer, counts.data(),
                                                 displs.data(), MPI_DOUBLE, comm_c_);
            const int local_gather_ok = gather_rc == MPI_SUCCESS ? 1 : 0;
            int global_gather_ok = 0;
            const int gather_status_rc = MPI_Allreduce(&local_gather_ok, &global_gather_ok, 1, MPI_INT, MPI_MIN, comm_c_);
            if (gather_status_rc != MPI_SUCCESS || global_gather_ok != 1) {
                if (gather_status_rc != MPI_SUCCESS) {
                    failure_detail =
                        "MPI_Allreduce failed while synchronizing MPI_Allgatherv status (error code " + std::to_string(gather_status_rc) + ")";
                } else {
                    failure_detail =
                        "MPI_Allgatherv failed on one or more ranks while assembling the replicated destination field "
                        "(local error code " +
                        std::to_string(gather_rc) + ")";
                }
                CECE_LOG_DEBUG("[DRIVER] rank-local regrid or replicated-field assembly failed!");
                return false;
            }
        } else {
            std::copy(local_destination.begin(), local_destination.end(), destination_layer + static_cast<size_t>(plan.j0) * nx_);
        }
    }

    // Transpose the gathered [level][j][i] full_destination into the LayoutLeft
    // (i, j, level) DualView layout exactly ONCE, into a single host mirror
    // buffer. This one buffer feeds the sole live consumer below (the core
    // import field) via deep_copy. The index math is unchanged:
    //   host(i, j, level) = full_destination[level*nx*ny + j*nx + i].
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace> transposed_host("assembled_field_host", nx_, ny_, field_nlev);
    for (int level = 0; level < field_nlev; ++level) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                transposed_host(i, j, level) = full_destination[static_cast<size_t>(level) * target_spatial + static_cast<size_t>(j) * nx_ + i];
            }
        }
    }

    // Also populate the Core import state with the same full field.
    auto* data = static_cast<cece::CeceInternalData*>(cece_core_data_ptr);
    auto core_it = data->import_state.fields.find(var_name);
    if (core_it == data->import_state.fields.end()) {
        cece::DualView3D field(var_name, nx_, ny_, field_nlev);
        data->import_state.fields[var_name] = field;
        core_it = data->import_state.fields.find(var_name);
    }

    auto& core_field = core_it->second;
    auto core_view = core_field.view_device();
    const bool local_core_shape_ready = core_view.extent(0) == static_cast<size_t>(nx_) && core_view.extent(1) == static_cast<size_t>(ny_) &&
                                        core_view.extent(2) == static_cast<size_t>(field_nlev);
    if (!local_core_shape_ready) {
        failure_detail = "core import field shape mismatch for '" + var_name + "': expected " + std::to_string(nx_) + "x" + std::to_string(ny_) +
                         "x" + std::to_string(field_nlev) + ", found " + std::to_string(core_view.extent(0)) + "x" +
                         std::to_string(core_view.extent(1)) + "x" + std::to_string(core_view.extent(2));
    }
    if (!collective_all_ready(comm_c_, local_core_shape_ready, "core import field shape validation", failure_detail)) return false;

    // Authoritative write of the core import field from the single host buffer.
    // This is now the SOLE authoritative write of the assembled field. The
    // former stream_view populate (Tier 2, task 9.1) has been removed: the
    // Finding-B read-site enumeration proved no consumer reads the data stored
    // in CeceIO::field_views_ / stream_view within a step (it is used only for
    // .extent(...) shape/metadata queries and the readiness gate above, both of
    // which are preserved). Dropping the stream_view deep_copy removes one
    // per-field-per-step copy without changing any value seen by a live
    // consumer, any layout, or any collective gate.
    Kokkos::deep_copy(core_view, transposed_host);
    core_field.modify_device();
    core_field.sync_host();

    ingest_buffer = std::move(full_destination);
    return true;
}

bool CeceDriverOrchestrator::AdvanceTime(const std::string& time_iso8601, void* cece_core_data_ptr) {
    std::string core_readiness_detail;
    if (!collective_all_ready(comm_c_, cece_core_data_ptr != nullptr, "CECE core-data readiness", core_readiness_detail)) {
        CECE_LOG_ERROR("[DRIVER FATAL] " + core_readiness_detail);
        return false;
    }

    // A. Advance the pipeline step
    dagr_->advance_step();
    Kokkos::fence();

    // Configuration is parsed exactly once at construction (Req 9.3); AdvanceTime
    // consults the resolved StreamConfig via stream_configs_ and never re-reads
    // the YAML file from disk.
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    int mpi_rank = 0;
    int mpi_size = 1;
    if (mpi_initialized && comm_c_ != MPI_COMM_NULL) {
        MPI_Comm_rank(comm_c_, &mpi_rank);
        MPI_Comm_size(comm_c_, &mpi_size);
    }

    // Parse the current simulation datetime once. Streams that declare a
    // temporal cadence (hourly/weekly/monthly) use these calendar fields to
    // select the correct file record; streams without a cadence keep the
    // legacy step-index cycling behaviour and ignore this.
    const SimDateTime sim_dt = parse_sim_datetime(time_iso8601);

    // B. Push CeceIO's newly computed emission views into CECE's data ingestor
    for (const auto& var_name : cece_io_->GetOutputVarNames()) {
        auto stream_view = cece_io_->GetFieldView(var_name);
        const int field_nlev = static_cast<int>(stream_view.extent(2));
        // Human-readable reason for the most recent read failure, propagated to
        // the fatal error message so the underlying AMIO status reaches CECE.
        std::string failure_detail;
        if (field_nlev < 1) {
            CECE_LOG_ERROR("[DRIVER FATAL] Field '" + var_name + "' has no configured levels");
            failure_detail = "field has no configured levels";
        }
        if (!collective_all_ready(comm_c_, field_nlev >= 1, "field-level metadata readiness for '" + var_name + "'", failure_detail)) return false;
        std::vector<double> ingest_buffer;
        bool read_success = false;

        // Read the resolved StreamConfig for this variable from the map filled
        // once at construction (Req 1.2, 9.3). A variable with no StreamConfig
        // entry is treated exactly like a stream whose input_file_path is empty:
        // the missing-configuration error is surfaced through the same
        // collective gate the empty-path case used, so all ranks agree (Req 1.4).
        auto cfg_it = stream_configs_.find(var_name);
        const bool has_stream_config = cfg_it != stream_configs_.end();
        const StreamConfig cfg = has_stream_config ? cfg_it->second : StreamConfig{};

        std::string input_file_path = cfg.input_file_path;
        std::string input_var_name = cfg.input_var_name;
        const std::string& mapalgo = cfg.mapalgo;
        const std::string& cadence = cfg.cadence;
        const std::string& tintalgo = cfg.tintalgo;

        // TWO-LEVEL keying for the shared caches:
        //   - handle_key (file/manifest-scoped) keys amio_handles_ and
        //     file_nt_cache_. Variables reading the same file/manifest share
        //     one open AMIO handle set and one record-count search even when
        //     their mapalgo differs (Req 11.2, 11.3).
        //   - stream_key (= handle_key + "|" + mapalgo) keys regrid_plans_.
        //     The plan cache splits only when mapalgo differs; everything else
        //     stays shared at the coarser handle_key level (Req 11.5, 11.6).
        //   - slice_caches_ and cece_ingestor_set_field remain keyed by
        //     var_name (Req 5, 12.1).
        // Both keys are identical on every rank because StreamConfig is
        // resolved identically from the same YAML (Req 8.1, 11.8).
        const std::string handle_key = HandleKey(cfg);
        const std::string stream_key = StreamKey(cfg);

        if (input_file_path.empty()) {
            CECE_LOG_ERROR("[DRIVER FATAL] Input file path not specified for stream variable '" + var_name + "' in configuration!");
            failure_detail = "input file path is not configured";
        }
        if (input_var_name.empty()) {
            input_var_name = var_name;
        }
        if (!collective_all_ready(comm_c_, !input_file_path.empty(), "stream configuration readiness for '" + var_name + "'", failure_detail)) {
            return false;
        }

        // Verify if the input file path exists and is accessible from this compute/login node
        std::error_code fs_ec;
        const bool local_file_ready = fs::exists(input_file_path, fs_ec);
        if (!local_file_ready) {
            CECE_LOG_ERROR("[DRIVER FATAL] File '" + input_file_path +
                           "' does not exist or is unreadable on this node! (System error: " + fs_ec.message() + ")");
            failure_detail = "input file does not exist or is unreadable: " + fs_ec.message();
        } else {
            CECE_LOG_DEBUG("[DRIVER] Input file '" + input_file_path + "' successfully verified on local filesystem.");
        }
        if (!collective_all_ready(comm_c_, local_file_ready, "input-file readiness for '" + var_name + "'", failure_detail)) return false;

        // Dynamically open and read using AMIO API
        // First-touch open (once per variable) via GetOrOpenHandleSet, which
        // builds the in-memory manifest, performs the MPI_COMM_SELF swap, and
        // tries the candidate data models. On subsequent steps it returns the
        // cached handle set with no re-open, no manifest write, and no barrier
        // (Req 2.1, 2.2, 9.1, 9.2). The dataset-open readiness collective still
        // runs every step so ranks agree that a usable handle set exists; the
        // actual open work only happens on the first touch.
        AmioHandleSet* handle_set = GetOrOpenHandleSet(handle_key, cfg, failure_detail);
        const bool local_open_ready = handle_set != nullptr && handle_set->dataset != nullptr && handle_set->core != nullptr;
        const bool amio_open_ready = collective_all_ready(comm_c_, local_open_ready, "AMIO dataset open for '" + var_name + "'", failure_detail);

        if (!amio_open_ready) {
            CECE_LOG_DEBUG("[DRIVER] amio open failed for " + input_file_path + ": " + failure_detail);
        } else {
            failure_detail.clear();
            // Use the retained dataset handle for record discovery, plan
            // building, and reads. It persists in amio_handles_ across timesteps
            // and is torn down in the destructor (task 10.1).
            amio_dataset_handle read_dataset = handle_set->dataset;

            // Determine this rank's contiguous destination latitude band [j0, j1)
            // via a simple block decomposition of the ny_ destination rows.
            const int band_base = ny_ / mpi_size;
            const int band_rem = ny_ % mpi_size;
            auto band_start = [&](int r) { return r * band_base + std::min(r, band_rem); };
            const int j0 = band_start(mpi_rank);
            const int j1 = band_start(mpi_rank + 1);

            // 1. Determine total timesteps from the input variable.
            //    Since AMIO doesn't expose a public function to query total timesteps,
            //    we use a binary search with amio_read on the input variable to identify
            //    the actual record limit (since reads beyond the record limit return AMIO_ERR_INVALID_INPUT).
            //    We cache the result in file_nt_cache_ to avoid binary search overhead on subsequent steps.
            int file_nt = 1;
            auto nt_it = file_nt_cache_.find(handle_key);
            if (nt_it != file_nt_cache_.end()) {
                file_nt = nt_it->second;
            } else {
                if (!input_var_name.empty()) {
                    int low = 1;
                    int high = 1000000;
                    int found_nt = 1;
                    while (low <= high) {
                        int mid = low + (high - low) / 2;
                        amio_view_handle v = nullptr;
                        amio_status_t rc = amio_read(read_dataset, input_var_name.c_str(), mid, nullptr, &v);
                        if (rc == AMIO_OK) {
                            amio_release_view(v);
                            found_nt = mid + 1;
                            low = mid + 1;
                        } else {
                            high = mid - 1;
                        }
                    }
                    file_nt = found_nt;
                }
                file_nt_cache_[handle_key] = file_nt;
            }
            bool file_records_ready =
                collective_all_ready(comm_c_, file_nt > 0, "AMIO record-count readiness for '" + var_name + "'", failure_detail);
            if (file_records_ready) {
                file_records_ready = collective_int_matches(comm_c_, file_nt, "AMIO record count for '" + var_name + "'", failure_detail);
            }

            // 2. Build (or reuse cached) interpolation weights for this rank's band.
            //    Weights depend only on the grids, so they are generated once and
            //    reused for every timestep.
            auto plan_it = regrid_plans_.find(stream_key);
            if (file_records_ready && (plan_it == regrid_plans_.end() || !plan_it->second.built)) {
                cece::io::RegridPlan plan;
                // An explicit passthrough is safe without reopening coordinate
                // variables only when the stream and gridspec resolve to the
                // exact same file.  main.cpp has already loaded and validated
                // the target coordinates from that file; read_slab below still
                // verifies every field's horizontal size before copying.
                std::error_code equivalent_ec;
                const bool exact_source_target_file = mapalgo == "passthrough" && !gridspec_file_.empty() &&
                                                      fs::equivalent(fs::path(input_file_path), fs::path(gridspec_file_), equivalent_ec) &&
                                                      !equivalent_ec;

                if (exact_source_target_file) {
                    plan.j0 = j0;
                    plan.j1 = j1;
                    plan.file_nx = nx_;
                    plan.file_ny = ny_;
                    plan.identity = true;
                    plan.built = true;
                    CECE_LOG_INFO("[DRIVER] passthrough verified stream file equals explicit gridspec file for '" + var_name +
                                  "'; using exact cell copy");
                    plan_it = regrid_plans_.emplace(stream_key, std::move(plan)).first;
                } else {
                    bool local_plan_built = false;
                    try {
                        local_plan_built =
                            cece::io::build_regrid_plan(read_dataset, nx_, ny_, target_lons_, target_lats_, mapalgo, j0, j1, gridspec_file_, plan);
                    } catch (const std::exception& error) {
                        failure_detail = "regrid plan construction threw an exception: " + std::string(error.what());
                    } catch (...) {
                        failure_detail = "regrid plan construction threw an unknown exception";
                    }
                    if (!local_plan_built) {
                        CECE_LOG_DEBUG("[DRIVER] build_regrid_plan failed for '" + var_name + "'");
                        if (failure_detail.empty()) {
                            failure_detail = "regrid plan construction failed (could not read source grid coordinates)";
                        }
                    } else {
                        plan_it = regrid_plans_.emplace(stream_key, std::move(plan)).first;
                    }
                }
            }
            const bool local_plan_ready = file_records_ready && plan_it != regrid_plans_.end() && plan_it->second.built;
            const bool all_plans_ready =
                collective_all_ready(comm_c_, local_plan_ready, "regrid-plan readiness for '" + var_name + "'", failure_detail);

            // 3. Read the bracketing record(s) for this timestep, blend in time on
            //    the SOURCE grid, then regrid ONCE. Because regridding is a linear
            //    operator, interpolating in time before space is mathematically
            //    identical to the reverse, but it costs a single regrid apply (not
            //    two) and keeps fill-value handling on the native grid.
            //
            //    The record bracket comes from the stream's temporal cadence:
            //      - no cadence declared  -> legacy step-index cycling (single read)
            //      - hourly / weekly      -> nearest discrete profile record
            //      - monthly + tintalgo=linear -> mid-month linear interpolation
            //        between the two bracketing climatological records.
            if (all_plans_ready) {
                const cece::io::RegridPlan& plan = plan_it->second;

                RecordBracket bracket = cadence_record_bracket(cadence, tintalgo, sim_dt, file_nt);
                if (!bracket.valid) {
                    const int t_idx = (file_nt > 0) ? (step_index_ % file_nt) : 0;
                    bracket.i0 = bracket.i1 = t_idx;
                    bracket.weight = 0.0;
                }
                const bool needs_upper_record = bracket.i1 != bracket.i0 && bracket.weight > 0.0;
                // Run the record-index / interp-mode collectives EVERY step so
                // all ranks resolve an identical bracket and therefore make the
                // same hit/miss decision (Req 6.3, 6.4).
                const bool bracket_ready = collective_int_matches(comm_c_, bracket.i0, "lower AMIO record index", failure_detail) &&
                                           collective_int_matches(comm_c_, bracket.i1, "upper AMIO record index", failure_detail) &&
                                           collective_int_matches(comm_c_, needs_upper_record ? 1 : 0, "AMIO interpolation mode", failure_detail);

                // Cache decision (Req 3, 5, 9.4): if the previously computed
                // result was for the same bracket, reuse its ingest buffer and
                // skip the disk read AND the regrid apply entirely. Because the
                // resolved bracket is identical across ranks, every rank makes
                // the same hit/miss decision, so no rank reads while another
                // skips (Req 6.3).
                auto& slice_cache = slice_caches_[var_name];
                const bool cache_hit = bracket_ready && slice_cache.valid && bracket_equal(bracket, slice_cache.last_bracket);

                if (cache_hit) {
                    CECE_LOG_DEBUG("[DRIVER] Reusing cached time slice " + std::to_string(bracket.i0) + " for field '" + var_name +
                                   "' (bracket unchanged; no read, no regrid)");
                    ingest_buffer = slice_cache.ingest_buffer;
                    read_success = true;
                } else {
                // Diagnostic: report which time slice(s) are being read from the file.
                if (bracket.i0 == bracket.i1 || bracket.weight == 0.0) {
                    CECE_LOG_INFO("[DRIVER] Reading time slice " + std::to_string(bracket.i0) + "/" + std::to_string(file_nt - 1) + " from '" +
                                  input_file_path + "' for field '" + var_name + "'" +
                                  (cadence.empty() ? " (cycling, step=" + std::to_string(step_index_) + ")"
                                                   : " (cadence=" + cadence + ", time=" + time_iso8601 + ")"));
                } else {
                    CECE_LOG_INFO("[DRIVER] Interpolating time slices " + std::to_string(bracket.i0) + " & " + std::to_string(bracket.i1) + "/" +
                                  std::to_string(file_nt - 1) + " (w=" + std::to_string(bracket.weight) + ") from '" + input_file_path +
                                  "' for field '" + var_name + "' (cadence=" + cadence + ", tintalgo=" + tintalgo + ", time=" + time_iso8601 + ")");
                }

                // Read one time record into a double buffer on the source grid.
                // AMIO removes the CF time dimension, but any remaining dimensions
                // before [lat, lon] are preserved as per-variable levels.
                auto read_slab = [&](int t_idx, std::vector<double>& out, int& slab_nx, int& slab_ny) -> bool {
                    amio_view_handle slab_view = nullptr;
                    amio_status_t rc = amio_read(read_dataset, input_var_name.c_str(), t_idx, nullptr, &slab_view);
                    if (rc != AMIO_OK) {
                        CECE_LOG_DEBUG("[DRIVER] amio_read('" + input_var_name + "', t=" + std::to_string(t_idx) +
                                       ") failed with rc = " + std::to_string(rc));
                        failure_detail =
                            std::string("amio_read('") + input_var_name + "') failed: rc=" + std::to_string(rc) + " (" + amio_strerror(rc) + ")";
                        return false;
                    }
                    const void* view_data = nullptr;
                    size_t view_size = 0;
                    rc = amio_view_data(slab_view, &view_data, &view_size);
                    if (rc != AMIO_OK) {
                        failure_detail = std::string("amio_view_data failed: rc=") + std::to_string(rc) + " (" + amio_strerror(rc) + ")";
                        amio_release_view(slab_view);
                        return false;
                    }
                    amio_shape_t read_shape{};
                    if (amio_view_shape(slab_view, &read_shape) != AMIO_OK) {
                        failure_detail = "amio_view_shape failed";
                        amio_release_view(slab_view);
                        return false;
                    }
                    if (read_shape.rank < 2) {
                        failure_detail = "AMIO field rank is less than two for '" + input_var_name + "'";
                        amio_release_view(slab_view);
                        return false;
                    }
                    const int fny = static_cast<int>(read_shape.extents[read_shape.rank - 2]);
                    const int fnx = static_cast<int>(read_shape.extents[read_shape.rank - 1]);
                    size_t total_elements = 1;
                    for (int d = 0; d < read_shape.rank; ++d) {
                        total_elements *= read_shape.extents[d];
                    }
                    const size_t spatial = static_cast<size_t>(fny) * fnx;
                    const size_t record_elements = static_cast<size_t>(field_nlev) * spatial;
                    if (spatial == 0 || record_elements == 0 || total_elements % record_elements != 0) {
                        failure_detail =
                            "AMIO field shape is incompatible with configured levels=" + std::to_string(field_nlev) + " for '" + input_var_name + "'";
                        amio_release_view(slab_view);
                        return false;
                    }
                    const size_t records_in_view = total_elements / record_elements;
                    const size_t record_index = records_in_view > 1 ? static_cast<size_t>(t_idx) : 0;
                    if (record_index >= records_in_view) {
                        failure_detail = "AMIO view does not contain requested record " + std::to_string(t_idx) + " for '" + input_var_name + "'";
                        amio_release_view(slab_view);
                        return false;
                    }
                    const size_t offset = record_index * record_elements;

                    const bool is_float = (view_size == total_elements * sizeof(float));
                    const bool is_double = (view_size == total_elements * sizeof(double));
                    if (!is_float && !is_double) {
                        failure_detail = "AMIO returned an unsupported element size for '" + input_var_name + "'";
                        amio_release_view(slab_view);
                        return false;
                    }

                    out.resize(record_elements);
                    if (is_float) {
                        const float* p = static_cast<const float*>(view_data) + offset;
                        for (size_t k = 0; k < record_elements; ++k) out[k] = static_cast<double>(p[k]);
                    } else {
                        const double* p = static_cast<const double*>(view_data) + offset;
                        for (size_t k = 0; k < record_elements; ++k) out[k] = p[k];
                    }
                    slab_nx = fnx;
                    slab_ny = fny;
                    amio_release_view(slab_view);
                    CECE_LOG_DEBUG("[DRIVER] Read slab t=" + std::to_string(t_idx) + " for '" + input_var_name + "': " + std::to_string(fny) + "x" +
                                   std::to_string(fnx) + "x" + std::to_string(field_nlev) + " (" + std::to_string(record_elements) +
                                   " elements, records_in_view=" + std::to_string(records_in_view) + ", " + (is_float ? "float32" : "float64") + ")");
                    return true;
                };

                // Read the lower record and, when interpolating, the upper record;
                // blend on the source grid with the bracket weight.
                std::vector<double> src;
                int file_nx = 0;
                int file_ny = 0;
                const bool local_lower_ready = bracket_ready && read_slab(bracket.i0, src, file_nx, file_ny);
                bool have_data = collective_all_ready(comm_c_, local_lower_ready, "lower AMIO slab readiness for '" + var_name + "'", failure_detail);
                if (have_data && needs_upper_record) {
                    std::vector<double> src1;
                    int upper_nx = 0;
                    int upper_ny = 0;
                    const bool local_upper_ready =
                        read_slab(bracket.i1, src1, upper_nx, upper_ny) && src1.size() == src.size() && upper_nx == file_nx && upper_ny == file_ny;
                    have_data = collective_all_ready(comm_c_, local_upper_ready, "upper AMIO slab readiness for '" + var_name + "'", failure_detail);
                    if (have_data) {
                        const double w = bracket.weight;
                        for (size_t k = 0; k < src.size(); ++k) {
                            src[k] = (1.0 - w) * src[k] + w * src1[k];
                        }
                    }
                }

                if (have_data) {
                    read_success = AssembleReplicatedField(var_name, plan, src, file_nx, file_ny, field_nlev, stream_view, cece_core_data_ptr,
                                                           ingest_buffer, failure_detail);
                    if (read_success) {
                        // Refresh the slice cache with the freshly computed
                        // ingest buffer and the bracket that produced it, so a
                        // later step resolving the same bracket can reuse it
                        // (Req 3.3, 9.4).
                        slice_cache.last_bracket = bracket;
                        slice_cache.ingest_buffer = ingest_buffer;
                        slice_cache.ingest_size = static_cast<size_t>(field_nlev) * nx_ * ny_;
                        slice_cache.valid = true;
                    }
                }
                }  // end cache-miss branch
            }
            read_success = collective_all_ready(comm_c_, read_success, "replicated field assembly for '" + var_name + "'", failure_detail);
            // The AMIO handle set persists in amio_handles_ across timesteps
            // (Req 2.2, 9.1); it is closed/finalized only in the destructor
            // (task 10.1). No per-step amio_close/amio_finalize, no manifest
            // write/delete, and no per-step MPI_Barrier occur here (Req 9.1, 9.2).
        }

        // Throw a fatal error on AMIO read failures
        if (!read_success) {
            std::string detail = failure_detail.empty() ? "AMIO open/read failed (no detail)" : failure_detail;
            CECE_LOG_ERROR("[FATAL ERROR] AMIO read failed for field '" + var_name + "' in file '" + input_file_path + "'. Reason: " + detail +
                           ". Idealized fallback is disabled!");
            return false;
        } else {
            CECE_LOG_INFO("[DRIVER] AMIO read succeeded for field '" + var_name + "' - loaded real data from " + input_file_path);
        }

        const size_t expected_ingest_size = static_cast<size_t>(field_nlev) * nx_ * ny_;
        const bool local_ingest_size_ready = ingest_buffer.size() == expected_ingest_size;
        if (!local_ingest_size_ready) {
            failure_detail = "internal ingest buffer size mismatch for field '" + var_name + "'";
        }
        if (!collective_all_ready(comm_c_, local_ingest_size_ready, "ingest-buffer readiness for '" + var_name + "'", failure_detail)) {
            CECE_LOG_ERROR("[DRIVER FATAL] " + failure_detail);
            return false;
        }

        // Ingest-copy consolidation (Req 3.1, 3.2, 3.4, 5.3): the driver facade
        // already wrote import_state.fields[var_name] directly and authoritatively
        // in AssembleReplicatedField, so the legacy ingestor round-trip here is
        // redundant. The `cece_ingestor_set_field` call (which populated the
        // separate field_cache_ for AMIO variables) and its guarding
        // `CECE ingestor readiness` collective gate were removed together as a
        // UNIT — the gate only guarded the now-deleted call, and it is removed
        // identically on every rank so no rank waits on a collective a peer
        // skipped. The surrounding `ingest-buffer readiness` gate above is
        // retained: it validates the assembled buffer and must still be reached
        // by every rank in the same order. `ingest_buffer` is still produced
        // above because the slice cache stores its own copy in AdvanceTime;
        // only the `cece_ingestor_set_field` consumer is removed here.
        //
        // Removing the SetField population of field_cache_ for AMIO variables
        // also naturally neutralizes IngestEmissionsInline's copy-back for those
        // fields: its HasCachedField(...) check now returns false, so it skips
        // them. No edit to IngestEmissionsInline itself is required.
        CECE_LOG_INFO("[DRIVER] Ingested field '" + var_name + "' with shape " + std::to_string(nx_) + "x" + std::to_string(ny_) + "x" +
                      std::to_string(field_nlev));
    }

    step_index_++;
    return true;
}

}  // namespace cece

extern "C" {
void amio_set_parent_communicator(MPI_Fint comm);

void cece_driver_create(const char* yaml_path, int path_len, int nx, int ny, int nz, const double* lon_coords, int lon_len, const double* lat_coords,
                        int lat_len, int mpi_comm_f, void** driver_ptr_out, int* rc) {
    if (rc) *rc = 0;
    try {
        std::string path(yaml_path, path_len);

        // 1. Pass custom parent communicator to AMIO
        amio_set_parent_communicator(static_cast<MPI_Fint>(mpi_comm_f));

        // 2. Convert Fortran MPI handle to C MPI_Comm
        MPI_Comm comm_c = MPI_Comm_f2c(static_cast<MPI_Fint>(mpi_comm_f));

        // 3. Create orchestrator using the custom communicator
        auto* driver = new cece::CeceDriverOrchestrator(path, nx, ny, nz, lon_coords, lon_len, lat_coords, lat_len, comm_c);
        *driver_ptr_out = static_cast<void*>(driver);
    } catch (const std::exception& e) {
        CECE_LOG_ERROR(std::string("cece_driver_create: ") + e.what());
        if (rc) *rc = -1;
    }
}

void cece_driver_advance_time(void* driver_ptr, const char* time_iso8601, int time_len, void* cece_core_data_ptr, int* rc) {
    if (rc) *rc = 0;
    try {
        auto* driver = static_cast<cece::CeceDriverOrchestrator*>(driver_ptr);
        std::string t_iso(time_iso8601, time_len);
        bool ok = driver->AdvanceTime(t_iso, cece_core_data_ptr);
        if (!ok && rc) *rc = -1;
    } catch (const std::exception& e) {
        CECE_LOG_ERROR(std::string("cece_driver_advance_time: ") + e.what());
        if (rc) *rc = -1;
    }
}

extern std::unique_ptr<cece::CeceStandaloneWriter> g_standalone_writer;

void cece_driver_destroy(void* driver_ptr) {
    if (driver_ptr) {
        delete static_cast<cece::CeceDriverOrchestrator*>(driver_ptr);
    }
    g_standalone_writer.reset();
}

}  // extern "C"
