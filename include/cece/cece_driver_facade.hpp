#ifndef CECE_DRIVER_FACADE_HPP
#define CECE_DRIVER_FACADE_HPP

#include <amio/amio.h>
#include <mpi.h>

#include <cstddef>
#include <dagr/dagr.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cece/cece_io.hpp"
#include "cece/cece_regridder_utils.hpp"

namespace cece {

/**
 * @brief Resolved input-record selection for a given simulation time.
 *
 * @c i0 / @c i1 are the lower/upper record indices and @c weight is the
 * fraction toward @c i1: the interpolated field is
 * @f$ (1-w)\,\mathrm{rec}[i_0] + w\,\mathrm{rec}[i_1] @f$. When @c weight is 0
 * (or @c i0 == @c i1) a single read of @c i0 suffices. @c valid is false when
 * the caller should fall back to legacy step-index cycling.
 *
 * This struct is shared between the facade header (for use as
 * SliceCacheEntry::last_bracket) and the facade translation unit's temporal
 * cadence helpers.
 */
struct RecordBracket {
    int i0 = 0;
    int i1 = 0;
    double weight = 0.0;
    bool valid = false;  ///< false -> caller falls back to legacy step-index cycling.
};

/**
 * @brief Resolved per-stream-variable configuration.
 *
 * Resolved once per stream variable at construction; depends only on
 * configuration + file, never on simulation time.
 */
struct StreamConfig {
    std::string input_file_path;        ///< stream["file"]; "" => missing (Req 1.4)
    std::string input_var_name;         ///< resolved file var name (falls back to model name)
    std::string mapalgo = "consd";      ///< default matches current AdvanceTime
    std::string cadence;                ///< "" => legacy step-index cycling
    std::string tintalgo = "nearest";   ///< "linear" | "nearest"
    std::string data_model = "enhanced";
    bool data_model_explicit = false;   ///< true => open with only data_model
    int amio_worker_threads = 1;        ///< driver-level, validated >= 1
    int amio_staging_buffer_count = 8;  ///< driver-level, validated >= 1
};

/**
 * @brief Retained AMIO resources for one stream variable, opened at most once.
 */
struct AmioHandleSet {
    amio_core_handle core = nullptr;        ///< from amio_init_from_string
    amio_dataset_handle dataset = nullptr;  ///< from amio_open_dataset_from_string
    std::string active_data_model;          ///< model that actually opened
    std::string manifest_content;           ///< in-memory manifest (never a file)
};

/**
 * @brief Most-recent read+regrid result plus the bracket that produced it.
 */
struct SliceCacheEntry {
    RecordBracket last_bracket;         ///< bracket that produced ingest_buffer
    bool valid = false;                 ///< false until first successful compute
    std::vector<double> ingest_buffer;  ///< field_nlev * nx_ * ny_, replicated
    size_t ingest_size = 0;             ///< == field_nlev * nx_ * ny_
};

class CeceDriverOrchestrator {
   public:
    CeceDriverOrchestrator(const std::string& config_file, int nx, int ny, int nz, const double* lon_coords, int lon_len, const double* lat_coords,
                           int lat_len, MPI_Comm comm_c);
    ~CeceDriverOrchestrator();

    // Prevent copy/move construction and assignment (Rule of Five)
    CeceDriverOrchestrator(const CeceDriverOrchestrator&) = delete;
    CeceDriverOrchestrator& operator=(const CeceDriverOrchestrator&) = delete;
    CeceDriverOrchestrator(CeceDriverOrchestrator&&) = delete;
    CeceDriverOrchestrator& operator=(CeceDriverOrchestrator&&) = delete;

    bool AdvanceTime(const std::string& time_iso8601, void* cece_core_data_ptr);

   private:
    using DeviceView3D = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

    bool AssembleReplicatedField(const std::string& var_name, const io::RegridPlan& plan, const std::vector<double>& source, int file_nx, int file_ny,
                                 int field_nlev, DeviceView3D stream_view, void* cece_core_data_ptr, std::vector<double>& ingest_buffer,
                                 std::string& failure_detail);

    // Parse config_file_ once at construction and populate stream_configs_ for
    // every model variable, resolving the same fields and defaults the legacy
    // inline AdvanceTime parse produced. Also resolves the driver-level
    // amio_worker_threads / amio_staging_buffer_count (validated >= 1) and
    // gridspec_file_ (Req 1.1-1.5, 4.3).
    void ResolveStreamConfigs();

    // Pure YAML->StreamConfig resolution shared by ResolveStreamConfigs() and
    // the Property 4 test. Parses config_file, fills out_configs (keyed by model
    // variable name) and out_gridspec_file using the exact same field resolution
    // and defaults as the legacy inline AdvanceTime parse. Static so it can be
    // exercised in isolation without constructing a full orchestrator (which
    // requires MPI/DAGR/CeceIO). This is the SAME code path production uses; it
    // is not a copy (Req 1.1-1.5, 4.3).
    static void ResolveStreamConfigsFromFile(const std::string& config_file, std::unordered_map<std::string, StreamConfig>& out_configs,
                                             std::string& out_gridspec_file);

    // Build the in-memory AMIO manifest YAML for one stream, byte-for-byte
    // identical to the manifest the legacy inline AdvanceTime code wrote to
    // disk (same keys, same order, same values), just returned as a string
    // instead of written to a file (Req 2.3, 2.5, 2.6).
    std::string BuildManifestContent(const StreamConfig& cfg, const std::string& data_model) const;

    // Compute the stream identity key for a given StreamConfig. The key is
    // input_file_path + "|" + mapalgo. Variables sharing the same input file
    // and mapping algorithm produce the same key and share the regrid plan,
    // AMIO handle set, and file record count caches.
    static std::string StreamKey(const StreamConfig& cfg);

    // Tolerance for comparing two resolved bracket weights for equality.
    static constexpr double kBracketWeightTol = 1e-12;

    // Return true iff two resolved brackets have identical record indices and
    // weights within kBracketWeightTol. Compares i0, i1, and weight only; the
    // `valid` field is intentionally NOT compared (Req 3.5, 6.3).
    static bool bracket_equal(const RecordBracket& a, const RecordBracket& b);

    // Return the retained AMIO handle set for the given Stream_Identity_Key
    // (input_file_path + "|" + mapalgo), opening it lazily on first touch and
    // reusing it on every subsequent call. Variables that share a stream share
    // a single open core+dataset handle (Req 2.1, 2.2, 3.1, 3.2, 7.1).
    //
    // On first touch this builds the in-memory manifest via BuildManifestContent
    // and opens via the STRING-based AMIO entry points (amio_init_from_string /
    // amio_open_dataset_from_string) — no manifest file is written to disk and
    // no per-step MPI_Barrier is issued. Only the open is wrapped in the
    // MPI_COMM_SELF parent-communicator swap; the communicator is restored to
    // comm_c_ afterward. Candidate data models are tried in order: {cfg.data_model}
    // when cfg.data_model_explicit, else {"enhanced", "classic"} (Req 2.5, 2.6).
    //
    // On success it caches {core, dataset, active_data_model, manifest_content}
    // in amio_handles_ and returns the pointer. On failure of a candidate it
    // closes/finalizes any partial handle and tries the next; if all candidates
    // fail it caches nothing, sets failure_detail (with amio_strerror), and
    // returns nullptr (Req 8.1). This helper only opens; it does not read or
    // regrid.
    AmioHandleSet* GetOrOpenHandleSet(const std::string& stream_key, const StreamConfig& cfg, std::string& failure_detail);

    // Release every retained AMIO handle set at shutdown. Iterates amio_handles_
    // and for each set closes the dataset (amio_close) then finalizes the core
    // (amio_finalize), each wrapped in its own best-effort try/catch so one
    // failing handle does not prevent the rest from tearing down; null
    // dataset/core are skipped. After the loop it clears amio_handles_,
    // stream_configs_, and slice_caches_. No manifest files are deleted because
    // none are ever created (all manifests are in-memory strings) (Req 7.2-7.5).
    void TeardownHandles();

    std::string config_file_;
    int nx_{0}, ny_{0}, nz_{0};
    std::vector<double> target_lons_;
    std::vector<double> target_lats_;
    int step_index_{0};
    MPI_Comm comm_c_{MPI_COMM_NULL};

    // Cached regridding plans keyed by Stream_Identity_Key
    // (input_file_path + "|" + mapalgo). Variables in the same stream share
    // one plan since the interpolation weights depend only on source grid,
    // target grid, and mapping algorithm — not on the variable name.
    std::unordered_map<std::string, io::RegridPlan> regrid_plans_;

    // File record counts keyed by Stream_Identity_Key. The binary-search
    // discovery runs at most once per stream.
    std::unordered_map<std::string, int> file_nt_cache_;

    // Loop-invariant work moved out of AdvanceTime:
    //  - stream_configs_ : YAML resolved once at construction, keyed by model
    //                      variable name (Req 1).
    //  - amio_handles_   : AMIO core/dataset opened lazily once, retained
    //                      across timesteps, keyed by Stream_Identity_Key
    //                      (input_file_path + "|" + mapalgo). Variables in the
    //                      same stream share one open core+dataset handle
    //                      (Req 2, 7).
    //  - slice_caches_   : most-recent read+regrid result, reused when the
    //                      resolved time bracket is unchanged. Keyed by model
    //                      variable name (NOT Stream_Identity_Key) because
    //                      each variable carries different data even within
    //                      the same stream (Req 3, 5).
    std::unordered_map<std::string, StreamConfig> stream_configs_;
    std::unordered_map<std::string, AmioHandleSet> amio_handles_;
    std::unordered_map<std::string, SliceCacheEntry> slice_caches_;

    // HELM Orchestration and pipeline components
    std::unique_ptr<dagr::GraphOrchestrator> dagr_;
    std::unique_ptr<io::CeceIO> cece_io_;
    std::string gridspec_file_;

    // Test-only access to the private static StreamKey helper. Grants the
    // property test (tests/test_stream_key_properties.cpp) permission to
    // invoke StreamKey without changing its signature/visibility. Has no
    // effect on production behavior. (Task 6.1)
    friend struct StreamKeyTestAccess;

    // Test-only access to the private static bracket_equal helper. Grants the
    // property test (tests/test_bracket_equal_properties.cpp) permission to
    // invoke bracket_equal without changing its signature/visibility. Has no
    // effect on production behavior. (Task 8.5)
    friend struct BracketEqualTestAccess;

    // Test-only access to the private static ResolveStreamConfigsFromFile
    // helper. Grants the Property 4 test
    // (tests/test_stream_config_resolution.cpp) permission to invoke the real
    // YAML->StreamConfig resolution against a chosen config path without
    // constructing a full orchestrator. Has no effect on production behavior.
    // (Task 8.2)
    friend struct StreamConfigTestAccess;

    // Test-only access to the private static bracket_equal helper for the
    // Property 6 slice-cache-equivalence test
    // (tests/test_slice_cache_equivalence.cpp). SliceCacheEntry / RecordBracket
    // are public structs in the cece namespace, so only bracket_equal needs
    // friend access. Has no effect on production behavior. (Task 9.2)
    friend struct SliceCacheTestAccess;

    // Test-only access to the private static bracket_equal helper for the
    // complementary "cache HIT skips read+regrid work" property test
    // (tests/test_cache_hit_skips_work.cpp). SliceCacheEntry / RecordBracket are
    // public structs in the cece namespace, so only bracket_equal needs friend
    // access. Has no effect on production behavior. (Task 13.3)
    friend struct CacheHitSkipTestAccess;

    // Test-only access to the private static bracket_equal helper for the
    // cross-rank reuse-decision MPI integration test
    // (tests/test_mpi_reuse_decision.cpp, Property 8). Grants that test
    // permission to invoke the REAL production bracket_equal so the per-rank
    // hit/miss decision is derived from production logic, not a copy.
    // RecordBracket is a public struct in the cece namespace, so only
    // bracket_equal needs friend access. Has no effect on production behavior.
    // (Task 12.3)
    friend struct CrossRankReuseTestAccess;
};

}  // namespace cece

extern "C" {
void cece_driver_create(const char* yaml_path, int path_len, int nx, int ny, int nz, const double* lon_coords, int lon_len, const double* lat_coords,
                        int lat_len, int mpi_comm_f, void** driver_ptr_out, int* rc);

void cece_driver_advance_time(void* driver_ptr, const char* time_iso8601, int time_len, void* cece_core_data_ptr, int* rc);

void cece_driver_destroy(void* driver_ptr);
}

#endif  // CECE_DRIVER_FACADE_HPP
