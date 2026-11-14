#include <amio/amio.h>
#include <netcdf.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "cece/cece_logger.hpp"
#include "cece/cece_standalone_writer.hpp"
#include "cece_output_batch.hpp"

extern "C" void amio_set_parent_communicator(MPI_Fint comm);

namespace cece {
namespace {
// Call only between AMIO operations: outstanding backend collectives must be
// complete before the application enters a collective on this communicator.
int AgreeFailure(MPI_Comm comm, int failed) {
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm);
    return failed;
}

std::vector<std::string> FieldNames(const CeceOutputConfig& config, const std::unordered_map<std::string, DualView3D>& fields) {
    std::vector<std::string> names;
    const bool all = config.fields.GetDataFields().empty();
    for (const auto& [name, view] : fields) {
        if (!IsCoordinateName(name) && (all || config.fields.Contains(name))) names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}
}  // namespace

bool CeceStandaloneWriter::ValidateOutputFields(const std::unordered_map<std::string, DualView3D>& fields) const {
    int failed = 0;
    std::string names;
    try {
        for (const auto& name : FieldNames(config_, fields)) {
            names += name;
            names.push_back('\0');
            const auto& view = fields.at(name);
            const auto h = view.view_host();
            if (h.extent(0) != static_cast<size_t>(nx_) || h.extent(1) != static_cast<size_t>(band_.ny_local) ||
                h.extent(2) != static_cast<size_t>(nz_))
                failed = 1;
        }
    } catch (const std::exception&) {
        failed = 1;
    }
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&finalized);
    if (initialized && !finalized && comm_ != MPI_COMM_NULL) {
        int rank = 0;
        MPI_Comm_rank(comm_, &rank);
        int length = static_cast<int>(names.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, comm_);
        std::string reference = rank == 0 ? names : std::string(length, '\0');
        MPI_Bcast(reference.data(), length, MPI_CHAR, 0, comm_);
        if (names != reference) failed = 1;
        failed = AgreeFailure(comm_, failed);
    }
    if (failed) CECE_LOG_ERROR("[CECE] Output fields differ across ranks or have invalid band-local shapes");
    return failed == 0;
}

int CeceStandaloneWriter::WriteParallelTimeStep(const std::unordered_map<std::string, DualView3D>& fields, double time_seconds,
                                                const std::string& manifest) {
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm writers = MPI_COMM_NULL;
    // The legacy region API requires positive extents. Empty bands wait for
    // the final status on comm_, without opening the shared file.
    MPI_Comm_split(comm_, band_.ny_local > 0 ? 0 : MPI_UNDEFINED, rank, &writers);
    int failed = 0;
    amio_core_handle core = nullptr;
    amio_dataset_handle dataset = nullptr;
    if (writers != MPI_COMM_NULL) {
        amio_set_parent_communicator(MPI_Comm_c2f(writers));
        std::vector<std::string> names;
        std::vector<std::string> schema_names;
        std::vector<amio_write_variable_t> schema;
        amio_shape_t local{};
        amio_write_region_t region{};
        size_t batch_limit = 1;
        try {
            names = FieldNames(config_, fields);
            schema_names.reserve(names.size() + 7);
            schema.reserve(names.size() + 7);
            auto add_schema = [&](const char* name, amio_dtype_t dtype, const amio_shape_t* shape) {
                schema_names.emplace_back(name);
                amio_write_variable_t variable{};
                variable.struct_size = sizeof(variable);
                variable.version = AMIO_WRITE_VARIABLE_VERSION;
                variable.dtype = dtype;
                variable.rank = shape->rank;
                std::copy_n(shape->extents, shape->rank, variable.global_extents);
                schema.push_back(variable);
            };
            if (rank == 0) {
                VisitCoordinateVariables(time_seconds, [&](const char* name, const void*, amio_dtype_t dtype, const amio_shape_t* shape) {
                    add_schema(name, dtype, shape);
                });
            } else {
                // Match the coordinate visitor's order without materializing a
                // global mesh and bounds on every writer rank.
                schema_names = {"lon", "lat", "lon_bnds", "lat_bnds"};
                if (ny_ == 1) schema_names.emplace_back("mesh");
                schema_names.emplace_back("lev");
                schema_names.emplace_back("time");
                schema.resize(schema_names.size());
            }
        } catch (const std::exception& e) {
            CECE_LOG_ERROR(std::string("[CECE] Preparing coordinate schema failed: ") + e.what());
            failed = 1;
        }
        failed = AgreeFailure(writers, failed);
        if (!failed) {
            // Names are assigned below; every pointer in these descriptors is
            // still null. Only the small fixed-layout schema is exchanged.
            MPI_Bcast(schema.data(), static_cast<int>(schema.size() * sizeof(amio_write_variable_t)), MPI_BYTE, 0, writers);
        }
        try {
            if (!failed) {
                auto add_schema = [&](const char* name, amio_dtype_t dtype, const amio_shape_t* shape) {
                    schema_names.emplace_back(name);
                    amio_write_variable_t variable{};
                    variable.struct_size = sizeof(variable);
                    variable.version = AMIO_WRITE_VARIABLE_VERSION;
                    variable.dtype = dtype;
                    variable.rank = shape->rank;
                    std::copy_n(shape->extents, shape->rank, variable.global_extents);
                    schema.push_back(variable);
                };
                local.rank = region.rank = ny_ == 1 ? 3 : 4;
                local.extents[0] = region.global_extents[0] = 1;
                local.extents[1] = region.global_extents[1] = nz_;
                if (ny_ == 1) {
                    local.extents[2] = region.global_extents[2] = nx_;
                } else {
                    local.extents[2] = band_.ny_local;
                    region.global_extents[2] = ny_;
                    region.offsets[2] = band_.j0;
                    local.extents[3] = region.global_extents[3] = nx_;
                }
                amio_shape_t global{};
                global.rank = region.rank;
                std::copy_n(region.global_extents, region.rank, global.extents);
                for (const auto& name : names) add_schema(name.c_str(), AMIO_DTYPE_F64, &global);
                if (schema.size() > 4096) throw std::runtime_error("AMIO prepared schema exceeds 4096 variables");
                for (size_t i = 0; i < schema.size(); ++i) schema[i].name = schema_names[i].c_str();
                // Maximum band size is rank-invariant, including surplus ranks.
                const int largest_band = *std::max_element(band_.row_counts.begin(), band_.row_counts.end());
                const size_t max_bytes = static_cast<size_t>(largest_band) * nx_ * nz_ * sizeof(double);
                batch_limit =
                    OutputBatchSize(ResolveStagingBufferCapacity(config_, nx_, largest_band, nz_), max_bytes, std::max(size_t{1}, names.size()));
            }
        } catch (const std::exception& e) {
            CECE_LOG_ERROR(std::string("[CECE] Preparing parallel output failed: ") + e.what());
            failed = 1;
        }
        failed = AgreeFailure(writers, failed);
        if (!failed) {
            failed = amio_init_from_string(manifest.c_str(), "yaml", &core) != AMIO_OK;
            failed = AgreeFailure(writers, failed);
        }
        if (!failed) {
            failed = amio_open_dataset_from_string(core, manifest.c_str(), "yaml", AMIO_MODE_WRITE, &dataset) != AMIO_OK;
            failed = AgreeFailure(writers, failed);
        }
        if (!failed) {
            failed = amio_prepare_write_schema(dataset, schema.data(), schema.size()) != AMIO_OK;
            failed = AgreeFailure(writers, failed);
        }
        for (size_t first = 0; !failed && first < names.size(); first += batch_limit) {
            const size_t count = std::min(batch_limit, names.size() - first);
            std::vector<std::vector<double>> payloads;
            std::vector<amio_region_write_t> writes;
            try {
                payloads.resize(count);
                writes.resize(count);
                for (size_t n = 0; n < count; ++n) {
                    auto& view = const_cast<DualView3D&>(fields.at(names[first + n]));
                    view.sync<Kokkos::HostSpace>();
                    auto host = view.view_host();
                    auto& payload = payloads[n];
                    payload.resize(static_cast<size_t>(nx_) * band_.ny_local * nz_);
                    for (int k = 0; k < nz_; ++k)
                        for (int j = 0; j < band_.ny_local; ++j)
                            for (int i = 0; i < nx_; ++i) payload[(static_cast<size_t>(k) * band_.ny_local + j) * nx_ + i] = host(i, j, k);
                    writes[n] = {names[first + n].c_str(), payload.data(), AMIO_DTYPE_F64, &local, &region};
                }
            } catch (const std::exception& e) {
                CECE_LOG_ERROR(std::string("[CECE] Packing parallel output failed: ") + e.what());
                failed = 1;
            }
            failed = AgreeFailure(writers, failed);
            if (failed) break;
            amio_io_handle pending = nullptr;
            failed = amio_write_region_batch(dataset, writes.data(), writes.size(), &pending) != AMIO_OK;
            // Successful admission snapshots every payload. Drain the accepted
            // operation before entering any application-level collective.
            if (pending && amio_wait(pending, 0) != AMIO_OK) failed = 1;
            failed = AgreeFailure(writers, failed);
        }
        if (dataset) {
            if (amio_flush(dataset, 0) != AMIO_OK) failed = 1;
            if (amio_close(dataset) != AMIO_OK) failed = 1;
        }
        if (core && amio_finalize(core) != AMIO_OK) failed = 1;
        failed = AgreeFailure(writers, failed);
        amio_set_parent_communicator(MPI_Comm_c2f(comm_));
        MPI_Comm_free(&writers);
    }
    failed = AgreeFailure(comm_, failed);
    // Small replicated coordinates have scalar/short axes that cannot be
    // partitioned across all writers with positive extents. Their schemas
    // were prepared collectively; fill them once, after every writer closes.
    if (!failed && rank == 0) {
        int ncid = -1;
        try {
            if (nc_open(ResolveFilename(time_seconds).c_str(), NC_WRITE, &ncid) != NC_NOERR)
                throw std::runtime_error("opening output for coordinate payloads");
            VisitCoordinateVariables(time_seconds, [&](const char* name, const void* values, amio_dtype_t dtype, const amio_shape_t* shape) {
                int id = -1;
                if (nc_inq_varid(ncid, name, &id) != NC_NOERR) throw std::runtime_error(std::string("coordinate schema missing: ") + name);
                size_t start[AMIO_MAX_RANK]{}, count[AMIO_MAX_RANK]{};
                for (int d = 0; d < shape->rank; ++d) count[d] = static_cast<size_t>(shape->extents[d]);
                const int rc = dtype == AMIO_DTYPE_I32 ? nc_put_vara_int(ncid, id, start, count, static_cast<const int*>(values))
                                                       : nc_put_vara_double(ncid, id, start, count, static_cast<const double*>(values));
                if (rc != NC_NOERR) throw std::runtime_error(std::string("writing coordinate ") + name + ": " + nc_strerror(rc));
            });
        } catch (const std::exception& e) {
            CECE_LOG_ERROR(std::string("[CECE] Parallel output coordinate failure: ") + e.what());
            failed = 1;
        }
        if (ncid >= 0 && nc_close(ncid) != NC_NOERR) failed = 1;
    }
    failed = AgreeFailure(comm_, failed);
    if (failed)
        CECE_LOG_ERROR("[CECE] Parallel output failed; partial output may remain");
    else if (rank == 0)
        CECE_LOG_INFO("[CECE] Wrote output with AMIO parallel region batches");
    return failed ? -1 : 0;
}
}  // namespace cece
