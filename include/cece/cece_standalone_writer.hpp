#ifndef CECE_STANDALONE_WRITER_HPP
#define CECE_STANDALONE_WRITER_HPP

#include <amio/amio_types.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cece/cece_band_decomposition.hpp"
#include "cece_compute.hpp"
#include "cece_config.hpp"

namespace cece {

// Local name for the opaque AMIO dataset handle used by the writer helpers.
using amio_dataset_handle_t = void*;

class CeceStandaloneWriter {
   public:
    explicit CeceStandaloneWriter(const CeceOutputConfig& config, MPI_Comm comm = MPI_COMM_SELF);
    ~CeceStandaloneWriter();

    CeceStandaloneWriter(const CeceStandaloneWriter&) = delete;
    CeceStandaloneWriter& operator=(const CeceStandaloneWriter&) = delete;
    CeceStandaloneWriter(CeceStandaloneWriter&&) = delete;
    CeceStandaloneWriter& operator=(CeceStandaloneWriter&&) = delete;

    int Initialize(const std::string& start_time_iso8601, int nx, int ny, int nz);

    int InitializeWithCoords(const std::string& start_time_iso8601, int nx, int ny, int nz, const std::vector<double>& lon_coords,
                             const std::vector<double>& lat_coords, const std::string& gridspec_file = "");

    int WriteTimeStep(const std::unordered_map<std::string, DualView3D>& export_fields, double time_seconds_since_start, int step_index);

    void Finalize();

    static size_t ResolveStagingBufferCapacity(const CeceOutputConfig& config, int nx, int ny, int nz);

    bool IsInitialized() const {
        return initialized_;
    }

   private:
    CeceOutputConfig config_;
    bool initialized_ = false;
    int record_count_ = 0;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    std::string start_time_iso8601_;
    std::vector<double> lon_coords_;
    std::vector<double> lat_coords_;
    bool use_custom_coords_ = false;
    MPI_Comm comm_ = MPI_COMM_SELF;
    std::string gridspec_file_;

    // Band decomposition of the global latitude rows [0, ny_) across comm_.
    // Computed in Initialize/InitializeWithCoords from ny_ + comm_ and used by
    // direct region writes, or the rank-0 gather when shared output is unavailable.
    BandDecomposition band_;

    std::string ResolveFilename(double time_seconds_since_start) const;

    // Write all coordinate variables (lon, lat, lon_bnds, lat_bnds, optional
    // UGRID mesh, lev, time) into an open WRITE-mode dataset. Extracted from
    // WriteTimeStep, whose body had grown past 600 lines; this is the
    // file-owner (rank-0 / single-process) coordinate section only. Throws
    // std::runtime_error (via check_amio_rc) on any AMIO write failure; the
    // caller degrades that to skip_file_work so the mandatory field gathers
    // still run. Steps 4-7 = lon, lat, bounds (+mesh), lev, time.
    void WriteCoordinateVariables(amio_dataset_handle_t dataset, double time_seconds);
    using CoordinateVisitor = std::function<void(const char*, const void*, amio_dtype_t, const amio_shape_t*)>;
    void VisitCoordinateVariables(double time_seconds, const CoordinateVisitor& write_coordinate);
    bool ValidateOutputFields(const std::unordered_map<std::string, DualView3D>& fields) const;
    int WriteParallelTimeStep(const std::unordered_map<std::string, DualView3D>& fields, double time_seconds, const std::string& manifest);
};

}  // namespace cece

#endif
