#include <gtest/gtest.h>
#include <mpi.h>
#include <netcdf.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "cece/cece_standalone_writer.hpp"

namespace {

struct FileSnapshot {
    std::map<std::string, std::vector<double>> values;
    std::map<std::string, std::vector<std::string>> dimensions;
    std::map<std::string, std::string> attributes;
};

// All ranks execute each case. Rank zero alone inspects the closed file.
void WriteBands(int ny, bool malformed, bool different_names, bool parallel = true, bool failing_open = false, FileSnapshot* snapshot = nullptr,
                bool curvilinear = false, bool coordinate_only = false) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (different_names && size == 1) GTEST_SKIP() << "requires multiple ranks";
    constexpr int nx = 5, nz = 3;
    const auto band = cece::BandDecomposition::compute(ny, MPI_COMM_WORLD);
    std::string directory;
    if (rank == 0) {
        directory = (std::filesystem::temp_directory_path() / ("cece_parallel_" + std::to_string(getpid()))).string();
        std::filesystem::create_directories(directory);
    }
    int length = static_cast<int>(directory.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    directory.resize(length);
    MPI_Bcast(directory.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);

    cece::CeceOutputConfig config;
    config.enabled = true;
    config.amio_parallel_output = parallel;
    config.directory = directory;
    config.filename_pattern = failing_open ? "missing/bands.nc" : "bands.nc";
    config.fields.SetTimeUnits("2020-01-01T00:00:00");
    config.global_attributes["title"] = "parallel regression";
    for (int f = 0; f < 7; ++f) {
        config.fields.push_back({"species" + std::to_string(f), {{"units", "kg m-2 s-1"}, {"long_name", "test emission"}}});
    }
    // The effective capacity fits one global field, exercising multiple batches.
    config.amio_staging_buffer_capacity_bytes = 1;
    config.amio_staging_buffer_count = 1;
    cece::CeceStandaloneWriter writer(config, MPI_COMM_WORLD);
    if (curvilinear) {
        std::vector<double> lon(nx * ny), lat(nx * ny);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                lon[j * nx + i] = -180.0 + 360.0 * (i + 0.5) / nx;
                lat[j * nx + i] = -90.0 + 180.0 * (j + 0.5) / ny;
            }
        ASSERT_EQ(writer.InitializeWithCoords("2020-01-01T00:00:00", nx, ny, nz, lon, lat), 0);
    } else if (ny == 1) {
        std::vector<double> lon(nx), lat(nx, 0.0);
        for (int i = 0; i < nx; ++i) lon[i] = -180.0 + 360.0 * (i + 0.5) / nx;
        ASSERT_EQ(writer.InitializeWithCoords("2020-01-01T00:00:00", nx, ny, nz, lon, lat), 0);
    } else {
        ASSERT_EQ(writer.Initialize("2020-01-01T00:00:00", nx, ny, nz), 0);
    }
    std::unordered_map<std::string, cece::DualView3D> fields;
    for (int f = 0; f < (coordinate_only ? 0 : 7); ++f) {
        const std::string name = "species" + std::to_string(f);
        const int local_ny = band.ny_local + ((malformed && rank == 0) ? 1 : 0);
        cece::DualView3D view(name, nx, local_ny, nz);
        auto h = view.view_host();
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < local_ny; ++j)
                for (int i = 0; i < nx; ++i) h(i, j, k) = f * 10000 + k * 1000 + (band.j0 + j) * 10 + i;
        view.modify_host();
        if (!(different_names && rank == 0 && f == 6)) fields.emplace(name, view);
    }
    const int result = writer.WriteTimeStep(fields, 3600.0, 0);
    EXPECT_EQ(result, (malformed || different_names || failing_open) ? -1 : 0);
    writer.Finalize();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !malformed && !different_names && !failing_open) {
        [&] {
            int ncid = -1;
            ASSERT_EQ(nc_open((directory + "/bands.nc").c_str(), NC_NOWRITE, &ncid), NC_NOERR);
            for (int f = 0; f < (coordinate_only ? 0 : 7); ++f) {
                int varid = -1;
                ASSERT_EQ(nc_inq_varid(ncid, ("species" + std::to_string(f)).c_str(), &varid), NC_NOERR);
                std::vector<double> values(nx * ny * nz);
                ASSERT_EQ(nc_get_var_double(ncid, varid, values.data()), NC_NOERR);
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i) EXPECT_EQ(values[(k * ny + j) * nx + i], f * 10000 + k * 1000 + j * 10 + i);
            }
            int unlimited = -1;
            EXPECT_EQ(nc_inq_unlimdim(ncid, &unlimited), NC_NOERR);
            EXPECT_NE(unlimited, -1);

            int timeid = -1;
            ASSERT_EQ(nc_inq_varid(ncid, "time", &timeid), NC_NOERR);
            double time = 0;
            EXPECT_EQ(nc_get_var_double(ncid, timeid, &time), NC_NOERR);
            EXPECT_EQ(time, 3600.0);
            for (const char* name : {"lon", "lat", "lev", "lon_bnds", "lat_bnds"}) {
                int id = -1;
                EXPECT_EQ(nc_inq_varid(ncid, name, &id), NC_NOERR);
            }
            size_t attr_length = 0;
            EXPECT_EQ(nc_inq_attlen(ncid, NC_GLOBAL, "title", &attr_length), NC_NOERR);
            std::string title(attr_length, '\0');
            EXPECT_EQ(nc_get_att_text(ncid, NC_GLOBAL, "title", title.data()), NC_NOERR);
            EXPECT_EQ(title, "parallel regression");
            if (snapshot) {
                int variables = 0;
                ASSERT_EQ(nc_inq_nvars(ncid, &variables), NC_NOERR);
                for (int id = 0; id < variables; ++id) {
                    char name[NC_MAX_NAME + 1]{};
                    int ndim = 0, natts = 0, dims[NC_MAX_VAR_DIMS]{};
                    ASSERT_EQ(nc_inq_var(ncid, id, name, nullptr, &ndim, dims, &natts), NC_NOERR);
                    size_t elements = 1;
                    for (int d = 0; d < ndim; ++d) {
                        char dimname[NC_MAX_NAME + 1]{};
                        size_t extent = 0;
                        ASSERT_EQ(nc_inq_dim(ncid, dims[d], dimname, &extent), NC_NOERR);
                        snapshot->dimensions[name].push_back(std::string(dimname) + ":" + std::to_string(extent));
                        elements *= extent;
                    }
                    auto& payload = snapshot->values[name];
                    payload.resize(elements);
                    ASSERT_EQ(nc_get_var_double(ncid, id, payload.data()), NC_NOERR);
                    for (int a = 0; a < natts; ++a) {
                        char attribute[NC_MAX_NAME + 1]{};
                        ASSERT_EQ(nc_inq_attname(ncid, id, a, attribute), NC_NOERR);
                        nc_type type = NC_NAT;
                        size_t len = 0;
                        ASSERT_EQ(nc_inq_att(ncid, id, attribute, &type, &len), NC_NOERR);
                        if (type == NC_CHAR) {
                            std::string text(len, '\0');
                            ASSERT_EQ(nc_get_att_text(ncid, id, attribute, text.data()), NC_NOERR);
                            snapshot->attributes[std::string(name) + ":" + attribute] = text;
                        }
                    }
                }
            }
            EXPECT_EQ(nc_close(ncid), NC_NOERR);
        }();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::filesystem::remove_all(directory);
}

TEST(ParallelStandaloneWriter, SerialAndParallelFilesAgree) {
    FileSnapshot serial, parallel;
    WriteBands(5, false, false, false, false, &serial);
    WriteBands(5, false, false, true, false, &parallel);
    EXPECT_EQ(serial.values, parallel.values);
    EXPECT_EQ(serial.dimensions, parallel.dimensions);
    EXPECT_EQ(serial.attributes, parallel.attributes);
}

TEST(ParallelStandaloneWriter, CurvilinearFilesAgree) {
    FileSnapshot serial, parallel;
    WriteBands(5, false, false, false, false, &serial, true);
    WriteBands(5, false, false, true, false, &parallel, true);
    EXPECT_EQ(serial.values, parallel.values);
    EXPECT_EQ(serial.dimensions, parallel.dimensions);
    EXPECT_EQ(serial.attributes, parallel.attributes);
}

TEST(ParallelStandaloneWriter, CoordinateOnlyTimeIsWritten) {
    WriteBands(5, false, false, true, false, nullptr, false, true);
}

TEST(ParallelStandaloneWriter, UnevenBandsAndBoundedBatches) {
    WriteBands(5, false, false);
}
TEST(ParallelStandaloneWriter, OpenFailureIsCollective) {
    WriteBands(5, false, false, true, true);
}
TEST(ParallelStandaloneWriter, SerialOpenFailureIsCollective) {
    WriteBands(5, false, false, false, true);
}
TEST(ParallelStandaloneWriter, SerialFallback) {
    WriteBands(5, false, false, false);
}
TEST(ParallelStandaloneWriter, UnstructuredEmptyBands) {
    WriteBands(1, false, false);
}
TEST(ParallelStandaloneWriter, EmptyBands) {
    WriteBands(2, false, false);
}
TEST(ParallelStandaloneWriter, InvalidShapeFailsCollectively) {
    WriteBands(5, true, false);
}
TEST(ParallelStandaloneWriter, DifferentNamesFailCollectively) {
    WriteBands(5, false, true);
}
}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int required = std::getenv("CECE_TEST_SERIALIZED_MPI") ? MPI_THREAD_SERIALIZED : MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, required, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    const int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
