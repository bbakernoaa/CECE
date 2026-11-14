// Measures real writer calls; input setup and independent readback are untimed.
#include <mpi.h>
#include <netcdf.h>
#include <sys/resource.h>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "cece/cece_logger.hpp"
#include "cece/cece_standalone_writer.hpp"
#include "cece_develop_standalone_writer.hpp"

namespace {
void Check(bool ok, const std::string& message) {
    if (!ok) {
        std::cerr << message << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double CpuSeconds() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec + 1e-6 * (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
}

long PeakRss() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

void Verify(const std::string& path, int nx, int ny, int nz, int fields) {
    int id = -1;
    Check(nc_open(path.c_str(), NC_NOWRITE, &id) == NC_NOERR, "readback open");
    std::vector<double> actual(static_cast<size_t>(nx) * ny * nz);
    for (int f = 0; f < fields; ++f) {
        int var = -1;
        Check(nc_inq_varid(id, ("species" + std::to_string(f)).c_str(), &var) == NC_NOERR, "readback variable");
        Check(nc_get_var_double(id, var, actual.data()) == NC_NOERR, "readback payload");
        for (size_t i = 0; i < actual.size(); ++i) Check(actual[i] == static_cast<double>(f) * 1000000000.0 + i, "readback value mismatch");
    }
    Check(nc_close(id) == NC_NOERR, "readback close");
}

template <class Writer>
void Measure(Writer& writer, const std::unordered_map<std::string, cece::DualView3D>& fields, const std::string& variant,
             const std::string& directory, int nx, int ny, int nz, int count, int repetitions, int rank, int ranks) {
    Check(writer.Initialize("2020-01-01T00:00:00", nx, ny, nz) == 0, "writer initialize");
    const long input_rss = PeakRss();
    std::vector<double> timings, cpus;
    long output_rss = 0;
    for (int rep = -1; rep < repetitions; ++rep) {
        Kokkos::fence();
        MPI_Barrier(MPI_COMM_WORLD);
        const double cpu_start = CpuSeconds();
        const double start = MPI_Wtime();
        Check(writer.WriteTimeStep(fields, 3600.0, rep + 1) == 0, "writer failed");
        const double elapsed = MPI_Wtime() - start;
        const double cpu = CpuSeconds() - cpu_start;
        double maximum = 0, total_cpu = 0;
        MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&cpu, &total_cpu, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        output_rss = std::max(output_rss, PeakRss());
        if (rank == 0) {
            if (rep >= 0) {
                timings.push_back(maximum);
                cpus.push_back(total_cpu);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    long max_rss = 0, root_rss = output_rss, max_input_rss = 0, sum_rss = 0;
    MPI_Reduce(&output_rss, &max_rss, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&output_rss, &sum_rss, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&input_rss, &max_input_rss, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    writer.Finalize();
    if (rank == 0) {
        Verify(directory + "/output.nc", nx, ny, nz, count);
        std::sort(timings.begin(), timings.end());
        const double median = timings[timings.size() / 2];
        const double bytes = static_cast<double>(nx) * ny * nz * count * sizeof(double);
        std::cout << std::setprecision(9) << "OUTPUT_BENCH {\"variant\":\"" << variant << "\",\"nx\":" << nx << ",\"ny\":" << ny << ",\"nz\":" << nz
                  << ",\"fields\":" << count << ",\"ranks\":" << ranks << ",\"repetitions\":" << repetitions << ",\"baseline_ref\":\""
                  << CECE_OUTPUT_BASELINE_REF << "\""
                  << ",\"median_s\":" << median << ",\"min_s\":" << timings.front() << ",\"max_s\":" << timings.back()
                  << ",\"payload_mib\":" << bytes / 1048576.0 << ",\"mib_per_s\":" << bytes / 1048576.0 / median
                  << ",\"mean_total_cpu_s\":" << std::accumulate(cpus.begin(), cpus.end(), 0.0) / cpus.size()
                  << ",\"max_rank_peak_rss_kib\":" << max_rss << ",\"root_peak_rss_kib\":" << root_rss << ",\"sum_rank_peak_rss_kib\":" << sum_rss
                  << ",\"max_input_peak_rss_kib\":" << max_input_rss << ",\"samples_s\":[";
        for (size_t i = 0; i < timings.size(); ++i) std::cout << (i ? "," : "") << timings[i];
        std::cout << "],\"readback\":\"PASS\"}" << std::endl;
        std::filesystem::remove_all(directory);
    }
}
}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    Check(provided == MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE required");
    Kokkos::initialize(argc, argv);
    cece::CeceLogger::GetInstance().SetLogLevel(cece::LogLevel::ERROR);
    {
        Check(argc == 9, "usage: benchmark_standalone_output develop|fallback|parallel nx ny nz fields repetitions directory staging_mib");
        const std::string variant = argv[1];
        std::string directory = argv[7];
        const int nx = std::stoi(argv[2]), ny = std::stoi(argv[3]), nz = std::stoi(argv[4]), count = std::stoi(argv[5]);
        const int repetitions = std::stoi(argv[6]), staging_mib = std::stoi(argv[8]);
        Check(nx > 0 && ny > 1 && nz > 0 && count > 0 && repetitions > 0 && staging_mib > 0 && staging_mib <= 1024,
              "positive benchmark dimensions required");
        int rank = 0, ranks = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        if (rank == 0) {
            try {
                std::filesystem::create_directories(directory);
                std::string pattern = std::filesystem::absolute(directory).string() + "/cece-output-XXXXXX";
                std::vector<char> writable(pattern.begin(), pattern.end());
                writable.push_back('\0');
                const char* owned = mkdtemp(writable.data());
                Check(owned != nullptr, "create owned benchmark directory");
                directory = owned;
            } catch (const std::exception& e) {
                Check(false, e.what());
            }
        }
        int directory_length = static_cast<int>(directory.size());
        MPI_Bcast(&directory_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        directory.resize(directory_length);
        MPI_Bcast(directory.data(), directory_length, MPI_CHAR, 0, MPI_COMM_WORLD);

        const auto band = cece::BandDecomposition::compute(ny, MPI_COMM_WORLD);
        cece::CeceOutputConfig config;
        config.enabled = true;
        config.directory = directory;
        config.filename_pattern = "output.nc";
        config.amio_worker_threads = 1;
        config.amio_staging_buffer_count = 2;
        config.amio_staging_buffer_capacity_bytes = static_cast<int>(static_cast<size_t>(staging_mib) * 1048576);
        config.amio_parallel_output = variant == "parallel";
        config.fields.SetTimeUnits("2020-01-01T00:00:00");
        std::unordered_map<std::string, cece::DualView3D> fields;
        for (int f = 0; f < count; ++f) {
            const std::string name = "species" + std::to_string(f);
            cece::DualView3D view(name, nx, band.ny_local, nz);
            auto h = view.view_host();
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < band.ny_local; ++j)
                    for (int i = 0; i < nx; ++i)
                        h(i, j, k) = static_cast<double>(f) * 1000000000.0 + (static_cast<size_t>(k) * ny + band.j0 + j) * nx + i;
            view.modify_host();
            fields.emplace(name, view);
        }
        if (variant == "develop") {
            cece::CeceDevelopStandaloneWriter writer(config, MPI_COMM_WORLD);
            Measure(writer, fields, variant, directory, nx, ny, nz, count, repetitions, rank, ranks);
        } else {
            Check(variant == "parallel" || variant == "fallback", "unknown variant");
            cece::CeceStandaloneWriter writer(config, MPI_COMM_WORLD);
            Measure(writer, fields, variant, directory, nx, ny, nz, count, repetitions, rank, ranks);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
}
