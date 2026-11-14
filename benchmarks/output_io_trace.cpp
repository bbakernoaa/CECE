// Optional LD_PRELOAD diagnostic. Timings overlap; do not add categories.
#include <amio/amio.h>
#include <dlfcn.h>
#include <mpi.h>
#include <netcdf.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>

namespace {
struct Counter {
    const char* name;
    std::atomic<unsigned long long> calls{0}, nanoseconds{0};
};
Counter counters[] = {
    {"MPI_Gatherv"}, {"MPI_Allreduce"}, {"nc_create"},   {"nc_create_par"}, {"nc_put_vara"}, {"nc_close"},  {"nc_sync"},
    {"amio_init"},   {"amio_open"},     {"amio_schema"}, {"amio_batch"},    {"amio_write"},  {"amio_wait"}, {"coordinate_generation"}};
template <class Function>
auto Trace(int index, Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    auto result = function();
    counters[index].calls.fetch_add(1, std::memory_order_relaxed);
    counters[index].nanoseconds.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count(),
                                          std::memory_order_relaxed);
    return result;
}
__attribute__((destructor)) void Report() {
    const char* directory = std::getenv("CECE_OUTPUT_TRACE_DIR");
    const char* rank = std::getenv("OMPI_COMM_WORLD_RANK");
    if (!directory || !rank) return;
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/rank-%s.json", directory, rank);
    FILE* file = std::fopen(path, "w");
    if (!file) return;
    std::fprintf(file, "{\"rank\":%s,\"categories\":{", rank);
    for (size_t i = 0; i < sizeof(counters) / sizeof(counters[0]); ++i)
        std::fprintf(file, "%s\"%s\":{\"calls\":%llu,\"seconds\":%.9f}", i ? "," : "", counters[i].name, counters[i].calls.load(),
                     counters[i].nanoseconds.load() * 1e-9);
    std::fprintf(file, "}}\n");
    std::fclose(file);
}
}  // namespace

#define FORWARD(index, name, signature, arguments)                                     \
    extern "C" auto name signature->decltype(name arguments) {                         \
        static auto next = reinterpret_cast<decltype(&name)>(dlsym(RTLD_NEXT, #name)); \
        if (!next) std::abort();                                                       \
        return Trace(index, [&] { return next arguments; });                           \
    }

FORWARD(0, MPI_Gatherv,
        (const void* send, int count, MPI_Datatype type, void* receive, const int* counts, const int* displacements, MPI_Datatype receive_type,
         int root, MPI_Comm comm),
        (send, count, type, receive, counts, displacements, receive_type, root, comm))
FORWARD(1, MPI_Allreduce, (const void* send, void* receive, int count, MPI_Datatype type, MPI_Op op, MPI_Comm comm),
        (send, receive, count, type, op, comm))
FORWARD(2, nc_create, (const char* path, int mode, int* id), (path, mode, id))
extern "C" int nc_create_par(const char*, int, MPI_Comm, MPI_Info, int*);
FORWARD(3, nc_create_par, (const char* path, int mode, MPI_Comm comm, MPI_Info info, int* id), (path, mode, comm, info, id))
FORWARD(4, nc_put_vara, (int id, int variable, const size_t* start, const size_t* count, const void* data), (id, variable, start, count, data))
FORWARD(5, nc_close, (int id), (id))
FORWARD(6, nc_sync, (int id), (id))
FORWARD(7, amio_init_from_string, (const char* manifest, const char* format, amio_core_handle* core), (manifest, format, core))
FORWARD(8, amio_open_dataset_from_string,
        (amio_core_handle core, const char* manifest, const char* format, int32_t mode, amio_dataset_handle* dataset),
        (core, manifest, format, mode, dataset))
FORWARD(9, amio_prepare_write_schema, (amio_dataset_handle dataset, const amio_write_variable_t* variables, size_t count),
        (dataset, variables, count))
FORWARD(10, amio_write_region_batch, (amio_dataset_handle dataset, const amio_region_write_t* writes, size_t count, amio_io_handle* io),
        (dataset, writes, count, io))
FORWARD(11, amio_write,
        (amio_dataset_handle dataset, const char* name, const void* data, amio_dtype_t type, const amio_shape_t* shape, amio_io_handle* io),
        (dataset, name, data, type, shape, io))
FORWARD(12, amio_wait, (amio_io_handle io, int64_t timeout), (io, timeout))

using Visitor = std::function<void(const char*, const void*, amio_dtype_t, const amio_shape_t*)>;
extern "C" void TraceCoordinates(void*, double, const Visitor&) asm(
    "_ZN4cece20CeceStandaloneWriter24VisitCoordinateVariablesEdRKSt8functionIFvPKcPKv12amio_dtype_ePK12amio_shape_sEE");
extern "C" void TraceCoordinates(void* writer, double time, const Visitor& visitor) {
    static auto next = reinterpret_cast<void (*)(void*, double, const Visitor&)>(
        dlsym(RTLD_NEXT, "_ZN4cece20CeceStandaloneWriter24VisitCoordinateVariablesEdRKSt8functionIFvPKcPKv12amio_dtype_ePK12amio_shape_sEE"));
    if (!next) std::abort();
    const auto start = std::chrono::steady_clock::now();
    next(writer, time, visitor);
    counters[13].calls.fetch_add(1, std::memory_order_relaxed);
    counters[13].nanoseconds.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count(),
                                       std::memory_order_relaxed);
}
