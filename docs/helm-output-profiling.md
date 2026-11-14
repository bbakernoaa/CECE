# HELM output profiling

The first implementation passed correctness tests but regressed writer throughput
on this host. Profiling identified and removed redundant field copies, a second
AXIS mesh/coordinate construction, and full-document YAML copies on configuration
lookups. The final measurements below include all paired trials, including the
noisy many-field trials. Both serial and parallel paths remain enabled and were measured.

## What was measured

The benchmark calls the real `CeceStandaloneWriter::WriteTimeStep`, including
manifest parsing, coordinate generation, schema preparation, MPI transfers,
NetCDF writes, completion, and file close. Input allocation and independent
NetCDF readback are outside the timed interval. Every run checks every output
field element against its expected value. A failed readback fails the run.

The baseline is the writer source at CECE develop commit
`4b1afa18578b229a7cbea9816828bd22d3481380`, compiled under a renamed class with
identical optimization flags and configuration types. The initial writer-only
comparison used the same HELM runtime on both sides. The final before/after
comparison selects saved, pre-optimization CECE/AMIO libraries for the baseline
and the optimized libraries for the candidate, so it includes the CONF fix.
The saved runtime uses HELM `eeb037b` (PR #41 before its profiling fixes).
The candidate fixes are committed in HELM `12ad43d`. Trials were captured
before commit: their `helm_ref` is the checked-out base, while the recorded
production source fingerprints identify the candidate changes and match the
final source. This is **not** a complete old-develop dependency build or a
whole-model run.

Raw result records contain the compiled baseline revision, source and binary
SHA256 fingerprints, runtime fingerprints, and `ldd` evidence that the saved
baseline libraries were selected. Debug information was stripped from saved
libraries; executable code was retained. These binary snapshots are local
artifacts and are not committed.

Environment: Linux x86-64 container, AMD EPYC 9V74 host, GCC 13, Kokkos 5.1.1,
OpenMPI 4.1.6, parallel NetCDF 4.9, `RelWithDebInfo` (`-O2`), one OpenMP thread
per rank, an eight-GiB memory limit and two-CPU quota. Storage is the local
container overlay. OpenMPI CMA was disabled because this sandbox rejects
`process_vm_readv`; all formal paired runs use the same transport settings.
These are warm-cache output measurements, without a power-loss durability claim.
They do not qualify HPC scaling, accelerator execution or object storage.

Each independently launched MPI process group performs one warm-up followed by
seven measured writes. Three paired rounds cover all workloads at one and two
ranks. Five additional paired rounds, with nine measured writes each, confirm
the noisy many-field case. Variant/case order is randomized with seed 41.

| Case | Grid / levels | Fields | Field payload per output |
| --- | --- | ---: | ---: |
| Small | 360 × 180 × 1 | 8 | 3.96 MiB |
| Layered | 720 × 360 × 8 | 8 | 126.56 MiB |
| Global | 1440 × 720 × 4 | 8 | 253.13 MiB |
| Many fields | 360 × 180 × 4 | 32 | 63.28 MiB |

## Results

At one rank the candidate uses the serial fallback; these rows measure that
path after its dense-copy and configuration fixes.

Elapsed time is the maximum over ranks. Times below are medians of the
independent process-group medians. Speedup is the median of the paired
baseline/candidate ratios, so it can differ from the ratio of displayed times.
The range contains every paired ratio; it is not a confidence interval.

| Case | Ranks | Baseline ms | Candidate ms | Paired speedup | Pair range | Rounds |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Small | 1 | 53.83 | 45.59 | 1.188× | 1.161–1.194× | 3 |
| Small | 2 | 53.91 | 43.01 | 1.253× | 1.218–1.381× | 3 |
| Layered | 1 | 142.25 | 137.12 | 1.040× | 1.011–1.070× | 3 |
| Layered | 2 | 167.85 | 151.23 | 1.085× | 1.022–1.148× | 3 |
| Global | 1 | 347.42 | 304.88 | 1.140× | 1.021–1.167× | 3 |
| Global | 2 | 362.68 | 324.91 | 1.122× | 1.116–1.180× | 3 |
| Many fields | 1 | 97.78 | 95.88 | 1.016× | 0.873–1.218× | 8 |
| Many fields | 2 | 109.41 | 100.72 | 1.086× | 0.892–1.328× | 8 |

The measured two-rank medians improve by 1.085–1.253×. The many-field case is
noisy: the additional five-round confirmation had a two-rank median speedup of
1.092× with every confirmation pair above 1.0×, but the complete eight-round
range still includes slower individual pairs. Single-rank many-field throughput
is effectively unchanged within noise. There is no claim that every individual
write or every production configuration is faster.

Memory is per-process peak RSS captured before validation reads. The sum of
rank peaks is an upper bound rather than a simultaneous job RSS measurement.
Parallel output creates a staging pool on each writer, so small outputs can
increase aggregate memory even while reducing rank-0 memory.

| Case, two ranks | Rank-0 baseline MiB | Rank-0 candidate MiB | Rank-0 reduction | Sum of peaks, baseline → candidate MiB |
| --- | ---: | ---: | ---: | --- |
| Small | 113.9 | 102.6 | 10% | 142.4 → 204.8 |
| Layered | 337.0 | 183.5 | 46% | 434.2 → 354.6 |
| Global | 455.9 | 265.4 | 42% | 625.0 → 495.7 |
| Many fields | 205.9 | 137.8 | 33% | 264.7 → 273.0 |

## Causes and fixes

- Dense `LayoutLeft(i,j,k)` host fields already have the NetCDF `[k,j,i]`
  contiguous order. Both output paths now borrow that source through AMIO's
  synchronous snapshot admission instead of packing an identical copy first.
  Non-dense/padded views still pack through their actual strides; MPI tests
  explicitly verify that padded allocations retain correct values.
- The parallel path generated coordinates to prepare the schema, then built
  the AXIS mesh again after field completion. It now retains coordinate
  payloads from the first visit. The call-level profile confirmed two visits
  per output before this fix and one afterward.
- CONF's resolver cloned the entire YAML document before every dotted lookup.
  Larger manifests amplified this cost across the many accessors used at open.
  It now traverses const handles and clones only the selected subtree. The
  isolated-snapshot behavior is preserved, including aliases and concurrent
  list reads. A deterministic regression reduced scalar-lookup allocations
  from 8,274 to 16 with 1,024 unrelated document entries; it also verifies root
  immutability and cyclic-alias isolation. The dimension parser also resolves
  its map once rather than copying it for every variable.
- A temporary `NC_NOFILL` experiment did not improve the many-field result
  (99.9 ms with default fill versus 102.5 ms without fill in that diagnostic
  run). It was not adopted; output fill semantics remain unchanged.

The optional `output_io_trace` library counts/times MPI gathers/allreduces,
NetCDF create/write/sync/close, AMIO admission/completion/schema calls and the
coordinate visitor. Categories can overlap across threads and nested calls;
they must not be added together as exclusive phases. It is a Linux/OpenMPI
profiling aid, not part of production execution.

## Reproduce

Build the optional benchmark in a dependency-equipped environment:

```sh
cmake -S . -B build -DCECE_BUILD_OUTPUT_BENCHMARK=ON
cmake --build build --target benchmark_standalone_output output_io_trace -j2
python3 scripts/profile_standalone_output.py \
  --binary "$PWD/build/benchmark_standalone_output" \
  --data-dir /tmp/cece-output-profile \
  --results /tmp/output-results.jsonl \
  --ranks 1 2 --rounds 3 --repetitions 7
```

This command compares writer implementations using a common current runtime.
To reproduce the joint dependency comparison, build the same benchmark harness
in a separate checkout with HELM pinned to `eeb037b`. Save that build's
`libcece_core.so`, `libcece_driver.so`, and `libamio.so.0` in one directory, then
pass it as `--baseline-library-path` to a candidate run. Both builds must use the
same toolchain, flags and other dependencies. The runner checks the baseline
loader selection and fingerprints the files. Add repeatable `--runtime-library`
arguments to fingerprint candidate libraries too.

For Docker, use `--container NAME` and paths visible inside the container.
Our launcher flags were `--allow-run-as-root --oversubscribe --bind-to none`;
HPC runs should use their own launcher flags. The benchmark creates and removes
an owned temporary child beneath `--data-dir`, leaving existing files intact.

For an instrumented diagnostic, create a trace directory and pass these
variables to the **MPI ranks**, rather than preloading the launcher:

```sh
mpiexec -n 2 env LD_PRELOAD="$PWD/build/liboutput_io_trace.so" \
  CECE_OUTPUT_TRACE_DIR=/tmp/cece-trace \
  build/benchmark_standalone_output parallel 720 360 8 8 7 /tmp/cece-output-profile 64
```

[Combined measurements](../benchmarks/results/helm-output-summary.json),
[main raw trials](../benchmarks/results/helm-output-final.jsonl), and
[many-field confirmation](../benchmarks/results/helm-output-many-fields-confirmation.jsonl)
are committed. The [initial diagnostic matrix](../benchmarks/results/helm-output-initial.jsonl)
and intermediate common-runtime/pre-CONF results are retained alongside them, so the initial regressions and the dependency contribution are
visible. All field readbacks passed. Correctness verification additionally
passed 55 CECE, 14 HELM NetCDF/MPI and 17 CONF CTest checks, including the new
snapshot-allocation and padded-view regressions.
