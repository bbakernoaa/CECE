#!/usr/bin/env python3
"""Run matched real-writer benchmarks; keep raw samples and paired summaries."""

import argparse
import hashlib
import json
import os
import random
import shlex
import statistics
import subprocess
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument(
        "--container", help="Optional Docker container containing the binary"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Output parent directory seen by the binary"
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["develop", "fallback", "parallel"],
        default=["develop", "parallel"],
    )
    parser.add_argument(
        "--case", action="append", default=[], help="name:nx:ny:nz:fields (repeatable)"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--staging-mib", type=int, default=64)
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--launcher-flags", default="")
    parser.add_argument(
        "--baseline-library-path", help="Optional saved runtime for develop only"
    )
    parser.add_argument(
        "--runtime-library",
        action="append",
        default=[],
        help="Candidate runtime path to fingerprint (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if (
        min(args.ranks + [args.rounds, args.repetitions, args.staging_mib]) <= 0
        or args.staging_mib > 1024
    ):
        parser.error(
            "positive ranks, rounds and repetitions, and 1..1024 staging MiB are required"
        )
    raw_cases = args.case or [
        "small:360:180:1:8",
        "layered:720:360:8:8",
        "global:1440:720:4:8",
        "many_fields:360:180:4:32",
    ]
    cases = []
    for raw in raw_cases:
        name, *shape = raw.split(":")
        if not name or not all(
            character.isalnum() or character in "_-" for character in name
        ):
            parser.error(
                "case names must contain letters, digits, underscores or hyphens"
            )
        if len(shape) != 4 or min(map(int, shape)) <= 0:
            parser.error("case must be name:nx:ny:nz:fields with positive sizes")
        cases.append((name, *map(int, shape)))
    return args, cases


def execution_prefix(args):
    return ["docker", "exec", args.container] if args.container else []


def record_provenance(args, repo):
    def fingerprint(path):
        return subprocess.check_output(
            [*execution_prefix(args), "sha256sum", path], text=True
        ).split()[0]

    source_paths = {
        "standalone_writer": "src/driver/cece_standalone_writer.cpp",
        "parallel_writer": "src/driver/cece_parallel_writer.cpp",
        "netcdf_driver": "extern/helm/libs/amio/src/drivers/netcdf/netcdf_driver.cpp",
        "conf_resolver": "extern/helm/libs/conf/src/detail/yaml_tree.cpp",
    }
    provenance = {
        "helm_ref": subprocess.check_output(
            ["git", "-C", str(repo / "extern/helm"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((repo / path).read_bytes()).hexdigest()
            for name, path in source_paths.items()
        },
        "binary_sha256": fingerprint(args.binary),
        "runtime_sha256": {path: fingerprint(path) for path in args.runtime_library},
        "baseline_library_path": args.baseline_library_path,
        "seed": args.seed,
        "note": "Exact baseline writer compiled under renamed class; runtime common unless baseline_library_path is set. Local cached output; no durability/whole-model throughput claim.",
    }
    if args.baseline_library_path:
        paths = [
            args.baseline_library_path + "/" + name
            for name in ["libcece_core.so", "libcece_driver.so", "libamio.so.0"]
        ]
        provenance["baseline_runtime_sha256"] = {
            path: fingerprint(path) for path in paths
        }
        loader = subprocess.check_output(
            [
                *execution_prefix(args),
                "env",
                "LD_LIBRARY_PATH=" + args.baseline_library_path,
                "ldd",
                args.binary,
            ],
            text=True,
        )
        for path in paths:
            if path not in loader:
                raise RuntimeError(
                    "Baseline loader did not select saved runtime: " + path
                )
        provenance["baseline_loader"] = loader
    return provenance


def benchmark_command(args, case, ranks, variant):
    _, nx, ny, nz, fields = case
    executable = [args.binary]
    if variant == "develop" and args.baseline_library_path:
        executable = [
            "env",
            "LD_LIBRARY_PATH=" + args.baseline_library_path,
            args.binary,
        ]
    command = [
        args.launcher,
        *shlex.split(args.launcher_flags),
        "-n",
        str(ranks),
        *executable,
        variant,
        str(nx),
        str(ny),
        str(nz),
        str(fields),
        str(args.repetitions),
        args.data_dir,
        str(args.staging_mib),
    ]
    if args.container:
        command = [
            "docker",
            "exec",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "OMP_PROC_BIND=false",
            "-e",
            "OMPI_MCA_btl_vader_single_copy_mechanism=none",
            args.container,
            *command,
        ]
    return command


def measure(args, case, ranks, variant, round_index, logs):
    command = benchmark_command(args, case, ranks, variant)
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ | {"OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false"},
    )
    (logs / f"{round_index}-{case[0]}-np{ranks}-{variant}.log").write_text(
        result.stdout
    )
    if result.returncode:
        raise RuntimeError(f"Benchmark failed: {shlex.join(command)}\n{result.stdout}")
    entry = json.loads(
        next(
            line.removeprefix("OUTPUT_BENCH ")
            for line in result.stdout.splitlines()
            if line.startswith("OUTPUT_BENCH ")
        )
    )
    entry.update(case=case[0], round=round_index)
    return entry


def comparison(case, ranks, variant, baseline, current):
    ratios = [
        baseline[entry["round"]]["median_s"] / entry["median_s"] for entry in current
    ]
    row = {
        "case": case,
        "ranks": ranks,
        "variant": variant,
        "paired_speedup_median": statistics.median(ratios),
        "paired_speedup_min": min(ratios),
        "paired_speedup_max": max(ratios),
    }
    for key, label in [
        ("median_s", "median_s"),
        ("root_peak_rss_kib", "root_peak_rss_kib"),
        ("sum_rank_peak_rss_kib", "sum_rank_peak_rss_kib"),
    ]:
        row["baseline_" + label] = statistics.median(
            entry[key] for entry in baseline.values()
        )
        row["candidate_" + label] = statistics.median(entry[key] for entry in current)
    return row


def summarize(entries, cases, args):
    summary = []
    for case in cases:
        for ranks in args.ranks:
            selected = [
                entry
                for entry in entries
                if entry["case"] == case[0] and entry["ranks"] == ranks
            ]
            baseline = {
                entry["round"]: entry
                for entry in selected
                if entry["variant"] == "develop"
            }
            for variant in args.variants:
                if variant != "develop" and baseline:
                    current = [
                        entry for entry in selected if entry["variant"] == variant
                    ]
                    summary.append(
                        comparison(case[0], ranks, variant, baseline, current)
                    )
    return summary


def main():
    args, cases = parse_arguments()
    provenance = record_provenance(args, Path(__file__).resolve().parents[1])
    args.results.parent.mkdir(parents=True, exist_ok=True)
    logs = args.results.with_suffix(".logs")
    logs.mkdir(exist_ok=True)
    entries = []
    rng = random.Random(args.seed)
    with args.results.open("w") as stream:
        for round_index in range(args.rounds):
            jobs = [
                (case, ranks, variant)
                for case in cases
                for ranks in args.ranks
                for variant in args.variants
            ]
            rng.shuffle(jobs)
            for case, ranks, variant in jobs:
                entry = measure(args, case, ranks, variant, round_index, logs)
                provenance["baseline_ref"] = entry["baseline_ref"]
                entry["provenance"] = provenance
                entries.append(entry)
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                print(
                    f"round={round_index + 1} {case[0]} np{ranks} {variant}: {entry['median_s']:.6f}s; readback={entry['readback']}",
                    flush=True,
                )
    summary = summarize(entries, cases, args)
    args.results.with_suffix(".summary.json").write_text(
        json.dumps({"provenance": provenance, "comparisons": summary}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
