#!/usr/bin/env python3
"""Verify the GFS notebook deliverables without needing a CECE kernel.

Run with any python3 that has nbformat installed (the host Homebrew python does);
nothing here imports cece, so this is checkable on a machine that cannot run the
notebook at all.

usage: python3 docs/notebooks/gfs_docker_checks.py [data|sh|nb|all]
                                             [--sync|--check-sync|--self-test]
"""

from __future__ import annotations

import ast
import re
import sys

DATA_PATH = "docs/notebooks/gfs_docker_data.py"
KERNEL_NAME = "cece-gfs"
CASE_KEYS = ("date", "hour", "product", "leads", "thresholds", "dt", "cache_dir", "climo_dir")
EXPECTED_PROBLEMS = 4


def strip_marks(text: str) -> str:
    """Drop '# %%' cell markers so a percent-format notebook script parses as python."""
    return "\n".join(
        "" if line.lstrip().startswith("# %%") else line for line in text.splitlines()
    )


def check_data_module(path: str = DATA_PATH) -> list[str]:
    problems: list[str] = []
    try:
        source = open(path).read()
    except OSError:
        return [f"{path}: missing"]
    try:
        tree = ast.parse(strip_marks(source))
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for needed in ("object_key", "url_for", "grib_path", "ensure_gribs", "parse_leads", "main"):
        if needed not in names:
            problems.append(f"{path}: {needed}() is missing")
    if "atmos" not in source:
        problems.append(f"{path}: the 2026 archive layout needs /<cycle-hour>/atmos/ in the key")
    if "0o666" not in source:
        problems.append(
            f"{path}: grib2io writes a '<file>_<sha>.grib2ioidx' sidecar beside the GRIB, "
            f"so downloads must be chmod 0o666 for a root-owned container to reuse them"
        )
    if "no-ssl-verify" not in source and "no_ssl_verify" not in source:
        problems.append(f"{path}: needs the --no-ssl-verify escape hatch (corporate CA stores)")
    problems += check_data_behaviour(path)
    return problems


def check_data_behaviour(path: str) -> list[str]:
    """Import the data module and exercise it (no network: the fetch is stubbed)."""
    import contextlib
    import importlib.util
    import io
    import tempfile
    from pathlib import Path

    try:
        spec = importlib.util.spec_from_file_location("gfs_docker_data_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except (ImportError, OSError, SyntaxError, AttributeError) as exc:
        return [f"{path}: cannot import for behaviour checks: {exc!r}"]
    problems: list[str] = []
    date, hour = "20261103", 12
    try:
        key0 = mod.object_key(date, hour, 0)
        want = "gfs.20261103/12/atmos/gfs.t12z.pgrb2.1p00.f000"
        if key0 != want:
            problems.append(f"{path}: object_key(lead 0) is {key0!r}, expected {want!r}")
        for lead, suffix in ((3, "f003"), (12, "f012")):
            key = mod.object_key(date, hour, lead)
            if not key.endswith(suffix):
                problems.append(f"{path}: object_key(lead {lead}) is {key!r}, must end {suffix}")
        url = mod.url_for(date, hour, 0)
        bucket = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        if not (url.startswith(bucket) and url.endswith(key0)):
            problems.append(f"{path}: url_for(lead 0) is {url!r}, expected {bucket}{key0}")
        with tempfile.TemporaryDirectory() as tmpdir:
            gp = Path(mod.grib_path(tmpdir, date, hour, 0))
            if Path(tmpdir) not in gp.parents:
                problems.append(f"{path}: grib_path() {gp} is not under the cache dir")
            if mod.parse_leads("0, 3, 6, 12") != [0, 3, 6, 12]:
                problems.append(f"{path}: parse_leads('0, 3, 6, 12') != [0, 3, 6, 12]")
            if not hasattr(mod, "_fetch"):
                return problems + [f"{path}: needs a single _fetch(url, tmp, no_ssl_verify) hook"]
            written = []
            orig_fetch, orig_min = mod._fetch, mod.MIN_BYTES

            def fake(url, tmp, no_ssl_verify=False):
                written.append(url)
                Path(tmp).write_bytes(b"GRIB")

            mod._fetch, mod.MIN_BYTES = fake, 1
            try:
                leads = [0, 3]
                stack = contextlib.ExitStack()
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                paths = [Path(p) for p in mod.ensure_gribs(tmpdir, date, hour, leads)]
                if len(paths) != len(leads) or not all(p.exists() for p in paths):
                    problems.append(f"{path}: ensure_gribs() returned missing files: {paths}")
                for p in paths:
                    if p.exists() and (p.stat().st_mode & 0o777) != 0o666:
                        mode = oct(p.stat().st_mode & 0o777)
                        problems.append(f"{path}: {p.name} mode is {mode}, expected 0o666")
                first = len(written)
                mod.ensure_gribs(tmpdir, date, hour, leads)
                stack.close()
                if first != len(leads) or len(written) != first:
                    problems.append(
                        f"{path}: cache not honoured: fetches {first} then {len(written)} "
                        f"for {len(leads)} leads"
                    )
            except Exception as exc:  # noqa: BLE001 - report, never traceback
                problems.append(f"{path}: ensure_gribs() raised {exc!r}")
            finally:
                stack.close()
                mod._fetch, mod.MIN_BYTES = orig_fetch, orig_min
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{path}: behaviour check raised {exc!r}")
    return problems


CHECKS = {"data": check_data_module}


def main(argv: list[str]) -> int:
    flags = [a for a in argv if a.startswith("--")]
    selectors = [a for a in argv if not a.startswith("--")]
    if "--sync" in flags or "--check-sync" in flags or "--self-test" in flags:
        import test_gfs_docker_checks as fixture  # noqa: F401  (Tasks 2+)

        handler = {
            "--sync": fixture.sync_cmd,
            "--check-sync": fixture.check_sync_cmd,
            "--self-test": fixture.self_test,
        }[next(iter(flags))]
        return handler(*selectors)
    targets = selectors or list(CHECKS)
    if "all" in targets:
        targets = list(CHECKS)
    problems: list[str] = []
    for name in targets:
        if name not in CHECKS:
            problems.append(f"unknown selector {name!r}; choices: {sorted(CHECKS)}")
            continue
        problems += CHECKS[name]()
    for problem in problems:
        print("FAIL", problem)
    if problems:
        return 1
    print("PASS", ",".join(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
