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
