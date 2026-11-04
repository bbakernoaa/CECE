#!/usr/bin/env python3
"""Fixtures that prove gfs_docker_checks actually discriminates.

Run: python3 docs/notebooks/gfs_docker_checks.py --self-test
"""

from __future__ import annotations

import os

import gfs_docker_checks as C

CASE_BLOCK = "\n".join(
    f'    "{key}": {"[0, 3, 6, 12]" if key == "leads" else ("[\'uthres\', \'uthres_sg\']" if key == "thresholds" else "0")},'
    for key in C.CASE_KEYS
)

BAD_PY = """# %%
import numpy as np
"""


def good_source(extra: str = "") -> str:
    """A notebook source that satisfies every token except the task-specific ones."""
    cells = ["# %% [markdown]\n# # GFS-forced FENGSHA\n", f"""# %%
import sys
sys.path.insert(0, "/gfs-cache/build/src/python")
CASE = {{
{CASE_BLOCK}
}}
"""]
    cells += [f"# %%\nassert CASE is not None, {i}\n" for i in range(C.MIN_ASSERTS)]
    cells += [f"# %%\nx_{i} = {i}\n" for i in range(C.MIN_CODE_CELLS - 1 - C.MIN_ASSERTS)]
    return "\n".join(cells) + extra

def check_self_test_contracts(tmp: str) -> list[str]:
    problems: list[str] = []
    good = os.path.join(tmp, "good.py")
    open(good, "w").write(good_source())
    good_nb = os.path.join(tmp, "good.ipynb")

    if C.sync_notebook(good, good_nb) != "written":
        problems.append("first sync did not write the notebook")
    if C.sync_notebook(good, good_nb) != "unchanged":
        problems.append("sync is not idempotent")
    found = C.check_notebook(good_nb, requires=C.ALL_TOKENS[:5])
    if found:
        problems.append(f"GOOD fixture was rejected: {found}")
    if C.check_sync(good, good_nb):
        problems.append("check_sync disagreed with a fresh sync")

    bad = os.path.join(tmp, "bad.py")
    open(bad, "w").write(BAD_PY)
    bad_nb = os.path.join(tmp, "bad.ipynb")
    C.sync_notebook(bad, bad_nb)
    n = len(C.check_notebook(bad_nb, requires=("bootstrap", "case", "asserts", "cells")))
    if n != C.EXPECTED_PROBLEMS:
        problems.append(f"BAD fixture produced {n} problems, expected {C.EXPECTED_PROBLEMS}")

    tampered = os.path.join(tmp, "tampered.ipynb")
    C.sync_notebook(good, tampered)
    open(tampered.replace(".ipynb", ".py"), "a").write("\n# %%\ny = 1\n")
    if not C.check_sync(tampered.replace(".ipynb", ".py"), tampered):
        problems.append("check_sync missed an out-of-sync pair")
    return problems


def self_test(*_) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        problems = check_self_test_contracts(tmp)
    for p in problems:
        print("SELF-TEST FAIL", p)
    if problems:
        return 1
    print(f"SELF-TEST pass: good fixture accepted, bad fixture rejected with exactly "
          f"{C.EXPECTED_PROBLEMS} problems, sync round-trip stable")
    return 0
