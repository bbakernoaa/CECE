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


def write(path: str, text: str) -> None:
    with open(path, "w") as fh:
        fh.write(text)


def minimal_good_script() -> str:
    stage_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'IMAGE="${CECE_IMAGE:-cece/cece-dev:latest}"',
        'ENV_NAME="${GFS_ENV_NAME:-cece-gfs}"',
        "MODE=${1:-lab}",
        'case "$MODE" in',
        "  --help)",
        "    ;;&",
        "  --nb)",
        '    EXTRA_ARGS="-e HOST_UID=123 -e HOST_GID=456"',
        "    ;;&",
        "  -c|--cmd)",
        '    EXTRA_ARGS=""',
        "    ;;&",
        "  --rebuild)",
        "    REBUILD=1",
        "    ;;&",
        "  --reset)",
        "    RESET=1",
        "    ;;&",
        "  --skip-build)",
        "    SKIP_BUILD=1",
        "    ;;&",
        "  lab)",
        '    PORT="-p 8888:8888"',
        "    ;;&",
        "esac",
        'docker run --platform linux/arm64 $PORT $EXTRA_ARGS "$IMAGE" bash <<\'STAGE\'',
        'micromamba create -n "$ENV_NAME" python=3.12 --override-channels',
        "micromamba run -n \"$ENV_NAME\" \"$PY\" -m pip install --no-deps extern/helm/libs/axis git+https://github.com/noaa-oar-arl/monetio@67b1071e30bbcbf2f8853b1c6ecc432c430a8df3 'numpy<3'",
        "git -C /work archive HEAD",
        'micromamba run -n "$ENV_NAME" cmake -S . -B build -DCECE_PYTHON_SIMULATION=OFF -DCECE_FETCH_DEPS=OFF -DBUILD_PYTHON_BINDINGS=ON pybind11',
        'micromamba run -n "$ENV_NAME" cmake --build build --target _cece_core',
        'micromamba run -n "$ENV_NAME" "$PY" -m ipykernel install',
        'micromamba run -n "$ENV_NAME" "$PY" -m jupyter nbconvert --to notebook',
        'micromamba run -n "$ENV_NAME" "$PY" gfs_docker_data.py',
        'chown "$HOST_UID:$HOST_GID" docs/notebooks/fengsha_gfs.ipynb',
        "# noaa-oar-arl/monetio",
        "# 67b1071e30bbcbf2f8853b1c6ecc432c430a8df3",
        "# 'numpy<3'",
        "STAGE",
    ]
    return "\n".join(stage_lines) + "\n"


def comments_only_script() -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines += ["# --help --nb --rebuild --reset --skip-build -c |--cmd )"]
    for pat, _ in C.SH_STAGES:
        lines.append(f"# {pat}")
    return "\n".join(lines) + "\n"


def check_self_test_contracts(tmp: str) -> list[str]:
    import json

    problems: list[str] = []
    good = os.path.join(tmp, "good.py")
    write(good, good_source())
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

    blank_py = os.path.join(tmp, "blank.py")
    blank_nb = os.path.join(tmp, "blank.ipynb")
    write(blank_py, "# %%\n\nvalue = 1\n")
    write(
        blank_nb,
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": "value = 1\n",
                        "outputs": [],
                        "execution_count": None,
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "name": C.KERNEL_NAME,
                        "display_name": C.KERNEL_DISPLAY,
                        "language": "python",
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
    )
    if not C.check_sync(blank_py, blank_nb):
        problems.append("blank-line mismatch at cell start was not detected")

    future_py = os.path.join(tmp, "future.py")
    future_nb = os.path.join(tmp, "future.ipynb")
    write(future_py, "# %%\nfrom __future__ import annotations\nanswer: int = 1\n")
    C.sync_notebook(future_py, future_nb)
    future_cells = C.parse_cells(future_nb)
    if not future_cells or "from __future__ import annotations" not in future_cells[0][1]:
        problems.append("from __future__ line was not preserved into synced notebook")

    bad = os.path.join(tmp, "bad.py")
    write(bad, BAD_PY)
    bad_nb = os.path.join(tmp, "bad.ipynb")
    C.sync_notebook(bad, bad_nb)
    n = len(C.check_notebook(bad_nb, requires=("bootstrap", "case", "asserts", "cells")))
    if n != C.EXPECTED_PROBLEMS:
        problems.append(f"BAD fixture produced {n} problems, expected {C.EXPECTED_PROBLEMS}")

    sh_good = os.path.join(tmp, "good.sh")
    sh_comment_only = os.path.join(tmp, "comment-only.sh")
    sh_double = os.path.join(tmp, "double-run.sh")
    sh_wrong_port = os.path.join(tmp, "wrong-port.sh")
    write(sh_good, minimal_good_script())
    write(sh_comment_only, comments_only_script())
    write(sh_double, minimal_good_script().replace("docker run", "docker run\ndocker run", 1))
    write(sh_wrong_port, minimal_good_script().replace('    EXTRA_ARGS=""', '    EXTRA_ARGS="-p 8888:8888"'))

    if C.check_script(sh_good):
        problems.append("minimal structurally-correct script was rejected")
    if not C.check_script(sh_comment_only):
        problems.append("comment-only tokens script was not rejected")
    if not C.check_script(sh_double):
        problems.append("script with two docker run calls was not rejected")
    if not C.check_script(sh_wrong_port):
        problems.append("script with -p in cmd branch was not rejected")

    tampered = os.path.join(tmp, "tampered.ipynb")
    C.sync_notebook(good, tampered)
    write(tampered.replace(".ipynb", ".py"), good_source("\n# %%\ny = 1\n"))
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
    print("SELF-TEST pass: sync and script fixtures reject all false-pass holes and keep valid cases green")
    return 0
