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
        "docker run --platform linux/arm64 $PORT $EXTRA_ARGS \"$IMAGE\" bash <<'STAGE'",
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
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'echo ok # --help --nb --rebuild --reset --skip-build -c | --cmd )',
    ]
    for pat, _ in C.SH_STAGES:
        lines.append(f"echo x # {pat}")
    return "\n".join(lines) + "\n"


def spaced_branch_script(with_cmd_port: bool = False) -> str:
    cmd = '    EXTRA_ARGS="-p 8888:8888"' if with_cmd_port else '    EXTRA_ARGS=""'
    return minimal_good_script().replace("  --nb)", "  --nb )").replace("  -c|--cmd)", "  -c | --cmd )").replace('    EXTRA_ARGS=""', cmd, 1)


def notebook_with_assert_comments_only() -> str:
    return good_source().replace("assert CASE is not None", "# assert CASE is not None")


def notebook_with_case_missing_key() -> str:
    return good_source().replace('    "climo_dir": 0,\n', "")


def notebook_with_case_extra_key() -> str:
    return good_source().replace("}\n", '    "extra_key": 1,\n}\n', 1)


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

    bad = os.path.join(tmp, "bad.py")
    write(bad, BAD_PY)
    bad_nb = os.path.join(tmp, "bad.ipynb")
    C.sync_notebook(bad, bad_nb)
    bad_found = sorted(C.check_notebook(bad_nb, requires=("bootstrap", "case", "asserts", "cells")))
    expected_bad = sorted([
        f"{bad_nb}: no sys.path bootstrap to the volume-built cece bindings",
        f"{bad_nb}: CASE is missing keys {sorted(C.CASE_KEYS)}",
        f"{bad_nb}: 0 runtime asserts, expected at least {C.MIN_ASSERTS}",
        f"{bad_nb}: 1 code cells, expected at least {C.MIN_CODE_CELLS}",
    ])
    if bad_found != expected_bad:
        problems.append(f"BAD fixture mismatch: got {bad_found}, expected {expected_bad}")

    sh_good = os.path.join(tmp, "good.sh")
    sh_comment_only = os.path.join(tmp, "comment-only.sh")
    sh_spaced_ok = os.path.join(tmp, "spaced-ok.sh")
    sh_spaced_bad = os.path.join(tmp, "spaced-bad.sh")
    write(sh_good, minimal_good_script())
    write(sh_comment_only, comments_only_script())
    write(sh_spaced_ok, spaced_branch_script(False))
    write(sh_spaced_bad, spaced_branch_script(True))

    if C.check_script(sh_good):
        problems.append("minimal structurally-correct script was rejected")
    if not C.check_script(sh_comment_only):
        problems.append("comment-tail-only tokens script was not rejected")
    if C.check_script(sh_spaced_ok):
        problems.append("spaced branch labels were not accepted")
    if not C.check_script(sh_spaced_bad):
        problems.append("spaced cmd branch with port was not rejected")

    aonly_py = os.path.join(tmp, "assert-comments.py")
    aonly_nb = os.path.join(tmp, "assert-comments.ipynb")
    write(aonly_py, notebook_with_assert_comments_only())
    C.sync_notebook(aonly_py, aonly_nb)
    aonly_found = C.check_notebook(aonly_nb, requires=("asserts",))
    if not any("runtime asserts" in p for p in aonly_found):
        problems.append("asserts in comments/strings were incorrectly counted")

    miss_py = os.path.join(tmp, "case-missing.py")
    miss_nb = os.path.join(tmp, "case-missing.ipynb")
    write(miss_py, notebook_with_case_missing_key())
    C.sync_notebook(miss_py, miss_nb)
    miss_found = C.check_notebook(miss_nb, requires=("case",))
    if not any("CASE is missing keys" in p for p in miss_found):
        problems.append("CASE missing-key notebook was not rejected")

    extra_py = os.path.join(tmp, "case-extra.py")
    extra_nb = os.path.join(tmp, "case-extra.ipynb")
    write(extra_py, notebook_with_case_extra_key())
    C.sync_notebook(extra_py, extra_nb)
    extra_found = C.check_notebook(extra_nb, requires=("case",))
    if not any("CASE has unexpected keys" in p for p in extra_found):
        problems.append("CASE extra-key notebook was not rejected")

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
                        "id": "code-000",
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
        problems.append("whitespace-only sync divergence was not detected")

    return problems


def self_test(*_) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        problems = check_self_test_contracts(tmp)
    for p in problems:
        print("SELF-TEST FAIL", p)
    if problems:
        return 1
    print("SELF-TEST pass: sync/script/notebook guards reject false-pass fixtures and keep valid cases green")
    return 0


# pytest-collectible tests while preserving fixture API for --self-test

def test_self_test_contracts_roundtrip(tmp_path):
    assert check_self_test_contracts(str(tmp_path)) == []


def test_bad_fixture_exact_problem_set(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(BAD_PY)
    bad_nb = tmp_path / "bad.ipynb"
    C.sync_notebook(str(bad), str(bad_nb))
    got = sorted(C.check_notebook(str(bad_nb), requires=("bootstrap", "case", "asserts", "cells")))
    expected = sorted([
        f"{bad_nb}: no sys.path bootstrap to the volume-built cece bindings",
        f"{bad_nb}: CASE is missing keys {sorted(C.CASE_KEYS)}",
        f"{bad_nb}: 0 runtime asserts, expected at least {C.MIN_ASSERTS}",
        f"{bad_nb}: 1 code cells, expected at least {C.MIN_CODE_CELLS}",
    ])
    assert got == expected


def test_comment_tail_script_is_rejected(tmp_path):
    sh = tmp_path / "comment-tail.sh"
    sh.write_text(comments_only_script())
    assert C.check_script(str(sh))


def test_spaced_case_labels_supported_and_cmd_port_blocked(tmp_path):
    sh_ok = tmp_path / "spaced-ok.sh"
    sh_bad = tmp_path / "spaced-bad.sh"
    sh_ok.write_text(spaced_branch_script(False))
    sh_bad.write_text(spaced_branch_script(True))
    assert C.check_script(str(sh_ok)) == []
    assert any("must not be set" in p for p in C.check_script(str(sh_bad)))


def test_asserts_in_comments_do_not_count(tmp_path):
    py = tmp_path / "assert-comments.py"
    nb = tmp_path / "assert-comments.ipynb"
    py.write_text(notebook_with_assert_comments_only())
    C.sync_notebook(str(py), str(nb))
    found = C.check_notebook(str(nb), requires=("asserts",))
    assert any("runtime asserts" in p for p in found)


def test_case_missing_and_extra_keys_are_distinct(tmp_path):
    py_m = tmp_path / "m.py"
    nb_m = tmp_path / "m.ipynb"
    py_e = tmp_path / "e.py"
    nb_e = tmp_path / "e.ipynb"
    py_m.write_text(notebook_with_case_missing_key())
    py_e.write_text(notebook_with_case_extra_key())
    C.sync_notebook(str(py_m), str(nb_m))
    C.sync_notebook(str(py_e), str(nb_e))
    m = C.check_notebook(str(nb_m), requires=("case",))
    e = C.check_notebook(str(nb_e), requires=("case",))
    assert any("CASE is missing keys" in p for p in m)
    assert any("CASE has unexpected keys" in p for p in e)
