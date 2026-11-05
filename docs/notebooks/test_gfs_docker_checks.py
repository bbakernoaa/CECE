#!/usr/bin/env python3
"""Fixtures that prove gfs_docker_checks actually discriminates.

Run: python3 docs/notebooks/gfs_docker_checks.py --self-test
"""

from __future__ import annotations

import os

import nbformat

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


def script_with_unquoted_docker_and_quoted_decoy() -> str:
    return (
        minimal_good_script()
        .replace("bash <<'STAGE'", "bash <<STAGE", 1)
        + "\ncat <<'STAGE'\ndecoy\nSTAGE\n"
    )




def script_with_commented_stage_body() -> str:
    lines = minimal_good_script().splitlines()
    out: list[str] = []
    in_stage = False
    for line in lines:
        if "docker run" in line and "<<'STAGE'" in line:
            in_stage = True
            out.append(line)
            continue
        if in_stage and line.strip() == "STAGE":
            in_stage = False
            out.append(line)
            continue
        if in_stage:
            out.append(f"# {line}" if line else "#")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def script_with_noop_stage_body() -> str:
    lines = minimal_good_script().splitlines()
    out: list[str] = []
    in_stage = False
    for line in lines:
        if "docker run" in line and "<<'STAGE'" in line:
            in_stage = True
            out.append(line)
            continue
        if in_stage and line.strip() == "STAGE":
            in_stage = False
            out.append(line)
            continue
        if in_stage and line.strip() and not line.lstrip().startswith("#"):
            out.append(f": {line}   # kept for the checker")
        else:
            out.append(line)
    return "\n".join(out) + "\n"

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
    sh_stage_commented = os.path.join(tmp, "stage-commented.sh")
    sh_stage_noop = os.path.join(tmp, "stage-noop.sh")
    write(sh_good, minimal_good_script())
    write(sh_comment_only, comments_only_script())
    write(sh_spaced_ok, spaced_branch_script(False))
    write(sh_spaced_bad, spaced_branch_script(True))
    write(sh_stage_commented, script_with_commented_stage_body())
    write(sh_stage_noop, script_with_noop_stage_body())

    if C.check_script(sh_good):
        problems.append("minimal structurally-correct script was rejected")
    if not C.check_script(sh_comment_only):
        problems.append("comment-tail-only tokens script was not rejected")
    if C.check_script(sh_spaced_ok):
        problems.append("spaced branch labels were not accepted")
    if not C.check_script(sh_spaced_bad):
        problems.append("spaced cmd branch with port was not rejected")
    if not C.check_script(sh_stage_commented):
        problems.append("commented-out stage body script was not rejected")
    if not C.check_script(sh_stage_noop):
        problems.append("no-op neutered stage body script was not rejected")

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

    # Finding A reproduction: executed notebook must be preserved when content matches.
    exec_py = os.path.join(tmp, "executed.py")
    exec_nb = os.path.join(tmp, "executed.ipynb")
    write(exec_py, "# %%\nprint('ok')\n")
    C.sync_notebook(exec_py, exec_nb)
    nb_exec = nbformat.read(exec_nb, as_version=4)
    nb_exec.cells[0].execution_count = 7
    nb_exec.cells[0].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="ok\\n")
    ]
    nbformat.write(nb_exec, exec_nb)
    if C.sync_notebook(exec_py, exec_nb) != "unchanged":
        problems.append("executed notebook was rewritten even though cell content was unchanged")
    reread = nbformat.read(exec_nb, as_version=4)
    if not reread.cells[0].outputs or reread.cells[0].execution_count != 7:
        problems.append("executed notebook outputs/execution_count were not preserved")
    if C.check_sync(exec_py, exec_nb):
        problems.append("check_sync reported drift for executed notebook with unchanged cell content")

    # Companion: genuine content edit after execution must rewrite and then report sync clean.
    write(exec_py, "# %%\nprint('edited')\n")
    if C.sync_notebook(exec_py, exec_nb) != "written":
        problems.append("content edit after execution did not force notebook rewrite")
    if C.check_sync(exec_py, exec_nb):
        problems.append("check_sync did not clear after rewriting edited notebook")

    # Finding B reproduction: quoted decoy must not satisfy docker heredoc gate.
    sh_decoy = os.path.join(tmp, "decoy.sh")
    write(sh_decoy, script_with_unquoted_docker_and_quoted_decoy())
    decoy_found = C.check_script(sh_decoy)
    if not any("docker stage" in p for p in decoy_found):
        problems.append("unquoted docker heredoc passed due to unrelated quoted decoy block")

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


def test_sync_preserves_executed_outputs_when_cell_content_is_unchanged(tmp_path):
    py = tmp_path / "n.py"
    nb = tmp_path / "n.ipynb"
    py.write_text("# %%\nprint('ok')\n")
    C.sync_notebook(str(py), str(nb))

    doc = nbformat.read(str(nb), as_version=4)
    doc.cells[0].execution_count = 3
    doc.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="ok\\n")]
    nbformat.write(doc, str(nb))

    assert C.sync_notebook(str(py), str(nb)) == "unchanged"
    reread = nbformat.read(str(nb), as_version=4)
    assert reread.cells[0].execution_count == 3
    assert reread.cells[0].outputs
    assert C.check_sync(str(py), str(nb)) == []


def test_sync_rewrites_on_real_content_edit_even_after_execution(tmp_path):
    py = tmp_path / "n.py"
    nb = tmp_path / "n.ipynb"
    py.write_text("# %%\nprint('ok')\n")
    C.sync_notebook(str(py), str(nb))

    doc = nbformat.read(str(nb), as_version=4)
    doc.cells[0].execution_count = 3
    doc.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="ok\\n")]
    nbformat.write(doc, str(nb))

    py.write_text("# %%\nprint('edited')\n")
    assert C.sync_notebook(str(py), str(nb)) == "written"
    assert C.check_sync(str(py), str(nb)) == []


def test_unquoted_docker_heredoc_is_not_accepted_by_decoy_quoted_block(tmp_path):
    sh = tmp_path / "decoy.sh"
    sh.write_text(script_with_unquoted_docker_and_quoted_decoy())
    found = C.check_script(str(sh))
    assert any("docker stage" in p for p in found)


def test_commented_stage_body_is_rejected(tmp_path):
    sh = tmp_path / "commented-stage.sh"
    sh.write_text(script_with_commented_stage_body())
    found = C.check_script(str(sh))
    assert found


def test_noop_neutered_stage_body_is_rejected(tmp_path):
    sh = tmp_path / "noop-stage.sh"
    sh.write_text(script_with_noop_stage_body())
    found = C.check_script(str(sh))
    assert found


def test_minimal_good_script_still_passes(tmp_path):
    sh = tmp_path / "good.sh"
    sh.write_text(minimal_good_script())
    assert C.check_script(str(sh)) == []


def plan_shaped_script(stage_extra: str = "", host_tail: str = "") -> str:
    """The layout Task 4 actually writes: RUN_ARGS array, lab-conditional port, heredoc."""
    stage = [
        'RUN() { micromamba run -n "${ENV_NAME}" "$@"; }',
        'PY=/env/envs/"${ENV_NAME}"/bin/python',
        '    /env/bin/micromamba create -y -n "${ENV_NAME}" -c conda-forge --override-channels \\',
        "        python=3.12 'numpy<3' pybind11 grib2io >/dev/null",
        '    RUN "$PY" -m pip install -q --no-deps /work/extern/helm/libs/axis',
        '    RUN "$PY" -m pip install -q --no-deps /cache/monetio  # noaa-oar-arl/monetio',
        "    git -C /work archive HEAD | tar -x -C /cache/src",
        '    RUN cmake -S /cache/src -B /cache/build -DBUILD_PYTHON_BINDINGS=ON \\',
        "        -DCECE_PYTHON_SIMULATION=OFF -DCECE_FETCH_DEPS=OFF",
        "    RUN cmake --build /cache/build --target _cece_core",
        '    RUN "$PY" -m ipykernel install --name "${ENV_NAME}"',
        '    RUN "$PY" -m jupyter nbconvert --to notebook --execute "${NOTEBOOK}"',
        '    RUN "$PY" -m jupyter kernelspec list >/dev/null',
        f'    chown "${{HOST_UID}}:${{HOST_GID}}" "${{NOTEBOOK}}" 2>/dev/null || true{stage_extra}',
    ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'IMAGE="${CECE_IMAGE:-cece/cece-dev:latest}"',
            'ENV_NAME="${GFS_ENV_NAME:-cece-gfs}"',
            'PORT="${GFS_PORT:-8888}"',
            "MODE=lab",
            "NOTEBOOK=/work/docs/notebooks/fengsha_gfs.ipynb",
            "DATA_HELPER=/work/docs/notebooks/gfs_docker_data.py",
            "RUN_ARGS=(--rm --platform linux/arm64",
            '    -e ENV_NAME="$ENV_NAME" -e NOTEBOOK="$NOTEBOOK"',
            '    -e MONETIO_REMOTE="https://github.com/noaa-oar-arl/monetio.git"',
            "    -e MONETIO_SHA=67b1071e30bbcbf2f8853b1c6ecc432c430a8df3",
            '    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)")',
            'if [ "$MODE" = lab ]; then RUN_ARGS+=(-p "$PORT:8888"); fi',
            'case "$1" in',
            "    --help|-h) usage; exit 0 ;;",
            "    --skip-build) SKIP_BUILD=1 ;;",
            "    --rebuild) REBUILD=1 ;;",
            "    --reset) docker volume rm $VOL_ENV ;;",
            "    --nb) MODE=nb ;;",
            '    -c|--cmd) MODE=cmd ;;',
            "esac",
            'docker run "${RUN_ARGS[@]}" \\',
            "    \"$IMAGE\" bash -euo pipefail -s <<'STAGE'",
            *stage,
            "STAGE",
            host_tail,
        ]
    )


def test_plan_shaped_script_passes(tmp_path):
    sh = tmp_path / "plan.sh"
    sh.write_text(plan_shaped_script())
    assert C.check_script(str(sh)) == []


def test_stage_requirements_cannot_be_moved_to_the_host(tmp_path):
    """A container stage emptied of its work must not pass by repeating it host-side."""
    import re

    emptied = re.sub(
        r"(<<'STAGE'\n).*?(\nSTAGE)",
        r"\1    echo nothing\n\2",
        plan_shaped_script(),
        flags=re.S,
    )
    host_decoy = (
        "\nmicromamba run -n x cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON "
        "-DCECE_PYTHON_SIMULATION=OFF -DCECE_FETCH_DEPS=OFF\n"
        "micromamba run -n x /env/bin/python -m pip install --no-deps "
        "git+https://github.com/noaa-oar-arl/monetio@67b1071e30bbcbf2f8853b1c6ecc432c430a8df3 'numpy<3'\n"
        "micromamba create -n x -c conda-forge --override-channels python=3.12 pybind11\n"
        "micromamba run -n x python -m jupyter nbconvert --to notebook fengsha_gfs.ipynb\n"
        'chown "$HOST_UID:$HOST_GID" fengsha_gfs.ipynb\n'
    )
    sh = tmp_path / "decoy.sh"
    sh.write_text(emptied + host_decoy)
    found = C.check_script(str(sh))
    assert len([x for x in found if "heredoc" in x]) >= 8, found


def test_port_published_for_nb_mode_is_rejected(tmp_path):
    script = plan_shaped_script().replace(
        'if [ "$MODE" = lab ]; then RUN_ARGS+=(-p "$PORT:8888"); fi',
        'if [ "$MODE" = nb ]; then RUN_ARGS+=(-p "$PORT:8888"); fi',
    )
    sh = tmp_path / "nbport.sh"
    sh.write_text(script)
    assert any("only be published for lab" in x for x in C.check_script(str(sh)))


def test_neutered_chown_is_rejected(tmp_path):
    script = plan_shaped_script().replace(
        '    chown "${HOST_UID}:${HOST_GID}"', '    : chown "${HOST_UID}:${HOST_GID}"'
    )
    sh = tmp_path / "nochown.sh"
    sh.write_text(script)
    assert any("chown" in x for x in C.check_script(str(sh)))


def test_missing_host_uid_env_is_rejected(tmp_path):
    script = plan_shaped_script().replace(
        '    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)")', '    -e MPLBACKEND=Agg)'
    )
    sh = tmp_path / "noid.sh"
    sh.write_text(script)
    assert any("HOST_UID/HOST_GID" in x for x in C.check_script(str(sh)))


def test_markdown_cells_render_as_markdown_not_commented_text(tmp_path):
    """The percent format comments every markdown line; the notebook must not keep that '#'.

    Otherwise the rendered notebook shows literal '#' characters instead of headings --
    the notebook IS the deliverable, invisible to substring gates but ruinous to read.
    """
    py = tmp_path / "nb.py"
    py.write_text(
        "# %% [markdown]\n# # Title\n#\n# ## Section one\n\n"
        + "# %%\nx = 1\n"
    )
    nb_path = tmp_path / "nb.ipynb"
    assert C.sync_notebook(str(py), str(nb_path)) == "written"
    cells = nbformat.read(str(nb_path), as_version=4).cells
    md = [c for c in cells if c.cell_type == "markdown"]
    assert [c.source for c in md] == ["# Title\n\n## Section one\n\n"], md[0].source
    assert C.check_sync(str(py), str(nb_path)) == []
    # The committed skeleton must be clean too: no markdown line may start with '# #'.
    if os.path.exists(C.PY_PATH):
        for kind, src in C.parse_cells(C.PY_PATH):
            if kind == "markdown":
                assert not any(ln.startswith("# #") for ln in src.splitlines()), src[:60]
