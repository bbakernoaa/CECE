#!/usr/bin/env python3
"""Fixtures that prove gfs_docker_checks actually discriminates.

Run: python3 docs/notebooks/gfs_docker_checks.py --self-test
"""

from __future__ import annotations

import os
import re

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
        "docker run -i --platform linux/arm64 -e XDG_DATA_HOME=/cache/xdg $PORT $EXTRA_ARGS \"$IMAGE\" bash <<'STAGE'",
        "export MAMBA_ROOT_PREFIX=/env",
        "export PATH=/env/bin:$PATH",
        'micromamba create -n "$ENV_NAME" python=3.12 fsspec cartopy --override-channels',
        "micromamba run -n \"$ENV_NAME\" \"$PY\" -m pip install --no-deps extern/helm/libs/axis git+https://github.com/noaa-oar-arl/monetio@67b1071e30bbcbf2f8853b1c6ecc432c430a8df3 'numpy<3'",
        "git -C /work archive HEAD",
        'micromamba run -n "$ENV_NAME" cmake -S . -B build -DCECE_PYTHON_SIMULATION=OFF -DCECE_FETCH_DEPS=OFF -DBUILD_PYTHON_BINDINGS=ON pybind11',
        'micromamba run -n "$ENV_NAME" cmake --build build --target _cece_core',
        'micromamba run -n "$ENV_NAME" "$PY" -m ipykernel install',
        'micromamba run -n "$ENV_NAME" "$PY" -m jupyter nbconvert --to notebook',
        'micromamba run -n "$ENV_NAME" "$PY" gfs_docker_data.py',
        'for p in /work/docs/notebooks/fengsha_gfs_kok.yaml /work/docs/notebooks/fengsha_gfs_webb.yaml; do'
        ' "$PY" -c "import cece,sys; c=cece.load_config(sys.argv[1]); assert c.validate()==\'Validation passed\'" "$p"; done',
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
        "        python=3.12 'numpy<3' pybind11 grib2io fsspec cartopy >/dev/null",
        '    RUN "$PY" -m pip install -q --no-deps /work/extern/helm/libs/axis',
        '    RUN "$PY" -m pip install -q --no-deps /cache/monetio  # noaa-oar-arl/monetio',
        "    git -C /work archive HEAD | tar -x -C /cache/src",
        '    RUN cmake -S /cache/src -B /cache/build -DBUILD_PYTHON_BINDINGS=ON \\',
        "        -DCECE_PYTHON_SIMULATION=OFF -DCECE_FETCH_DEPS=OFF",
        "    RUN cmake --build /cache/build --target _cece_core",
        '    RUN "$PY" -m ipykernel install --name "${ENV_NAME}"',
        '    for cfg in /work/docs/notebooks/fengsha_gfs_kok.yaml /work/docs/notebooks/fengsha_gfs_webb.yaml; do',
        '        RUN "$PY" -c \'import cece,sys; c=cece.load_config(sys.argv[1]);'
        ' assert c.validate()=="Validation passed"\' "$cfg"',
        "    done",
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
            "RUN_ARGS=(--rm -i --platform linux/arm64 -e XDG_DATA_HOME=/cache/xdg",
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
            "export MAMBA_ROOT_PREFIX=/env",
            'export PATH=/env/bin:"${PATH}"',
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


def test_second_docker_run_invocation_is_rejected(tmp_path):
    """A second, unchecked docker run call must not slip through just because the
    first one looks fine: every requirement before this fix was only ever evaluated
    against the first match, so a smuggled second call was invisible."""
    script = plan_shaped_script() + '\ndocker run bad bash <<\'STAGE\'\necho bad\nSTAGE\n'
    sh = tmp_path / "two-docker-runs.sh"
    sh.write_text(script)
    found = C.check_script(str(sh))
    assert found
    assert any("exactly one docker run" in x for x in found), found


def test_host_uid_forged_inside_stage_is_rejected(tmp_path):
    """HOST_UID/HOST_GID must come from the host-side RUN_ARGS array that docker run
    actually expands, not from a stage line that merely echoes the same text: the
    container never receives an echoed value, and the stage's own chown then fails
    under 'set -u', leaving the executed notebook root-owned."""
    script = plan_shaped_script().replace(
        '    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)")', ")"
    ).replace(
        f'    chown "${{HOST_UID}}:${{HOST_GID}}" "${{NOTEBOOK}}" 2>/dev/null || true',
        '    echo "-e HOST_UID=1 -e HOST_GID=1"\n'
        f'    chown "${{HOST_UID}}:${{HOST_GID}}" "${{NOTEBOOK}}" 2>/dev/null || true',
    )
    sh = tmp_path / "forged-hostuid.sh"
    sh.write_text(script)
    found = C.check_script(str(sh))
    assert any("HOST_UID/HOST_GID" in x for x in found), found


def test_main_rejects_unknown_flag(capsys):
    assert C.main(["data", "--bogus"]) == 2
    err = capsys.readouterr().err
    assert "unknown flag" in err and "--bogus" in err


def test_main_rejects_unknown_flag_after_print(tmp_path, capsys):
    path = tmp_path / "f.txt"
    path.write_text("hello")
    assert C.main(["--print", str(path), "--bogus"]) == 2
    err = capsys.readouterr().err
    assert "unknown flag" in err and "--bogus" in err


def test_main_rejects_conflicting_mode_flags(capsys):
    assert C.main(["--sync", "--check-sync"]) == 2
    err = capsys.readouterr().err
    assert "at most one of" in err


def test_docker_run_inside_command_substitution_is_rejected(tmp_path):
    """`ignored=$(docker run ...)` starts a container from mid-line.

    Anchoring the "exactly one docker run" count to line starts let a second
    invocation hide inside a substitution while the first, compliant call absorbed
    every requirement.
    """
    script = plan_shaped_script().replace(
        'docker run "${RUN_ARGS[@]}" \\',
        'ignored=$(docker run --rm evil sh -c "unreviewed")\ndocker run "${RUN_ARGS[@]}" \\',
        1,
    )
    assert script != plan_shaped_script()
    sh = tmp_path / "sub-docker-run.sh"
    sh.write_text(script)
    found = C.check_script(str(sh))
    assert any("exactly one docker run" in x for x in found), found


def test_quoted_mention_of_docker_run_is_not_an_invocation(tmp_path):
    """Usage/help prose may name `docker run` without being counted as a call."""
    script = plan_shaped_script().replace(
        "MODE=lab", 'usage() { echo "this script uses docker run once"; }', 1
    )
    assert script != plan_shaped_script()
    sh = tmp_path / "quoted-docker-mention.sh"
    sh.write_text(script)
    assert C.check_script(str(sh)) == []


def test_fragmented_spellings_of_docker_run_are_counted(tmp_path):
    """`d''ocker run` / `do"cker" run` are the same command to bash; blanking quoted
    spans made them invisible, so a second invocation could hide behind fragmentation."""
    for line in (
        "d''ocker run --rm image true",
        'do"cker" run --rm image true',
        '"docker" run --rm image true',
        "/usr/bin/docker run --rm image true",
        "if docker run --rm image true; then :; fi",
        "x=1 docker run --rm image true",
        "ignored=$(d''ocker run --rm evil sh -c 'x')",
        "sudo docker run img",
        "docker container run img",
        "docker --debug run img",
        "{ docker run --rm evil true; }",
        "case 1 in 1) docker run --rm evil true;; esac",
        "2>/dev/null docker run --rm evil true",
        "nohup docker run --rm evil true &",
    ):
        assert C.docker_run_in_line(line), line
    # Quoted *prose* and non-words must stay invisible, and indirection (an alias, a
    # variable holding the command name) is documented as out of scope.
    for line in (
        'echo "this script uses docker run once"',
        "echo 'literal docker run in single quotes'",
        "echo docker-running-status",
        "my-docker-run-helper",
        "docker runner img",
        "echo hi docker run img",
        "D=docker; $D run img",
    ):
        assert not C.docker_run_in_line(line), line


def test_fragmented_docker_run_is_not_two_invocations_when_only_one_exists(tmp_path):
    """Counting must stay honest: the compliant script has one call even though its
    quoted arguments contain words, and a fragmented second call pushes it to two."""
    script = plan_shaped_script()
    sh = tmp_path / "one.sh"
    sh.write_text(script)
    assert len([x for x in C.check_script(str(sh)) if "exactly one docker run" in x]) == 0
    frag = script.replace(
        'docker run "${RUN_ARGS[@]}" \\',
        "d''ocker run --rm evil true\ndocker run \"${RUN_ARGS[@]}\" \\",
        1,
    )
    assert frag != script
    sh2 = tmp_path / "two.sh"
    sh2.write_text(frag)
    assert any("exactly one docker run" in x for x in C.check_script(str(sh2)))


def test_operators_inside_quotes_are_not_command_separators(tmp_path):
    """`echo "docker run; docker run"` starts no container.

    Reconstructing quote fragments must not resurrect the operators that were literal
    characters inside the quotes, or a compliant script gets rejected for prose.
    """
    for line in (
        'echo "docker run; docker run"',
        'printf %s "x && docker run"',
        'echo "a; docker run b"',
        "echo 'docker run | sh'",
    ):
        assert not C.docker_run_in_line(line), line
    script = plan_shaped_script().replace(
        "MODE=lab",
        'usage() { echo "this script wraps: docker run; docker run again"; }',
        1,
    )
    assert script != plan_shaped_script()
    sh = tmp_path / "quoted-ops.sh"
    sh.write_text(script)
    assert C.check_script(str(sh)) == []


def test_docker_run_in_double_quoted_substitution_is_counted(tmp_path):
    # bash expands and runs $( ) / backticks inside double quotes
    sh = tmp_path / "gfs-docker.sh"
    for hidden in (
        'echo "$(docker run --rm evil true)"',
        'echo "`docker run --rm evil true`"',
        'ignored=$(docker run --rm evil sh -c "x")',
        'X="$(docker run --rm evil true)"',
    ):
        sh.write_text(minimal_good_script() + "\n" + hidden + "\n")
        problems = C.check_script(str(sh))
        assert any("exactly one docker run" in p for p in problems), (hidden, problems)


def test_docker_run_text_inside_double_quotes_is_not_a_call(tmp_path):
    sh = tmp_path / "gfs-docker.sh"
    for prose in (
        'echo "we use docker run once"',
        'echo "docker run; docker run"',
        'printf %s "x && docker run"',
        'echo "a; docker run b"',
    ):
        sh.write_text(minimal_good_script() + "\n" + prose + "\n")
        problems = C.check_script(str(sh))
        assert problems == [], (prose, problems)


COUNTEDFormS = None  # placeholder to keep the diff readable


def _count_calls(text: str) -> int:
    return sum(1 for _ in C.docker_run_call_sites(text))


def test_wrapper_and_flag_forms_are_counted(tmp_path):
    sh = tmp_path / "gfs-docker.sh"
    forms = [
        "{ docker run --rm evil true; }",
        "case 1 in 1) docker run --rm evil true;; esac",
        ">/tmp/out docker run --rm evil true",
        "2>/dev/null docker run --rm evil true",
        "sudo docker run --rm evil true",
        "env A=1 docker run --rm evil true",
        "echo a | xargs docker run --rm evil",
        "docker --debug run --rm evil true",
        "docker -H unix:///var/run/docker.sock run --rm evil true",
        "docker container run --rm evil true",
        'bash -c \'docker run --rm evil true\'',
        'eval "docker run --rm evil true"',
        "docker run y; docker run x",  # two calls on one line
    ]
    for form in forms:
        sh.write_text(minimal_good_script() + "\n" + form + "\n")
        problems = C.check_script(str(sh))
        assert any("exactly one docker run" in p for p in problems), (form, problems)


def test_non_invocations_stay_invisible(tmp_path):
    sh = tmp_path / "gfs-docker.sh"
    prose = [
        'echo "we use docker run once"',
        'echo "docker run; docker run"',
        'printf %s "x && docker run"',
        'echo hi docker run img',  # echo's argument, not a command
        "echo docker-running-status",
        "my-docker-run-helper",
        "docker runner img",
        'x="${DOCKER:-docker} run ${IMAGE:-img}"',  # quoted text, inert
        '# the docker run call publishes the port',
        # a real multi-line double quoted string: bash keeps the state per line
        'x="\ntwo\nlines\n',
    ]
    for text in prose:
        sh.write_text(minimal_good_script() + "\n" + text + "\n")
        problems = C.check_script(str(sh))
        assert problems == [], (text, problems)


def test_call_sites_counted_per_line_not_per_file(tmp_path):
    sh = tmp_path / "gfs-docker.sh"
    sh.write_text(minimal_good_script())
    assert C.check_script(str(sh)) == []
    # the same script with a second call added on its own line fails shut
    sh.write_text(minimal_good_script() + "\nnohup docker run --rm evil true &\n")
    assert any("exactly one docker run" in p for p in C.check_script(str(sh)))


def test_quoted_multiline_strings_and_comments_do_not_leak_state():
    # bash really does continue a quote across newlines, so a call after one is hidden
    hidden = 'x="\nfoo\ndocker run --rm evil true\n"\n'
    assert len(C.docker_run_call_sites(hidden)) == 0
    # ... and the next real call after it is still found
    assert len(set(C.docker_run_call_sites(hidden + "\ndocker run --rm real true\n"))) == 1
    # an apostrophe inside a comment must not swallow the rest of the file
    commented = "# it's a comment\nif [ -n \"x\" ]; then :; fi\ndocker run --rm real true\n"
    assert len(set(C.docker_run_call_sites(commented))) == 1
    # a real call inside a comment is not a call
    assert len(set(C.docker_run_call_sites("# docker run --rm evil true\n"))) == 0


def sites(text: str) -> int:
    return len(set(C.docker_run_call_sites(text)))


def test_ansi_c_quoting_does_not_hide_later_calls():
    # $'...' treats \' as an escaped quote, so the string closes on this line
    assert sites("x=$'a\\'b'\ndocker run x\n") == 1
    assert sites("echo $'it\\'s' docker run x\n") == 0  # echo argument is inert text
    assert sites("$'a\\'b'\nnohup docker run --rm evil true &\n") == 1


def test_command_split_by_line_continuation_is_counted():
    assert sites("doc\\\nker run x\n") == 1
    assert sites("doc\\\nker run a; docker run b\n") == 2


def test_heredoc_bodies_are_data_not_commands():
    assert sites("cat <<EOF\ndocker run x\nEOF\n") == 0
    assert sites("docker run img <<'STAGE'\necho hi\nSTAGE\n") == 1
    assert sites("cat <<-EOF\n\tdocker run x\n\tEOF\ndocker run real\n") == 1


def test_parameter_expansion_and_trap_strings_are_counted():
    assert sites("ignored=${x:+docker run --rm evil true}\n") == 1
    assert sites("x=1 ${y:+docker run --rm evil true}\n") == 1
    assert sites("trap 'docker run --rm evil true' EXIT\n") == 1
    # a double-quoted expansion value is data, not a command line
    assert sites('cmd="${y:+docker run --rm evil}"\nexec bash -c "$cmd"\n') == 0


def test_multiple_heredocs_on_one_line_are_all_data():
    # cat <<A <<B reads body A then body B; neither line is a host command
    two = "cat <<A <<B\ndocker run 1\nA\ndocker run 2\nB\ndocker run 3\n"
    assert sites(two) == 1
    assert sites("cat <<A\nx\nA\ndocker run real\n") == 1


def test_unquoted_heredoc_expands_command_substitution_only():
    assert sites("cat <<A\n$(docker run --rm evil true)\nA\n") == 1
    assert sites("cat <<A\n`docker run --rm evil true`\nA\n") == 1
    # a quoted delimiter keeps the whole body literal
    assert sites("cat <<'A'\n$(docker run --rm evil true)\ndocker run x\nA\n") == 0
    # the body of the plan-style STAGE heredoc is not host code either
    assert sites("docker run img <<'STAGE'\necho $(date)\nSTAGE\n") == 1


def test_stdin_stage_needs_interactive_stdin(tmp_path):
    """Without -i Docker never attaches stdin, so the quoted STAGE heredoc is discarded and
    the container exits 0 having run nothing -- the silent no-op the first cold run hit."""
    no_i = plan_shaped_script().replace("RUN_ARGS=(--rm -i --platform", "RUN_ARGS=(--rm --platform")
    sh = tmp_path / "no-i.sh"
    sh.write_text(no_i)
    found = C.check_script(str(sh))
    assert any("docker run must pass -i" in x for x in found), found
    good = tmp_path / "good.sh"
    good.write_text(plan_shaped_script())
    assert C.check_script(str(good)) == []


def test_interactive_flag_spellings_and_non_qualifying_ones(tmp_path):
    """--interactive and a literal -i on the docker line both qualify; an -i that is not a
    standalone flag (in a value, or a longer word) must not satisfy the check."""
    base = plan_shaped_script()
    long_form = base.replace("RUN_ARGS=(--rm -i --platform", "RUN_ARGS=(--rm --interactive --platform")
    on_line = base.replace('    "$IMAGE" bash -euo pipefail -s', '    "$IMAGE" -i bash -euo pipefail -s')
    for variant in (long_form, on_line):
        sh = tmp_path / "v.sh"
        sh.write_text(variant)
        assert C.check_script(str(sh)) == [], C.check_script(str(sh))
        assert len(C.docker_run_call_sites(variant)) == 1
    decoys = (
        # a balanced -e assignment whose *value* contains -i is not the flag being asked for
        base.replace("RUN_ARGS=(--rm -i --platform", 'RUN_ARGS=(--rm -e OPT="-i x" --platform'),
        base.replace("RUN_ARGS=(--rm -i --platform", "RUN_ARGS=(--rm --init-x --platform"),
        base.replace("RUN_ARGS=(--rm -i --platform", "RUN_ARGS=(--rm -init --platform"),
    )
    for variant in decoys:
        sh = tmp_path / "d.sh"
        sh.write_text(variant)
        problems = C.check_script(str(sh))
        assert C.docker_run_call_sites(variant), (variant, problems)
        assert any("docker run must pass -i" in x for x in problems), (variant, problems)


def test_interactive_required_for_bare_stdin_stage_too(tmp_path):
    """The minimal fixture spells the stage as a bare ``bash <<'STAGE'`` (no -s), which reads
    stdin exactly the same way, so the requirement cannot depend on the -s spelling."""
    no_i = minimal_good_script().replace("docker run -i --platform", "docker run --platform")
    sh = tmp_path / "bare.sh"
    sh.write_text(no_i)
    assert any("docker run must pass -i" in x for x in C.check_script(str(sh)))


def test_minimal_fixture_still_passes(tmp_path):
    sh = tmp_path / "m.sh"
    sh.write_text(minimal_good_script())
    assert C.check_script(str(sh)) == []


# --------------------------------------------------------------------------- configs gate

GOOD_KOK = """
driver:
  start_time: "2026-11-03T12:00:00"
  end_time: "2026-11-04T00:00:00"
  timestep_seconds: 3600
physics_schemes:
  - name: fengsha
    language: cpp
    options:
      use_brittle_impaction: true
"""

# The gate requires the Webb file to pin the three constants only its branch reads, so the
# good Webb fixture has to carry them too -- a bare `true`->`false` replace would not.
GOOD_WEBB = GOOD_KOK.replace(
    "      use_brittle_impaction: true\n",
    "      use_brittle_impaction: false\n      alpha: 1.0\n      gamma: 1.0\n      kvhmax: 2.45e-4\n",
)


def _write(tmp_path, name, text):
    """Config fixtures live in tmp_path; `write()` above takes a path string instead."""
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_check_fengsha_configs_accepts_matching_pair(tmp_path):
    kok = _write(tmp_path, "fengsha_gfs_kok.yaml", GOOD_KOK)
    webb = _write(tmp_path, "fengsha_gfs_webb.yaml", GOOD_WEBB)
    assert C.check_fengsha_configs((kok, webb)) == []


def test_check_fengsha_configs_flags_driver_drift(tmp_path):
    kok = _write(tmp_path, "fengsha_gfs_kok.yaml", GOOD_KOK)
    webb = _write(tmp_path, "fengsha_gfs_webb.yaml",
                  GOOD_WEBB.replace('end_time: "2026-11-04T00:00:00"',
                                    'end_time: "2026-11-04T06:00:00"'))
    found = C.check_fengsha_configs((kok, webb))
    assert len(found) == 1 and "driver" in found[0], found


def test_check_fengsha_configs_flags_wrong_brittle_flag(tmp_path):
    kok = _write(tmp_path, "fengsha_gfs_kok.yaml", GOOD_KOK)
    webb = _write(tmp_path, "fengsha_gfs_webb.yaml", GOOD_KOK)  # both true
    found = C.check_fengsha_configs((kok, webb))
    assert any("use_brittle_impaction" in p for p in found), found


def test_check_fengsha_configs_flags_missing_scheme(tmp_path):
    kok = _write(tmp_path, "fengsha_gfs_kok.yaml", GOOD_KOK.replace("name: fengsha", "name: other"))
    webb = _write(tmp_path, "fengsha_gfs_webb.yaml", GOOD_WEBB)
    found = C.check_fengsha_configs((kok, webb))
    assert any("fengsha" in p for p in found), found


def test_real_config_files_pass_the_gate():
    assert C.check_fengsha_configs() == []


def test_check_fengsha_configs_flags_unpinned_webb_constant(tmp_path):
    """The Webb branch must pin alpha/gamma/kvhmax rather than lean on the C++ defaults."""
    kok = _write(tmp_path, "fengsha_gfs_kok.yaml", GOOD_KOK)
    webb = _write(tmp_path, "fengsha_gfs_webb.yaml",
                  GOOD_WEBB.replace("      kvhmax: 2.45e-4\n", ""))
    found = C.check_fengsha_configs((kok, webb))
    assert len(found) == 1 and "kvhmax" in found[0], found


def test_check_fengsha_configs_flags_missing_file(tmp_path):
    found = C.check_fengsha_configs((str(tmp_path / "fengsha_gfs_kok.yaml"),
                                     str(tmp_path / "fengsha_gfs_webb.yaml")))
    assert len(found) == 2 and all("missing" in p for p in found), found


# ------------------------------------------------------------------- launcher owns no case

CASE_VARS_GONE = (
    "CASE_DATE", "CASE_HOUR", "CASE_LEADS", "CECE_PYTHONPATH", "GFS_CACHE", "--case",
)


def test_launcher_no_longer_carries_the_case(tmp_path):
    """The cycle lives in the CECE configs; the launcher must not re-export it."""
    src = open("docs/notebooks/gfs-docker.sh").read()
    leftovers = [name for name in CASE_VARS_GONE if name in src]
    assert leftovers == [], leftovers


def test_probe_validates_both_configs():
    # Same slicing check_script uses: strip comments, then take the quoted STAGE heredoc body.
    src = open("docs/notebooks/gfs-docker.sh").read()
    stage = " ".join(C.quoted_stage_body(C.drop_noop_prefixed_lines(
        C.strip_shell_comments(src, strip_heredoc_body_comments=True))).splitlines())
    assert "fengsha_gfs_kok.yaml" in stage and "fengsha_gfs_webb.yaml" in stage
    assert "load_config" in stage


def test_usage_covers_exactly_the_header_comment_block():
    """`usage()` prints a hardcoded line range; deleting a usage line without retuning it
    silently starts printing shell code (or stops before the last note)."""
    src = open("docs/notebooks/gfs-docker.sh").read().splitlines()
    last_comment = 1
    for i, line in enumerate(src[1:], start=2):
        if not line.startswith("#"):
            break
        last_comment = i
    m = re.search(r"usage\(\) \{ sed -n '(\d+),(\d+)p'", "\n".join(src))
    assert m, "usage() must print the header comment block with sed -n '<a>,<b>p'"
    assert int(m.group(1)) == 2, m.group(1)
    assert int(m.group(2)) == last_comment, (
        f"usage() stops at line {m.group(2)} but the header comment block ends at "
        f"{last_comment} — it would print shell code: {src[int(m.group(2)):last_comment]!r}")


def test_retired_case_knobs_are_a_gate_failure(tmp_path):
    sh = tmp_path / "d.sh"
    sh.write_text(minimal_good_script().replace(
        'IMAGE="${CECE_IMAGE:-cece/cece-dev:latest}"',
        'IMAGE="${CECE_IMAGE:-cece/cece-dev:latest}"\nCASE_DATE="${CASE_DATE:-20261103}"'))
    assert any("retired" in p for p in C.check_script(str(sh))), C.check_script(str(sh))
