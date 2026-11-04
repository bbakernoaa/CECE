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
import os
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
    except Exception as exc:  # noqa: BLE001 - any import-time failure is a problem, not a traceback
        return [f"{path}: cannot import for behaviour checks: {exc!r}"]
    problems: list[str] = []
    date, hour = "20261103", 12
    try:
        bucket = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        with tempfile.TemporaryDirectory() as tmpdir:
            for lead in (0, 3, 12):
                name = f"gfs.t12z.pgrb2.1p00.f{lead:03d}"
                want_key = f"gfs.20261103/12/atmos/{name}"
                key = mod.object_key(date, hour, lead)
                if key != want_key:
                    problems.append(f"{path}: object_key(lead {lead}) is {key!r}, expected {want_key!r}")
                url = mod.url_for(date, hour, lead)
                if url != bucket + want_key:
                    problems.append(f"{path}: url_for(lead {lead}) is {url!r}, expected {bucket + want_key!r}")
                gp = Path(mod.grib_path(tmpdir, date, hour, lead))
                want_gp = Path(tmpdir) / want_key.replace("/", "_")
                if gp != want_gp:
                    problems.append(f"{path}: grib_path(lead {lead}) is {str(gp)!r}, expected {str(want_gp)!r}")
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
            stack = contextlib.ExitStack()
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                leads = [0, 3]
                paths = [Path(p) for p in mod.ensure_gribs(tmpdir, date, hour, leads)]
                if len(paths) != len(leads) or not all(p.exists() for p in paths):
                    problems.append(f"{path}: ensure_gribs() returned missing files: {paths}")
                for p in paths:
                    if p.exists() and (p.stat().st_mode & 0o777) != 0o666:
                        mode = oct(p.stat().st_mode & 0o777)
                        problems.append(f"{path}: {p.name} mode is {mode}, expected 0o666")
                first = len(written)
                mod.ensure_gribs(tmpdir, date, hour, leads)
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


PY_PATH = "docs/notebooks/fengsha_gfs.py"
NB_PATH = "docs/notebooks/fengsha_gfs.ipynb"
SH_PATH = "docs/notebooks/gfs-docker.sh"
MIN_CODE_CELLS = 12
MIN_ASSERTS = 5
ALL_TOKENS = ("kernelspec", "bootstrap", "case", "asserts", "cells", "report", "maps", "caveats")
KERNEL_DISPLAY = "Python 3 (CECE GFS)"


def parse_cells(path: str) -> list[tuple[str, str]]:
    """Cell (kind, source) pairs, from a percent-marked .py or an .ipynb."""
    if path.endswith(".ipynb"):
        import nbformat

        nb = nbformat.read(path, as_version=4)
        return [
            (c.cell_type, c.source if isinstance(c.source, str) else "".join(c.source))
            for c in nb.cells
        ]
    cells: list[tuple[str, str]] = []
    kind, buf = None, []
    with open(path) as fh:
        for line in fh.readlines():
            mark = re.match(r"^#\s*%%\s*(\[markdown\])?", line)
            if mark:
                if kind is not None:
                    cells.append((kind, "".join(buf)))
                kind = "markdown" if mark.group(1) else code_kind()
                buf = []
            elif kind is not None:
                buf.append(line)
    if kind is not None:
        cells.append((kind, "".join(buf)))
    return cells


def code_kind() -> str:
    return "code"


def render_ipynb(py_path: str):
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": KERNEL_NAME,
        "display_name": KERNEL_DISPLAY,
        "language": "python",
    }
    nb.metadata["jupytext"] = {"formats": "ipynb,py:percent"}
    for kind, source in parse_cells(py_path):
        source = source if source.endswith("\n") else source + "\n"
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source))
    return nb


def sync_notebook(py_path: str = PY_PATH, nb_path: str = NB_PATH) -> str:
    """Regenerate the .ipynb from the .py unless cell sources and kernelspec already match.

    Executing the notebook (Task 7) adds outputs; outputs are not compared, so a run does not
    make the pair "out of sync" — only an edited source does.
    """
    import nbformat

    rendered = render_ipynb(py_path)
    os.makedirs(os.path.dirname(nb_path) or ".", exist_ok=True)
    if os.path.exists(nb_path):
        current = nbformat.read(nb_path, as_version=4)
        same = [(c.cell_type, c.source) for c in current.cells] == [
            (c.cell_type, c.source) for c in rendered.cells
        ]
        if same and current.metadata.get("kernelspec", {}).get("name") == KERNEL_NAME:
            return "unchanged"
    with open(nb_path, "w") as fh:
        nbformat.write(rendered, fh)
    return "written"


def normalize_cell_trailing_newline(source: str) -> str:
    return source[:-1] if source.endswith("\n") else source


def check_sync(py_path: str = PY_PATH, nb_path: str = NB_PATH) -> list[str]:
    problems: list[str] = []
    if not os.path.exists(py_path):
        return [f"{py_path}: missing"]
    if not os.path.exists(nb_path):
        return [f"{nb_path}: missing (run --sync)"]
    try:
        from_nb = parse_cells(nb_path)
    except Exception as exc:  # nbformat raises several unrelated types
        return [f"{nb_path}: unreadable notebook for sync check: {exc}"]
    try:
        from_py = parse_cells(py_path)
    except Exception as exc:
        return [f"{py_path}: unreadable notebook script for sync check: {exc}"]
    if len(from_nb) != len(from_py):
        problems.append(f"{nb_path}: {len(from_nb)} cells vs {len(from_py)} in {py_path} — run --sync")
        return problems
    for i, ((ka, sa), (kb, sb)) in enumerate(zip(from_nb, from_py)):
        if ka != kb or normalize_cell_trailing_newline(sa) != normalize_cell_trailing_newline(sb):
            problems.append(f"cell {i} differs from {py_path} ({ka} vs {kb}) — run --sync")
            break
    return problems


def check_notebook(path: str = NB_PATH, requires: tuple[str, ...] = ("kernelspec",)) -> list[str]:
    problems: list[str] = []
    if not path.endswith(".ipynb"):
        return [f"{path}: expected a real .ipynb (sync first)"]
    try:
        cells = parse_cells(path)
        import nbformat

        nb = nbformat.read(path, as_version=4)
    except Exception as exc:  # missing file, malformed json, schema error
        return [f"{path}: unreadable notebook: {exc}"]
    if not cells:
        return [f"{path}: no cells found"]
    code = [src for kind, src in cells if kind == "code"]
    md = "\n".join(src for kind, src in cells if kind == "markdown")
    joined = "\n".join(code)
    if "kernelspec" in requires:
        spec = nb.metadata.get("kernelspec", {})
        if spec.get("name") != KERNEL_NAME or spec.get("display_name") != KERNEL_DISPLAY:
            problems.append(
                f"{path}: kernelspec is {spec.get('name')!r}/{spec.get('display_name')!r}, "
                f"expected {KERNEL_NAME!r}/{KERNEL_DISPLAY!r}"
            )
    if "bootstrap" in requires:
        if "sys.path.insert" not in joined or "build/src/python" not in joined:
            problems.append(f"{path}: no sys.path bootstrap to the volume-built cece bindings")
    if "case" in requires:
        missing = [k for k in CASE_KEYS if not re.search(rf'"{k}"\s*:', joined)]
        if missing:
            problems.append(f"{path}: CASE is missing keys {missing}")
    if "asserts" in requires and joined.count("assert ") < MIN_ASSERTS:
        problems.append(
            f"{path}: {joined.count('assert ')} runtime asserts, expected at least {MIN_ASSERTS}"
        )
    if "cells" in requires and len(code) < MIN_CODE_CELLS:
        problems.append(f"{path}: {len(code)} code cells, expected at least {MIN_CODE_CELLS}")
    if "report" in requires and joined.count("build_report(") < 2:
        problems.append(f"{path}: build_report() must be defined and called")
    if "maps" in requires and "pcolormesh" not in joined:
        problems.append(f"{path}: no pcolormesh map cell")
    if "caveats" in requires:
        for phrase in ("erodibility = 1", "frozen", "lake_fraction", "nearest"):
            if phrase not in md:
                problems.append(f"{path}: caveats text omits {phrase!r}")
    return problems


SH_STAGES = [
    (r'^IMAGE="\$\{CECE_IMAGE:-cece/cece-dev:latest\}"$', "the existing dev image is reused, not redefined"),
    (r'--platform linux/arm64', "build and run platform must match, or mounts break with an exec-format error"),
    (r'--override-channels', "avoid the defaults channel (license wall, different build pins)"),
    (r"python=3\.12", "grib2io is capped at 3.13 on osx-arm64; conda solves 3.12"),
    (r'^ENV_NAME="\$\{GFS_ENV_NAME:-cece-gfs\}"$', "the kernelspec name the notebook expects"),
    (r"'numpy<3'", "numpy 3 breaks grib2io and xarray here"),
    (r"pybind11", "conda pybind11 avoids the FetchContent /usr/include/python3.12 failure"),
    (r"extern/helm/libs/axis", "axis-regrid provides regrid_rectilinear"),
    (r"--no-deps.*monetio|monetio.*--no-deps", "MONETIO must not upgrade numpy under the bindings"),
    (r"noaa-oar-arl/monetio", "the only clonable MONETIO remote (NOAA-EMC prompts for credentials)"),
    (r"67b1071e30bbcbf2f8853b1c6ecc432c430a8df3", "MONETIO pinned to compatibility-lock.json"),
    (r"-DCECE_PYTHON_SIMULATION=OFF", "the image ships a static Kokkos; ON fails at src/python/CMakeLists.txt:46"),
    (r"-DCECE_FETCH_DEPS=OFF", "use the image-provided dependencies instead of downloading them"),
    (r"-DBUILD_PYTHON_BINDINGS=ON", "the notebook imports _cece_core"),
    (r"micromamba run -n", "cmake and pip must run inside the env or the .so links against the wrong python"),
    (r"gfs_docker_data\.py", "GRIB staging must reuse the shared helper, not duplicate the URL"),
    (r'git -C /work archive HEAD', "a dirty worktree snapshot breaks the build (deleted cmake/*.cmake)"),
    (r"--skip-build", "escape hatch documented in the usage header"),
    (r"--rebuild", "forces a fresh bindings build"),
    (r"-m jupyter", "Lab and nbconvert must run from the env python, not apt's jupyter"),
]


def strip_shell_comments(src: str) -> str:
    out = []
    for i, line in enumerate(src.splitlines()):
        stripped = line.lstrip()
        if stripped.startswith("#") and not (i == 0 and stripped.startswith("#!")):
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if src.endswith("\n") else "")


def branch_block(src: str, label: str) -> str:
    m = re.search(rf"^\s*{re.escape(label)}\)\n(.*?)(?=^\s*;;&?\s*$|^\s*esac\s*$)", src, re.M | re.S)
    return m.group(1) if m else ""


def check_script(path: str = SH_PATH) -> list[str]:
    problems: list[str] = []
    try:
        src = open(path).read()
    except OSError:
        return [f"{path}: missing"]
    clean = strip_shell_comments(src)
    if not src.startswith("#!/usr/bin/env bash"):
        problems.append(f"{path}: must start with #!/usr/bin/env bash")
    if "set -euo pipefail" not in clean:
        problems.append(f"{path}: needs 'set -euo pipefail'")
    for flag in ("--help", "--nb", "--rebuild", "--reset", "--skip-build", "-c |--cmd )"):
        if flag.replace(" ", "") not in clean.replace(" ", ""):
            problems.append(f"{path}: flag {flag} is not dispatched")
    for pat, why in SH_STAGES:
        if not re.search(pat, clean, re.M):
            problems.append(f"{path}: missing required stage: {pat} — {why}")
    if re.search(r'python3? -m pip', clean):
        problems.append(
            f"{path}: system python3 must not pip-install into the conda env — use \"$PY\" -m pip"
        )
    if len(re.findall(r"^\s*docker\s+run\b", clean, re.M)) != 1:
        problems.append(f"{path}: must contain exactly one docker run invocation")
    if "<<'STAGE'" not in clean or not re.search(r"^STAGE\s*$", clean, re.M):
        problems.append(f"{path}: docker stage must be a single quoted heredoc <<'STAGE' ... STAGE")

    nb_block = branch_block(clean, "--nb")
    cmd_block = branch_block(clean, "-c|--cmd")
    lab_block = branch_block(clean, "lab")
    if "-p 8888:8888" in nb_block or "-p 8888:8888" in cmd_block:
        problems.append(f"{path}: -p 8888:8888 must not be set in --nb/-c|--cmd branches")
    if "-p 8888:8888" not in lab_block:
        problems.append(f"{path}: lab branch must publish -p 8888:8888")
    if "-e HOST_UID=" not in nb_block or "-e HOST_GID=" not in nb_block:
        problems.append(f"{path}: --nb branch must pass HOST_UID/HOST_GID environment variables")
    if not re.search(r"chown\s+.*\$HOST_UID:\$HOST_GID.*ipynb", clean):
        problems.append(f"{path}: stage must chown the executed notebook with HOST_UID/HOST_GID")
    return problems


def sync_cmd(*paths: str) -> int:
    py_path = paths[0] if paths else PY_PATH
    nb_path = paths[1] if len(paths) > 1 else NB_PATH
    print(f"{sync_notebook(py_path, nb_path)} {nb_path}")
    return 0


def check_sync_cmd(*paths: str) -> int:
    problems = check_sync(*(paths or (PY_PATH, NB_PATH)))
    for p in problems:
        print("FAIL", p)
    if problems:
        return 1
    print("PASS sync")
    return 0


CHECKS = {
    "data": check_data_module,
    "sh": check_script,
    "nb": lambda: check_notebook(requires=("kernelspec", "bootstrap", "case", "asserts"))
    + check_sync(),
    "nbfinal": lambda: check_notebook(requires=ALL_TOKENS) + check_sync(),
}


def main(argv: list[str]) -> int:
    flags = [a for a in argv if a.startswith("--")]
    selectors = [a for a in argv if not a.startswith("--")]
    if "--sync" in flags or "--check-sync" in flags or "--self-test" in flags:
        import test_gfs_docker_checks as fixture

        handler = {
            "--sync": sync_cmd,
            "--check-sync": check_sync_cmd,
            "--self-test": fixture.self_test,
        }[next(iter(f for f in flags if f in ("--sync", "--check-sync", "--self-test")))]
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
