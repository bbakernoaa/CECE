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
    return [(kind_, percent_markdown_to_text(src) if kind_ == "markdown" else src) for kind_, src in cells]


def percent_markdown_to_text(source: str) -> str:
    """Turn a percent-format markdown cell (every line commented with ``#``) into markdown.

    Without this the generated notebook renders literal ``#`` characters instead of
    headings, and the .py/.ipynb pair is no longer comparable cell-for-cell.
    """
    out = []
    for line in source.splitlines():
        if line.startswith("#"):
            out.append(line[1:].removeprefix(" "))
        else:
            out.append(line)
    text = "\n".join(out)
    return text + "\n" if source.endswith("\n") and text else text


def code_kind() -> str:
    return "code"


def normalize_cell_trailing_newline(source: str) -> str:
    """Allow only nbformat's single trailing newline convention difference."""
    return source[:-1] if source.endswith("\n") else source


def _cells_match_ignoring_execution_artifacts(
    notebook_cells: list[tuple[str, str]],
    script_cells: list[tuple[str, str]],
) -> bool:
    if len(notebook_cells) != len(script_cells):
        return False
    for (nb_kind, nb_source), (py_kind, py_source) in zip(notebook_cells, script_cells):
        if nb_kind != py_kind:
            return False
        if normalize_cell_trailing_newline(nb_source) != normalize_cell_trailing_newline(py_source):
            return False
    return True


def _notebook_matches_script_content(nb_path: str, py_path: str) -> bool:
    import nbformat

    nb = nbformat.read(nb_path, as_version=4)
    spec = nb.metadata.get("kernelspec", {}) or {}
    if spec.get("name") != KERNEL_NAME:
        return False
    nb_cells = [
        (c.cell_type, c.source if isinstance(c.source, str) else "".join(c.source))
        for c in nb.cells
    ]
    py_cells = parse_cells(py_path)
    return _cells_match_ignoring_execution_artifacts(nb_cells, py_cells)


def render_ipynb(py_path: str):
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": KERNEL_NAME,
        "display_name": KERNEL_DISPLAY,
        "language": "python",
    }
    nb.metadata["jupytext"] = {"formats": "ipynb,py:percent"}
    for idx, (kind, source) in enumerate(parse_cells(py_path)):
        source = source if source.endswith("\n") else source + "\n"
        cell_id = f"{kind}-{idx:03d}"
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(source, id=cell_id))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source, id=cell_id))
    return nb


def _rendered_notebook_text(py_path: str) -> str:
    import nbformat

    return nbformat.writes(render_ipynb(py_path))


def _current_notebook_text(nb_path: str) -> str:
    import nbformat

    current = nbformat.read(nb_path, as_version=4)
    return nbformat.writes(current)


def sync_notebook(py_path: str = PY_PATH, nb_path: str = NB_PATH) -> str:
    """Sync notebook content from the percent script while preserving execution artifacts."""
    import nbformat

    os.makedirs(os.path.dirname(nb_path) or ".", exist_ok=True)
    if os.path.exists(nb_path):
        try:
            if _notebook_matches_script_content(nb_path, py_path):
                return "unchanged"
        except Exception:
            pass
    rendered_text = nbformat.writes(render_ipynb(py_path))
    with open(nb_path, "w") as fh:
        fh.write(rendered_text)
    return "written"


def check_sync(py_path: str = PY_PATH, nb_path: str = NB_PATH) -> list[str]:
    if not os.path.exists(py_path):
        return [f"{py_path}: missing"]
    if not os.path.exists(nb_path):
        return [f"{nb_path}: missing (run --sync)"]
    try:
        parse_cells(py_path)
    except Exception as exc:
        return [f"{py_path}: unreadable notebook script for sync check: {exc}"]
    try:
        if _notebook_matches_script_content(nb_path, py_path):
            return []
    except Exception as exc:
        return [f"{nb_path}: unreadable notebook for sync check: {exc}"]
    return [f"{nb_path}: differs from {py_path} content/kernelspec.name — run --sync"]


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

    assert_count = 0
    case_keys: set[str] | None = None
    for idx, src in enumerate(code):
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            problems.append(f"{path}: code cell {idx} has syntax error: {exc}")
            continue
        assert_count += sum(isinstance(n, ast.Assert) for n in ast.walk(tree))
        for stmt in tree.body:
            value_node = None
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name) and t.id == "CASE":
                        value_node = stmt.value
                        break
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "CASE":
                value_node = stmt.value
            if isinstance(value_node, ast.Dict):
                parsed_keys = set()
                for k in value_node.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        parsed_keys.add(k.value)
                case_keys = parsed_keys

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
        expected = set(CASE_KEYS)
        found = case_keys or set()
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        if missing:
            problems.append(f"{path}: CASE is missing keys {missing}")
        if unexpected:
            problems.append(f"{path}: CASE has unexpected keys {unexpected}")
    if "asserts" in requires and assert_count < MIN_ASSERTS:
        problems.append(f"{path}: {assert_count} runtime asserts, expected at least {MIN_ASSERTS}")
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


def _strip_inline_comment_outside_quotes(line: str) -> str:
    out = []
    in_single = False
    in_double = False
    escaped = False
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_double:
            out.append(ch)
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            continue
        if ch == "#" and not in_single and not in_double:
            return "".join(out).rstrip()
        out.append(ch)
    return "".join(out).rstrip()


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

# These describe how the host invokes Docker (image, platform, env name, MONETIO pin
# passed with -e, and the CLI escape hatches), so they legitimately live outside the
# bootstrap heredoc. Every other stage requirement is a container instruction and is
# checked inside the heredoc body only: scanning the whole file would let a host-side
# decoy satisfy a requirement that the container never executes.
SH_FILE_SCOPED = frozenset(
    {
        r'^IMAGE="\$\{CECE_IMAGE:-cece/cece-dev:latest\}"$',
        r'--platform linux/arm64',
        r'^ENV_NAME="\$\{GFS_ENV_NAME:-cece-gfs\}"$',
        r"noaa-oar-arl/monetio",
        r"67b1071e30bbcbf2f8853b1c6ecc432c430a8df3",
        r"--skip-build",
        r"--rebuild",
        r"gfs_docker_data\.py",
    }
)


def strip_shell_comments(src: str, *, strip_heredoc_body_comments: bool = False) -> str:
    out = []
    heredoc_end = None
    for i, line in enumerate(src.splitlines()):
        if heredoc_end is not None:
            body_clean = _strip_inline_comment_outside_quotes(line) if strip_heredoc_body_comments else line
            if body_clean.lstrip().startswith("#"):
                body_clean = ""
            if body_clean.strip() != "" or line.strip() == heredoc_end:
                out.append(body_clean if line.strip() != heredoc_end else line.strip())
            if line.strip() == heredoc_end:
                heredoc_end = None
            continue
        stripped = line.lstrip()
        if stripped.startswith("#") and not (i == 0 and stripped.startswith("#!")):
            continue
        clean = _strip_inline_comment_outside_quotes(line)
        if clean.strip() == "" and not (i == 0 and stripped.startswith("#!")):
            continue
        out.append(clean)
        import re as _re
        m = _re.search(r'<<\s*([\'\"]?)([A-Za-z_][A-Za-z0-9_]*)\1', clean)
        if m:
            heredoc_end = m.group(2)
    return "\n".join(out) + ("\n" if src.endswith("\n") else "")


def drop_noop_prefixed_lines(src: str) -> str:
    kept = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if re.match(r"^(?::|true|false)(?:\s|$)", stripped):
            continue
        kept.append(line)
    return "\n".join(kept) + ("\n" if src.endswith("\n") else "")

def branch_block(src: str, label: str) -> str:
    label_pat = r"\s*\|\s*".join(re.escape(part.strip()) for part in label.split("|"))
    m = re.search(
        rf"^\s*{label_pat}\s*\)\s*\n(.*?)(?=^\s*;;&?\s*$|^\s*esac\s*$)",
        src,
        re.M | re.S,
    )
    return m.group(1) if m else ""


_DOCKER_RUN_RE = re.compile(r"(?<![-\w])docker\s+run\b(?![-\w])")


def _unquoted_view(text: str) -> str:
    """Blank out quoted spans so words inside quotes cannot look like commands.

    Escaped characters inside double quotes are kept in step so `\"docker` does not
    flip the quote state.
    """
    out = []
    quote = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is None:
            if ch in ("'", '"'):
                quote = ch
                out.append(" ")
            else:
                out.append(ch)
        else:
            if ch == "\\" and quote == '"':
                out.append(" ")
                if i + 1 < len(text) and text[i + 1] == '"':
                    out.append(" ")
                    i += 2
                    continue
            elif ch == quote:
                quote = None
                out.append(" ")
            else:
                out.append(" ")
        i += 1
    return "".join(out)


def docker_run_in_line(line: str) -> bool:
    """True if the line contains an *executable* ``docker run``, not just the words.

    A line-start match is not enough: `ignored=$(docker run --rm evil ...)` starts a
    container from mid-line, and `if docker run ...; then` hides one behind a shell
    keyword. Either would let a second, unreviewed invocation slip past the
    exactly-one rule while the compliant call satisfied every requirement. Quoted
    text is blanked first, so usage prose that merely names the command does not count.
    """
    return bool(_DOCKER_RUN_RE.search(_unquoted_view(line)))


def docker_run_command(src: str) -> tuple[str, int, int] | None:
    """The single ``docker run`` invocation: its text, start offset, and the offset
    just after its last line.

    Shell line continuations are followed, so a heredoc operator written on a
    continuation line (the usual formatting) still belongs to the container call.
    Returns None unless there is *exactly* one ``docker run`` line in the script —
    zero is obviously wrong, but more than one is equally dangerous: a second,
    unchecked invocation could run anything while every requirement below is
    satisfied by the first. Treating "not exactly one" as "no sole call exists"
    means every Docker-bound check (heredoc ownership, STAGE terminator, stage-body
    scoping, host-side env vars) fails shut instead of silently checking only the
    first of several invocations.
    """
    lines = src.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if docker_run_in_line(ln)]
    if len(starts) != 1:
        return None
    i = starts[0]
    start = sum(len(ln) for ln in lines[:i])
    text = ""
    while i < len(lines):
        text += lines[i]
        if not lines[i].rstrip("\r\n").rstrip().endswith("\\"):
            break
        i += 1
    end = start + len(text)
    return text, start, end


def _balanced_paren_assignment(text: str, assign_start: int) -> str:
    """From the position of a ``NAME=(`` / ``NAME+=(`` match, return through the
    matching close paren, tracking depth so a nested ``$(...)`` command substitution
    (e.g. ``-e HOST_UID="$(id -u)"``) doesn't end the match early."""
    idx = text.index("(", assign_start)
    depth = 0
    for i in range(idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[assign_start : i + 1]
    return text[assign_start:]


def _assignment_value(text: str, assign_start: int) -> str:
    """From the position of a ``NAME=`` match, return the whole assignment: a
    parenthesised array (depth-balanced), a quoted string, or a bare word."""
    eq_idx = text.index("=", assign_start)
    after = eq_idx + 1
    if after < len(text) and text[after] == "(":
        return _balanced_paren_assignment(text, assign_start)
    if after < len(text) and text[after] in "\"'":
        quote = text[after]
        end = text.find(quote, after + 1)
        if end == -1:
            return text[assign_start:]
        return text[assign_start : end + 1]
    end = after
    while end < len(text) and text[end] not in " \t\r\n\\":
        end += 1
    return text[assign_start:end]


def docker_host_scope(src: str) -> str:
    """The ``docker run`` command text plus any variable or bash array it expands
    (e.g. a ``RUN_ARGS=( ... -e HOST_UID=... )`` array referenced as
    ``"${RUN_ARGS[@]}"``, or a plain ``EXTRA_ARGS="-e HOST_UID=..."`` string
    referenced as ``$EXTRA_ARGS``), restricted to text that precedes the container
    call.

    This is the only text host-side requirements (like forwarding HOST_UID/HOST_GID
    into the container) may be satisfied from. A heredoc stage line that merely
    echoes ``-e HOST_UID=...`` is never part of the host invocation and must not
    count: the container never actually receives it.
    """
    cmd = docker_run_command(src)
    if cmd is None:
        return ""
    text, start, _end = cmd
    preamble = src[:start]
    scope = text
    for name in set(re.findall(r"\$\{?(\w+)(?:\[@\])?\}?", text)):
        for m in re.finditer(rf"^[ \t]*{re.escape(name)}\+?=", preamble, re.M):
            scope += "\n" + _assignment_value(preamble, m.start())
    return scope


def quoted_stage_body(src: str) -> str:
    """Body of the quoted ``STAGE`` heredoc fed to the container, or "".

    An empty result means the container instructions are not inside a quoted heredoc
    belonging to the docker run call, so every container-scoped requirement fails
    instead of being satisfied by a host-side decoy.
    """
    cmd = docker_run_command(src)
    if cmd is None:
        return ""
    text, _start, after = cmd
    if re.search(r"<<\s*'STAGE'", text) is None:
        return ""
    head = src[after:]
    m = re.search(r"^[ \t]*STAGE[ \t]*\r?\n?", head, re.M)
    return head[: m.start()] if m else ""


def check_script(path: str = SH_PATH) -> list[str]:
    problems: list[str] = []
    try:
        src = open(path).read()
    except OSError:
        return [f"{path}: missing"]
    clean = strip_shell_comments(src)
    requirement_clean = drop_noop_prefixed_lines(strip_shell_comments(src, strip_heredoc_body_comments=True))
    # Comment/no-op stripping happens first for the whole file, then the heredoc body is
    # sliced out, so a commented-out or ``:``-neutered container command cannot satisfy
    # a stage requirement from inside the body either.
    stage_body = quoted_stage_body(requirement_clean)
    stage_text = " ".join(stage_body.splitlines())
    if not src.startswith("#!/usr/bin/env bash"):
        problems.append(f"{path}: must start with #!/usr/bin/env bash")
    if "set -euo pipefail" not in clean:
        problems.append(f"{path}: needs 'set -euo pipefail'")
    for flag in ("--help", "--nb", "--rebuild", "--reset", "--skip-build", "-c |--cmd )"):
        if flag.replace(" ", "") not in clean.replace(" ", ""):
            problems.append(f"{path}: flag {flag} is not dispatched")
    for pat, why in SH_STAGES:
        if pat in SH_FILE_SCOPED:
            if not re.search(pat, requirement_clean, re.M):
                problems.append(f"{path}: missing required stage: {pat} — {why}")
            continue
        if not stage_text and pat not in SH_FILE_SCOPED:
            problems.append(f"{path}: quoted docker STAGE heredoc is missing or empty, cannot check {pat}")
            continue
        if not re.search(pat, stage_text):
            problems.append(f"{path}: missing required stage in the docker heredoc: {pat} — {why}")
    if stage_body and re.search(r"python3? -m pip", stage_text):
        problems.append(
            f"{path}: system python3 must not pip-install into the conda env — use \"$PY\" -m pip"
        )
    cmd = docker_run_command(src)
    if cmd is None:
        problems.append(f"{path}: must contain exactly one docker run invocation")
    else:
        text, _start, after = cmd
        hd = re.search(r"<<(?!-)(\s*)(?P<delim>'STAGE'|\"STAGE\"|\\STAGE|STAGE)(?![A-Za-z0-9_])", text)
        if not hd or hd.group("delim") == "STAGE":
            problems.append(f"{path}: docker stage must be a quoted heredoc on docker run (<<'STAGE' ... STAGE)")
        elif not re.search(r"^STAGE\s*$", src[after:], re.M):
            problems.append(f"{path}: docker stage heredoc from docker run is missing its STAGE terminator")

    # Port publishing is lab-only, however the script spells the lab condition (a
    # "lab)" case branch or an [ "$MODE" = lab ] test on the RUN_ARGS line itself).
    port_lines = [ln for ln in clean.splitlines() if re.search(r"-p\s+\S*8888", ln)]
    if not port_lines:
        problems.append(f"{path}: lab mode must publish -p 8888:8888")
    nb_block = branch_block(clean, "--nb")
    cmd_block = branch_block(clean, "-c|--cmd")
    lab_block = branch_block(clean, "lab")
    for line in port_lines:
        lab_scoped = bool(re.search(r'[=!\[]\s*"?lab\b', line)) or line in lab_block
        if not lab_scoped:
            if line in nb_block or line in cmd_block:
                problems.append(f"{path}: -p 8888:8888 must not be set in --nb/-c|--cmd branches")
            else:
                problems.append(f"{path}: -p 8888:8888 may only be published for lab mode: {line.strip()}")
    host_scope = docker_host_scope(requirement_clean)
    if "-e HOST_UID=" not in host_scope or "-e HOST_GID=" not in host_scope:
        problems.append(f"{path}: docker run must pass HOST_UID/HOST_GID environment variables")
    if not re.search(r"chown\s+\S*\$\{?HOST_UID\}?:\$\{?HOST_GID\}?", stage_text):
        problems.append(f"{path}: stage must chown the executed notebook with HOST_UID/HOST_GID")
    elif "ipynb" not in requirement_clean:
        problems.append(f"{path}: stage must hand back the executed notebook (an .ipynb path)")
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


KNOWN_FLAGS = ("--sync", "--check-sync", "--self-test", "--print")


def main(argv: list[str]) -> int:
    # Reject any flag outside the documented contract, and reject combining more than
    # one mode flag, before anything else runs — a typo must never silently fall
    # through to a passing selector-only invocation (e.g. "data --bogus" => PASS data).
    flags = [a for a in argv if a.startswith("--")]
    unknown = [f for f in flags if f not in KNOWN_FLAGS]
    if unknown:
        print(f"FAIL unknown flag {unknown[0]!r}; choices: {sorted(KNOWN_FLAGS)}", file=sys.stderr)
        return 2
    mode_flags = sorted(set(f for f in flags if f in KNOWN_FLAGS))
    if len(mode_flags) > 1:
        print(
            f"FAIL at most one of {sorted(KNOWN_FLAGS)} may be given, got {mode_flags}",
            file=sys.stderr,
        )
        return 2

    # Handle --print PATH specially since it takes an argument
    if "--print" in argv:
        try:
            print_idx = argv.index("--print")
            if print_idx + 1 >= len(argv):
                print("FAIL --print requires a PATH argument", file=sys.stderr)
                return 1
            path = argv[print_idx + 1]
            with open(path, "r") as f:
                print(f.read(), end="")
            return 0
        except (IOError, OSError) as e:
            print(f"FAIL --print {path}: {e}", file=sys.stderr)
            return 1

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
