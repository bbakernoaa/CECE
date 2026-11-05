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
    (r"MAMBA_ROOT_PREFIX=/env", "micromamba must create the env under the volume, not the container home"),
    (r"export PATH=/env/bin", "the stage calls bare micromamba, which only exists in the volume"),
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


# Command-word boundaries bash honours before `docker`: line start, operators, group/
# subshell/case-arm punctuation, redirections, shell keywords, prefix wrappers and a
# `VAR=value` assignment. Between the binary and `run` bash accepts global options
# (`--debug`, `-H tcp://x`, `-D value`) and the `container` subcommand, so ``docker
# container run``, ``docker --debug run`` and two calls on one line are invocations too.
_DOCKER_RUN_RE = re.compile(
    r"(?:^[ \t]*|[;&|]\s*|\(\s*|\$\(\s*|`\s*|\{[ \t]*|\}[ \t]*|\)[ \t]*"
    r"|[<>][^\s]*[ \t]+"
    r"|(?:(?:^|(?<=\s))(?:if|then|else|elif|while|until|for|do|time|!)\s+)"
    r"|(?:(?:^|(?<=\s))(?:sudo|nohup|env|exec|command|xargs)\s+)"
    r"|(?:(?:^|(?<=\s))\w+=\S*\s+)"
    r"|(?:\$\{[^}]*[=+?]))"
    r"(?:[\w.$@/-]+/)?docker"
    r"(?:[ \t]+(?:--?[A-Za-z][\w-]*(?:[= ][^ \t]+)?|container))*"
    r"[ \t]+run\b(?![-\w])",
    re.M,
)


_QUOTE_FILLER = "\x01"  # not matched by \s, so quoted prose cannot form a command boundary
_OP_FILLER = "x"


def _shell_word_view(text: str, quote: str | None = None, stack: list | None = None,
                     no_opaque: bool = False) -> str | tuple[str, str | None, list]:
    """Rebuild the words bash would see, honouring each quoting level.

    With an initial ``quote``/``stack`` the scan continues a context opened on an
    earlier line, which is what bash does for a valid multi-line double-quoted string:
    ``x="\n...\ndocker run IMG"`` is one inert word, not a call. ``no_opaque`` drops
    the blanking of quoted text for callers that rescan a string known to be executed
    by an ``eval``/``-c`` wrapper.

    Quoting is not invisible to a command word: bash runs ``d''ocker run``,
    ``do"cker" run`` and ``"docker" run`` exactly like ``docker run``, so quote
    delimiters are dropped and adjacent fragments join into one word. What each level
    hides differs:

    * inside single quotes nothing is active, so ``echo 'docker run; docker run'``
      starts no container: whitespace and every operator become fillers;
    * inside double quotes whitespace still cannot split a word and ``;``/``|``/``&&``
      stay literal, but ``$( )`` and backtick substitutions *do* execute, so
      ``echo "$(docker run --rm evil true)"`` is a real second invocation and the
      interior of the substitution is scanned with normal word splitting.
    """
    initial_quote = quote
    out = []
    if stack is None:
        stack = []      # quote state per open nesting level ('", "'", or None)
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and quote != "'":
            out.append(text[i + 1])
            i += 2
            continue
        if quote == "`" and ch != "`":
            out.append(ch)  # inside a backtick substitution anything can start a command
            i += 1
            continue
        if no_opaque:
            if quote is not None and ch == quote:
                quote = stack.pop() if stack else None
            else:
                out.append(ch)
            i += 1
            continue
        if quote == "'" and ch != "'":
            if ch == "\\":  # $'a\'b': the escaped quote does not close the string
                out.append(_QUOTE_FILLER * 2)
                i += 2
                continue
            out.append(_QUOTE_FILLER if (ch.isspace() or ch in "&|;<>!$`()") else ch)
            i += 1
            continue
        if quote == '"' and ch not in('"', "$", "`"):
            if ch.isspace():
                out.append(_QUOTE_FILLER)
            elif ch in "&|;<>!":
                out.append(_OP_FILLER)
            else:
                out.append(ch)
            i += 1
            continue
        if ch == "$" and i + 1 < n and text[i + 1] == "'" and quote is None:
            stack.append(None)
            quote = "'"  # ANSI-C: behaves like single quotes but honours backslash escapes
            out.append("''")  # keeps it a word character, never a command boundary
            i += 2
            continue
        if ch in ("'", '"'):
            if quote == ch:
                quote = stack.pop() if stack else None
            else:
                if quote is not None:
                    stack.append(quote)
                quote = ch
            i += 1
            continue
        if ch == "#" and (i == 0 or text[i - 1] in " \t"):
            # a comment runs to end of line: bash never parses it, so it must neither
            # start a command nor leave a quote open for the next line
            nl = text.find("\n", i)
            i = n if nl == -1 else nl
            out.append(" ")
            continue
        if ch == "$" and i + 1 < n and text[i + 1] == "(":
            if quote is not None:
                stack.append(quote)
            quote = None
            out.append("$( ")  # becomes a command opener for the regex, spaces matter not
            i += 2
            continue
        if ch == "`":
            if quote == "`":
                quote = stack.pop() if stack else None
            else:
                if quote is not None:
                    stack.append(quote)
                quote = "`"
                out.append("` ")  # substitution interior: normal word splitting
            i += 1
            continue
        if ch == ")" and stack:
            quote = stack.pop()
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    view = "".join(out)
    if quote is None and not stack and initial_quote is None:
        return view
    return view, quote, stack


def docker_run_in_line(line: str, quote: str | None = None,
                       stack: list | None = None) -> bool | tuple[bool, str | None, list]:
    """True if the line contains an *executable* ``docker run``, not just the words.

    A line-start match is not enough: bash also runs the command from mid-line
    (``ignored=$(docker run ...)``), behind a keyword (``if docker run ...; then``), a
    wrapper (``sudo``/``env``/``xargs``/``nohup``), punctuation (``{ docker run; }``,
    ``case 1 in 1) docker run ...``), a leading redirection, or with global options
    between the binary and ``run`` (``docker container run``, ``docker --debug run``).
    Quoted text is blanked first, so usage prose that merely names the command does not
    count, and a quote left open by an earlier line keeps blanking this one.
    """
    result = _shell_word_view(line, quote=quote, stack=stack)
    if isinstance(result, tuple):
        view, quote, stack = result
    else:
        view, quote, stack = result, None, []
    if _shell_word_view(line + "\n", quote=quote, stack=stack)[1:] != (None, []):
        # an unterminated quote reaching a later line (also a bash syntax error) would
        # otherwise hide a call that follows it: rescan without that context instead.
        view = _shell_word_view(line)
    found = bool(_DOCKER_RUN_RE.search(view))
    if quote is None:
        return found
    return found, quote, stack


_CODE_WRAPPER_RE = re.compile(
    r"\b(?:eval|exec|trap|(?:ba|z|da|k|a)sh|sh|python3?(?:[ \t]+-c)?|node|perl|ruby)\b"
)
def _ends_continuation(text: str) -> bool:
    """True if a physical line ends in an odd run of backslashes (bash continues)."""
    body = text.rstrip("\r\n")
    return body.endswith("\\") and (len(body) - len(body.rstrip("\\"))) % 2 == 1


_HEREDOC_OP_RE = re.compile(
    r"<<(?P<dash>[-~]?)[ \t]*(?P<quote>['\x22]?)(?P<esc>\\?)(?P<delim>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)
_SUBSTITUTION_RE = re.compile(r"\$\(([^)]*)\)|`([^`]*)`")


_QUOTED_ARG_RE = re.compile(r"'([^']*)'|" + '"([^"]*)"' + r"|`([^`]*)`")


def _docker_calls_in_code(text: str) -> int:
    """Number of ``docker [flags] run`` calls in text bash would execute as code.

    Used for the string argument of ``eval``/``sh -c`` wrappers, where quoting hides
    nothing: bash reparses the string as a command line, so the command-boundary
    requirement only applies where a call could actually sit at a word boundary.
    """
    return len(_DOCKER_RUN_RE.findall("( " + text)) + len(_DOCKER_RUN_RE.findall(text))


def _code_string_parts(line: str) -> tuple[list[str], str | None]:
    """Quoted arguments of an eval/``-c`` wrapper on ``line``, plus any string left open.

    Returns ([fragments], pending_open_string_or_None). A wrapper that is merely quoted
    inside another command's argument is not a wrapper here, because the outer command
    would only print it; the scan runs on the raw line, whose first word is checked.
    """
    first = line.strip().split(" ", 1)[0] if line.strip() else ""
    if not _CODE_WRAPPER_RE.search(rf"{re.escape(first)}") or first in ("echo", "printf"):
        return [], None
    if not re.search(r"\b(?:eval|exec|trap)\b|[ \t]-(?:c|e)[ \t]", line):
        return [], None
    frags = [g for m in _QUOTED_ARG_RE.finditer(line) for g in m.groups() if g]
    pending = None
    for q in ('"', "'"):
        if line.count(q) % 2 == 1:
            tail = line.rsplit(q, 1)[1]
            pending = q + tail
            break
    return frags, pending


def _close_code_string(pending: str, line: str) -> tuple[str, bool]:
    """Append ``line`` to an open eval/-c string; return (content, closed_here)."""
    q, body = pending[0], pending[1:]
    end = line.find(q)
    joined = q + body + "\n" if body else q + body
    if end == -1:
        return joined + line.rstrip("\r\n"), False
    return joined + line[:end], True


def _heredoc_queue(text: str) -> list[tuple[str, bool, bool]]:
    r"""Heredoc bodies opened on ``text``: (delimiter, tab-strip, delimiter-was-quoted).

    ``cat <<A <<B`` opens two bodies, read in the order the redirections appear, so they
    must be consumed as a queue -- remembering one delimiter at a time made the second
    body look like commands, a false positive that would reject a compliant script.
    Whether the delimiter was quoted decides if the body is inert data or still expands
    ``$( )``, so this walks the raw text with its own quote/comment tracking instead of
    reading the blanked view (where ``<<'A'`` and ``<<A`` are indistinguishable).
    """
    found: list[tuple[str, bool, bool]] = []
    quote = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            if ch == "\\" and quote == '"' and i + 1 < n:
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "#" and (i == 0 or text[i - 1] in " \t\n"):
            break  # the rest of the line is a comment, no redirection here
        if ch in "'\"":
            quote = ch
            i += 1
            continue
        if ch == "<" and i + 1 < n and text[i + 1] == "<":
            m = _HEREDOC_OP_RE.match(text, i)
            if m:
                found.append((m.group("delim"), bool(m.group("dash")),
                              m.group("quote") != "" or m.group("esc") != ""))
                i = m.end()
                continue
        i += 1
    return found


def _scan_units(src: str) -> list[tuple[int, str, str]]:
    r"""(offset, text, kind) units of ``src``: "code", "data" or "expand" heredoc body.

    Bash joins a line ending in ``\`` with the next one before splitting words, so
    ``doc\`` + ``ker run x`` really is a ``docker run``. A heredoc body is stdin data,
    never parsed as commands by this shell, so an unquoted ``cat <<EOF`` body naming
    ``docker run`` is not a call and its unbalanced quotes must not poison later lines --
    that is a "data"/"expand" unit. With an *unquoted* delimiter bash still expands
    ``$( )`` inside the body ("expand"), which can start a command; a quoted ``<<'EOF'``
    body stays wholly literal ("data").
    """
    lines = src.splitlines(keepends=True)
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    units: list[tuple[int, str, str]] = []
    queue: list[tuple[str, bool, bool]] = []
    i = 0
    while i < len(lines):
        if queue:
            delim, dash, quoted = queue[0]
            body = lines[i].lstrip("\t") if dash else lines[i]
            closed = body.rstrip("\r\n").strip("\t") == delim
            units.append((offsets[i], lines[i], "data" if quoted else "expand"))
            if closed:
                queue.pop(0)
            i += 1
            continue
        j = i
        text = lines[j]
        while _ends_continuation(text) and j + 1 < len(lines):
            # bash deletes the backslash and the newline, joining the two words
            text = text.rstrip("\r\n")[:-1] + lines[j + 1]
            j += 1
        units.append((offsets[i], text, "code"))
        queue = _heredoc_queue(text)
        i = j + 1
    return units


def docker_run_call_sites(src: str) -> list[int]:
    """Offsets of every executable ``docker run`` in ``src``, one entry per invocation.

    The scan walks code units (continuation-joined lines outside heredoc bodies)
    carrying their quote context, so a call is counted where bash would execute it --
    at a line start, mid-line behind a wrapper (``sudo``/``env``/``xargs``), a keyword,
    a redirection, a substitution, group/case punctuation, a ``${x:+...}`` expansion, or
    twice on one line -- while quoted prose and heredoc bodies stay invisible. Inside an
    unquoted-delimiter body only ``$( )``/backtick text is scanned, since that is all bash
    expands there. The argument of ``eval``/``trap``/``-c`` is scanned as code, including
    when that string continues over several lines.
    """
    sites: list[int] = []
    quote, stack = None, []
    pending: str | None = None  # eval/-c string still open from an earlier unit
    for offset, chunk, kind in _scan_units(src):
        if kind != "code":
            if kind == "expand" and pending is None:
                for m in _SUBSTITUTION_RE.finditer(chunk):
                    frag = m.group(1) or m.group(2) or ""
                    if _docker_calls_in_code(frag):
                        sites.append(offset + m.start())
            continue
        result = _shell_word_view(chunk, quote=quote, stack=stack)
        view, quote, stack = result if isinstance(result, tuple) else (result, None, [])
        if pending is None:  # an eval/-c interior is scanned once, as code
            for m in _DOCKER_RUN_RE.finditer(view):
                sites.append(offset + m.start())
        if pending is not None:
            content, closed = _close_code_string(pending, chunk)
            added = content[len(pending):]  # only this chunk can introduce a new call
            if _docker_calls_in_code(added) or (
                pending and _docker_calls_in_code(pending[-1:] + added)
            ):
                sites.append(offset)
            pending = None if closed else content
        else:
            frags, opened = _code_string_parts(chunk)
            if any(_docker_calls_in_code(f) for f in frags) or (
                opened and _docker_calls_in_code(opened)
            ):
                sites.append(offset)
            pending = opened
    return sites


def docker_run_command(src: str) -> tuple[str, int, int] | None:
    """The single ``docker run`` invocation: its text, start offset, and the offset
    just after its last line.

    Shell line continuations are followed, so a heredoc operator written on a
    continuation line (the usual formatting) still belongs to the container call.
    Returns None unless there is *exactly* one ``docker run`` call in the script --
    zero is obviously wrong, but more than one is equally dangerous: a second,
    unchecked invocation could run anything while every requirement below is
    satisfied by the first. Treating "not exactly one" as "no sole call exists"
    means every Docker-bound check (heredoc ownership, STAGE terminator, stage-body
    scoping, host-side env vars) fails shut instead of silently checking only the
    first of several invocations.
    """
    sites = sorted(set(docker_run_call_sites(src)))
    if len(sites) != 1:
        return None
    start = sites[0]
    lines = src.splitlines(keepends=True)
    i = next(k for k, ln in enumerate(lines) if sum(len(p) for p in lines[:k]) <= start
             < sum(len(p) for p in lines[:k + 1]))
    start = sum(len(ln) for ln in lines[:i])
    text = ""
    while i < len(lines):
        text += lines[i]
        if not lines[i].rstrip("\r\n").rstrip().endswith("\\"):
            break
        i += 1
    return text, start, start + len(text)


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


def interactive_flag_present(scope: str) -> bool:
    """True when -i/--interactive appears as its own argument in *scope*.

    Tokenised on whitespace instead of matched as a substring: ``-e OPT="-i x"`` carries `-i`
    inside a value, and `--init`/`-init` merely start with it, neither of which attaches stdin.
    """
    for line in scope.splitlines():
        for tok in re.split(r"[ \t]+", line.strip()):
            if tok.strip("\"'") in ("-i", "--interactive"):
                return True
    return False


def docker_arg_scope(clean: str, cmd_text: str) -> str:
    """The docker run command plus every shell array it interpolates ("${RUN_ARGS[@]}").

    Flags are commonly assembled in an array rather than on the ``docker run`` line, so a
    requirement on the command must look at those assignments too -- but only the arrays this
    command actually expands, so an unrelated ``-i`` elsewhere in the file cannot satisfy it.
    """
    parts = [cmd_text]
    for name in sorted(set(re.findall(r"\$\{(\w+)\[@\]\}", cmd_text))):
        for m in re.finditer(r"(?<![\w])" + re.escape(name) + r"\+?=\(", clean):
            depth, i, quote = 0, m.end() - 1, ""
            while i < len(clean):
                ch = clean[i]
                if quote:
                    if ch == "\\":
                        i += 2
                        continue
                    if ch == quote:
                        quote = ""
                elif ch in "\"'":
                    quote = ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            parts.append(clean[m.start() : i + 1])
    return "\n".join(parts)


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
        if hd:
            # The stage reaches the container on stdin, and Docker only wires stdin up when -i
            # (--interactive) is given. Without it the heredoc is discarded unread, the container
            # exits 0, and every later "verified" claim rests on a run that did nothing at all.
            if not interactive_flag_present(docker_arg_scope(clean, text)):
                problems.append(
                    f"{path}: docker run must pass -i (or --interactive): the quoted STAGE heredoc is "
                    "read from stdin by the container, and without -i Docker never attaches stdin, so the "
                    "stage is discarded and the run exits 0 having done nothing"
                )

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
