#!/usr/bin/env bash
# Bootstrap and run the CECE container for docs/notebooks/fengsha_gfs.ipynb.
#
#   docs/notebooks/gfs-docker.sh                      # Jupyter Lab on http://localhost:8888
#   docs/notebooks/gfs-docker.sh --nb                 # execute the notebook headless, in place
#   docs/notebooks/gfs-docker.sh -c 'python -V'       # one-off command inside the env
#   docs/notebooks/gfs-docker.sh --case 20261103 12 0,3,6,12
#   docs/notebooks/gfs-docker.sh --rebuild            # force a fresh _cece_core build
#   docs/notebooks/gfs-docker.sh --skip-build         # reuse the cached bindings
#   docs/notebooks/gfs-docker.sh --reset              # drop both volumes
#
# State lives in two named volumes, so the host worktree never receives Linux artefacts.
# Cold start is ~15 min (conda solve + bindings build); warm starts are seconds.
#
# Load-bearing choices, each of them bought with a failed build:
#   * -i on docker run: the stage is fed to `bash -s` through stdin, and without -i Docker
#     leaves stdin detached, so the whole heredoc is discarded and the container exits 0
#     having run nothing at all -- a silent, successful-looking no-op;
#   * the platform flag must match the image arch, or the mount fails with an exec-format error;
#   * -DCECE_PYTHON_SIMULATION=OFF because the image ships a *static* Kokkos, which
#     src/python/CMakeLists.txt refuses for that target (bindings themselves are unaffected);
#   * conda-forge pybind11 because FetchContent else points at /usr/include/python3.12;
#   * pip installs are --no-deps so MONETIO/AXIS cannot downgrade the conda numpy;
#   * GRIBs are downloaded, never read remotely: grib2io's xarray backend hands the s3:// URI to
#     builtin open(), and virtualizarr-style reads hung for minutes.
set -euo pipefail

IMAGE="${CECE_IMAGE:-cece/cece-dev:latest}"
ENV_NAME="${GFS_ENV_NAME:-cece-gfs}"
VOL_ENV="${GFS_VOL_ENV:-cece-gfs-env}"
VOL_CACHE="${GFS_VOL_CACHE:-cece-gfs-cache}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PORT="${GFS_PORT:-8888}"
NOTEBOOK=/work/docs/notebooks/fengsha_gfs.ipynb
DATA_HELPER=/work/docs/notebooks/gfs_docker_data.py
CASE_DATE="${CASE_DATE:-20261103}"
CASE_HOUR="${CASE_HOUR:-12}"
CASE_LEADS="${CASE_LEADS:-0,3,6,12}"

# The org is noaa-oar-arl: NOAA-EMC/MONETIO prompts for credentials and cannot be cloned here.
MONETIO_REMOTE="${MONETIO_REMOTE:-https://github.com/noaa-oar-arl/monetio.git}"
# = monetio_v1_commit in tests/data/interfaces/compatibility-lock.json: pin it, MONETIO's readers
# change shape between releases and this notebook depends on 67b1071e's gfs reader.
MONETIO_SHA="${MONETIO_SHA:-67b1071e30bbcbf2f8853b1c6ecc432c430a8df3}"

MODE=lab
RUN_CMD=""
WANT_BUILD=1
SKIP_BUILD=0
REBUILD=0

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --nb) MODE=nb; shift ;;
        --rebuild) REBUILD=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --reset)
            docker volume rm -f "$VOL_ENV" "$VOL_CACHE" >/dev/null
            echo "reset: removed $VOL_ENV $VOL_CACHE"
            exit 0 ;;
        --case)
            CASE_DATE="$2"; CASE_HOUR="$3"; CASE_LEADS="$4"; shift 4 ;;
        -c|--cmd) MODE=cmd; RUN_CMD="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "missing image $IMAGE: docker build -t $IMAGE ." >&2
    exit 1
}
docker volume create "$VOL_ENV" >/dev/null
docker volume create "$VOL_CACHE" >/dev/null

# Rebuild when HEAD moves *or* tracked source/notebook files differ from HEAD: a pure worktree
# snapshot cannot work here (this repo can have tracked files deleted locally), so the tree is
# git -C /work archive HEAD plus an overlay of the current src/ and include/.
DIRTY_HASH="$( { git -C "$REPO" rev-parse HEAD;
                 git -C "$REPO" status --porcelain -- src include docs/notebooks; } | sha1sum | cut -d' ' -f1 )"

if [ "$SKIP_BUILD" = 1 ] && [ "$REBUILD" = 1 ]; then
    echo "--skip-build and --rebuild contradict each other" >&2
    exit 2
fi

RUN_ARGS=(--rm -i --platform linux/arm64
    -v "$REPO":/work -v "$VOL_ENV":/env -v "$VOL_CACHE":/cache
    -e ENV_NAME="$ENV_NAME" -e MONETIO_REMOTE="$MONETIO_REMOTE" -e MONETIO_SHA="$MONETIO_SHA"
    -e CASE_DATE="$CASE_DATE" -e CASE_HOUR="$CASE_HOUR" -e CASE_LEADS="$CASE_LEADS"
    -e DIRTY_HASH="$DIRTY_HASH" -e MODE="$MODE" -e RUN_CMD="$RUN_CMD"
    -e NOTEBOOK="$NOTEBOOK" -e DATA_HELPER="$DATA_HELPER"
    -e CECE_PYTHONPATH=/cache/build/src/python -e PYTHONPATH=/cache/build/src/python
    -e GFS_CACHE=/cache/gfs -e MPLBACKEND=Agg -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
    -e REBUILD="$REBUILD" -e SKIP_BUILD="$SKIP_BUILD"
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)")
if [ "$MODE" = lab ]; then RUN_ARGS+=(-p "$PORT:8888"); fi

docker run "${RUN_ARGS[@]}" \
    "$IMAGE" bash -euo pipefail -s <<'STAGE'
# Both lines are load-bearing: -n resolves to "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME" (which is
# where $PY points), and micromamba is downloaded into the volume rather than installed, so it
# is not on PATH. Without them `micromamba create` silently builds an env under $HOME and every
# later RUN fails with "micromamba: command not found".
export MAMBA_ROOT_PREFIX=/env
export PATH=/env/bin:"${PATH}"
RUN() { micromamba run -n "${ENV_NAME}" "$@"; }
PY=/env/envs/"${ENV_NAME}"/bin/python

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    curl ca-certificates git unzip tar cmake ninja-build >/dev/null

if [ ! -x /env/bin/micromamba ]; then
    echo "[env] micromamba"
    mkdir -p /env/bin
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xj -C /env bin/micromamba
fi
if [ ! -d /env/envs/"${ENV_NAME}" ]; then
    echo "[env] conda solve (minutes)"
    /env/bin/micromamba create -y -q -n "${ENV_NAME}" -c conda-forge --override-channels \
        python=3.12 'numpy<3' grib2io xarray netcdf4 pandas scipy pyyaml matplotlib-base \
        ipykernel jupyterlab nbconvert pip pybind11 >/dev/null
fi
# Tripwire for the failure class above: an env that landed somewhere else looks like success.
[ -x "$PY" ] || { echo "[env] FATAL: $PY missing after create (root prefix ${MAMBA_ROOT_PREFIX})" >&2; exit 1; }

if ! RUN "$PY" -c 'import axis' 2>/dev/null; then
    echo "[env] axis-regrid (sanctioned regrid_rectilinear backend)"
    RUN "$PY" -m pip install -q --no-deps /work/extern/helm/libs/axis
fi
if ! RUN "$PY" -c 'import monetio' 2>/dev/null; then
    echo "[env] MONETIO ${MONETIO_SHA:0:7}"
    rm -rf /cache/monetio && mkdir -p /cache/monetio
    git -C /cache/monetio init -q
    git -C /cache/monetio remote add origin "${MONETIO_REMOTE}"
    git -C /cache/monetio fetch -q --depth 1 origin "${MONETIO_SHA}"
    git -C /cache/monetio checkout -q FETCH_HEAD
    RUN "$PY" -m pip install -q --no-deps /cache/monetio
fi
RUN "$PY" -m ipykernel install --name "${ENV_NAME}" --display-name "Python 3 (CECE GFS)" >/dev/null

if [ "${REBUILD}" = 1 ]; then rm -rf /cache/src /cache/build /cache/build.stamp; fi
if [ "${SKIP_BUILD}" = 1 ]; then
    echo "[cece] --skip-build: reusing the cached bindings"
elif [ ! -f /cache/build.stamp ] || [ "$(cat /cache/build.stamp)" != "${DIRTY_HASH}" ]; then
    echo "[cece] snapshot + _cece_core"
    rm -rf /cache/src /cache/build && mkdir -p /cache/src
    git -C /work archive HEAD | tar -x -C /cache/src
    tar -C /work -cf - src include 2>/dev/null | tar -x -C /cache/src 2>/dev/null || true
    cp -r /work/extern/helm /work/extern/yaml-cpp /cache/src/extern/ 2>/dev/null || true
    RUN cmake -S /cache/src -B /cache/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DCECE_BUILD_TESTING=OFF \
        -DENABLE_ESMF=OFF -DCECE_PYTHON_SIMULATION=OFF -DCECE_FETCH_DEPS=OFF \
        -DPython3_EXECUTABLE="${PY}" >/dev/null
    RUN cmake --build /cache/build --target _cece_core -j"$(nproc)"
    echo "${DIRTY_HASH}" > /cache/build.stamp
else
    echo "[cece] bindings current"
fi
# Resolved *after* the build: a glob expanded while /cache/build is still empty keeps the
# literal pattern, so the early spelling made a successful build look like a missing .so.
SO=$(echo /cache/build/src/python/cece/_cece_core*.so)
ls "${SO}" >/dev/null

echo "[data] GRIB leads ${CASE_LEADS} for ${CASE_DATE} t${CASE_HOUR}z"
RUN "$PY" "${DATA_HELPER}" --cache /cache/gfs --date "${CASE_DATE}" --hour "${CASE_HOUR}" \
    --leads "${CASE_LEADS}"

RUN "$PY" - <<'PROBE'
import sys
sys.path.insert(0, "/cache/build/src/python")
import numpy as np
import cece, monetio, grib2io, axis
from cece import Grid
from cece.preparation import regrid_rectilinear  # noqa: F401
g = Grid.structured(longitude=np.arange(-180.0, 180.0, 1.0), latitude=np.arange(90.0, -91.0, -1.0))
print(f"PROBE ok py{sys.version.split()[0]} monetio={monetio.__version__} grib2io={grib2io.__version__} grid={g.ncol}")
PROBE

case "${MODE}" in
    nb)
        echo "[run] nbconvert --execute ${NOTEBOOK}"
        "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.kernel_name="${ENV_NAME}" --ExecutePreprocessor.timeout=1800 \
            "${NOTEBOOK}"
        rc=$?
        # nbconvert rewrites the notebook into the bind-mounted repo as root; hand it back to the
        # host user, or the next git operation on the host cannot write it.
        chown "${HOST_UID}:${HOST_GID}" "${NOTEBOOK}" 2>/dev/null || true
        exit "$rc"
        ;;
    cmd)
        echo "[run] ${RUN_CMD}"
        exec bash -c "${RUN_CMD}"
        ;;
    lab)
        echo "[run] Jupyter Lab on port 8888 (host ${GFS_PORT_HOST:-8888})"
        exec "$PY" -m jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/work \
            --ServerApp.token='' --ServerApp.password='' --ServerApp.root_dir=/work
        ;;
esac
STAGE
