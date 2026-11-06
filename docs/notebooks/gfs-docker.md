# GFS → FENGSHA in the `cece/cece-dev` container

[fengsha_gfs.ipynb](./fengsha_gfs.ipynb) runs the compiled C++ FENGSHA dust scheme against a real
GFS forecast cycle instead of a synthetic column. It is a data-pipeline demonstration; the
emissions magnitudes are an upper bound (see §7 in the notebook) because the BES erodibility
field is not available in this path. Everything runs inside the existing `cece/cece-dev:latest`
image on Apple silicon; the launcher is [gfs-docker.sh](./gfs-docker.sh).

## What the path does

1. GRIB2 files for the forecast cycle are downloaded once and cached in the `/cache/gfs` named
   volume, by the notebook itself (§2 calls `gfs_docker_data.ensure_gribs`). The cycle is *not* a
   launcher flag or an environment variable: it is the `driver:` block of
   [fengsha_gfs_kok.yaml](./fengsha_gfs_kok.yaml) and [fengsha_gfs_webb.yaml](./fengsha_gfs_webb.yaml)
   (`start_time`, `end_time`, `timestep_seconds`), which must stay identical — the launcher's
   `PROBE` stage and the notebook both assert it. Changing the cycle means editing `start_time` /
   `end_time` in **both** files, plus `leads` in the notebook's `CASE`. The recorded case is
   20261103 t12z, leads 0, 3, 6, 12, so the committed notebook outputs stay reproducible.
2. [MONETIO](https://github.com/noaa-oar-arl/monetio)'s `gfsreader` (via `grib2io`) decodes
   them into xarray. MONETIO itself is a pinned git checkout (`67b1071e`, from
   `tests/data/interfaces/compatibility-lock.json`), installed `pip --no-deps` inside the
   env so it cannot upgrade numpy under the compiled bindings.
3. Two NESDIS dust-source climatologies on a 0.1° (~10 km) grid — `uthres` and `uthres_sg` for
   threshold velocity, plus clay/sand/silt, snow, frozen-soil, lakes — are aligned onto the 1°
   GFS grid with `axis-regrid`'s `regrid_rectilinear`, `method="nearest"`. Nearest is *not*
   conservative; land edges step instead of blending — this is called out in the notebook.
4. Fourteen FENGSHA inputs are derived per lead and stepped through `Model(["fengsha"])`.
5. Two figures render: a 2×2 world panel of the inputs and a 2×4 Sahel emission zoom.
   Both are drawn with [Cartopy](https://github.com/SciTools/cartopy)'s Natural Earth 50 m coastlines and
   political borders so the geography is legible; the shapefile cache lives on the `/cache`
   volume.

## Launch modes

`gfs-docker.sh` takes an optional first argument that decides what the container does:

| Command | Mode |
|---|---|
| `docs/notebooks/gfs-docker.sh` | JupyterLab on <http://localhost:8888>, notebook bind-mount |
| `docs/notebooks/gfs-docker.sh --nb` | execute [fengsha_gfs.ipynb](./fengsha_gfs.ipynb) headless, in place |
| `docs/notebooks/gfs-docker.sh -c '<cmd>'` | one-off shell command inside the conda env |

The launcher deviates from the plan's spelling `-- '<cmd>'`: `-c`/`--cmd` is used instead so
`--` remains free for future passthrough of extra `docker run` arguments. `usage()` prints the
same list you see here and stays in lock-step with the file header.

Additional flags can be combined with any of the above:

| Flag | Meaning |
|---|---|
| `--rebuild` | discard `/cache/build`, `/cache/src`, and the stamp, and recompile `_cece_core` |
| `--skip-build` | explicitly reuse the cached bindings; the default `--nb` path already skips a rebuild when the content-hash stamp matches, this makes it unconditional |
| `--reset` | remove both named volumes next run; slow, but reproduces a fresh machine |

There is no `--case` flag: the cycle lives in the two CECE configs (see step 1 above), and the
launcher only checks that they load, agree, and set `use_brittle_impaction` opposite on each other.

The rebuild trigger is a content hash over `git rev-parse HEAD:src HEAD:include HEAD:cmake`
plus `git status --porcelain` on those directories. Notebook-only commits do not touch the
stamp, so `--nb` on a warm cache skips the C++ build and lands in seconds.

## Cold start on a fresh machine

Two named volumes are created on first run — `cece-gfs-env` (micromamba env, `MAMBA_ROOT_PREFIX=/env`)
and `cece-gfs-cache` (`/cache/gfs`, `/cache/build`, `/cache/xdg/cartopy`). Rough timings on
Apple silicon from measured runs:

| Stage | Cold | Warm |
|---|---|---|
| conda solve (grib2io, xarray, cartopy, numpy<3, …) | ≈ 3 min | skipped |
| apt + micromamba download | ≈ 20 s | skipped |
| pip install of `axis-regrid` + `fsspec` + pinned MONETIO checkout | ≈ 20 s | skipped when the git SHA matches |
| GRIB header / lead download per lead file (42–45 MB) | ≈ 15 s per lead | cached |
| `_cece_core` cmake + ninja build | ≈ 7 min | skipped when stamp matches |
| Cartopy Natural Earth shapefile download (50 m coast + borders) | ≈ 5 s once | cached in `/cache/xdg/cartopy` |
| Notebook execute (8 cases + 2 figures + Cartopy maps) | ≈ 15 s | ≈ 15 s |

End-to-end warm `--nb` reruns take ≈ 18 s wall clock (measured). `--reset` (drops both volumes)
exercises the whole cold path again; a full cold on a fresh machine runs ≈ 11 min.

## Environment variables you might want

| Variable | Default | Purpose |
|---|---|---|
| `CECE_IMAGE` | `cece/cece-dev:latest` | swap to a locally tagged image |
| `GFS_PORT` | `8888` | host port mapping for JupyterLab |
| `GFS_ENV_NAME` | `cece-gfs` | conda env name *and* the `ipykernel` spec the notebook selects |
| `GFS_VOL_ENV` | `cece-gfs-env` | named volume holding the conda env |
| `GFS_VOL_CACHE` | `cece-gfs-cache` | named volume holding GRIBs, builds, NE shapefiles |
| `MONETIO_REMOTE` / `MONETIO_SHA` | `https://github.com/noaa-oar-arl/monetio.git` / `67b1071e…` | MONETIO pin (must match compatibility-lock.json) |
| `OMP_NUM_THREADS` | `10` | Kokkos openmp threads inside the container |

The forecast cycle is not here: `docs/notebooks/fengsha_gfs_kok.yaml` and
`docs/notebooks/fengsha_gfs_webb.yaml` carry it in their `driver:` blocks and are the only place
to change it. Nothing in the notebook reads an environment variable.

`$XDG_DATA_HOME=/cache/xdg` is pinned in the launcher so Cartopy caches its Natural Earth
shapefiles on the volume rather than in the container's `/root/.local/share`. First draw on a
cold `/cache`: ~5 s while the 50 m coastline and borders zip files are pulled from S3; later
draws are offline.

## Why only the container

GFS decoding needs `grib2io`, which conda-forge does not build for Python 3.14 (or 3.13) on
`osx-arm64`. The macOS host cannot install it at all, so the notebook cannot run outside the
container. The image is Linux/arm64, Python 3.12, and every pip install happens inside a
micromamba env built for that interpreter, which is why `--platform linux/arm64` is on every
`docker run` here. Building the same env natively on macOS would drop `_cece_core` on
`libgrib2` symbol differences instead.

The `cece/cece-dev:latest` image already carries Kokkos, ESMF, and a working CMake + Ninja
chain for the CECE bindings, so the launcher only has to install the Python-side dependencies
and build `_cece_core` on the volume — it does not define a Dockerfile.

## Verifying the launcher hasn't drifted

The launcher (`gfs-docker.sh`), the notebook (`fengsha_gfs.ipynb` + `fengsha_gfs.py` generated
from it), and the plan's §5 verbatim bash block are kept in lockstep by
[gfs_docker_checks.py](./gfs_docker_checks.py):

```bash
python3 docs/notebooks/gfs_docker_checks.py --check-sync   # script -> notebook cell-by-cell
python3 docs/notebooks/gfs_docker_checks.py nbfinal        # ≥12 code cells, ≥5 asserts,
                                                           # tokens for every §
python3 docs/notebooks/gfs_docker_checks.py sh             # launcher matches SH_STAGES
python3 -m pytest docs/notebooks/test_gfs_docker_checks.py # 44 gate tests
```

`sh` checks that load-bearing stage fragments (`docker run -i`, `MAMBA_ROOT_PREFIX=/env`,
`export PATH=/env/bin`, no `python3 -m pip` outside the env, `fsspec`, `cartopy`,
`-DCECE_PYTHON_SIMULATION=OFF`, the `--no-deps` pattern around MONETIO, the pinned SHA, …) all
appear inside the *quoted* heredoc, not in a comment or a decoy `cat <<'STAGE'` further down
the file. Every rule in `SH_STAGES` corresponds to a real cold-start failure; the comment in
[gfs-docker.sh](./gfs-docker.sh#L12) at the top of the launcher lists them.

## Files this path owns

- [gfs-docker.sh](./gfs-docker.sh) — the launcher
- [gfs_docker_data.py](./gfs_docker_data.py) — MONETIO download / cache helper the launcher
  calls before starting the notebook
- [gfs_docker_checks.py](./gfs_docker_checks.py) — gate library and CLI
- [test_gfs_docker_checks.py](./test_gfs_docker_checks.py) — pytest suite for the gate
- [fengsha_gfs.py](./fengsha_gfs.py) — the executable script that generates
  [fengsha_gfs.ipynb](./fengsha_gfs.ipynb)
- [fengsha_gfs.ipynb](./fengsha_gfs.ipynb) — the notebook, committed *with* its outputs so a
  reader who does not build Docker sees the executed numbers

Other notebook pages under [docs/notebooks/](./) (e.g. [fengsha.ipynb](./fengsha.ipynb)) use
synthetic column inputs and no launcher; they run in the host's default Python kernel.
