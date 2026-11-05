# %% [markdown]
# # GFS-forced FENGSHA mineral dust emission in CECE
#
# The real-data twin of [`fengsha.ipynb`](./fengsha.ipynb). Every FENGSHA input comes from a
# named GFS GRIB field (decoded by [MONETIO](https://github.com/noaa-oar-arl/monetio)) or from
# the NESDIS soil / Baker `feff` climatologies, on a global 1° grid. Design, provenance and
# caveats: `docs/superpowers/specs/2026-11-04-gfs-monetio-fengsha-design.md`.
#
# This notebook is a **data-path test, not a dust forecast**: GFS supplies no erodibility, no
# lake fraction and no vegetation, so the numbers show that real meteorology can drive the
# scheme — not what the world emitted on 3 November.
#
# ## 0. Run me in the container
#
# Launch with `docs/notebooks/gfs-docker.sh` (kernel `cece-gfs`). The host kernel cannot work:
# conda-forge caps `grib2io` at Python 3.13 on osx-arm64 while the host CECE build is 3.14.

# %%
import os
import sys

for _p in (os.environ.get("CECE_PYTHONPATH"), "/cache/build/src/python", "build/src/python"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

import cece  # noqa: E402
from cece import Grid, Model  # noqa: E402
from cece.preparation import regrid_rectilinear  # noqa: E402
import monetio  # noqa: E402

REPO = "/work" if os.path.isdir("/work/docs/notebooks") else os.getcwd()
CACHE = os.environ.get("GFS_CACHE", "/cache/gfs")
print(f"python  {sys.version.split()[0]}  {sys.executable}")
print(f"cece    {cece.__file__}")
print(f"monetio {monetio.__version__}")
print(f"repo    {REPO}")
print(f"cache   {CACHE}")

# %% [markdown]
# ## 1. Configuration
#
# The cycle is the one used to design this notebook, so the printed output of a fresh run can be
# compared with the design's recorded numbers. `monetio.load` needs both filter keys: without
# `productDefinitionTemplateNumber` the GRIB reader refuses a cube that mixes instantaneous and
# accumulation records. One file per read because merged files keep the first file's scalar
# `validDate`/`leadTime`.

# %%
CASE = {
    "date": "20261103",
    "hour": 12,
    "product": "pgrb2.1p00",
    "leads": [0, 3, 6, 12],
    "thresholds": ["uthres", "uthres_sg"],
    "dt": 3600.0,
    "cache_dir": CACHE,
    "climo_dir": REPO,
}
SURFACE_FILTER = {"typeOfFirstFixedSurface": 1, "productDefinitionTemplateNumber": 0}
SOIL_FILTER = {"typeOfFirstFixedSurface": 106, "productDefinitionTemplateNumber": 0}
SENTINEL = 900.0  # NESDIS writes 999 into uthres/uthres_sg; anything >= this is "no data"
R_EARTH = 6.371e6
KDRY = 287.04  # dry air gas constant, J kg-1 K-1 -> rho = p / (KDRY * T)

assert CASE["leads"] == sorted(set(CASE["leads"])), "leads must be unique and ascending"
assert CASE["hour"] in (0, 6, 12, 18), "GFS is delivered at 6-hourly cycle hours"
assert set(CASE["thresholds"]) <= {"uthres", "uthres_sg"}, "unknown threshold set"
assert CASE["dt"] > 0, "dt must be positive seconds"
assert CASE["product"] == "pgrb2.1p00", "only the 1-degree product is in scope here"

KOK = {
    "particle_radii": [2.15e-6, 4.875e-6, 9.575e-6, 1.375e-4, 6.25e-4],
    "bin_lower_edges": [1.2e-6, 3.1e-6, 6.65e-6, 2.5e-5, 1.0e-4],
    "bin_upper_edges": [3.1e-6, 6.65e-6, 1.25e-4, 2.5e-4, 1.0e-3],
    "use_brittle_impaction": True,
    "c_tune": 0.05,
}
print(CASE)



# %%
