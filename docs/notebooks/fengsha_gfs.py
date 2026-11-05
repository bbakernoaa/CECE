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

# Pick the first candidate that exists and make it win outright. Reordering inserts is not enough:
# PYTHONPATH already puts /cache/build/src/python in sys.path *behind* the script directory, so a
# `not in sys.path` guard skips re-inserting it and the repo's stale host build tree
# (build/src/python, bind-mounted at /work) lands at the front and shadows the container bindings
# with a 3.14-macos `_cece_core` -- a circular-import ImportError that only appears in the container.
_CECE_PATHS = (os.environ.get("CECE_PYTHONPATH"), "/cache/build/src/python", "build/src/python")
_CECE_PATH = next((_p for _p in _CECE_PATHS if _p and os.path.isdir(_p)), None)
if _CECE_PATH:
    sys.path = [_p for _p in sys.path if _p not in _CECE_PATHS]
    sys.path.insert(0, _CECE_PATH)

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



# %% [markdown]
# ## 2. Read one file per lead
#
# MONETIO hands back whatever GRIB shortNames the product carries, with `PRES/TMP/HGT` renamed to
# `pressure/temperature/height`. Two rules that come from measurement, not from the docs:
#
# * the `productDefinitionTemplateNumber: 0` filter is mandatory, otherwise the reader raises
#   *"filter to a single typeOfFirstFixedSurface"* because accumulation and categorical records
#   (PDT 8) share the surface;
# * a file is decoded per lead. Merging several files produces a correct `time` dimension but
#   copies the **first** file's scalar `validDate`/`leadTime`/`refDate` onto the whole cube, which
#   would mislabel every lead but the first. The asserts below are what detect that regression.

# %%
# Ordered deliberately: the helper import must come after the path insert, which only matters in
# notebook mode, where the kernel's cwd is the repo root rather than the script's own directory.
sys.path.insert(0, os.path.join(REPO, "docs", "notebooks"))

from pathlib import Path

import pandas as pd
import gfs_docker_data as gdata

SURFACE_NEEDED = ["FRICV", "GUST", "LAND", "SNOD", "HPBL", "SOTYP", "VEG", "pressure", "temperature"]
SOIL_NEEDED = ["SOILL", "TSOIL"]
LEADS = {}
for lead in CASE["leads"]:
    (path,) = gdata.ensure_gribs(CASE["cache_dir"], CASE["date"], CASE["hour"], [lead])
    surface = monetio.load("gfs", files=[str(path)], filters=SURFACE_FILTER)
    soil = monetio.load("gfs", files=[str(path)], filters=SOIL_FILTER)
    valid = pd.Timestamp(str(np.datetime_as_string(np.asarray(surface.validDate.values), unit="s")))
    lead_s = int(np.asarray(surface.leadTime.values).astype("timedelta64[s]").astype(int))
    LEADS[lead] = {"surf": surface, "soil": soil, "valid": valid, "lead_s": lead_s, "path": Path(path)}

# %%
for lead in CASE["leads"]:
    surf, soil = LEADS[lead]["surf"], LEADS[lead]["soil"]
    assert set(surf.dims) == {"y", "x"} and surf.sizes["y"] == 181 and surf.sizes["x"] == 360, surf.dims
    missing = [n for n in SURFACE_NEEDED if n not in surf.data_vars]
    assert not missing, f"lead {lead}: surface cube lacks {missing}"
    assert set(soil.dims) == {"level", "y", "x"} and soil.sizes["level"] == 4, soil.dims
    missing = [n for n in SOIL_NEEDED if n not in soil.data_vars]
    assert not missing, f"lead {lead}: soil cube lacks {missing}"
    assert int(np.asarray(soil.valueOfFirstFixedSurface)[0]) == 0, "level 0 must be the 0-0.1 m layer"
    assert LEADS[lead]["lead_s"] == 3600 * lead, (lead, LEADS[lead]["lead_s"])
    expected = pd.Timestamp(f"{CASE['date']} {CASE['hour']:02d}:00") + pd.Timedelta(seconds=lead * 3600)
    assert LEADS[lead]["valid"] == expected, (lead, LEADS[lead]["valid"], expected)
    print(f"f{lead:03d}  valid {LEADS[lead]['valid']}  lead {LEADS[lead]['lead_s'] // 60:5d} min  "
          f"surface {len(surf.data_vars)} vars  soil {len(soil.data_vars)} x {soil.sizes['level']} levels")

# %% [markdown]
# ## 3. Axes, model grid and climatology alignment
#
# GFS gives 2-D broadcast coordinates on `(y, x)` with latitudes descending from +90 and longitudes
# 0…359. CECE wants monotonic 1-D `latitude`/`longitude`, so the axes are read off one row or column
# and the longitudes are rolled to −180…179 to overlap the climatologies (−179.95…179.95).
#
# The climatology cells are 0.1° **centred** at ±0.05 from the whole-degree model nodes, so every
# target cell has two equally near sources and `nearest` deterministically picks one. It is nearest,
# **not** conservative: a 1° cell is described by one 0.1° cell, not by their area mean.

# %%
first = LEADS[CASE["leads"][0]]["surf"]
LAT1 = np.asarray(first.latitude, dtype=float)[:, 0]
LON_G = np.asarray(first.longitude, dtype=float)[0, :]
assert LAT1[0] == 90.0 and LAT1[-1] == -90.0 and np.all(np.diff(LAT1) < 0), "latitudes must descend"
ROLL = int(np.searchsorted(LON_G, 180.0))
LON1 = np.concatenate([LON_G[ROLL:] - 360.0, LON_G[:ROLL]])
assert LON_G[0] == 0.0 and LON_G[-1] == 359.0 and ROLL == 180, (LON_G[0], LON_G[-1], ROLL)
# The rolled axis runs -180..179 (the brief asserted -1.0 for the last node, which is the roll of
# 359 rather than of 179: LON_G[:180] = 0..179 is what follows the negative half).
assert LON1[0] == -180.0 and LON1[-1] == 179.0 and np.all(np.diff(LON1) > 0), "roll broke monotonicity"


def to_grid(value):
    """(y, x) field on the GFS scan -> DataArray on (latitude, longitude), NaN over ocean -> 0.

    Ocean NaN becomes 0 because §4 masks those columns out of the land mask anyway; keeping NaN
    here is not an option because CECE rejects non-finite inputs outright.
    """
    array = np.asarray(value, dtype=float)
    array = np.concatenate([array[:, ROLL:], array[:, :ROLL]], axis=-1)
    return xr.DataArray(
        array, dims=("latitude", "longitude"), coords={"latitude": LAT1, "longitude": LON1}
    ).fillna(0.0)


GRID = Grid.structured(longitude=LON1, latitude=LAT1)
assert GRID.dims == ("latitude", "longitude") and GRID.ncol == len(LAT1) * len(LON1) == 65160, GRID.dims
for lead in CASE["leads"][1:]:
    surf = LEADS[lead]["surf"]
    assert np.array_equal(np.asarray(surf.latitude, dtype=float)[:, 0], LAT1)
    assert np.array_equal(np.asarray(surf.longitude, dtype=float)[0, :], LON_G)
print(f"grid {GRID.dims} {len(LAT1)}x{len(LON1)} = {GRID.ncol} columns, lon {LON1[0]}…{LON1[-1]}")

# %%
# Land fraction per lead: the check that the longitude roll and the (y, x) -> (lat, lon) relabel
# did not scramble the field (a mix-up would change the percentage, not just its geography).
for lead in CASE["leads"]:
    land = to_grid(LEADS[lead]["surf"].LAND)
    assert land.shape == (181, 360) and bool(np.isfinite(land).all())
    print(f"f{lead:03d}  land columns {int((land > 0.5).sum()):6d} "
          f"({100 * float((land > 0.5).mean()):5.2f}% of the globe)")

# %%
nesdis = xr.open_dataset(os.path.join(CASE["climo_dir"], "FENGSHA_2022_NESDIS_inputs_10km_v4.nc"))
baker = xr.open_dataset(
    os.path.join(CASE["climo_dir"], "FENGSHA_BakerMethod_2020_2026_monthlyclimo_new_new.nc")
)
assert {"clayfrac", "sandfrac", "uthres", "uthres_sg"} <= set(nesdis.data_vars), list(nesdis.data_vars)
assert set(baker.feff.sizes) == {"time", "lat", "lon"} and baker.sizes["time"] == 12


def climo(var, month=None):
    """NESDIS/Baker naming (lat/lon, optional time) -> (latitude, longitude) on the 0.1° grid."""
    # Conditional: UNUSABLE below feeds arrays that are already renamed, and an unconditional
    # rename of lat/lon raises "not a variable or dimension in this dataset".
    da = var.rename({k: v for k, v in (("lat", "latitude"), ("lon", "longitude")) if k in var.dims}).astype(float)
    if "time" in da.dims:
        da = da.isel(time=month)
    return da.transpose("latitude", "longitude")


def nearest(da):
    """Explicit nearest alignment onto the model grid.

    999-sentinel and NaN cells are zeroed first: regrid_rectilinear rejects non-finite input, and a
    zeroed cell is only ever consumed by a column §4 has already masked out.
    """
    return regrid_rectilinear(da.where(np.isfinite(da) & (da < SENTINEL), 0.0), GRID, method="nearest")


MONTH = LEADS[CASE["leads"][0]]["valid"].month - 1
CLAY, SAND = nearest(climo(nesdis.clayfrac)), nearest(climo(nesdis.sandfrac))
UTHRES, UTHRES_SG = nearest(climo(nesdis.uthres)), nearest(climo(nesdis.uthres_sg))
FEFF = nearest(climo(baker.feff, month=MONTH))
UNUSABLE = climo(
    (~np.isfinite(nesdis.clayfrac.rename({"lat": "latitude", "lon": "longitude"})))
    | (~np.isfinite(nesdis.sandfrac.rename({"lat": "latitude", "lon": "longitude"})))
    | (nesdis.uthres.rename({"lat": "latitude", "lon": "longitude"}) >= SENTINEL)
    | (nesdis.uthres_sg.rename({"lat": "latitude", "lon": "longitude"}) >= SENTINEL)
    | (~np.isfinite(climo(baker.feff, month=MONTH)))
).astype(float)
INVALID = nearest(UNUSABLE) > 0.5

for name, da in (("clay", CLAY), ("sand", SAND), ("uthres", UTHRES), ("feff", FEFF)):
    assert da.dims == ("latitude", "longitude") and da.shape == (181, 360), (name, da.dims, da.shape)
    assert bool(np.isfinite(da).all()), f"{name}: still carries non-finite values after alignment"
assert bool(((CLAY + SAND) <= 1.0 + 1e-9).all()), "clay+sand must not exceed 1 where soils are defined"
assert bool((CLAY >= 0).all()) and bool((SAND >= 0).all())
print(f"climatology month index {MONTH} ({LEADS[CASE['leads'][0]]['valid']:%b %Y})")
print(f"invalid share of the 1-degree grid: {100 * float(INVALID.mean()):5.2f}% (ocean + 999 sentinel)")
print(f"clay {float(CLAY.min()):.3f}-{float(CLAY.max()):.3f}  sand {float(SAND.min()):.3f}-{float(SAND.max()):.3f}  "
      f"uthres {float(UTHRES.min()):.3f}-{float(UTHRES.max()):.3f}  feff {float(FEFF.min()):.3f}-{float(FEFF.max()):.3f}")

# %%
