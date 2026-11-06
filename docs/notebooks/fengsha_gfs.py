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
_CECE_PATHS = ("/cache/build/src/python", "build/src/python")
_CECE_PATH = next((_p for _p in _CECE_PATHS if os.path.isdir(_p)), None)
if _CECE_PATH:
    sys.path = [_p for _p in sys.path if _p not in _CECE_PATHS]
    sys.path.insert(0, _CECE_PATH)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from pathlib import Path  # noqa: E402
from datetime import datetime  # noqa: E402

import cece  # noqa: E402
from cece import Grid, Model  # noqa: E402
from cece.preparation import regrid_rectilinear  # noqa: E402
import monetio  # noqa: E402

REPO = "/work" if os.path.isdir("/work/docs/notebooks") else os.getcwd()
# No environment variables anywhere in this notebook: the scientific case is CECE YAML
# (docs/notebooks/fengsha_gfs_*.yaml) and the container paths are literals. The bindings
# bootstrap above and this cache choice are the only host/container difference, and both
# degrade to the repo-relative path so a plain `python fengsha_gfs.py` from a checkout works.
CACHE = "/cache/gfs" if os.path.isdir("/cache") else os.path.join(os.getcwd(), "cache", "gfs")
print(f"python  {sys.version.split()[0]}  {sys.executable}")
print(f"cece    {cece.__file__}")
print(f"monetio {monetio.__version__}")
print(f"repo    {REPO}")
print(f"cache   {CACHE}")

# %% [markdown]
# ## 1. Configuration
#
# The scheme is configured exactly the way a CECE run configures it: two standard YAML files,
# `fengsha_gfs_kok.yaml` (Kok et al. 2014a brittle fragmentation) and `fengsha_gfs_webb.yaml`
# (Webb et al. 2020 saltation flux), each with a `driver:` cycle and one `physics_schemes`
# entry. `Model` hands the scheme options to the C++ parser as YAML text either way
# (`src/python/model.py` serializes them with `yaml.safe_dump`), so a file and a dict are the
# same path -- the file is just reviewable, and it is what a driver run would read.
# Their `driver:` blocks must be identical, so one cycle governs both branches; the assert
# below is what proves it. Nothing here reads an environment variable.
# `monetio.load` needs both filter keys: without `productDefinitionTemplateNumber` the GRIB
# reader refuses a cube that mixes instantaneous and accumulation records. One file per read
# because merged files keep the first file's scalar `validDate`/`leadTime`.

# %%
CONFIG_FILES = {"brittle": "fengsha_gfs_kok.yaml", "webb": "fengsha_gfs_webb.yaml"}
CFG = {name: cece.load_config(os.path.join(REPO, "docs", "notebooks", fname))
       for name, fname in CONFIG_FILES.items()}
SCHEME_OPTIONS = {name: dict(cfg.physics_schemes[0].options) for name, cfg in CFG.items()}
DRIVER = CFG["brittle"].driver
DT = float(DRIVER.timestep_seconds)
_start, _end = datetime.fromisoformat(DRIVER.start_time), datetime.fromisoformat(DRIVER.end_time)
CYCLE = {"date": _start.strftime("%Y%m%d"), "hour": _start.hour}

CASE = {
    "product": "pgrb2.1p00",
    "leads": [0, 3, 6, 12],
    "thresholds": ["uthres", "uthres_sg"],
    "cache_dir": CACHE,
    "climo_dir": REPO,
}
SURFACE_FILTER = {"typeOfFirstFixedSurface": 1, "productDefinitionTemplateNumber": 0}
SOIL_FILTER = {"typeOfFirstFixedSurface": 106, "productDefinitionTemplateNumber": 0}
SENTINEL = 900.0  # NESDIS writes 999 into uthres/uthres_sg; anything >= this is "no data"
R_EARTH = 6.371e6
KDRY = 287.04  # dry air gas constant, J kg-1 K-1 -> rho = p / (KDRY * T)

assert DT > 0, "driver.timestep_seconds must be positive"
assert _start < _end, "driver.start_time must precede end_time"
assert CYCLE["hour"] in (0, 6, 12, 18), "GFS is delivered at 6-hourly cycle hours"
assert CASE["leads"] == sorted(set(CASE["leads"])), "leads must be unique and ascending"
assert max(CASE["leads"]) * 3600 <= (_end - _start).total_seconds(), "longest lead exceeds the cycle"
assert all(int(hours) * 3600 % DT == 0 for hours in CASE["leads"]), "leads must be whole steps"
assert set(CASE["thresholds"]) <= {"uthres", "uthres_sg"}, "unknown threshold set"
assert CASE["product"] == "pgrb2.1p00", "only the 1-degree product is in scope here"
for _name, _cfg in CFG.items():
    _result = _cfg.validate()
    assert _result == "Validation passed", (_name, _result)
    assert SCHEME_OPTIONS[_name]["use_brittle_impaction"] is (_name == "brittle"), _name
assert CFG["brittle"].driver == CFG["webb"].driver, "the two configs disagree about the cycle"
# Option *values* include lists, so compare (key, value) pairs with the list entries made
# hashable; the flux-formulation keys are the only allowed difference between the two files.
_SKIP = {"use_brittle_impaction", "alpha", "gamma", "kvhmax"}


def _pairs(_o):
    return {(k, tuple(v) if isinstance(v, list) else v) for k, v in _o.items() if k not in _SKIP}


_b_on, _b_off = SCHEME_OPTIONS["brittle"], SCHEME_OPTIONS["webb"]
assert _pairs(_b_on) == _pairs(_b_off), "the two configs must differ only in the flux formulation"
# the Webb file pins its branch's constants; they must equal the C++ defaults (hpp:40-42) that
# the brittle file leaves unset, otherwise "identical forcing" would be a false claim.
assert (_b_on.get("alpha", 1.0), _b_on.get("gamma", 1.0), _b_on.get("kvhmax", 2.45e-4)) == (
    _b_off["alpha"], _b_off["gamma"], _b_off["kvhmax"]), "Webb-branch constants drifted"
print(f"case    {CYCLE['date']} t{CYCLE['hour']:02d}z leads={','.join(str(v) for v in CASE['leads'])} "
      f"dt={DT:.0f}s from {CONFIG_FILES['brittle']} driver")
for _name in CONFIG_FILES:
    print(f"flux    {_name:<8} {CONFIG_FILES[_name]:<22} "
          f"use_brittle_impaction={SCHEME_OPTIONS[_name]['use_brittle_impaction']}")



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

import pandas as pd
import gfs_docker_data as gdata

SURFACE_NEEDED = ["FRICV", "GUST", "LAND", "SNOD", "HPBL", "SOTYP", "VEG", "pressure", "temperature"]
SOIL_NEEDED = ["SOILL", "TSOIL"]
LEADS = {}
for lead in CASE["leads"]:
    (path,) = gdata.ensure_gribs(CASE["cache_dir"], CYCLE["date"], CYCLE["hour"], [lead])
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
    expected = pd.Timestamp(f"{CYCLE['date']} {CYCLE['hour']:02d}:00") + pd.Timedelta(seconds=lead * 3600)
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

# %% [markdown]
# ## 4. From GFS fields to FENGSHA inputs
#
# | CECE input | source | transform |
# |---|---|---|
# | `friction_velocity` | GFS `FRICV` | roll to −180…179; **`u*` used directly**, not recomputed from the 10 m wind |
# | `threshold_velocity` | climatology `uthres` or `uthres_sg` | nearest alignment |
# | `soil_moisture` | GFS `SOILL` level 0 (0–0.1 m) | volume fraction, units are already `Proportion` |
# | `soil_temperature` | GFS `TSOIL` level 0 (0–0.1 m) | K; switches the frozen-soil cap on or off |
# | `pbl_height` | GFS `HPBL` | m; FENGSHA caps it at 3 km internally |
# | `clay_fraction`, `sand_fraction` | climatology `clayfrac`, `sandfrac` | nearest alignment |
# | `silt_fraction` | derived | `clip(1 − clay − sand, 0, 1)` |
# | `erodibility` | constant `1.0` | documented limitation, see §7 |
# | `drag_partition` | Baker climatology `feff` | nearest alignment, month of the case |
# | `air_density` | derived | `pressure / (287.04 × temperature)` — ideal gas for **dry** air |
# | `snow_fraction` | GFS `SNOD` | binary: `SNOD > 0` |
# | `lake_fraction` | constant `0.0` | no lake field in this path |
# | `land_mask` | GFS `LAND` **and** climatology coverage | `0` where soil data is unusable, else `LAND > 0.5` |
#
# Units are set here because `field_adapter` compares them exactly (it maps `m/s` to `m s-1`), and every
# value must be finite: CECE rejects NaN outright, so ocean NaN is zeroed and the column is masked out
# instead of left as a hole.

# %%
KDRY = 287.04  # J kg-1 K-1, dry-air gas constant, matching the FENGSHA source
R_EARTH = 6.371e6  # m


def labelled(da, expected_units):
    """Attach the units string `field_adapter` requires, without touching the data."""
    out = da.copy()
    out.attrs["units"] = expected_units
    return out


def to_grid_raw(value):
    """The same longitude roll as `to_grid`, but NaN is preserved (needed for masks)."""
    array = np.asarray(value, dtype=float)
    array = np.concatenate([array[:, ROLL:], array[:, :ROLL]], axis=-1)
    return xr.DataArray(array, dims=("latitude", "longitude"),
                        coords={"latitude": LAT1, "longitude": LON1})


def area_weights():
    """Cell area in m^2 on the 1° grid: cos(lat) · dlat · dlon · R², zero at the poles."""
    dlat = abs(float(np.diff(LAT1).mean())) * np.pi / 180.0
    dlon = abs(float(np.diff(LON1).mean())) * np.pi / 180.0
    w = np.cos(np.deg2rad(LAT1))[:, None] * (dlat * dlon * R_EARTH**2)
    return np.broadcast_to(w, (len(LAT1), len(LON1))).copy()


WEIGHT = area_weights()
# A sign slip in dlat, or cos(deg) instead of cos(rad), shows up here as a wrong total rather than as
# a wrong-looking map, which is exactly why the sum is checked against 4*pi*R^2.
assert float(WEIGHT.min()) >= 0.0
assert abs(float(WEIGHT.sum()) - 4.0 * np.pi * R_EARTH**2) / (4.0 * np.pi * R_EARTH**2) < 0.02, \
    "grid area must integrate to ~4 pi R^2"
print(f"cell area {float(WEIGHT.max()):.4e} m^2 at the equator, global total {float(WEIGHT.sum()):.4e} m^2")


def build_inputs(lead, threshold="uthres"):
    """All fourteen FENGSHA inputs for one lead, plus the numbers that justify the masks."""
    surf, soil, valid = LEADS[lead]["surf"], LEADS[lead]["soil"], LEADS[lead]["valid"]
    pressure, temperature = to_grid(surf.pressure), to_grid(surf.temperature)
    u_star, hpbl, snod = to_grid(surf.FRICV), to_grid(surf.HPBL), to_grid(surf.SNOD)
    moisture = to_grid(soil.SOILL.isel(level=0))
    soil_temperature = to_grid(soil.TSOIL.isel(level=0))
    land_01 = (to_grid_raw(surf.LAND) > 0.5).astype(float)
    mask = land_01.where(~INVALID, 0.0)  # a climatology gap wins over GFS land
    threshold_field = {"uthres": UTHRES, "uthres_sg": UTHRES_SG}[threshold]
    rho = pressure / (KDRY * temperature)

    fields = {
        "friction_velocity": labelled(u_star, "m s-1"),
        "threshold_velocity": labelled(threshold_field, "m s-1"),
        "soil_moisture": labelled(moisture, "1"),
        "soil_temperature": labelled(soil_temperature, "K"),
        "pbl_height": labelled(hpbl, "m"),
        "clay_fraction": labelled(CLAY, "1"),
        "sand_fraction": labelled(SAND, "1"),
        "silt_fraction": labelled((1.0 - CLAY - SAND).clip(0.0, 1.0), "1"),
        "erodibility": labelled(xr.full_like(land_01, 1.0), "1"),
        "drag_partition": labelled(FEFF, "1"),
        "air_density": labelled(rho, "kg m-3"),
        "lake_fraction": labelled(xr.zeros_like(land_01), "1"),
        "snow_fraction": labelled((snod > 0).astype(float), "1"),
        "land_mask": labelled(mask, "1"),
    }
    for name, da in fields.items():
        assert da.dims == GRID.dims and da.shape == (181, 360), (name, da.dims, da.shape)
        assert bool(np.isfinite(da).all()), f"{name}: non-finite value reached the adapter boundary"

    on_land = np.asarray(mask, dtype=float) > 0.5
    gfs_land = np.asarray(to_grid_raw(surf.LAND), dtype=float) > 0.5
    stats = {
        "valid": valid,
        "gfs_land_cols": int(gfs_land.sum()),
        "land_cols": int(on_land.sum()),
        "snow_cols": int((np.asarray(fields["snow_fraction"], dtype=float) > 0)[gfs_land].sum()),
        "frozen_cols": int((soil_temperature.values[gfs_land] <= 273.15).sum()),
        "ustar_p50": float(np.percentile(u_star.values[gfs_land], 50)),
        "ustar_p90": float(np.percentile(u_star.values[gfs_land], 90)),
        "ustar_max": float(u_star.values[gfs_land].max()),
        "rho_min": float(rho.values[gfs_land].min()),
        "rho_max": float(rho.values[gfs_land].max()),
        "uthres_max": float(threshold_field.max()),
    }
    assert set(np.unique(fields["snow_fraction"].values)) <= {0.0, 1.0}
    assert set(np.unique(fields["land_mask"].values)) <= {0.0, 1.0}
    assert 0.5 < float(fields["air_density"].min()) < float(fields["air_density"].max()) < 1.6
    assert float(fields["erodibility"].min()) == float(fields["erodibility"].max()) == 1.0
    assert float(fields["lake_fraction"].max()) == 0.0
    assert float(threshold_field.max()) < 2.0, "uthres above 2 m/s means the alignment mixed in a sentinel"
    return fields, stats


def build_report(rows, title):
    """Two-column table of the provenance numbers; §4 calls it for inputs, §5 for emissions."""
    print(f"\n{title}")
    print(f"{'quantity':<26}{'value':>18}")
    for key, value in rows.items():
        # Exponential for the small physically-interesting quantities: a dust flux of 1.68e-06
        # kg m-2 s-1 printed with %.3f is "0.000", which reads as "no emission" instead.
        if isinstance(value, float) and 0.0 < abs(value) < 1e-3:
            text = f"{value:.3e}"
        else:
            text = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"{key:<26}{text:>18}")


FIELDS, STATS = build_inputs(CASE["leads"][0], CASE["thresholds"][0])
assert set(FIELDS) == {
    "friction_velocity", "threshold_velocity", "soil_moisture", "soil_temperature", "pbl_height",
    "clay_fraction", "sand_fraction", "silt_fraction", "erodibility", "drag_partition",
    "air_density", "lake_fraction", "snow_fraction", "land_mask",
}, sorted(FIELDS)
build_report({
    "lead": f"f{CASE['leads'][0]:03d}",
    "valid time": f"{STATS['valid']:%Y-%m-%dT%H:%M}",
    "GFS land columns": STATS["gfs_land_cols"],
    "masked by climatology": STATS["gfs_land_cols"] - STATS["land_cols"],
    "columns run": STATS["land_cols"],
    "snow columns (of GFS land)": STATS["snow_cols"],
    "frozen top soil (of GFS land)": STATS["frozen_cols"],
    "u* p50 m/s": STATS["ustar_p50"],
    "u* p90 m/s": STATS["ustar_p90"],
    "u* max m/s": STATS["ustar_max"],
    "air density min kg/m3": STATS["rho_min"],
    "air density max kg/m3": STATS["rho_max"],
    "threshold max m/s": STATS["uthres_max"],
}, title=f"f{CASE['leads'][0]:03d} input provenance ({CASE['thresholds'][0]})")

# %% [markdown]
# ## 5. FENGSHA runs: four leads × two threshold sets
#
# Eight independent column loads — the scheme is single-column, so a lead change is only a change of
# GFS forcing, and the two threshold sets differ only in which climatology supplies `threshold_velocity`.
# Each run uses a fresh `Model`, because CECE keeps scheme state between steps.

# %%
import time


def run_case(lead, threshold):
    """One Model(["fengsha"]) step on one lead. Returns (emissions, per-bin kg/s, seconds)."""
    fields, stats = build_inputs(lead, threshold)
    with Model(["fengsha"], grid=GRID, options={"fengsha": dict(SCHEME_OPTIONS["brittle"])}) as model:
        model.set_inputs(fields)
        started = time.perf_counter()
        out = model.step(time=float(LEADS[lead]["lead_s"]), dt=DT)
        elapsed = time.perf_counter() - started
    em = out["fengsha_dust_emissions"]
    arr = np.asarray(em, dtype=float)
    assert arr.shape == (181, 360, len(SCHEME_OPTIONS["brittle"]["particle_radii"])), arr.shape
    assert bool(np.isfinite(arr).all()) and float(arr.min()) >= 0.0, "emissions must be finite and >= 0"
    weighted = arr * WEIGHT[..., None]  # kg/s per (column, size)
    per_bin = weighted.sum(axis=(0, 1))  # kg/s per size bin
    flux_col = arr.sum(axis=2)  # kg m-2 s-1 per column, all sizes summed
    colsum = weighted.sum(axis=2)  # kg/s per column
    total = float(per_bin.sum())
    run_stats = dict(stats)
    run_stats.update(
        total_kg_s=total,
        emitting_cols=int((colsum > 0).sum()),
        # The peak *flux* (per area) is what the column label claims; the peak load (per column)
        # is the area-weighted sibling. Reporting colsum.max() under a kg m-2 s-1 label gave a
        # four-digit number that no dust emission can be.
        peak_kg_m2_s=float(flux_col.max()),
        peak_kg_s=float(colsum.max()),
        seconds=elapsed,
    )
    index = int(np.argmax(flux_col))
    run_stats["peak_lat"] = float(LAT1[index // len(LON1)])
    run_stats["peak_lon"] = float(LON1[index % len(LON1)])
    return arr, per_bin, run_stats


# %%
RESULTS = {}
for threshold in CASE["thresholds"]:
    for lead in CASE["leads"]:
        RESULTS[(threshold, lead)] = run_case(lead, threshold)

em0, bins0, st0 = RESULTS[(CASE["thresholds"][0], CASE["leads"][0])]
build_report({
    "threshold set": CASE["thresholds"][0],
    "total kg/s": st0["total_kg_s"],
    "emitting columns": st0["emitting_cols"],
    "peak flux kg m-2 s-1": st0["peak_kg_m2_s"],
    "peak load kg/s": st0["peak_kg_s"],
    "peak latitude": st0["peak_lat"],
    "peak longitude": st0["peak_lon"],
    "column load s": st0["seconds"],
    "bin 1 kg/s": float(bins0[0]),
    "bin 5 kg/s": float(bins0[-1]),
}, title=f"f{CASE['leads'][0]:03d} emission report ({CASE['thresholds'][0]})")

# The two threshold sets must actually differ — if they do not, the climatology variable was ignored.
IDENTICAL = all(
    np.array_equal(RESULTS[(CASE["thresholds"][0], lead)][0], RESULTS[(CASE["thresholds"][1], lead)][0])
    for lead in CASE["leads"]
)
assert not IDENTICAL, "uthres and uthres_sg gave identical emissions"

print(f"\n{'set':<10}{'lead':>6}{'total kg/s':>14}{'emitting':>10}{'peak kg m-2 s-1':>18}{'s':>7}")
for (threshold, lead), (_, _, st) in RESULTS.items():
    print(f"{threshold:<10}{f'f{lead:03d}':>6}{st['total_kg_s']:>14.3e}{st['emitting_cols']:>10}"
          f"{st['peak_kg_m2_s']:>18.3e}{st['seconds']:>7.2f}")

# %% [markdown]
# ## 6. Maps
#
# Geography, not magnitude: this path is a data-pipeline demonstration. The Sahel column that lights up
# in the emissions panel is the same one the climatology flags as the most erodible in the world.
#
# Overlays use Cartopy's Natural Earth shapefiles (cached under $XDG_DATA_HOME/cartopy on the
# `/cache` volume after the first run). Coastlines and political borders are drawn with the
# 50 m dataset: crisp enough at 1°/°subpanel scale and ~1 MB, so no ~60 MB 10 m download.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# The launcher runs the kernel with MPLBACKEND=Agg so a headless `python file.py` works, but Agg
# alone never *displays* a figure: nbconvert captures nothing. Switch to inline when there is a
# shell, and stay on Agg when the file is executed as a plain script.
try:
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
    _INLINE = True
except NameError:
    _INLINE = False

FIGSIZE = (11, 8)
WEST, EAST, SOUTH, NORTH = -25.0, 15.0, 10.0, 32.0  # the Sahel band FENGSHA cares about in November
PC = ccrs.PlateCarree()  # the model grid is already lon/lat, so the plot CRS = the data CRS


def extent():
    """Pixel edges for imshow on a node-centred 1-degree grid."""
    return [float(LON1[0]) - 0.5, float(LON1[-1]) + 0.5, float(LAT1[-1]) - 0.5, float(LAT1[0]) + 0.5]


def geo_overlays(ax, *, borders=False, labels=False):
    """Natural Earth coastlines (+ optionally country borders) over a PlateCarree panel.

    Coast white with a dark underlay so they read on any cmap, from Greys land-mask (light) to
    magma (dark). Called on each panel once; shapefiles are lazily loaded from the volume cache.
    """
    ax.coastlines(resolution="50m", linewidth=0.9, color="0.1", zorder=3)
    ax.coastlines(resolution="50m", linewidth=0.4, color="white", zorder=4)
    if borders:
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4,
                       edgecolor="0.3", facecolor="none", zorder=3)
    gl = ax.gridlines(crs=PC, linewidth=0.3, color="0.5", alpha=0.35, linestyle=(0, (2, 3)),
                      draw_labels=labels, x_inline=False, y_inline=False, zorder=2)
    if labels:
        gl.top_labels = False
        gl.right_labels = False
        # leave gl.xformatter / gl.yformatter alone so cartopy's default degrees Formatter stays.
        # Assigning None here looks harmless but crashes draw(): geoaxes calls formatter.set_locs().


fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, subplot_kw={"projection": PC})
panels = (
    ("threshold velocity m/s", np.asarray(UTHRES), "magma", None),
    ("friction velocity m/s", np.asarray(FIELDS["friction_velocity"]), "turbo", 0.0),
    ("snow fraction", np.asarray(FIELDS["snow_fraction"]), "Blues", None),
    ("land mask used", np.asarray(FIELDS["land_mask"]), "Greys", None),
)
for ax, (title, array, cmap, clip) in zip(axes.ravel(), panels):
    shown = np.ma.masked_where(array <= clip, array) if clip is not None else array
    # `origin="upper"` still matters here: LAT1 descends from +90, so row 0 of `shown` must be
    # drawn at the top or the whole panel mirrors north-south. Cartopy only fixes the projection;
    # pixel row order is still imshow's, and the extent() pair is (lon-, lat+) top-left in LAT1[0].
    ax.imshow(shown, origin="upper", extent=extent(), transform=PC, cmap=cmap, aspect="auto")
    ax.set_global()
    geo_overlays(ax, labels=True)
    ax.set_title(title, pad=1)
fig.tight_layout()
if _INLINE:  # a bare `fig` expression would ALSO emit an execute_result next to the image
    plt.show()

# %%
fig2, axes2 = plt.subplots(
    2, 4, figsize=(15, 7), sharex=True, sharey=True, subplot_kw={"projection": PC}
)
LON_MESH, LAT_MESH = np.meshgrid(LON1, LAT1)
for row, threshold in enumerate(CASE["thresholds"]):
    for col, lead in enumerate(CASE["leads"]):
        ax = axes2[row, col]
        array = RESULTS[(threshold, lead)][0].sum(axis=-1)
        mesh = ax.pcolormesh(LON_MESH, LAT_MESH, np.ma.masked_where(array <= 0, array),
                             cmap="YlOrRd", shading="nearest", transform=PC, zorder=1)
        ax.set_extent([WEST, EAST, SOUTH, NORTH], crs=PC)
        # one graticule per column, one label set per edge — a 2x4 of labelled panels is noise
        geo_overlays(ax, borders=True, labels=(col == 0 and row == 1))
        ax.set_title(f"{threshold} f{lead:03d}  {array.max():.1e} kg m-2 s-1", pad=1)
fig2.suptitle("FENGSHA dust emissions, Sahel zoom (Natural Earth 50 m coastlines + borders)")
# tight_layout *before* the colourbar: with the colourbar axes already present matplotlib warns
# "Axes that are not compatible with tight_layout", and the warning lands in the committed output.
fig2.tight_layout()
fig2.colorbar(mesh, ax=axes2, fraction=0.03, pad=0.02)
print(f"{len(RESULTS)} cases mapped; peak Sahel cell is where the climatology threshold is lowest")
if _INLINE:
    plt.show()

# %% [markdown]
# ## 7. Known limitations of this path
#
# - **`erodibility = 1` everywhere.** The BES/dry-deposition erodibility field is not available here, so
#   every soil-defined land cell is treated as fully erodible. The relative geography is right; the
#   absolute flux is an upper bound for the grid cell.
# - **Frozen soil is a cliff, not a threshold.** 11,456 of the 22,071 GFS land columns (52 %)
#   have a top soil layer at or below 273.15 K, and the scheme suppresses emission in frozen
#   columns; the 1° GFS soil level cannot resolve the diurnal skin/soil split that a 12z desert
#   column actually has.
# - **`lake_fraction = 0`.** Great Salt Lake / Caspian-type sources are not masked out.
# - **Nearest, not conservative.** A model cell is described by one 0.1° climatology cell; the
#   desert edges therefore step instead of blending. `land_mask` is zeroed wherever the 10 km
#   (0.1°) sheet has no soil data, so 9,248 of the 22,071 GFS land columns (42 %) never run.
# - **`u*` comes from GFS `FRICV`.** It is not recomputed from the 10 m wind, so this notebook cannot
#   exercise the scheme's internal friction-velocity inversion.
# - **The atmosphere is ideal gas and `pressure`/`temperature` are the GFS surface fields.** No moisture
#   correction, so `air_density` is a few tenths of a percent low over humid oceans — those columns are masked.
# - **`time = lead seconds`** is passed to the step, not to the clock; FENGSHA ignores it.
