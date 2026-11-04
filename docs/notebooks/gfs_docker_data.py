# %% [markdown]
# GFS archive access for the FENGSHA GFS notebook, shared by `gfs-docker.sh` (which warms the
# default case) and the notebook (which fetches whatever the user edits CASE to). Remote
# *decoding* is not usable: grib2io's xarray backend hands the `s3://` URI to builtin `open()`,
# so the file has to exist locally before MONETIO can read it.

# %%
from __future__ import annotations

import argparse
import ssl
import urllib.request
from pathlib import Path

BUCKET = "noaa-gfs-bdp-pds"

# Post-2025 GFS archive layout: one flat directory per cycle hour. These files are 42-45 MB.
TEMPLATE = "gfs.{date}/{hour:02d}/atmos/gfs.t{hour:02d}z.pgrb2.1p00.f{lead:03d}"

MIN_BYTES = 8 * 1024 * 1024  # a truncated download is worse than no download at all


def object_key(date: str, hour: int, lead: int) -> str:
    return TEMPLATE.format(date=date, hour=hour, lead=lead)


def url_for(date: str, hour: int, lead: int) -> str:
    return f"https://{BUCKET}.s3.amazonaws.com/" + object_key(date, hour, lead)


def grib_path(cache_dir: str, date: str, hour: int, lead: int) -> Path:
    """Cache filename with the key's slashes folded out, so the cycle stays in the name."""
    return Path(cache_dir) / object_key(date, hour, lead).replace("/", "_")


def _urlopen(url: str, no_ssl_verify: bool):
    if no_ssl_verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return urllib.request.urlopen(url, timeout=1800, context=ctx)
    return urllib.request.urlopen(url, timeout=1800)


def ensure_gribs(cache_dir: str, date: str, hour: int, leads, no_ssl_verify: bool = False):
    """Return a local path per lead, fetching only what is missing."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for lead in leads:
        dest = grib_path(cache_dir, date, hour, lead)
        if dest.is_file() and dest.stat().st_size >= MIN_BYTES:
            print(f"cached  {dest.name}  {dest.stat().st_size / 1e6:6.1f} MB", flush=True)
        else:
            url = url_for(date, hour, lead)
            tmp = dest.with_suffix(dest.suffix + ".part")
            print(f"fetch   {url}", flush=True)
            with _urlopen(url, no_ssl_verify) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            size = tmp.stat().st_size if tmp.exists() else 0
            if size < MIN_BYTES:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"{url}: got {size} bytes")
            tmp.replace(dest)
            print(f"    ->  {dest.name}  {dest.stat().st_size / 1e6:6.1f} MB", flush=True)
        dest.chmod(0o666)
        out.append(dest)
    return out


def parse_leads(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch GFS 1-degree GRIBs from the public archive")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--hour", type=int, required=True)
    ap.add_argument("--leads", default="0,3,6,12")
    ap.add_argument("--no-ssl-verify", action="store_true")
    args = ap.parse_args(argv)
    paths = ensure_gribs(
        args.cache, args.date, args.hour, parse_leads(args.leads), args.no_ssl_verify
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
