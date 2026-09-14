import struct
import time
from datetime import datetime, timezone
from timezonefinder import TimezoneFinder
import zoneinfo

LATS = 1440  # 180 / 0.125
LONS = 2880  # 360 / 0.125
STEP = 0.125  # ~14 km at equator


def get_utc_offset_15min_units(
    tf: TimezoneFinder, lat: float, lon: float, target_date: datetime
) -> int:
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if not tz_name:
        return 0  # Default oceans to UTC (+0)

    try:
        tz = zoneinfo.ZoneInfo(tz_name)
        offset_seconds = target_date.astimezone(tz).utcoffset().total_seconds()
        offset_units = int(round(offset_seconds / 900.0))
        return max(-128, min(127, offset_units))
    except Exception:
        return 0


def main():
    print(
        f"Initializing TimezoneFinder for F1440 Grid ({LATS}x{LONS} = {LATS * LONS:,} points)..."
    )
    tf = TimezoneFinder(in_memory=True)
    ref_date = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    raw_grid = bytearray()
    start_time = time.time()

    print("Rasterizing globe (North to South, West to East)...")
    for row in range(LATS):
        lat = 90.0 - (row + 0.5) * STEP
        for col in range(LONS):
            lon = -180.0 + (col + 0.5) * STEP
            offset_units = get_utc_offset_15min_units(tf, lat, lon, ref_date)
            raw_grid.extend(struct.pack("b", offset_units))

        if (row + 1) % 360 == 0:
            print(f"Progress: {((row + 1) / LATS) * 100:.1f}%")

    print(f"Rasterization completed in {time.time() - start_time:.2f}s.")

    output_filename = "utc_grid_f1440.rle"
    print(f"Compressing into {output_filename} using Run-Length Encoding...")

    compressed = bytearray()
    current_val = struct.unpack("b", bytes([raw_grid[0]]))[0]
    run_length = 0

    for byte_val in raw_grid:
        signed_val = struct.unpack("b", bytes([byte_val]))[0]
        if signed_val == current_val and run_length < 65535:
            run_length += 1
        else:
            compressed.extend(struct.pack("<Hb", run_length, current_val))
            current_val = signed_val
            run_length = 1

    compressed.extend(struct.pack("<Hb", run_length, current_val))

    with open(output_filename, "wb") as f:
        f.write(compressed)

    print(f"Success! Saved to {output_filename} ({len(compressed) / 1024:.2f} KB)")


if __name__ == "__main__":
    main()
