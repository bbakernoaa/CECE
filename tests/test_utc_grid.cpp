// SPDX-License-Identifier: Apache-2.0
// CECE — Chemical Emissions Coupling Engine
// Feature 001 (local-time support) — T013 [US1]: RLE UTC-offset grid decoder tests.
//
// Decodes the real data/utc_grid_f1440.rle and checks the (lat,lon) probe
// offsets against an embedded reference produced by the generator's Python
// source (scripts/python/utcoffset_generator.py), plus the all-or-nothing
// length guarantee (FR-003, FR-008) and the nearest-cell mapping (FR-004).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "cece/utc_grid.hpp"

#ifndef CECE_SOURCE_DIR
#define CECE_SOURCE_DIR "."
#endif

namespace cece {
namespace {

std::string GridPath() {
    return std::string(CECE_SOURCE_DIR) + "/data/utc_grid_f1440.rle";
}

/// Reference offsets (quarter-hours) at named probe points, verified against
/// the generator's own expansion of the shipped file.
struct Probe {
    double lat;
    double lon;
    int quarter_hours;
};

TEST(UtcGridDecode, ExpandsToFullGlobalGrid) {
    const auto grid = DecodeUtcGridRle(GridPath());
    EXPECT_EQ(grid.size(), static_cast<std::size_t>(kUtcGridCells));
    // Observed range of the shipped snapshot: -12h .. +14h in quarter-hours.
    int lo = 127, hi = -128;
    for (const auto v : grid) {
        lo = std::min(lo, static_cast<int>(v));
        hi = std::max(hi, static_cast<int>(v));
    }
    EXPECT_EQ(lo, -48);
    EXPECT_EQ(hi, 56);
}

TEST(UtcGridDecode, ProbeCitiesMatchReference) {
    const auto grid = DecodeUtcGridRle(GridPath());
    const std::vector<Probe> probes = {
        {40.0, -74.0, -20},   // New York  (UTC-5)
        {51.5, -0.125, 0},    // London    (UTC+0)
        {28.6, 77.2, 22},     // Delhi     (UTC+5.5)
        {35.68, 139.7, 36},   // Tokyo     (UTC+9)
        {0.0, 0.0, 0},        // Atlantic (0,0)
        {-33.86, 151.2, 44},  // Sydney    (UTC+11)
        {52.5, 13.4, 4},      // Berlin    (UTC+1)
        {90.0, -180.0, -48},  // N-pole / date-line corner
        {-45.0, 179.99, 48},  // far south, near date line
    };
    for (const auto& p : probes) {
        const int row = UtcGridRowForLat(p.lat);
        const int col = UtcGridColForLon(p.lon);
        const int got = static_cast<int>(grid[static_cast<std::size_t>(row) * kUtcGridCols + col]);
        EXPECT_EQ(got, p.quarter_hours) << "probe lat=" << p.lat << " lon=" << p.lon;
    }
}

TEST(UtcGridMapping, RowColFloorAndClamp) {
    // Cell-center floor mapping: row = floor((90 - lat)/0.125), clamped.
    EXPECT_EQ(UtcGridRowForLat(90.0), 0);
    EXPECT_EQ(UtcGridRowForLat(89.9375), 0);  // center of row 0
    EXPECT_EQ(UtcGridRowForLat(89.9), 0);
    EXPECT_EQ(UtcGridRowForLat(-90.0), kUtcGridRows - 1);
    EXPECT_EQ(UtcGridRowForLat(0.0), kUtcGridRows / 2);  // 720
    // Column: col = floor(mod(lon+180,360)/0.125), clamped; wraps conventions.
    EXPECT_EQ(UtcGridColForLon(-180.0), 0);
    EXPECT_EQ(UtcGridColForLon(180.0), 0);               // wraps to -180
    EXPECT_EQ(UtcGridColForLon(0.0), kUtcGridCols / 2);  // 1440
    EXPECT_EQ(UtcGridColForLon(179.99), kUtcGridCols - 1);
    EXPECT_EQ(UtcGridColForLon(190.0), UtcGridColForLon(-170.0));  // modulo wrap
}

TEST(UtcGridMapping, MapToNativeProducesSecondsBandLocal) {
    const auto grid = DecodeUtcGridRle(GridPath());
    // Two columns at NY and Tokyo longitudes, single equator row, whole band.
    const std::vector<double> lons = {-74.0, 139.7};
    const std::vector<double> lats = {40.0};
    const auto secs = MapToNativeOffsetsSec(grid, lons, lats, /*nx=*/2, /*j0=*/0, /*ny_local=*/1);
    ASSERT_EQ(secs.size(), 2u);
    EXPECT_EQ(secs[0], -20 * kUtcOffsetQuarterHoursPerSecond);  // NY -5h
    EXPECT_EQ(secs[1], 36 * kUtcOffsetQuarterHoursPerSecond);   // Tokyo +9h
    // Output layout is out[i + j*nx]; band-local j offset by j0.
    const std::vector<double> lats6 = {89.0, 88.0, 87.0, 86.0, 85.0, 40.0};
    const auto band = MapToNativeOffsetsSec(grid, lons, lats6, /*nx=*/2, /*j0=*/5, /*ny_local=*/1);
    ASSERT_EQ(band.size(), 2u);
    EXPECT_EQ(band[0], secs[0]);
    EXPECT_EQ(band[1], secs[1]);
}

TEST(UtcGridDecode, TruncatedFileThrows) {
    // Write a deliberately truncated copy of the real file: fewer cells than
    // 1440x2880 must be rejected wholesale (never a partial grid masking 0s).
    std::ifstream in(GridPath(), std::ios::binary);
    ASSERT_TRUE(static_cast<bool>(in));
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_GT(bytes.size(), 30u);
    const std::string tmp = "/tmp/cece_utc_grid_truncated.rle";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
    }
    EXPECT_THROW(DecodeUtcGridRle(tmp), std::runtime_error);
    std::remove(tmp.c_str());
}

TEST(UtcGridDecode, MissingFileThrows) {
    EXPECT_THROW(DecodeUtcGridRle("/tmp/cece_does_not_exist_" + std::to_string(__LINE__) + ".rle"), std::runtime_error);
}

}  // namespace
}  // namespace cece
