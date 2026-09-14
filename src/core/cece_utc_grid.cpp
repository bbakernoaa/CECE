/**
 * @file cece_utc_grid.cpp
 * @brief Decoder + native-grid mapper for the static UTC-offset RLE grid.
 *
 * Implements the file-format contract in
 * specs/001-local-time-support/contracts/utc-grid-file.md: a headerless
 * sequence of little-endian 3-byte tokens (uint16 run_length, int8 offset),
 * expanded all-or-nothing to a dense 1440x2880 int8 field, plus the exact
 * inverse of the generator's cell-center floor mapping (research D1) onto the
 * native simulation grid.
 *
 * Deliberately free of Kokkos / config / MPI dependencies so the decoder is
 * unit-testable in isolation.
 */

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "cece/utc_grid.hpp"

namespace cece {
namespace {

/// Read exactly n bytes into buf; returns false on short read.
bool ReadExact(std::FILE* f, void* buf, std::size_t n) {
    return std::fread(buf, 1, n, f) == n;
}

}  // namespace

std::vector<std::int8_t> DecodeUtcGridRle(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("utc_grid: cannot open RLE file '" + path + "'");
    }

    std::vector<std::int8_t> grid;
    grid.reserve(kUtcGridCells);

    unsigned char token[3];
    std::size_t run = 0;
    while (std::fread(token, 1, 3, f) == 3) {
        // Little-endian uint16 run_length, int8 offset (matches struct '<Hb').
        const std::uint16_t run_length = static_cast<std::uint16_t>(token[0]) | (static_cast<std::uint16_t>(token[1]) << 8);
        const std::int8_t offset = static_cast<std::int8_t>(token[2]);
        run += run_length;
        if (run > static_cast<std::size_t>(kUtcGridCells)) {
            std::fclose(f);
            throw std::runtime_error("utc_grid: RLE file '" + path + "' expands beyond " + std::to_string(kUtcGridCells) + " cells (corrupt)");
        }
        grid.resize(grid.size() + run_length, offset);
    }
    const bool clean_eof = std::feof(f) != 0;
    std::fclose(f);

    if (!clean_eof) {
        throw std::runtime_error("utc_grid: read error on RLE file '" + path + "'");
    }
    // Reject a trailing partial token as corruption (all-or-nothing, FR-003).
    // Also reject any length mismatch — never a partially-populated grid.
    if (grid.size() != static_cast<std::size_t>(kUtcGridCells)) {
        throw std::runtime_error("utc_grid: RLE file '" + path + "' decoded to " + std::to_string(grid.size()) + " cells, expected " +
                                 std::to_string(kUtcGridCells));
    }
    return grid;
}

int UtcGridRowForLat(double lat) {
    // Cell-center inverse of lat = 90 - (row + 0.5) * step (generator convention).
    double row = std::floor((90.0 - lat) / kUtcGridStepDeg);
    if (row < 0.0) row = 0.0;
    if (row > kUtcGridRows - 1) row = kUtcGridRows - 1;
    return static_cast<int>(row);
}

int UtcGridColForLon(double lon) {
    // Wrap to [-180, 180) first so 0..360 conventions and exact 180 land correctly.
    double wrapped = std::fmod(lon + 180.0, 360.0);
    if (wrapped < 0.0) wrapped += 360.0;
    double col = std::floor(wrapped / kUtcGridStepDeg);
    if (col < 0.0) col = 0.0;
    if (col > kUtcGridCols - 1) col = kUtcGridCols - 1;
    return static_cast<int>(col);
}

std::vector<std::int32_t> MapToNativeOffsetsSec(const std::vector<std::int8_t>& grid, const std::vector<double>& native_lons,
                                                const std::vector<double>& native_lats, int nx, int j0, int ny_local) {
    if (grid.size() != static_cast<std::size_t>(kUtcGridCells)) {
        throw std::invalid_argument("utc_grid: MapToNativeOffsetsSec expects a dense grid of " + std::to_string(kUtcGridCells) + " cells, got " +
                                    std::to_string(grid.size()));
    }
    if (nx <= 0 || ny_local < 0 || native_lons.size() != static_cast<std::size_t>(nx)) {
        throw std::invalid_argument("utc_grid: MapToNativeOffsetsSec lon array size mismatch (nx=" + std::to_string(nx) +
                                    ", lons=" + std::to_string(native_lons.size()) + ")");
    }
    if (static_cast<std::size_t>(j0) + static_cast<std::size_t>(ny_local) > native_lats.size()) {
        throw std::invalid_argument("utc_grid: MapToNativeOffsetsSec band [" + std::to_string(j0) + ", " + std::to_string(j0 + ny_local) +
                                    ") exceeds native_lats size " + std::to_string(native_lats.size()));
    }

    std::vector<std::int32_t> out(static_cast<std::size_t>(nx) * ny_local, 0);
    for (int j = 0; j < ny_local; ++j) {
        const int row = UtcGridRowForLat(native_lats[static_cast<std::size_t>(j0 + j)]);
        for (int i = 0; i < nx; ++i) {
            const int col = UtcGridColForLon(native_lons[static_cast<std::size_t>(i)]);
            out[static_cast<std::size_t>(j) * nx + i] =
                static_cast<std::int32_t>(grid[static_cast<std::size_t>(row) * kUtcGridCols + col]) * kUtcOffsetQuarterHoursPerSecond;
        }
    }
    return out;
}

}  // namespace cece
