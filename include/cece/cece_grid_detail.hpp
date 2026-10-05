// SPDX-License-Identifier: Apache-2.0
// CECE — Chemical Emissions Coupling Engine
// Copyright (c) HELM Project Contributors

#ifndef CECE_GRID_DETAIL_HPP
#define CECE_GRID_DETAIL_HPP

namespace cece {
namespace detail {

/// Wrap a longitude into [-180, 180). Applied to every coordinate source
/// (YAML/stream/gridspec files and ESMF-extracted grids alike) so both
/// drivers hand the core identically normalized arrays.
constexpr double wrap_longitude(double lon) {
    if (lon >= 180.0) {
        return lon - 360.0;
    }
    if (lon < -180.0) {
        return lon + 360.0;
    }
    return lon;
}

/// Convert a radian-flagged coordinate to degrees. ESMF spherical-rad grids
/// and radian-coded gridspec files both flow through this single rule.
constexpr double radians_to_degrees(double rad) {
    return rad * 180.0 / 3.14159265358979323846;
}

}  // namespace detail
}  // namespace cece

#endif  // CECE_GRID_DETAIL_HPP
