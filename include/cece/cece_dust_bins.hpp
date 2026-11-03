#ifndef CECE_DUST_BINS_HPP
#define CECE_DUST_BINS_HPP

/**
 * @file cece_dust_bins.hpp
 * @brief Configuration-driven dust size-bin geometry shared by the dust schemes.
 * @author CECE Development Team
 * @date 2025
 * @version 1.0
 */

#include <cmath>
#include <conf/value.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "cece/cece_kokkos_config.hpp"
#include "cece/cece_logger.hpp"

namespace cece {

/**
 * @namespace dustbins
 * @brief Helpers that turn scheme configuration into a dust size-bin geometry.
 *
 * @details
 * The number of dust size bins is a property of the model grid and its fields,
 * not of a physics scheme: every scheme takes it from the size dimension of the
 * emission field (`emissions.extent(2)`). What a scheme does need from
 * configuration is the bin *geometry* - the representative radius and the
 * lower/upper edge of each bin - because the threshold, drag and collection
 * calculations depend on particle size. This namespace resolves that geometry
 * from the user's configuration (a helm `conf::Value`, parsed from the YAML
 * control file) so that no scheme carries a hard-coded bin count or a
 * hard-coded default bin table.
 *
 * Resolution order (see `dustbins::Resolve`):
 *   1. `particle_radii` + `bin_lower_edges` + `bin_upper_edges` -> full geometry;
 *      unless `bin_weights` is also given, the Kok et al. (2014b) brittle
 *      fragmentation distribution is integrated over each bin for the weights.
 *   2. `particle_radii` alone -> representative radii only; schemes that need
 *      per-bin sizes read them from their own size input (for example the model
 *      `particle_radius` field) at run time.
 *   3. `bin_weights` alone -> mass fractions only, applied to whatever bins the
 *      emission field has.
 *   4. nothing supplied -> equal weights over the field bins (logged once per
 *      scheme, so the fallback is visible rather than silent).
 *
 * All sizes are in metres. A sequence whose values cannot be radii in metres is
 * rejected with an explicit message rather than used at the wrong scale.
 */
namespace dustbins {

/// @brief A resolved dust size-bin geometry supplied through configuration.
struct DustBins {
    /// @brief True when `radii`, `lower_edges` and `upper_edges` are all filled.
    bool has_geometry = false;

    /// @brief True when `weights` holds per-bin emission mass fractions.
    bool has_weights = false;

    /// @brief True when the weights came from the Kok et al. (2014b) distribution.
    bool weights_are_kok = false;

    /// @brief Representative particle radius of each bin [m].
    std::vector<double> radii;

    /// @brief Lower edge radius of each bin [m].
    std::vector<double> lower_edges;

    /// @brief Upper edge radius of each bin [m].
    std::vector<double> upper_edges;

    /// @brief Emission mass fraction carried by each bin [-], normalised to sum 1.
    std::vector<double> weights;

    /// @brief Human-readable description of how the bins were chosen.
    std::string source;
};

/**
 * @brief Compute the Kok et al. (2014b) brittle-fragmentation size distribution
 * normalised over the configured bins.
 * @details
 * Mass size distribution of the emitted dust from brittle fragmentation theory
 * (Kok et al. 2012, 2014b), as used by CTSM's `Leung_2023` dust scheme
 * (Leung et al. 2024):
 *   D(d) ~ d [1 + erf( ln(d/d_mm) / (sqrt(2) ln s) )] exp[-(d/lambda)^3] ln(r_up/r_low)
 * with median mass diameter d_mm = 3.4 um, geometric standard deviation s = 3.0
 * and crack-propagation length lambda = 12 um. The aggregate emission flux is
 * split across bins by these weights; CTSM Eq. 2.31.31 does the equivalent with
 * the fixed source-mode/bin matrix M_ij, while here the bins are integrated
 * directly.
 * @param radii Representative bin radii [m]
 * @param lower_edges Bin lower edge radii [m]
 * @param upper_edges Bin upper edge radii [m]
 * @return Normalized weights summing to 1.0, or all zeros if the distribution
 * integrates to zero over the given bins
 */
inline std::vector<double> compute_kok_distribution(const std::vector<double>& radii, const std::vector<double>& lower_edges,
                                                    const std::vector<double>& upper_edges) {
    constexpr double mmd = 3.4;      // median mass diameter d_mm [um] (Kok et al. 2014b)
    constexpr double stddev = 3.0;   // geometric standard deviation s [-]
    constexpr double lambda = 12.0;  // crack propagation length lambda [um]
    const double factor = 1.0 / (std::sqrt(2.0) * std::log(stddev));

    const int nbins = static_cast<int>(radii.size());
    std::vector<double> dist(nbins, 0.0);
    double total = 0.0;

    for (int n = 0; n < nbins; ++n) {
        const double diameter = 2.0 * radii[n] * 1.0e6;  // m -> um
        const double dlam = diameter / lambda;
        const double width = std::log(upper_edges[n] / lower_edges[n]);  // ln(r_up/r_low) is scale-invariant
        dist[n] = diameter * (1.0 + std::erf(factor * std::log(diameter / mmd))) * std::exp(-dlam * dlam * dlam) * width;
        total += dist[n];
    }

    if (total > 0.0) {
        for (double& w : dist) w /= total;
    }
    return dist;
}

namespace detail {

/**
 * @brief Read a configured sequence into a std::vector<double>.
 * @param node Configuration node to look the key up in
 * @param key Sequence key name
 * @return The values, or an empty vector when the key is absent
 */
inline std::vector<double> read_sequence(const conf::Value& node, const std::string& key) {
    const conf::Value entry = node[key];
    if (!entry.is_defined()) return {};
    if (entry.kind() != conf::Node_Kind::Sequence) {
        throw std::invalid_argument("dust bins: '" + key + "' must be a sequence of numbers");
    }
    return entry.as_double_list();
}

/**
 * @brief Reject values that cannot plausibly be a radius in metres.
 * @param key Configuration key name, used in the error message
 * @param values Candidate radii/edges [m]
 */
inline void check_metres(const std::string& key, const std::vector<double>& values) {
    for (const double v : values) {
        if (!(v > 0.0)) {
            throw std::invalid_argument("dust bins: '" + key + "' contains non-positive value " + std::to_string(v));
        }
        if (v > 1.0e-3) {
            throw std::invalid_argument("dust bins: '" + key + "' contains " + std::to_string(v) +
                                        ", which is not a plausible bin radius in metres - give radii/edges in m (1 um = 1.0e-6 m)");
        }
    }
}

}  // namespace detail

/**
 * @brief Resolve the dust size-bin geometry from a scheme's configuration.
 * @details
 * Implements the resolution order described in the `dustbins` namespace
 * documentation. A partial geometry (for example `particle_radii` without
 * `bin_upper_edges`) is rejected rather than silently completed, so that a typo
 * in the control file cannot quietly change the bin definition.
 * @param config Scheme options node (may be undefined)
 * @param scheme_name Name used in error messages
 * @return The resolved geometry; `has_geometry`/`has_weights` say what was filled
 */
inline DustBins Resolve(const conf::Value& config, const std::string& scheme_name) {
    const std::string prefix = scheme_name + ": dust bins";

    if (config["num_bins"].is_defined()) {
        throw std::invalid_argument(prefix +
                                    ": the 'num_bins' option was removed - the bin count comes from the size"
                                    " dimension of the emission field. Configure the bin geometry instead with"
                                    " 'particle_radii', 'bin_lower_edges' and 'bin_upper_edges'.");
    }

    DustBins bins;
    bins.radii = detail::read_sequence(config, "particle_radii");
    bins.lower_edges = detail::read_sequence(config, "bin_lower_edges");
    bins.upper_edges = detail::read_sequence(config, "bin_upper_edges");
    bins.weights = detail::read_sequence(config, "bin_weights");

    const bool has_radii = !bins.radii.empty();
    const bool has_lower = !bins.lower_edges.empty();
    const bool has_upper = !bins.upper_edges.empty();
    const bool has_weights = !bins.weights.empty();

    if ((has_lower || has_upper) && !has_radii) {
        throw std::invalid_argument(prefix + ": 'bin_lower_edges'/'bin_upper_edges' require 'particle_radii'");
    }
    if (has_radii && (has_lower != has_upper)) {
        throw std::invalid_argument(prefix + ": 'bin_lower_edges' and 'bin_upper_edges' must be given together");
    }
    if (has_radii && has_lower && has_upper && !(bins.radii.size() == bins.lower_edges.size() && bins.radii.size() == bins.upper_edges.size())) {
        throw std::invalid_argument(prefix +
                                    ": 'particle_radii', 'bin_lower_edges' and 'bin_upper_edges' must have the"
                                    " same length (got " +
                                    std::to_string(bins.radii.size()) + ", " + std::to_string(bins.lower_edges.size()) + ", " +
                                    std::to_string(bins.upper_edges.size()) + ")");
    }
    if (has_radii) detail::check_metres("particle_radii", bins.radii);
    if (has_lower) {
        detail::check_metres("bin_lower_edges", bins.lower_edges);
        detail::check_metres("bin_upper_edges", bins.upper_edges);
        for (size_t n = 0; n < bins.radii.size(); ++n) {
            if (!(bins.upper_edges[n] > bins.lower_edges[n])) {
                throw std::invalid_argument(prefix + ": bin " + std::to_string(n) + " has invalid edges [" + std::to_string(bins.lower_edges[n]) +
                                            ", " + std::to_string(bins.upper_edges[n]) + "] (need 0 < lower < upper)");
            }
            if (!(bins.radii[n] >= bins.lower_edges[n] && bins.radii[n] <= bins.upper_edges[n])) {
                throw std::invalid_argument(prefix + ": bin " + std::to_string(n) + " representative radius " + std::to_string(bins.radii[n]) +
                                            " lies outside its edges [" + std::to_string(bins.lower_edges[n]) + ", " +
                                            std::to_string(bins.upper_edges[n]) + "]");
            }
        }
        bins.has_geometry = true;
    }

    if (has_weights) {
        if (bins.has_geometry && bins.weights.size() != bins.radii.size()) {
            throw std::invalid_argument(prefix + ": 'bin_weights' has " + std::to_string(bins.weights.size()) + " entries but " +
                                        std::to_string(bins.radii.size()) + " bins were configured");
        }
        double sum = 0.0;
        for (const double w : bins.weights) {
            if (!(w >= 0.0)) {
                throw std::invalid_argument(prefix + ": 'bin_weights' must be non-negative");
            }
            sum += w;
        }
        if (sum <= 0.0) {
            throw std::invalid_argument(prefix + ": 'bin_weights' must contain at least one positive value");
        }
        for (double& w : bins.weights) w /= sum;  // accept raw fractions or un-normalised masses
        bins.has_weights = true;
    }

    if (bins.has_geometry && !bins.has_weights) {
        bins.weights = compute_kok_distribution(bins.radii, bins.lower_edges, bins.upper_edges);
        bins.weights_are_kok = true;
        bins.has_weights = true;
        bins.source = "Kok et al. (2014b) brittle-fragmentation weights integrated over the configured bins";
    } else if (bins.has_geometry) {
        bins.source = "configured bin geometry with configured weights";
    } else if (bins.has_weights) {
        bins.source = "configured weights, applied directly to the emission field bins";
    } else {
        bins.source = "no bin configuration - equal weights over the emission field bins";
        CECE_LOG_WARNING(prefix +
                         ": no 'particle_radii'/'bin_weights' configured, emitting with equal weights across the "
                         "size bins of the emission field. Configure the bin geometry explicitly if the scheme needs "
                         "size-resolved thresholds.");
    }
    CECE_LOG_INFO(prefix + ": " + bins.source);
    return bins;
}

/**
 * @brief Build a Kokkos view from a host vector.
 * @param name Kokkos view label
 * @param values Host vector (may be empty)
 * @return A 1-D Kokkos::View<double*> holding a copy of `values`
 */
inline Kokkos::View<double*, Kokkos::DefaultExecutionSpace> MakeView(const std::string& name, const std::vector<double>& values) {
    const int n = static_cast<int>(values.size());
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> view(name, n);
    if (n == 0) return view;
    auto host = Kokkos::create_mirror_view(view);
    for (int i = 0; i < n; ++i) host(i) = values[i];
    Kokkos::deep_copy(view, host);
    return view;
}

}  // namespace dustbins
}  // namespace cece

#endif  // CECE_DUST_BINS_HPP
