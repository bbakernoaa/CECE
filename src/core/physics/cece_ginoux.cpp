/**
 * @file cece_ginoux.cpp
 * @brief Native C++ implementation of the Ginoux (GOCART2G) dust emission scheme.
 *
 * Implements the Ginoux dust emission algorithm using Kokkos for GPU-portable
 * parallel execution. Includes helper functions for Marticorena dry-soil
 * threshold velocity and Ginoux moisture modification.
 *
 * References:
 * - Marticorena, B., et al. (1997), Modeling wind erosion, Annales Geophysicae, 15, 1381-1388.
 * - Ginoux, P., et al. (2001), Sources and distributions of dust aerosols, JGR, 106, 20255-20273.
 *
 * @author CECE Development Team
 * @date 2025
 * @version 1.0
 */

#include "cece/physics/cece_ginoux.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>

#include "cece/cece_dust_bins.hpp"
#include "cece/cece_logger.hpp"
#include "cece/cece_physics_factory.hpp"

namespace cece {

/// @brief Self-registration for the Ginoux dust emission scheme.
static PhysicsRegistration<GinouxScheme> register_ginoux("ginoux");

}  // namespace cece

/// @brief Compute Marticorena dry-soil threshold friction velocity.
/// @param diameter Particle diameter [m]
/// @param soil_density Soil particle density [kg/m³] (default 2650)
/// @param grav Gravity [m/s²]
/// @param air_dens Air density [kg/m³] (default 1.25)
/// @return Threshold friction velocity [m/s]
KOKKOS_INLINE_FUNCTION
double ginoux_marticorena_threshold(double diameter, double soil_density, double grav, double air_dens) {
    double reynol = 1331.0 * Kokkos::pow(100.0 * diameter, 1.56) + 0.38;
    return 0.13 * Kokkos::sqrt(soil_density * grav * diameter / air_dens) *
           Kokkos::sqrt(1.0 + 6.0e-7 / (soil_density * grav * Kokkos::pow(diameter, 2.5))) / Kokkos::sqrt(1.928 * Kokkos::pow(reynol, 0.092) - 1.0);
}

/// @brief Adjust threshold for soil moisture using Ginoux modification.
/// @param u_thresh0 Dry-soil threshold velocity [m/s]
/// @param gwettop Surface soil wetness [1]
/// @return Moisture-adjusted threshold velocity [m/s], or 0 if gwettop >= 0.5
KOKKOS_INLINE_FUNCTION
double ginoux_moisture_threshold(double u_thresh0, double gwettop) {
    if (gwettop >= 0.5) return 0.0;
    return Kokkos::max(0.0, u_thresh0 * (1.2 + 0.2 * Kokkos::log10(Kokkos::max(1.0e-3, gwettop))));
}

namespace cece {

/**
 * @brief Initialize the Ginoux scheme from YAML configuration.
 *
 * Reads the emission tuning constant and gravity, and resolves the optional
 * per-bin `particle_radii` override with `cece::dustbins::Resolve`. The bin
 * count is not configured here: it is the size dimension of the emission field.
 *
 * @param config YAML configuration node with Ginoux parameters
 * @param diag_manager Diagnostic manager for tracking outputs
 */
void GinouxScheme::Initialize(const YAML::Node& config, CeceDiagnosticManager* diag_manager) {
    BasePhysicsScheme::Initialize(config, diag_manager);

    if (config["ch_du"]) ch_du_ = config["ch_du"].as<double>();
    if (config["grav"]) grav_ = config["grav"].as<double>();

    // Optional per-bin radii override. The bin count is never configured here: it
    // is the size dimension of the emission field, and Run() checks the override
    // length against it. Without the override the model `radius` field supplies
    // the size of each bin.
    const auto bins = dustbins::Resolve(config, "GinouxScheme");
    if (!bins.radii.empty()) {
        particle_radii_ = dustbins::MakeView("ginoux_particle_radii", bins.radii);
        has_particle_radii_ = true;
    }

    std::cout << "GinouxScheme: Initialized. ch_du=" << ch_du_ << " grav=" << grav_ << " bins=" << bins.source << "\n";
}

/**
 * @brief Compute size-resolved Ginoux dust emissions for one timestep.
 *
 * Kokkos kernel over the (nx, ny) grid. For each land cell, the 10 m wind is
 * compared with the Marticorena dry-soil threshold modified by the Ginoux
 * soil-moisture factor; above it the emission is
 * `ch_du · du_src · (1 − f_lake) · u10 · u10 · (u10 − u_thresh)` per bin `n`,
 * with the bin radius taken from the configured `particle_radii` override
 * when its length matches the size dimension of the emission field, and from
 * the model `radius` field otherwise.
 *
 * @param import_state Input fields: 10 m wind components, land mask, surface
 *                     wetness, lake fraction, dust source function and particle
 *                     radius (the latter optional when `particle_radii` is
 *                     configured with a matching bin count)
 * @param export_state Output field `ginoux_dust_emissions` [kg m-2 s-1], 3D
 *                     (nx × ny × nbins); written in place
 */
void GinouxScheme::Run(CeceImportState& import_state, CeceExportState& export_state) {
    // Resolve import fields
    auto u10m = ResolveImport("u10m", import_state);
    auto v10m = ResolveImport("v10m", import_state);
    auto gwettop = ResolveImport("surface_soil_wetness", import_state);
    auto oro = ResolveImport("land_mask", import_state);
    auto fraclake = ResolveImport("lake_fraction", import_state);
    auto du_src = ResolveImport("dust_source", import_state);
    auto radius = ResolveImport("particle_radius", import_state);

    // Resolve export field
    auto emissions = ResolveExport("ginoux_dust_emissions", export_state);

    // The model `particle_radius` field is only required when no configured
    // override of matching length supplies the bin sizes.
    auto cfg_radii = particle_radii_;
    const bool use_cfg_radii = has_particle_radii_ && static_cast<int>(cfg_radii.extent(0)) == static_cast<int>(emissions.extent(2));
    const bool have_field_radii = (radius.data() != nullptr);

    // Early return if any field is missing
    if (u10m.data() == nullptr || v10m.data() == nullptr || gwettop.data() == nullptr || oro.data() == nullptr || fraclake.data() == nullptr ||
        du_src.data() == nullptr || emissions.data() == nullptr || (!use_cfg_radii && !have_field_radii)) {
        CECE_LOG_WARNING("GinouxScheme::Run: One or more required fields could not be resolved. Skipping execution.");
        return;
    }

    int nx = static_cast<int>(emissions.extent(0));
    int ny = static_cast<int>(emissions.extent(1));
    int nbins = static_cast<int>(emissions.extent(2));

    // Capture config parameters for the lambda
    double ch_du = ch_du_;
    double grav = grav_;

    constexpr double LAND = 1.0;
    constexpr double air_dens = 1.25;
    constexpr double soil_density = 2650.0;

    // Bin sizes come from the configured override when its length matches the
    // field, otherwise from the model `radius` field.
    if (has_particle_radii_ && !use_cfg_radii) {
        CECE_LOG_WARNING("GinouxScheme::Run: " + std::to_string(static_cast<int>(cfg_radii.extent(0))) +
                         " configured particle_radii"
                         " do not match the " +
                         std::to_string(nbins) +
                         " bins of the emission field; using the model"
                         " 'particle_radius' field instead.");
    }

    Kokkos::parallel_for(
        "GinouxKernel", Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {nx, ny}), KOKKOS_LAMBDA(int i, int j) {
            // Skip if not on land
            double oro_val = oro(i, j, 0);
            if (Kokkos::round(oro_val) != LAND) return;

            double w10m = Kokkos::sqrt(u10m(i, j, 0) * u10m(i, j, 0) + v10m(i, j, 0) * v10m(i, j, 0));

            double gwettop_val = gwettop(i, j, 0);
            double fraclake_val = fraclake(i, j, 0);
            double du_src_val = du_src(i, j, 0);

            for (int n = 0; n < nbins; ++n) {
                const double bin_radius = use_cfg_radii ? cfg_radii(n) : radius(0, 0, n);
                double diameter = 2.0 * bin_radius;

                // Marticorena dry-soil threshold velocity
                double u_thresh0 = ginoux_marticorena_threshold(diameter, soil_density, grav, air_dens);

                // Ginoux moisture modification
                double u_thresh = ginoux_moisture_threshold(u_thresh0, gwettop_val);

                // u_thresh == 0 means gwettop >= 0.5, no emission
                if (u_thresh <= 0.0) continue;

                if (w10m > u_thresh) {
                    emissions(i, j, n) = (1.0 - fraclake_val) * w10m * w10m * (w10m - u_thresh);
                    emissions(i, j, n) = ch_du * du_src_val * emissions(i, j, n);
                }
            }
        });

    Kokkos::fence();
    MarkModified("ginoux_dust_emissions", export_state);
}

}  // namespace cece
