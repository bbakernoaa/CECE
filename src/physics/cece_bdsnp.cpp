#include "cece/physics/cece_bdsnp.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "cece/cece_physics_factory.hpp"

namespace cece {

/// @brief Self-registration for the BDSNP soil NO scheme.
static PhysicsRegistration<BDSNPScheme> register_scheme("bdsnp");

/**
 * @brief Calculate soil temperature-dependent emission factor.
 *
 * @param tk Soil temperature [K]
 * @return Temperature-dependent emission factor (dimensionless)
 */
KOKKOS_INLINE_FUNCTION
double bdsnp_soil_temp_term(double tk) {
    double tc = tk - 273.16;
    if (tc <= 0.0) {
        return 0.0;
    }
    if (tc >= 30.0) {
        tc = 30.0;
    }
    return std::exp(0.103 * tc);
}

/**
 * @brief Calculate soil moisture-dependent NO emission factor.
 *
 * @param theta Water-filled pore space fraction [0-1]
 * @param arid Flag indicating arid region (1) or not (0)
 * @param nonarid Flag indicating nonarid region (1) or not (0)
 * @return Moisture-dependent emission factor (dimensionless)
 */
KOKKOS_INLINE_FUNCTION
double bdsnp_soil_wet_term(double theta, int arid, int nonarid) {
    if (arid == 1) {
        return 8.24 * theta * std::exp(-12.5 * theta * theta);
    } else if (nonarid == 1) {
        return 5.5 * theta * std::exp(-5.55 * theta * theta);
    }
    return 0.0;
}

/**
 * @brief Calculate pulse multiplier for NO emissions.
 *
 * @param theta Current soil moisture
 * @param theta_prev Previous soil moisture
 * @param pfactor Pulse factor state
 * @param dryperiod Dry period state
 * @param dt_hours Timestep in hours
 * @return Pulse multiplier
 */
KOKKOS_INLINE_FUNCTION
double bdsnp_pulse_multiplier(double theta, double theta_prev, double& pfactor, double& dryperiod,
                              double dt_hours) {
    if (theta < 0.3 && pfactor == 1.0) {
        double moistdiff = theta - theta_prev;
        if (moistdiff > 0.01) {
            pfactor = 13.01 * std::log(dryperiod) - 53.6;
            if (pfactor < 1.0) pfactor = 1.0;
            dryperiod = 0.001;
        } else {
            dryperiod += dt_hours;
        }
    } else if (pfactor != 1.0) {
        pfactor = pfactor * std::exp(-0.068 * dt_hours);
        if (theta < 0.3) dryperiod += dt_hours;
        if (pfactor < 1.0) pfactor = 1.0;
    }
    return pfactor;
}

void BDSNPScheme::Initialize(const YAML::Node& config, CeceDiagnosticManager* diag_manager) {
    BasePhysicsScheme::Initialize(config, diag_manager);

    if (config["fert_scale"]) {
        fert_scale_ = config["fert_scale"].as<double>();
    }

    std::cout << "BDSNPScheme: Initialized." << "\n";
}

void BDSNPScheme::Run(CeceImportState& import_state, CeceExportState& export_state) {
    auto soilt = ResolveImport("soil_temperature", import_state);
    auto soilm = ResolveImport("soil_moisture", import_state);
    auto soilm_prev = ResolveImport("soil_moisture_prev", import_state);
    auto landtype = ResolveImport("landtype", import_state);
    auto arid = ResolveImport("arid", import_state);
    auto nonarid = ResolveImport("nonarid", import_state);
    auto fert = ResolveImport("fertilizer", import_state);
    auto depn = ResolveImport("nitrogen_dep", import_state);
    auto crf = ResolveImport("canopy_reduction", import_state);

    // State fields (could also be stored internally if persistent)
    auto pfactor = ResolveImport("pulse_factor", import_state);
    auto dryperiod = ResolveImport("dry_period", import_state);

    auto bdsnp_no = ResolveExport("bdsnp_no", export_state);

    if (soilt.data() == nullptr || soilm.data() == nullptr || bdsnp_no.data() == nullptr) {
        return;
    }

    int nx = static_cast<int>(bdsnp_no.extent(0));
    int ny = static_cast<int>(bdsnp_no.extent(1));

    // Prepare views if optional fields are missing
    bool has_prev = soilm_prev.data() != nullptr;
    bool has_landtype = landtype.data() != nullptr;
    bool has_arid = arid.data() != nullptr;
    bool has_nonarid = nonarid.data() != nullptr;
    bool has_fert = fert.data() != nullptr;
    bool has_depn = depn.data() != nullptr;
    bool has_crf = crf.data() != nullptr;
    bool has_state = pfactor.data() != nullptr && dryperiod.data() != nullptr;

    double dt_hours = 1.0;  // Assume 1 hour timestep for now, can be read from config
    double fert_scale = fert_scale_;

    // Device copies of arrays
    Kokkos::Array<double, 24> k_a_biome;
    for (int i = 0; i < 24; ++i) {
        k_a_biome[i] = a_biome_[i];
    }

    Kokkos::parallel_for(
        "BDSNPKernel",
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(int i, int j) {
            double tk = soilt(i, j, 0);
            double theta = soilm(i, j, 0);
            double theta_prev = has_prev ? soilm_prev(i, j, 0) : theta;

            int lt =
                has_landtype ? static_cast<int>(landtype(i, j, 0)) : 10;  // Default to some biome
            if (lt < 1) lt = 1;
            if (lt > 24) lt = 24;

            int is_arid = has_arid ? static_cast<int>(arid(i, j, 0)) : 0;
            int is_nonarid =
                has_nonarid ? static_cast<int>(nonarid(i, j, 0)) : 1;  // Default to nonarid

            double fert_val = has_fert ? fert(i, j, 0) : 0.0;
            double depn_val = has_depn ? depn(i, j, 0) : 0.0;
            double crf_val = has_crf ? crf(i, j, 0) : 0.0;

            double pf = has_state ? pfactor(i, j, 0) : 1.0;
            double dp = has_state ? dryperiod(i, j, 0) : 0.0;

            double t_term = bdsnp_soil_temp_term(tk);
            double w_term = bdsnp_soil_wet_term(theta, is_arid, is_nonarid);
            double pulse = bdsnp_pulse_multiplier(theta, theta_prev, pf, dp, dt_hours);

            if (has_state) {
                pfactor(i, j, 0) = pf;
                dryperiod(i, j, 0) = dp;
            }

            double a_biome_val = k_a_biome[lt - 1];

            // FERTADD
            double a_fert = (fert_val + depn_val) / (86400.0 * 365.0) * fert_scale;

            double soilnox = (a_biome_val + a_fert) * (t_term * w_term * pulse) * (1.0 - crf_val);

            // Output in nmol/m2/s
            bdsnp_no(i, j, 0) += soilnox / 14.0;
        });

    Kokkos::fence();
    MarkModified("bdsnp_no", export_state);
    if (has_state) {
        // Technically these are imports modified, but CECE usually writes states back if mapped
        // correctly.
    }
}

}  // namespace cece
