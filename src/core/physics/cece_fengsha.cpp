/**
 * @file cece_fengsha.cpp
 * @brief Native C++ implementation of the FENGSHA dust emission scheme.
 *
 * Implements the FENGSHA dust emission algorithm using Kokkos for GPU-portable
 * parallel execution: saltation-driven size-resolved emission with Fécan soil
 * moisture correction, MB95 vertical-to-horizontal flux ratio, drag partitioning
 * of wind stress by rocks/vegetation, and Comola-style emission intermittency.
 *
 * The threshold, drag-partition and intermittency helpers follow the Leung_2023
 * formulation documented in the CTSM 5.0 Technical Note, section 2.31 "Dust
 * Emission":
 * https://escomp.github.io/CTSM/tech_note/Dust/CLM50_Tech_Note_Dust.html#dust-emission-flux
 * Equation numbers cited below as (Eq. 2.31.x) refer to that note. Two total
 * emission fluxes are available and are selected with `use_brittle_impaction`:
 * the Kok et al. (2014a) / Leung_2023 brittle-fragmentation flux (Eq. 2.31.8),
 * and the Webb et al. (2020) horizontal saltation flux used by the original
 * FENGSHA formulation. See FengshaScheme::Run.
 *
 * Bibliography below follows the reference list of that note (§2.35); bracketed
 * text says which part of FENGSHA each source provides.
 *
 * References:
 * - Comola, F., Kok, J.F., Chamecki, M. and Martin, R.L. (2019), GRL, 46(22),
 *   13430–13440, doi:10.1029/2019GL085739. [intermittency eta, saltation height
 *   z_sal = 0.1 m, Eq. 2.31.22–2.31.30]
 * - Darmenova, K., Sokolik, I.N., Shao, Y., Marticorena, B. and Bergametti, G.
 *   (2009), JGR: Atmospheres, 114(D14), doi:10.1029/2008JD011236. [b1, b2, X
 *   constants of the sandblasting term]
 * - Fécan, F., Marticorena, B. and Bergametti, G. (1998), Annales Geophysicae,
 *   17(1), 149–157, doi:10.1007/s00585-999-0149-7. [soil moisture correction
 *   F_m(w), Eq. 2.31.5–2.31.6]
 * - Kok, J.F., Parteli, E.J., Michaels, T.I. and Karam, D.B. (2012), Reports on
 *   Progress in Physics, 75(10), 106901, doi:10.1088/0034-4885/75/10/106901.
 *   [B*_it = 0.81 impact-threshold ratio, Eq. 2.31.7]
 * - Kok, J.F., Mahowald, N.M., et al. (2014a), ACP, 14(23), 13023–13041,
 *   doi:10.5194/acp-14-13023-2014. [brittle-fragmentation emission flux,
 *   Eq. 2.31.8–2.31.13]
 * - Kok, J.F., Albani, S., Mahowald, N.M. and Ward, D.S. (2014b), ACP, 14(23),
 *   13043–13061, doi:10.5194/acp-14-13043-2014. [size distribution D(n),
 *   revised brittle-fragmentation form]
 * - Leung, D.M., Kok, J.F., et al. (2023), ACP, 23(11), 6487–6523,
 *   doi:10.5194/acp-23-6487-2023. [scheme structure, eta, drag partition F_eff]
 * - Leung, D.M., Kok, J.F., et al. (2024), ACP, 24(4), 2287–2318,
 *   doi:10.5194/acp-24-2287-2024. [CTSM implementation and tuning]
 * - Marticorena, B. and Bergametti, G. (1995), JGR, 100(D8), 16415–16430,
 *   doi:10.1029/95JD00690. [k_v/h vertical-to-horizontal flux ratio; clay
 *   sandblasting factor; drag partition, Eq. 2.31.19]
 * - Okin, G.S. (2008), JGR: Earth Surface, 113(F2), doi:10.1029/2007JF000758.
 *   [vegetation drag partition, Eq. 2.31.20]
 * - Panofsky, H.A., Tennekes, H., Lenschow, D.H. and Wyngaard, J.C. (1977),
 *   Boundary-Layer Meteorology, 11(3), 355–361, doi:10.1007/BF02186086.
 *   [convective shear standard deviation, Eq. 2.31.26]
 * - Pierre, C., Bergametti, G., Marticorena, B., Kergoat, L., Mougin, E. and
 *   Hiernaux, P. (2014), JGR: Earth Surface, 119(10), 2291–2313,
 *   doi:10.1002/2014JF003177. [vegetation gap-length K, Eq. 2.31.21]
 * - Prigent, C., Tegen, I., Aires, F., Marticorena, B. and Zribi, M. (2005),
 *   JGR: Atmospheres, 110(D9), doi:10.1029/2004JD005370. [aeolian roughness
 *   length z_0a dataset]
 * - Shao, Y. and Lu, H. (2000), JGR: Atmospheres, 105(D17), 22437–22443,
 *   doi:10.1029/2000JD900304. [dry fluid threshold u_*ft0, Eq. 2.31.2]
 * - Zender, C.S., Bian, H. and Newman, D. (2003), JGR, 108(D14), 4416,
 *   doi:10.1029/2002JD002775. [DEAD; f_clay' sandblasting efficiency,
 *   Eq. 2.31.9; bin partition, Eq. 2.31.31–2.31.32]
 *
 * Not in the CTSM reference list (from the FENGSHA formulation itself):
 * - Webb, N.P., et al. (2020), Current Opinion in Environmental Sustainability,
 *   44, 138–146. [horizontal saltation flux q, used when
 *   `use_brittle_impaction` is off]
 *
 * @author Barry Baker
 * @date 2025
 * @version 1.0
 */

#include "cece/physics/cece_fengsha.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "cece/cece_dust_bins.hpp"
#include "cece/cece_logger.hpp"
#include "cece/cece_physics_factory.hpp"

namespace {

/// @brief Impact-threshold ratio B*_it: u_*it = B*_it u_*ft0 (CTSM Eq. 2.31.7).
/// Earth value from Kok et al. (2012); moisture-independent by construction.
constexpr double B_STAR_IT = 0.81;

/// @brief Aeolian roughness length z_0a [m] used to map friction velocities to
/// the saltation height (CTSM §2.31.4 sets this constant for simplicity).
constexpr double Z0_AEOLIAN = 1.0e-4;

/// @brief Default planetary boundary layer height z_i [m] used when no PBL height
/// field is supplied (CTSM Leung_2023 constant, Eq. 2.31.26).
constexpr double PBLH_DEFAULT = 1000.0;

/// @brief Sentinel Obukhov length [m] used when no Obukhov length field is
/// supplied: large positive value gives the neutral 12^(1/3) shear limit of
/// Panofsky et al. (1977) in Eq. 2.31.26.
constexpr double NEUTRAL_OBL = 1.0e6;

/// @brief Reference surface air density ρ_0atm [kg m⁻³] used to standardise the
/// fluid threshold (CTSM Eq. 2.31.13; Kok et al. 2014a).
constexpr double RHO_ATM_REF = 1.225;

/// @brief Standardised threshold friction velocity u_*st0 [m s⁻¹] for brittle
/// fragmentation, i.e. the reference erodibility of the soil (CTSM Eq. 2.31.12;
/// Kok et al. 2014a). CTSM uses this fixed value rather than a site threshold.
[[maybe_unused]] constexpr double UST_ST0 = 0.16;

/// @brief Fragmentation-exponent constant C_κ [-] (CTSM Eq. 2.31.12).
constexpr double C_KAPPA = 2.7;

/// @brief Upper limit imposed on the fragmentation exponent κ in CTSM, since
/// Kok et al. (2014a) had no measurements above κ ≈ 3 (CTSM §2.31.2).
constexpr double KAPPA_MAX = 2.5;

/// @brief Compute the sandblasting-efficiency clay factor f_clay'.
/// @details
/// CTSM offers two forms (see Eq. 2.31.9 and Eq. 2.31.10):
///   Zender et al. (2003) / Kok et al. (2014a):  f_clay' = max(f_clay, 0.2)
///   Leung et al. (2024) CESM tuning:            f_clay' = max(0.1 + 0.5 f_clay, 0.2)
/// The second form limits how strongly the abundance of fine (clay) particles can
/// scale the total emission flux.
/// @param clay clay mass fraction of the top soil layer [kg kg⁻¹]
/// @param ctsm_form when true use the Leung et al. (2024) tuned form (Eq. 2.31.10),
///        otherwise the Zender/Kok form (Eq. 2.31.9)
/// @return clay factor f_clay' [-]
KOKKOS_INLINE_FUNCTION
double fengsha_clay_factor(double clay, bool ctsm_form) {
    const double raw = ctsm_form ? (0.1 + 0.5 * clay) : clay;
    return Kokkos::max(raw, 0.2);
}

}  // namespace

namespace cece {

/// @brief Self-registration for the FENGSHA dust emission scheme.
static PhysicsRegistration<FengshaScheme> register_fengsha("fengsha");

/// @brief Convert volumetric to gravimetric soil moisture (CTSM Eq. 2.31.3–2.31.4).
/// @details
///   w = θ · ρ_liq / ρ_bulk,   ρ_bulk = ρ_p (1 − θ_sat)
/// CTSM uses ρ_p = 2650 kg m⁻³; FENGSHA keeps the DEAD/Zender (2003) particle
/// density ρ_p = 1700 kg m⁻³ and the porosity parameterisation
/// θ_sat = 0.489 − 0.126·f_sand, which is also used for the saltation flux.
/// θ is the (water + ice) volumetric moisture of the top soil layer.
/// @param vsoil Volumetric soil moisture fraction θ [m³ m⁻³]
/// @param sandfrac Fractional sand content f_sand [1]
/// @return Gravimetric soil moisture w [percent, kg water kg soil⁻¹ × 100]
KOKKOS_INLINE_FUNCTION
double fengsha_soil_moisture_vol2grav(double vsoil, double sandfrac) {
    constexpr double rhow = 1000.0;          // density of liquid water ρ_liq [kg m⁻³]
    constexpr double rhop = 1700.0;          // soil particle density ρ_p [kg m⁻³] (Zender et al. 2003)
    double vsat = 0.489 - 0.126 * sandfrac;  // saturation/porosity θ_sat [1]
    return 100.0 * vsoil * rhow / (rhop * (1.0 - vsat));
}

/// @brief Fécan soil moisture correction factor Fm on the fluid threshold
///        (CTSM Eq. 2.31.5–2.31.6; Fécan et al. 1998).
/// @details
///   u_*ft = u_*ft0 · Fm(w)                                   (Eq. 2.31.1)
///   Fm(w) = 1                                   for w ≤ w_t
///   Fm(w) = sqrt(1 + 1.21 [100 (w − w_t)]^0.68) for w > w_t
/// with w and w_t in percent here, so the 100 factor is already folded into the
/// gravimetric conversion. The minimum moisture threshold is
///   w_t = b · f_clay (14 f_clay + 17)
/// which is the original Fécan et al. (1998) form with b as the soil-type
/// coefficient (FENGSHA config key `drylimit_factor`). Note CTSM's Leung_2023
/// tuning instead uses a = 1/f_clay (Eq. 2.31.6); `drylimit_factor` can be set
/// per-run to reproduce either behaviour.
/// @param slc Liquid water content, volumetric fraction θ [1]
/// @param sand Fractional sand content [1]
/// @param clay Fractional clay content f_clay [1]
/// @param b Drylimit factor (Fécan soil-type coefficient a_clay) [1]
/// @return Soil moisture correction factor Fm [1], ≥ 1
KOKKOS_INLINE_FUNCTION
double fengsha_moisture_correction_fecan(double slc, double sand, double clay, double b) {
    double grvsoilm = fengsha_soil_moisture_vol2grav(slc, sand);  // w [percent]
    double drylimit = b * clay * (14.0 * clay + 17.0);            // w_t [percent]
    double excess = Kokkos::max(0.0, grvsoilm - drylimit);
    return Kokkos::sqrt(1.0 + 1.21 * Kokkos::pow(excess, 0.68));
}

/// @brief Vertical-to-horizontal saltation flux ratio k_v/h (Marticorena &
///        Bergametti 1995, §3.2; adopted by Zender et al. 2003 and Leung et al. 2024).
/// @details
///   k_v/h = 10^(13.4 f_clay − 6)   for f_clay ≤ 0.2
///   k_v/h = kvhmax                 for f_clay > 0.2
/// The clay breakpoint and the maximum ratio cap both reflect the observation
/// that the rebound vertical flux saturates over clay-rich (non-erodible) soils.
/// @param clay Fractional clay content f_clay [1]
/// @param kvhmax Maximum flux ratio [1] (FENGSHA config key `kvhmax`)
/// @return Vertical-to-horizontal dust flux ratio k_v/h [1]
KOKKOS_INLINE_FUNCTION
double fengsha_flux_v2h_ratio_mb95(double clay, double kvhmax) {
    constexpr double clay_thresh = 0.2;  // clay breakpoint [1] (MB95 / Zender et al. 2003)
    if (clay > clay_thresh) {
        return kvhmax;
    }
    return Kokkos::pow(10.0, 13.4 * clay - 6.0);
}

/// @brief Rock/bare-soil drag partition factor F_eff (CTSM Eq. 2.31.19).
/// @details
/// Splits the grid-mean wind stress into the stress felt by the soil surface,
///   u_*s = u_* · F_eff                                     (Eq. 2.31.16)
/// after Marticorena & Bergametti (1995), valid for low-roughness surfaces and
/// only for cells with VAI ≤ VAI_thr (Eq. 2.31.15):
///   f_rock = 1 − ln(z_0a / z_0s) / ln[ b1 (X / z_0s)^b2 ]
/// b1, b2 and X follow Darmenova et al. (2009). z_0a is the satellite-derived
/// aeolian roughness of Prigent et al. (2005) and must not be confused with the
/// model aerodynamic roughness length Z0.
/// @param z0 aeolian roughness length z_0a [m]
/// @param z0s idealized soil roughness length z_0s [m]
/// @return feff = f_rock [1], clipped to [0, 1]
/// @note FengshaScheme::Run consumes the combined `drag_partition` field produced
///       by the driver; these helpers are provided for drivers/tests that need to
///       build F_eff themselves (CTSM Eq. 2.31.17 selects f_rock for bare-soil
///       patches and Eq. 2.31.18 selects f_veg for short-vegetation patches).
KOKKOS_INLINE_FUNCTION
double fengsha_mb95_drag_partition(double z0, double z0s) {
    constexpr double b1 = 0.7;  // coefficient (Darmenova et al. 2009) [-]
    constexpr double b2 = 0.8;  // coefficient (Darmenova et al. 2009) [-]
    constexpr double X = 10.0;  // distance downstream of the roughness discontinuity [m]
    if (z0 <= 0.0 || z0s <= 0.0) {
        return 1.0;  // no resolvable roughness contrast: full stress reaches the soil
    }
    double ratio = Kokkos::log(z0 / z0s) / Kokkos::log(b1 * Kokkos::pow(X / z0s, b2));
    return Kokkos::max(0.0, Kokkos::min(1.0, 1.0 - ratio));
}

/// @brief Short-vegetation drag partition factor F_eff (CTSM Eq. 2.31.20–2.31.21;
///        Okin 2008; Pierre et al. 2014; Leung et al. 2023).
/// @details
///   f_veg = (K + f0 c) / (K + c)                            (Eq. 2.31.20)
///   K = 2 (1/f_v − 1) = 2 (VAI_thr / VAI − 1)               (Eq. 2.31.21)
///   f_v = VAI / VAI_thr                                     (Eq. 2.31.15)
/// K is the mean normalised gap length between plant obstacles: it shrinks to 0
/// as VAI → VAI_thr, so F_eff → f0 in the dense-vegetation limit and → 1 for
/// bare ground. CTSM sets VAI_thr = 0.6 m² m⁻² (Leung et al. 2024 uses 1.0).
/// @param VAI vegetation area index = LAI + SAI [m² m⁻²]
/// @return feff = f_veg [1], clipped to [0, 1]
KOKKOS_INLINE_FUNCTION
double fengsha_leung_drag_partition(double VAI) {
    constexpr double f0 = 0.33;     // wind dissipation right behind an obstacle [-]
    constexpr double c = 4.8;       // e-folding momentum recovery distance after a plant [m]
    constexpr double VAIthr = 0.6;  // vegetation area index threshold for dust emission [m² m⁻²]
    if (VAI <= 0.0) {
        return 1.0;  // no obstacles: full stress reaches the soil surface
    }
    double K = 2.0 * (VAIthr / VAI - 1.0);
    double feff = (K + f0 * c) / (K + c);
    return Kokkos::max(0.0, Kokkos::min(1.0, feff));
}

/// @brief Emission intermittency factor eta (CTSM Eq. 2.31.22–2.31.30; Comola et
///        al. 2019; Panofsky et al. 1977; Leung et al. 2023).
/// @details
/// Fraction of the model timestep during which dust emission is active, because
/// sub-timestep boundary-layer turbulence drives the instantaneous wind across
/// the emission thresholds repeatedly. Friction velocities are first mapped to
/// the saltation height z_sal by the log law of the wall:
///   u_s  = (u_*s  / k) ln(z_sal / z_0a)                      (Eq. 2.31.22)
///   u_ft = (u_*ft / k) ln(z_sal / z_0a)                      (Eq. 2.31.23)
///   u_it = (u_*it / k) ln(z_sal / z_0a)                      (Eq. 2.31.24)
/// The instantaneous wind is taken Gaussian, ~N(u_s, sigma_us), with the
/// fluctuation strength from similarity theory (Panofsky et al. 1977):
///   sigma_us = u_*s (12 − 0.5 z_i/L)^(1/3)   for 12 − 0.5 z_i/L ≥ 0   (Eq. 2.31.26)
/// Then
///   alpha = { exp[ (u_ft² − u_it² − 2 u_s (u_ft − u_it)) / (2 sigma_us²) ] + 1 }^-1
///                                                            (Eq. 2.31.28, fluid-threshold crossing fraction)
///   P_ft = 0.5 [1 + erf((u_ft − u_s) / (sqrt(2) sigma_us))]  (Eq. 2.31.29)
///   P_it = 0.5 [1 + erf((u_it − u_s) / (sqrt(2) sigma_us))]  (Eq. 2.31.30)
///   eta  = 1 − P_ft + alpha (P_ft − P_it)                    (Eq. 2.31.27)
/// alpha weights the hysteresis zone between u_it and u_ft, where saltation is
/// self-sustaining but cannot be re-initiated. eta multiplies the emission flux
/// F_d (Eq. 2.31.8), which goes to zero once u_*s drops below u_*it.
/// @param z0       aeolian roughness length z_0a [m]; CTSM fixes this at 1e-4 m
///                 for simplicity (do not confuse with the model Z0)
/// @param L        Monin-Obukhov length z_i/L term of Eq. 2.31.26 [m]
/// @param PBLH     planetary boundary layer height z_i [m]; CTSM uses 1000 m
/// @param ustar_s  soil surface friction velocity u_*s = u_* F_eff [m s⁻¹] (Eq. 2.31.16)
/// @param ustar_ft idealized threshold velocity modified by soil moisture
///                 u_*ft = u_*ft0 Fm [m s⁻¹] (Eq. 2.31.1)
/// @param ustar_it impact/dynamic threshold u_*it = B*_it u_*ft0, B*_it = 0.81
///                 [m s⁻¹] (Eq. 2.31.7; Kok et al. 2012)
/// @return eta, the fraction of the timestep with active emission [0, 1]
KOKKOS_INLINE_FUNCTION
double fengsha_calc_eta(double z0, double L, double PBLH, double ustar_s, double ustar_ft, double ustar_it) {
    constexpr double z_sal = 0.1;  // saltation height [m] (Comola et al. 2019)
    constexpr double k = 0.41;     // von Kármán constant [-]
    constexpr double onethird = 1.0 / 3.0;
    constexpr double sqrt2 = 1.4142135623730951;

    // Emission shuts off entirely below the impact threshold (CTSM §2.31.1).
    if (ustar_s < ustar_it || ustar_it <= 0.0 || z0 <= 0.0) {
        return 0.0;
    }

    // Log law of the wall to the saltation height (Eq. 2.31.22–2.31.24).
    double logfac = Kokkos::log(z_sal / z0) / k;
    double U_s = ustar_s * logfac;
    double U_ft = ustar_ft * logfac;
    double U_it = ustar_it * logfac;

    // Similarity-theory wind fluctuation (Eq. 2.31.26); only valid for
    // 12 − 0.5 z_i/L >= 0, with the neutral shear limit of 12^(1/3) otherwise.
    double stab = 12.0 - 0.5 * PBLH / L;
    if (stab < 0.0 || !Kokkos::isfinite(stab)) {
        stab = 12.0;
    }
    double sigma_s = ustar_s * Kokkos::pow(stab, onethird);

    // Fluid-threshold crossing fraction (Eq. 2.31.28).
    double alpha_arg = (U_ft * U_ft - U_it * U_it - 2.0 * U_s * (U_ft - U_it)) / (2.0 * sigma_s * sigma_s);
    double alpha = 1.0 / (Kokkos::exp(alpha_arg) + 1.0);

    // Cumulative probabilities of not exceeding each threshold (Eq. 2.31.29–2.31.30).
    double pft = 0.5 * (1.0 + Kokkos::erf((U_ft - U_s) / (sqrt2 * sigma_s)));
    double pit = 0.5 * (1.0 + Kokkos::erf((U_it - U_s) / (sqrt2 * sigma_s)));

    // Intermittency factor (Eq. 2.31.27).
    double eta = 1.0 - pft + alpha * (pft - pit);
    return Kokkos::max(0.0, Kokkos::min(1.0, eta));
}

/**
 * @brief Vertical dust mass emission flux from the brittle-impaction cascade,
 *        Kok et al. (2014a), as used by CTSM's Leung_2023 scheme.
 * @details
 *   F_d = eta · C_tune · C_d · f_bare · f_clay' · rho_atm
 *         · (u_*s² − u_*it²) / u_*it² · (u_*s / u_*it)^kappa   (CTSM Eq. 2.31.8)
 *
 * with the soil erodibility coefficient and fragmentation exponent
 *   C_d   = C_d0 · exp[−C_e (u_*st − u_*st0) / u_*st0]         (Eq. 2.31.11)
 *   kappa = min(C_κ · (u_*st − u_*st0) / u_*st0, 2.5)          (Eq. 2.31.12)
 *   u_*st = u_*ft · sqrt(rho_atm / rho_0atm)                   (Eq. 2.31.13)
 *
 * so the flux scales roughly as u_*s^(2+kappa): emission becomes far more
 * wind-sensitive over less erodible (larger u_*ft) soil. Note that the
 * bombardment threshold u_*it — not the fluid threshold — sets the denominator,
 * and that F_d is zero below it.
 *
 * @param eta    emission intermittency factor, fengsha_calc_eta (Eq. 2.31.27)
 * @param rho_a  surface air density rho_atm [kg m⁻³]
 * @param ust_ft fluid threshold u_*ft including the soil moisture effect
 *               (Eq. 2.31.1), which drives the transient erodibility C_d
 * @param ust_st0 reference standardised threshold u_*st0 [m s⁻¹]; CTSM uses the
 *               constant 0.16 (UST_ST0), not a site-specific threshold
 * @param fclay_prime sandblasting clay factor f_clay' (fengsha_clay_factor)
 * @param fbare  bare-soil fraction f_bare (Eq. 2.31.14) [0–1]
 * @param ctune  tuning constant C_tune (CTSM default 0.05)
 * @param ust_s  saltation friction velocity u_*s = u_* F_eff (Eq. 2.31.16) [m/s]
 * @param ust_it impact threshold u_*it = B*_it u_*ft0 (Eq. 2.31.7) [m/s]
 * @return total vertical dust mass flux F_d [kg m⁻² s⁻¹]; 0 below threshold
 */
KOKKOS_INLINE_FUNCTION
double fengsha_brittle_impaction_flux(double eta, double rho_a, double ust_ft, double ust_st0, double fclay_prime, double fbare, double ctune,
                                      double ust_s, double ust_it) {
    constexpr double cd0 = 4.4e-5;  // C_d0, erodibility constant (Kok et al. 2014a)
    constexpr double ce = 2.0;      // C_e, erodibility decay constant
    if (ust_s <= ust_it || ust_it <= 0.0 || ust_st0 <= 0.0 || rho_a <= 0.0) return 0.0;

    // Standardised wet fluid threshold at reference air density (Eq. 2.31.13).
    const double ust_st = ust_ft * Kokkos::sqrt(rho_a / RHO_ATM_REF);
    const double st_excess = (ust_st - ust_st0) / ust_st0;

    // Transient soil erodibility (Eq. 2.31.11) and fragmentation exponent
    // (Eq. 2.31.12), the latter capped because Kok et al. (2014a) has no
    // measurements above kappa ~ 3.
    const double cd = cd0 * Kokkos::exp(-ce * st_excess);
    const double kappa = Kokkos::min(C_KAPPA * st_excess, KAPPA_MAX);

    const double ratio = (ust_s * ust_s - ust_it * ust_it) / (ust_it * ust_it);
    const double dflux = eta * ctune * cd * fbare * fclay_prime * rho_a * ratio * Kokkos::pow(ust_s / ust_it, kappa);
    return Kokkos::isfinite(dflux) && dflux > 0.0 ? dflux : 0.0;
}

/**
 * @brief Initialize the FENGSHA scheme from configuration.
 *
 * Reads the scheme tuning parameters (see docs/fengsha.md for the full table) and
 * resolves the dust size-bin geometry with `cece::dustbins::Resolve`. Nothing is
 * keyed to a fixed bin count: with `particle_radii`, `bin_lower_edges` and
 * `bin_upper_edges` configured, the Kok et al. (2014b) brittle-fragmentation
 * distribution is integrated over each configured bin (or `bin_weights` used if
 * also given); with only `bin_weights`, those fractions are used directly; with
 * nothing configured, the kernel falls back to equal weights over however many
 * size bins the emission field has. A `num_bins` option is rejected outright.
 *
 * @param config Scheme configuration (helm `conf::Value`) with FENGSHA parameters
 * @param diag_manager Diagnostic manager for tracking outputs
 */
void FengshaScheme::Initialize(const conf::Value& config, CeceDiagnosticManager* diag_manager) {
    BasePhysicsScheme::Initialize(config, diag_manager);

    if (config["alpha"]) alpha_ = config["alpha"].as_double();
    if (config["gamma"]) gamma_ = config["gamma"].as_double();
    if (config["kvhmax"]) kvhmax_ = config["kvhmax"].as_double();
    if (config["grav"]) grav_ = config["grav"].as_double();
    if (config["drylimit_factor"]) drylimit_factor_ = config["drylimit_factor"].as_double();
    if (config["use_intermittency"]) use_intermittency_ = config["use_intermittency"].as_bool();
    if (config["use_brittle_impaction"]) use_brittle_impaction_ = config["use_brittle_impaction"].as_bool();
    if (config["c_tune"]) c_tune_ = config["c_tune"].as_double();

    // Size-bin geometry is entirely configuration-driven (see cece_dust_bins.hpp);
    // the bin count itself is taken from the emission field in Run().
    const auto bins = dustbins::Resolve(config, "FengshaScheme");
    if (bins.has_weights) {
        bin_weights_ = dustbins::MakeView("fengsha_bin_weights", bins.weights);
        has_bin_weights_ = true;
    }

    std::cout << "FengshaScheme: Initialized. alpha=" << alpha_ << " gamma=" << gamma_ << " kvhmax=" << kvhmax_
              << " brittle_impaction=" << use_brittle_impaction_ << " bins=" << bins.source << "\n";
}

/**
 * @brief Compute size-resolved FENGSHA dust emissions for one timestep.
 *
 * Kokkos kernel over the (nx, ny) grid. For each land cell with non-negligible
 * erodibility, the emission is accumulated into `fengsha_dust_emissions` as
 * `E(n) += D(n) · F`, with D(n) the Kok et al. (2014b) bin weight and F the
 * size-integrated vertical dust mass flux, selected by
 * `use_brittle_impaction`:
 *
 *   - true:  F = F_d, the brittle-fragmentation flux of Kok et al. (2014a),
 *     CTSM Eq. 2.31.8 (fengsha_brittle_impaction_flux);
 *   - false: F = eta · alpha · f_bare · SSM^gamma · rho_atm · k_v/h · q,
 *     the original FENGSHA form, with k_v/h the Marticorena & Bergametti (1995)
 *     vertical-to-horizontal flux ratio and q the Webb et al. (2020, Eq. 9)
 *     horizontal saltation flux.
 *
 * In both cases f_bare is the bare-soil fraction (CTSM Eq. 2.31.14), u_*s = u_*
 * F_eff is the drag-partitioned friction velocity (Eq. 2.31.16) and u_*ft the
 * Fécan-corrected fluid threshold (Eq. 2.31.1); eta is the emission
 * intermittency factor (Eq. 2.31.27) when `use_intermittency` is enabled and 1
 * otherwise.
 *
 * The number of bins is the size dimension of the emission field, so the loop is
 * valid for any bin count. D(n) is the configured weight (Kok et al. 2014b
 * weights when a bin geometry was configured, otherwise the configured
 * `bin_weights`); when no weights are configured, or their length disagrees with
 * the field, the flux is split equally over the field bins, which keeps the
 * size-integrated total unchanged.
 *
 * @param import_state Input fields: friction velocity, threshold, soil moisture,
 *                     texture, erodibility, drag partition, air density, lake and
 *                     snow fractions, land mask; optionally soil temperature, LAI,
 *                     SAI, Obukhov length and PBL height.
 * @param export_state Output field `fengsha_dust_emissions` [kg m-2 s-1], 3D
 *                     (nx × ny × nbins); updated in place across timesteps.
 */
void FengshaScheme::Run(CeceImportState& import_state, CeceExportState& export_state) {
    // Resolve required import fields
    auto ustar = ResolveImport("friction_velocity", import_state);
    auto uthrs = ResolveImport("threshold_velocity", import_state);
    auto slc = ResolveImport("soil_moisture", import_state);
    auto clay = ResolveImport("clay_fraction", import_state);
    auto sand = ResolveImport("sand_fraction", import_state);
    auto silt = ResolveImport("silt_fraction", import_state);  // resolved for interface completeness; FENGSHA uses clay/sand only
    auto ssm = ResolveImport("erodibility", import_state);
    auto rdrag = ResolveImport("drag_partition", import_state);
    auto airdens = ResolveImport("air_density", import_state);
    auto fraclake = ResolveImport("lake_fraction", import_state);
    auto fracsnow = ResolveImport("snow_fraction", import_state);
    auto oro = ResolveImport("land_mask", import_state);

    // Resolve optional import fields. When absent, the scheme falls back to the
    // constants CTSM's Leung_2023 implementation hard-codes (see fengsha_calc_eta),
    // and soil is assumed unfrozen.
    auto soilt = ResolveImport("soil_temperature", import_state);
    auto lai = ResolveImport("leaf_area_index", import_state);
    auto sai = ResolveImport("stem_area_index", import_state);
    auto obl = ResolveImport("obukhov_length", import_state);
    auto pbl = ResolveImport("pbl_height", import_state);
    const bool has_soilt = (soilt.data() != nullptr);
    const bool has_veg = (lai.data() != nullptr);
    const bool has_obl = (obl.data() != nullptr);
    const bool has_pbl = (pbl.data() != nullptr);

    // Resolve export field
    auto emissions = ResolveExport("fengsha_dust_emissions", export_state);

    // Early return if any required field is missing
    if (ustar.data() == nullptr || uthrs.data() == nullptr || slc.data() == nullptr || clay.data() == nullptr || sand.data() == nullptr ||
        silt.data() == nullptr || ssm.data() == nullptr || rdrag.data() == nullptr || airdens.data() == nullptr || fraclake.data() == nullptr ||
        fracsnow.data() == nullptr || oro.data() == nullptr || emissions.data() == nullptr) {
        CECE_LOG_WARNING("FengshaScheme::Run: One or more required fields could not be resolved. Skipping execution.");
        return;
    }

    int nx = static_cast<int>(emissions.extent(0));
    int ny = static_cast<int>(emissions.extent(1));
    int nbins = static_cast<int>(emissions.extent(2));

    // Capture config parameters for the lambda
    double alpha = alpha_;
    double gamma_param = gamma_;
    double kvhmax = kvhmax_;
    [[maybe_unused]] double grav = grav_;  // reserved for the alpha/g sandblasting prefactor
    double drylimit_factor = drylimit_factor_;
    double c_tune = c_tune_;
    bool use_intermittency = use_intermittency_;
    bool use_brittle_impaction = use_brittle_impaction_;

    // Bin weights: configured weights are used only when their length matches the
    // size dimension of the emission field; otherwise the flux is split equally
    // over the field bins so that the total is still conserved (weights sum to 1).
    auto bin_weights = bin_weights_;
    int nweights = 0;
    if (has_bin_weights_) {
        const int configured = static_cast<int>(bin_weights.extent(0));
        if (configured == nbins) {
            nweights = configured;
        } else {
            CECE_LOG_WARNING("FengshaScheme::Run: " + std::to_string(configured) + " configured size bins do not match the " + std::to_string(nbins) +
                             " bins of the emission field; splitting the flux equally instead.");
        }
    } else if (!warned_equal_bins_) {
        CECE_LOG_WARNING(
            "FengshaScheme::Run: no bin configuration supplied, distributing emissions with equal weights over "
            "the " +
            std::to_string(nbins) + " bins of the emission field.");
        warned_equal_bins_ = true;
    }

    constexpr double LAND = 1.0;
    constexpr double SSM_THRESH = 1.0e-2;

    Kokkos::parallel_for(
        "FengshaKernel", Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {nx, ny}), KOKKOS_LAMBDA(int i, int j) {
            // Skip if not on land
            double oro_val = oro(i, j, 0);
            if (Kokkos::round(oro_val) != LAND) return;

            // Skip cells with negligible sediment supply / erodibility (SSM)
            double ssm_val = ssm(i, j, 0);
            if (ssm_val < SSM_THRESH) return;

            double clay_val = clay(i, j, 0);
            double sand_val = sand(i, j, 0);
            double rdrag_val = rdrag(i, j, 0);
            if (clay_val < 0.0 || sand_val < 0.0 || rdrag_val < 0.0) return;

            // Bare-soil fraction f_bare (CTSM Eq. 2.31.14):
            //   f_bare = (1 - f_lake)(1 - f_sno)(1 - f_v) * w_liq / (w_liq + w_ice)
            // The moisture-ratio term is approximated by a hard shutoff of frozen
            // soil, since FENGSHA does not carry separate liquid/ice soil water.
            // f_v = VAI / VAI_thr is the vegetation cover fraction (Eq. 2.31.15).
            constexpr double T_FREEZE = 273.15;  // freezing point of water [K]
            constexpr double VAIthr = 0.6;       // vegetation area index threshold for dust emission [m2 m-2]
            double fveg = 0.0;
            if (has_veg) {
                double VAI = lai(i, j, 0) + (sai.data() != nullptr ? sai(i, j, 0) : 0.0);
                fveg = Kokkos::max(0.0, Kokkos::min(1.0, VAI / VAIthr));
            }
            double fracland = 0.0;
            if (!has_soilt || soilt(i, j, 0) > T_FREEZE) {
                fracland = Kokkos::max(0.0, Kokkos::min(1.0, 1.0 - fraclake(i, j, 0))) * Kokkos::max(0.0, Kokkos::min(1.0, 1.0 - fracsnow(i, j, 0))) *
                           Kokkos::max(0.0, Kokkos::min(1.0, 1.0 - fveg));
            }
            if (fracland <= 0.0) return;

            // Drag-partition-adjusted soil-surface friction velocity
            // u_*s = u_* F_eff (CTSM Eq. 2.31.16).
            const double ustar_s = ustar(i, j, 0) * rdrag_val;

            // Vertical-to-horizontal mass flux ratio k_v/h (MB95), used by the
            // Webb-branch saltation flux below.
            const double kvh = fengsha_flux_v2h_ratio_mb95(clay_val, kvhmax);

            // Soil-moisture effect F_m(w) applied to the dry fluid threshold
            // u_*ft0 (Eq. 2.31.1, 2.31.5) supplied by the driver as
            // `threshold_velocity`; u_*it = B*_it u_*ft0 (Eq. 2.31.7).
            const double smois = slc(i, j, 0);
            const double h = fengsha_moisture_correction_fecan(smois, sand_val, clay_val, drylimit_factor);
            const double u_ft = uthrs(i, j, 0) * h;
            const double u_it = B_STAR_IT * uthrs(i, j, 0);

            // Emission intermittency eta: fraction of the timestep with active
            // emission under sub-timestep turbulent wind fluctuations (Eq. 2.31.27).
            double eta = 1.0;
            if (use_intermittency) {
                const double L_val = has_obl ? obl(i, j, 0) : NEUTRAL_OBL;
                const double z_i = has_pbl ? pbl(i, j, 0) : PBLH_DEFAULT;
                eta = fengsha_calc_eta(Z0_AEOLIAN, L_val, z_i, ustar_s, u_ft, u_it);
            }

            // Total (size-integrated) vertical dust mass flux.
            double total_emissions = 0.0;
            if (use_brittle_impaction) {
                // Kok et al. (2014a) brittle-fragmentation flux, CTSM Eq. 2.31.8.
                // The erodibility reference u_*st0 is here taken from the driver's
                // `threshold_velocity` (the local dry fluid threshold); CTSM instead
                // uses the fixed standardised value UST_ST0 = 0.16 m/s, so the two
                // differ wherever the supplied threshold is not 0.16 m/s.
                const double fclay_prime = fengsha_clay_factor(clay_val, /*ctsm_form=*/true);
                total_emissions =
                    fengsha_brittle_impaction_flux(eta, airdens(i, j, 0), u_ft, uthrs(i, j, 0), fclay_prime, fracland, c_tune, ustar_s, u_it);
            } else {
                // Horizontal saltation flux (Webb et al. 2020, Eq. 9)
                double ust_t = u_ft / rdrag_val;
                double q = airdens(i, j, 0) / grav * Kokkos::pow(ustar_s, 3.0) * (1 - Kokkos::pow(ust_t / ustar(i, j, 0), 2.0)) *
                           (1.0 + ust_t / ustar(i, j, 0));
                total_emissions = eta * alpha * fracland * Kokkos::pow(ssm_val, gamma_param) * kvh * q;
            }
            if (total_emissions <= 0.0) return;

            // Distribute the size-integrated flux across the size bins.
            for (int n = 0; n < nbins; ++n) {
                const double weight = nweights > 0 ? bin_weights(n) : 1.0 / nbins;
                emissions(i, j, n) += weight * total_emissions;
            }
        });

    Kokkos::fence();
    MarkModified("fengsha_dust_emissions", export_state);
}

}  // namespace cece
