/**
 * @file cece_fengsha.hpp
 * @brief Native C++ implementation of the FENGSHA dust emission scheme.
 * @author CECE Development Team
 * @date 2025
 * @version 1.0
 */
#ifndef CECE_FENGSHA_HPP
#define CECE_FENGSHA_HPP

#include <Kokkos_Core.hpp>

#include "cece/physics_scheme.hpp"

namespace cece {

/**
 * @class FengshaScheme
 * @brief Native C++ implementation of the FENGSHA dust emission scheme.
 *
 * Implements the FENGSHA algorithm using Kokkos for GPU-portable parallel
 * execution, including Fécan moisture correction, MB95 flux ratio, drag
 * partitioning, emission intermittency (Comola et al. 2019), and per-bin
 * emission distribution. The total flux is either the brittle-fragmentation
 * flux of Kok et al. (2014a) as in CTSM's Leung_2023 scheme
 * (`use_brittle_impaction: true`) or the original FENGSHA saltation-driven
 * flux (Webb et al. 2020).
 */
class FengshaScheme : public BasePhysicsScheme {
   public:
    FengshaScheme() = default;
    ~FengshaScheme() override = default;

    void Initialize(const conf::Value& config, CeceDiagnosticManager* diag_manager) override;
    void Run(CeceImportState& import_state, CeceExportState& export_state) override;

   private:
    double alpha_ = 1.0;
    double gamma_ = 1.0;
    double kvhmax_ = 2.45e-4;
    double grav_ = 9.81;
    double drylimit_factor_ = 1.0;

    /// @brief Tuning constant C_tune of the brittle-impaction flux (CTSM Eq. 2.31.8).
    double c_tune_ = 0.05;

    /// @brief Emission intermittency factor (Comola et al. 2019; CTSM Eq. 2.31.27).
    bool use_intermittency_ = true;

    /// @brief Select the total dust flux: true = Kok et al. (2014a) brittle
    /// fragmentation (CTSM Eq. 2.31.8); false = Marticorena & Bergametti (1995)
    /// k_v/h ratio with the Webb et al. (2020) horizontal saltation flux.
    bool use_brittle_impaction_ = false;

    /// @brief Emission mass fraction of each size bin [-], normalised to sum 1.
    /// @details
    /// Resolved from configuration by `cece::dustbins::Resolve` (see
    /// cece_dust_bins.hpp) and sized from the `bin_weights` / `particle_radii`
    /// entries, never from a hard-coded bin count. The kernel writes bin `n` only
    /// for `n` below both this length and the size dimension of the emission
    /// field, and warns when the two disagree.
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> bin_weights_;

    /// @brief True when `bin_weights_` came from the configuration rather than
    /// from the equal-weights fallback derived at run time from the field bins.
    bool has_bin_weights_ = false;

    /// @brief Guards the one-time warning about the equal-weights fallback.
    bool warned_equal_bins_ = false;
};

}  // namespace cece

#endif  // CECE_FENGSHA_HPP
