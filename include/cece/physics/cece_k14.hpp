/**
 * @file cece_k14.hpp
 * @brief Native C++ implementation of the K14 (Kok et al., 2014) dust emission scheme.
 *
 * Implements the K14 dust emission algorithm using Kokkos for GPU-portable
 * parallel execution. Includes helper functions for Kok vertical dust flux,
 * MacKinnon drag partition, smooth roughness lookup, and Laurent erodibility.
 *
 * References:
 * - Kok, J.F., et al. (2012), An improved dust emission model, ACP, 12, 7413–7430.
 * - Kok, J.F., et al. (2014), An improved dust emission model — Part 2, ACP, 14, 13023–13041.
 * - MacKinnon, D.J., et al. (2004), A geomorphological approach, JGR, 109, F01013.
 * - Laurent, B., et al. (2008), Modeling mineral dust emissions, ACP, 8, 395–409.
 * - Fécan, F., et al. (1999), Annales Geophysicae, 17, 149–157.
 *
 * @author CECE Development Team
 * @date 2025
 * @version 1.0
 */
#ifndef CECE_K14_HPP
#define CECE_K14_HPP

#include <Kokkos_Core.hpp>

#include "cece/physics_scheme.hpp"

namespace cece {

/**
 * @class K14Scheme
 * @brief GPU-portable K14 (Kok et al., 2014) dust emission scheme using Kokkos.
 *
 * Implements the complete K14 algorithm using Kokkos for GPU-portable parallel
 * execution, including Shao threshold friction velocity, Zender gravimetric
 * moisture, Fécan moisture correction, clay parameterization (opt_clay),
 * MacKinnon drag partition, Laurent erodibility, IGBP vegetation masking,
 * and Kok (2012, 2014) vertical dust flux.
 */
class K14Scheme : public BasePhysicsScheme {
   public:
    K14Scheme() = default;
    ~K14Scheme() override = default;

    void Initialize(const conf::Value& config, CeceDiagnosticManager* diag_manager) override;
    void Run(CeceImportState& import_state, CeceExportState& export_state) override;

   private:
    double ch_du_ = 0.8e-9;
    double f_w_ = 1.0;
    double f_c_ = 1.0;
    double uts_gamma_ = 1.65e-4;
    double undef_ = 1.0e15;
    double grav_ = 9.81;
    double von_karman_ = 0.4;
    int opt_clay_ = 0;
    /// @brief Emission mass fraction of each size bin [-], normalised to sum 1.
    /// @details
    /// Resolved from configuration by `cece::dustbins::Resolve` (see
    /// cece_dust_bins.hpp). The bin count is not stored here: it is the size
    /// dimension of the emission field, and the kernel only uses these weights
    /// when their length matches it.
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> bin_weights_;

    /// @brief True when `bin_weights_` was populated from configuration.
    bool has_bin_weights_ = false;

    /// @brief Guards the one-time warning about the equal-weights fallback.
    bool warned_equal_bins_ = false;
};

}  // namespace cece

#endif  // CECE_K14_HPP
