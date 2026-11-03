/**
 * @file cece_k14_fortran.hpp
 * @brief Fortran bridge header for the K14 (Kok et al., 2014) dust emission scheme.
 */
#ifndef CECE_K14_FORTRAN_HPP
#define CECE_K14_FORTRAN_HPP

#include <vector>

#include "cece/physics_scheme.hpp"

namespace cece {

/**
 * @class K14FortranScheme
 * @brief Fortran bridge implementation of the K14 dust emission scheme.
 *
 * Calls the K14 Fortran kernel (k14_kernel_mod) via extern "C" linkage,
 * syncing DualViews between host and device as needed.
 */
class K14FortranScheme : public BasePhysicsScheme {
   public:
    K14FortranScheme() = default;
    ~K14FortranScheme() override = default;

    void Initialize(const YAML::Node& config, CeceDiagnosticManager* diag_manager) override;
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
    /// cece_dust_bins.hpp); sized from the configured geometry, never from a
    /// hard-coded bin count. The bridge truncates/validates against the field's
    /// size dimension before calling the Fortran kernel.
    std::vector<double> bin_weights_;

    /// @brief True when `bin_weights_` was populated from configuration.
    bool has_bin_weights_ = false;
};

}  // namespace cece

#endif  // CECE_K14_FORTRAN_HPP
