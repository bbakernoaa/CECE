#ifndef CECE_BDSNP_HPP
#define CECE_BDSNP_HPP

#include "cece/physics_scheme.hpp"

namespace cece {

/**
 * @class BDSNPScheme
 * @brief Native C++ implementation of the BDSNP soil NO emission scheme.
 *
 * Implements the Berkeley Dalhousie Soil NO Parameterization (BDSNP)
 * for calculating soil nitric oxide emissions. The parameterization models
 * NO emissions as a function of biome type, soil temperature, soil moisture,
 * fertilizer application, and pulsing events following dry periods.
 */
class BDSNPScheme : public BasePhysicsScheme {
   public:
    BDSNPScheme() = default;
    ~BDSNPScheme() override = default;

    void Initialize(const YAML::Node& config, CeceDiagnosticManager* diag_manager) override;
    void Run(CeceImportState& import_state, CeceExportState& export_state) override;

   private:
    double fert_scale_ = 0.0068;
    double a_biome_[24] = {0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.09, 0.09, 0.01, 0.84, 0.84, 0.24,
                           0.42, 0.62, 0.03, 0.36, 0.36, 0.35, 1.66, 0.08, 0.44, 0.57, 0.57, 0.57};
};

}  // namespace cece

#endif  // CECE_BDSNP_HPP
