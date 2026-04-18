#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <Kokkos_Core.hpp>

#include "cece/cece_physics_factory.hpp"
#include "cece/io/cece_diagnostic_manager.hpp"
#include "cece/physics/cece_bdsnp.hpp"

class BDSNPTest : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }
};

TEST_F(BDSNPTest, BasicExecution) {
    cece::BDSNPScheme bdsnp;

    YAML::Node config;
    config["name"] = "bdsnp";
    config["fert_scale"] = 0.0068;

    cece::CeceDiagnosticManager diag;
    bdsnp.Initialize(config, &diag);

    cece::CeceImportState import_state;
    cece::CeceExportState export_state;

    cece::DualView3D soilt("soil_temperature", 2, 2, 1);
    cece::DualView3D soilm("soil_moisture", 2, 2, 1);
    cece::DualView3D landtype("landtype", 2, 2, 1);

    auto h_soilt = soilt.view_host();
    auto h_soilm = soilm.view_host();
    auto h_landtype = landtype.view_host();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            h_soilt(i, j, 0) = 293.15;
            h_soilm(i, j, 0) = 0.25;
            h_landtype(i, j, 0) = 10;
        }
    }

    soilt.modify<Kokkos::HostSpace>();
    soilm.modify<Kokkos::HostSpace>();
    landtype.modify<Kokkos::HostSpace>();

    soilt.sync<Kokkos::DefaultExecutionSpace>();
    soilm.sync<Kokkos::DefaultExecutionSpace>();
    landtype.sync<Kokkos::DefaultExecutionSpace>();

    import_state.fields["soil_temperature"] = soilt;
    import_state.fields["soil_moisture"] = soilm;
    import_state.fields["landtype"] = landtype;

    cece::DualView3D bdsnp_no("bdsnp_no", 2, 2, 1);
    Kokkos::deep_copy(bdsnp_no.view_device(), 0.0);
    bdsnp_no.modify<Kokkos::DefaultExecutionSpace>();
    export_state.fields["bdsnp_no"] = bdsnp_no;

    bdsnp.Run(import_state, export_state);

    bdsnp_no.sync<Kokkos::HostSpace>();
    auto h_bdsnp_no = bdsnp_no.view_host();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_GT(h_bdsnp_no(i, j, 0), 0.0);
        }
    }
}
