#include <gtest/gtest.h>

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
    void TearDown() override {
        // Kokkos::finalize() is typically handled by main
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

    // Create required input fields
    Kokkos::View<double***> soilt("soil_temperature", 2, 2, 1);
    Kokkos::View<double***> soilm("soil_moisture", 2, 2, 1);

    // Create optional input fields
    Kokkos::View<double***> landtype("landtype", 2, 2, 1);

    auto h_soilt = Kokkos::create_mirror_view(soilt);
    auto h_soilm = Kokkos::create_mirror_view(soilm);
    auto h_landtype = Kokkos::create_mirror_view(landtype);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            h_soilt(i, j, 0) = 293.15;  // 20 C
            h_soilm(i, j, 0) = 0.25;    // 25% WFPS
            h_landtype(i, j, 0) = 10;   // Some active biome
        }
    }

    Kokkos::deep_copy(soilt, h_soilt);
    Kokkos::deep_copy(soilm, h_soilm);
    Kokkos::deep_copy(landtype, h_landtype);

    import_state.fields["soil_temperature"] = soilt;
    import_state.fields["soil_moisture"] = soilm;
    import_state.fields["landtype"] = landtype;

    // Create export field
    Kokkos::View<double***> bdsnp_no("bdsnp_no", 2, 2, 1);
    Kokkos::deep_copy(bdsnp_no, 0.0);
    export_state.fields["bdsnp_no"] = bdsnp_no;

    // Run scheme
    bdsnp.Run(import_state, export_state);

    // Verify results
    auto h_bdsnp_no = Kokkos::create_mirror_view(bdsnp_no);
    Kokkos::deep_copy(h_bdsnp_no, bdsnp_no);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_GT(h_bdsnp_no(i, j, 0), 0.0);
        }
    }
}
