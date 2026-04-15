#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <fstream>

#include "cece/cece_physics_factory.hpp"
#include "cece/io/cece_diagnostic_manager.hpp"
#include "cece/physics/cece_megan.hpp"

class MeganSpeciationTest : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }

        // Create a dummy speciation yaml
        std::ofstream spec_file("dummy_spec.yaml");
        spec_file << "speciation:\n"
                  << "  ISOP:\n"
                  << "    - source: ISOP\n"
                  << "      factor: 1.0\n"
                  << "  OLE:\n"
                  << "    - source: MBO\n"
                  << "      factor: 0.5\n";
        spec_file.close();
    }
    void TearDown() override {
        std::remove("dummy_spec.yaml");
    }
};

TEST_F(MeganSpeciationTest, SpeciationExecution) {
    cece::MeganScheme megan;

    YAML::Node config;
    config["name"] = "megan";
    config["speciation_file"] = "dummy_spec.yaml";
    YAML::Node aefs;
    aefs["ISOP"] = 1.0e-9;
    aefs["MBO"] = 0.5e-9;
    config["aefs"] = aefs;

    cece::CeceDiagnosticManager diag;
    megan.Initialize(config, &diag);

    cece::CeceImportState import_state;
    cece::CeceExportState export_state;

    // Create required input fields
    Kokkos::View<double***> temp("temperature", 2, 2, 1);
    Kokkos::View<double***> lai("leaf_area_index", 2, 2, 1);
    Kokkos::View<double***> pardr("par_direct", 2, 2, 1);
    Kokkos::View<double***> pardf("par_diffuse", 2, 2, 1);
    Kokkos::View<double***> suncos("solar_cosine", 2, 2, 1);

    auto h_temp = Kokkos::create_mirror_view(temp);
    auto h_lai = Kokkos::create_mirror_view(lai);
    auto h_pardr = Kokkos::create_mirror_view(pardr);
    auto h_pardf = Kokkos::create_mirror_view(pardf);
    auto h_suncos = Kokkos::create_mirror_view(suncos);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            h_temp(i, j, 0) = 300.0;
            h_lai(i, j, 0) = 2.0;
            h_pardr(i, j, 0) = 200.0;
            h_pardf(i, j, 0) = 100.0;
            h_suncos(i, j, 0) = 0.5;
        }
    }

    Kokkos::deep_copy(temp, h_temp);
    Kokkos::deep_copy(lai, h_lai);
    Kokkos::deep_copy(pardr, h_pardr);
    Kokkos::deep_copy(pardf, h_pardf);
    Kokkos::deep_copy(suncos, h_suncos);

    import_state.fields["temperature"] = temp;
    import_state.fields["leaf_area_index"] = lai;
    import_state.fields["par_direct"] = pardr;
    import_state.fields["par_diffuse"] = pardf;
    import_state.fields["solar_cosine"] = suncos;

    // Create export fields
    Kokkos::View<double***> isop("MEGAN_ISOP", 2, 2, 1);
    Kokkos::View<double***> ole("MEGAN_OLE", 2, 2, 1);
    Kokkos::deep_copy(isop, 0.0);
    Kokkos::deep_copy(ole, 0.0);
    export_state.fields["MEGAN_ISOP"] = isop;
    export_state.fields["MEGAN_OLE"] = ole;

    // Run scheme
    megan.Run(import_state, export_state);

    // Verify results
    auto h_isop = Kokkos::create_mirror_view(isop);
    auto h_ole = Kokkos::create_mirror_view(ole);
    Kokkos::deep_copy(h_isop, isop);
    Kokkos::deep_copy(h_ole, ole);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_GT(h_isop(i, j, 0), 0.0);
            EXPECT_GT(h_ole(i, j, 0), 0.0);
            // ISOP aef is double MBO aef, and MBO factor is 0.5
            // so ISOP should be roughly 4x OLE in this specific dummy setup
            EXPECT_GT(h_isop(i, j, 0), h_ole(i, j, 0));
        }
    }
}
