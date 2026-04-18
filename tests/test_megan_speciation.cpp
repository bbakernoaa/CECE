#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

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

    cece::DualView3D temp("temperature", 2, 2, 1);
    cece::DualView3D lai("leaf_area_index", 2, 2, 1);
    cece::DualView3D pardr("par_direct", 2, 2, 1);
    cece::DualView3D pardf("par_diffuse", 2, 2, 1);
    cece::DualView3D suncos("solar_cosine", 2, 2, 1);

    auto h_temp = temp.view_host();
    auto h_lai = lai.view_host();
    auto h_pardr = pardr.view_host();
    auto h_pardf = pardf.view_host();
    auto h_suncos = suncos.view_host();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            h_temp(i, j, 0) = 300.0;
            h_lai(i, j, 0) = 2.0;
            h_pardr(i, j, 0) = 200.0;
            h_pardf(i, j, 0) = 100.0;
            h_suncos(i, j, 0) = 0.5;
        }
    }

    temp.modify<Kokkos::HostSpace>();
    lai.modify<Kokkos::HostSpace>();
    pardr.modify<Kokkos::HostSpace>();
    pardf.modify<Kokkos::HostSpace>();
    suncos.modify<Kokkos::HostSpace>();

    temp.sync<Kokkos::DefaultExecutionSpace>();
    lai.sync<Kokkos::DefaultExecutionSpace>();
    pardr.sync<Kokkos::DefaultExecutionSpace>();
    pardf.sync<Kokkos::DefaultExecutionSpace>();
    suncos.sync<Kokkos::DefaultExecutionSpace>();

    import_state.fields["temperature"] = temp;
    import_state.fields["leaf_area_index"] = lai;
    import_state.fields["par_direct"] = pardr;
    import_state.fields["par_diffuse"] = pardf;
    import_state.fields["solar_cosine"] = suncos;

    cece::DualView3D isop("MEGAN_ISOP", 2, 2, 1);
    cece::DualView3D ole("MEGAN_OLE", 2, 2, 1);
    Kokkos::deep_copy(isop.view_device(), 0.0);
    Kokkos::deep_copy(ole.view_device(), 0.0);
    isop.modify<Kokkos::DefaultExecutionSpace>();
    ole.modify<Kokkos::DefaultExecutionSpace>();

    export_state.fields["MEGAN_ISOP"] = isop;
    export_state.fields["MEGAN_OLE"] = ole;

    megan.Run(import_state, export_state);

    isop.sync<Kokkos::HostSpace>();
    ole.sync<Kokkos::HostSpace>();

    auto h_isop = isop.view_host();
    auto h_ole = ole.view_host();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_GT(h_isop(i, j, 0), 0.0);
            EXPECT_GT(h_ole(i, j, 0), 0.0);
            EXPECT_GT(h_isop(i, j, 0), h_ole(i, j, 0));
        }
    }
}
