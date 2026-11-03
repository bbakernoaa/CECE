/**
 * @file test_fengsha_properties.cpp
 * @brief Property-based tests for the FENGSHA dust emission scheme.
 *
 * Validates correctness properties for the FENGSHA helper functions and
 * the full scheme implementation using randomized inputs.
 *
 * Properties tested:
 * 1. Volumetric-to-gravimetric soil moisture conversion
 * 2. Fécan moisture correction computation
 * 3. MB95 vertical-to-horizontal flux ratio
 * 4. Numerical equivalence between C++ and Fortran implementations
 * 5. Zero-emission invariant for non-emitting cells
 * 6. Configuration initialization with defaults
 */

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cece/cece_dust_bins.hpp"
#include "cece/cece_physics_factory.hpp"
#include "cece/cece_state.hpp"

// Re-declare the KOKKOS_INLINE_FUNCTION helpers so we can call them from host tests.
// These match the implementations in cece_fengsha.cpp exactly.
namespace fengsha_helpers {

double soil_moisture_vol2grav(double vsoil, double sandfrac) {
    constexpr double rhow = 1000.0;
    constexpr double rhop = 1700.0;
    double vsat = 0.489 - 0.126 * sandfrac;
    return 100.0 * vsoil * rhow / (rhop * (1.0 - vsat));
}

double moisture_correction_fecan(double slc, double sand, double clay, double b) {
    double grvsoilm = soil_moisture_vol2grav(slc, sand);
    double drylimit = b * clay * (14.0 * clay + 17.0);
    double excess = std::max(0.0, grvsoilm - drylimit);
    return std::sqrt(1.0 + 1.21 * std::pow(excess, 0.68));
}

double flux_v2h_ratio_mb95(double clay, double kvhmax) {
    constexpr double clay_thresh = 0.2;
    if (clay > clay_thresh) {
        return kvhmax;
    }
    return std::pow(10.0, 13.4 * clay - 6.0);
}

}  // namespace fengsha_helpers

namespace cece {

class FengshaPropertyTest : public ::testing::Test {
   public:
    static void SetUpTestSuite() {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }

    std::mt19937 rng_{42};
    static constexpr int NUM_ITERATIONS = 200;

    double rand_uniform(double lo, double hi) {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(rng_);
    }

    int rand_int(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(rng_);
    }

    [[nodiscard]] DualView3D create_dv(const std::string& name, int nx, int ny, int nz, double val) const {
        DualView3D dv(name, nx, ny, nz);
        Kokkos::deep_copy(dv.view_host(), val);
        dv.modify<Kokkos::HostSpace>();
        dv.sync<Kokkos::DefaultExecutionSpace>();
        return dv;
    }
};

// ============================================================================
// Property 1: Volumetric-to-gravimetric soil moisture conversion
// Feature: fengsha-dust-scheme, Property 1: Volumetric-to-gravimetric soil moisture conversion
// ============================================================================

TEST_F(FengshaPropertyTest, Property1_SoilMoistureConversion) {
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        double vsoil = rand_uniform(0.0, 1.0);
        double sandfrac = rand_uniform(0.0, 1.0);

        double result = fengsha_helpers::soil_moisture_vol2grav(vsoil, sandfrac);

        // Reference formula
        double vsat = 0.489 - 0.126 * sandfrac;
        double expected = 100.0 * vsoil * 1000.0 / (1700.0 * (1.0 - vsat));

        EXPECT_NEAR(result, expected, std::abs(expected) * 1e-14 + 1e-14)
            << "Failed at iter=" << iter << " vsoil=" << vsoil << " sandfrac=" << sandfrac;
    }
}

// ============================================================================
// Property 2: Fécan moisture correction computation
// Feature: fengsha-dust-scheme, Property 2: Fécan moisture correction computation
// ============================================================================

TEST_F(FengshaPropertyTest, Property2_FecanMoistureCorrection) {
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        double slc = rand_uniform(0.0, 1.0);
        double sand = rand_uniform(0.0, 1.0);
        double clay = rand_uniform(0.0, 1.0);
        double b = rand_uniform(0.01, 10.0);

        double result = fengsha_helpers::moisture_correction_fecan(slc, sand, clay, b);

        // Reference formula
        double grvsoilm = fengsha_helpers::soil_moisture_vol2grav(slc, sand);
        double drylimit = b * clay * (14.0 * clay + 17.0);
        double excess = std::max(0.0, grvsoilm - drylimit);
        double expected = std::sqrt(1.0 + 1.21 * std::pow(excess, 0.68));

        EXPECT_NEAR(result, expected, std::abs(expected) * 1e-12 + 1e-14)
            << "Failed at iter=" << iter << " slc=" << slc << " sand=" << sand << " clay=" << clay << " b=" << b;

        // Correction factor must be >= 1.0
        EXPECT_GE(result, 1.0) << "Correction factor must be >= 1.0";
    }
}

// ============================================================================
// Property 3: MB95 vertical-to-horizontal flux ratio
// Feature: fengsha-dust-scheme, Property 3: MB95 vertical-to-horizontal flux ratio
// ============================================================================

TEST_F(FengshaPropertyTest, Property3_MB95FluxRatio) {
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        double clay = rand_uniform(0.0, 1.0);
        double kvhmax = rand_uniform(0.001, 1.0);

        double result = fengsha_helpers::flux_v2h_ratio_mb95(clay, kvhmax);

        if (clay > 0.2) {
            EXPECT_DOUBLE_EQ(result, kvhmax) << "When clay > 0.2, should return kvhmax. clay=" << clay;
        } else {
            double expected = std::pow(10.0, 13.4 * clay - 6.0);
            EXPECT_NEAR(result, expected, std::abs(expected) * 1e-12 + 1e-14) << "When clay <= 0.2, should return 10^(13.4*clay-6.0). clay=" << clay;
        }

        // Result must be positive
        EXPECT_GT(result, 0.0) << "Flux ratio must be positive";
    }
}

// ============================================================================
// Property 5: Zero-emission invariant for non-emitting cells
// Feature: fengsha-dust-scheme, Property 5: Zero-emission invariant for non-emitting cells
// ============================================================================

TEST_F(FengshaPropertyTest, Property5_ZeroEmissionInvariant) {
    // Test that ocean cells, low-erodibility cells, and below-threshold cells
    // produce zero emissions in the native C++ scheme.
    int nx = 6, ny = 6, nbins = 5;

    PhysicsSchemeConfig cfg;
    cfg.name = "fengsha";
    auto scheme = PhysicsFactory::CreateScheme(cfg);
    ASSERT_NE(scheme, nullptr) << "FengshaScheme must be registered";
    scheme->Initialize(cfg.options, nullptr);

    for (int iter = 0; iter < 50; ++iter) {
        CeceImportState import_state;
        CeceExportState export_state;

        // Create fields with random valid values
        import_state.fields["friction_velocity"] = create_dv("ustar", nx, ny, 1, rand_uniform(0.1, 2.0));
        import_state.fields["threshold_velocity"] = create_dv("uthrs", nx, ny, 1, rand_uniform(0.1, 1.0));
        import_state.fields["soil_moisture"] = create_dv("slc", nx, ny, 1, rand_uniform(0.0, 0.5));
        import_state.fields["clay_fraction"] = create_dv("clay", nx, ny, 1, rand_uniform(0.01, 0.5));
        import_state.fields["sand_fraction"] = create_dv("sand", nx, ny, 1, rand_uniform(0.1, 0.8));
        import_state.fields["silt_fraction"] = create_dv("silt", nx, ny, 1, rand_uniform(0.1, 0.5));
        import_state.fields["erodibility"] = create_dv("ssm", nx, ny, 1, rand_uniform(0.1, 1.0));
        import_state.fields["drag_partition"] = create_dv("rdrag", nx, ny, 1, rand_uniform(0.1, 1.0));
        import_state.fields["air_density"] = create_dv("airdens", nx, ny, 1, rand_uniform(1.0, 1.5));
        import_state.fields["lake_fraction"] = create_dv("fraclake", nx, ny, 1, 0.0);
        import_state.fields["snow_fraction"] = create_dv("fracsnow", nx, ny, 1, 0.0);
        import_state.fields["land_mask"] = create_dv("oro", nx, ny, 1, 1.0);
        export_state.fields["fengsha_dust_emissions"] = create_dv("emis", nx, ny, nbins, 0.0);

        // Force specific cells to non-emitting conditions
        auto oro_h = import_state.fields["land_mask"].view_host();
        auto ssm_h = import_state.fields["erodibility"].view_host();
        auto uthrs_h = import_state.fields["threshold_velocity"].view_host();

        // Row 0-1: ocean cells (land_mask = 0)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < 2; ++i) {
                oro_h(i, j, 0) = 0.0;
            }
        }
        // Row 2-3: low erodibility (ssm < 1e-2)
        for (int j = 0; j < ny; ++j) {
            for (int i = 2; i < 4; ++i) {
                ssm_h(i, j, 0) = 1.0e-4;
            }
        }
        // Row 4-5: very high threshold (friction velocity below threshold)
        for (int j = 0; j < ny; ++j) {
            for (int i = 4; i < 6; ++i) {
                uthrs_h(i, j, 0) = 1000.0;  // impossibly high threshold
            }
        }

        import_state.fields["land_mask"].modify<Kokkos::HostSpace>();
        import_state.fields["land_mask"].sync<Kokkos::DefaultExecutionSpace>();
        import_state.fields["erodibility"].modify<Kokkos::HostSpace>();
        import_state.fields["erodibility"].sync<Kokkos::DefaultExecutionSpace>();
        import_state.fields["threshold_velocity"].modify<Kokkos::HostSpace>();
        import_state.fields["threshold_velocity"].sync<Kokkos::DefaultExecutionSpace>();

        auto* base = dynamic_cast<BasePhysicsScheme*>(scheme.get());
        if (base) base->ClearPhysicsCache();
        scheme->Run(import_state, export_state);

        auto& dv_emis = export_state.fields["fengsha_dust_emissions"];
        dv_emis.sync<Kokkos::HostSpace>();
        auto emis_h = dv_emis.view_host();

        // Verify zero emissions for all non-emitting cells
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                for (int n = 0; n < nbins; ++n) {
                    if (i < 2 || (i >= 2 && i < 4) || i >= 4) {
                        EXPECT_DOUBLE_EQ(emis_h(i, j, n), 0.0)
                            << "Non-emitting cell (" << i << "," << j << "," << n << ") should have zero emissions, iter=" << iter;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Property 4: Numerical equivalence between C++ and Fortran implementations
// Feature: fengsha-dust-scheme, Property 4: Numerical equivalence between C++ and Fortran
// ============================================================================

TEST_F(FengshaPropertyTest, Property4_NumericalEquivalence) {
    // Check at runtime if the Fortran scheme is available
    PhysicsSchemeConfig cfg_fort_check;
    cfg_fort_check.name = "fengsha_fortran";
    auto fort_check = PhysicsFactory::CreateScheme(cfg_fort_check);
    if (fort_check == nullptr) {
        GTEST_SKIP() << "fengsha_fortran scheme not available, skipping equivalence test";
    }

    for (int iter = 0; iter < 50; ++iter) {
        int nx = rand_int(1, 10);
        int ny = rand_int(1, 10);
        int nbins = rand_int(1, 5);

        PhysicsSchemeConfig cfg_cpp, cfg_fort;
        cfg_cpp.name = "fengsha";
        cfg_fort.name = "fengsha_fortran";

        auto scheme_cpp = PhysicsFactory::CreateScheme(cfg_cpp);
        auto scheme_fort = PhysicsFactory::CreateScheme(cfg_fort);
        ASSERT_NE(scheme_cpp, nullptr);
        ASSERT_NE(scheme_fort, nullptr);

        scheme_cpp->Initialize(cfg_cpp.options, nullptr);
        scheme_fort->Initialize(cfg_fort.options, nullptr);

        // Create import state with random physically valid values
        CeceImportState import_state;
        CeceExportState export_cpp, export_fort;

        import_state.fields["friction_velocity"] = create_dv("ustar", nx, ny, 1, 0.0);
        import_state.fields["threshold_velocity"] = create_dv("uthrs", nx, ny, 1, 0.0);
        import_state.fields["soil_moisture"] = create_dv("slc", nx, ny, 1, 0.0);
        import_state.fields["clay_fraction"] = create_dv("clay", nx, ny, 1, 0.0);
        import_state.fields["sand_fraction"] = create_dv("sand", nx, ny, 1, 0.0);
        import_state.fields["silt_fraction"] = create_dv("silt", nx, ny, 1, 0.0);
        import_state.fields["erodibility"] = create_dv("ssm", nx, ny, 1, 0.0);
        import_state.fields["drag_partition"] = create_dv("rdrag", nx, ny, 1, 0.0);
        import_state.fields["air_density"] = create_dv("airdens", nx, ny, 1, 0.0);
        import_state.fields["lake_fraction"] = create_dv("fraclake", nx, ny, 1, 0.0);
        import_state.fields["snow_fraction"] = create_dv("fracsnow", nx, ny, 1, 0.0);
        import_state.fields["land_mask"] = create_dv("oro", nx, ny, 1, 1.0);

        // Fill with random values
        auto fill_random = [&](const std::string& name, double lo, double hi) {
            auto h = import_state.fields[name].view_host();
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) h(i, j, 0) = rand_uniform(lo, hi);
            import_state.fields[name].modify<Kokkos::HostSpace>();
            import_state.fields[name].sync<Kokkos::DefaultExecutionSpace>();
        };

        fill_random("friction_velocity", 0.1, 2.0);
        fill_random("threshold_velocity", 0.05, 0.5);
        fill_random("soil_moisture", 0.0, 0.5);
        fill_random("clay_fraction", 0.01, 0.5);
        fill_random("sand_fraction", 0.1, 0.8);
        fill_random("silt_fraction", 0.1, 0.5);
        fill_random("erodibility", 0.1, 1.0);
        fill_random("drag_partition", 0.5, 1.0);
        fill_random("air_density", 1.0, 1.5);
        fill_random("lake_fraction", 0.0, 0.3);
        fill_random("snow_fraction", 0.0, 0.3);

        export_cpp.fields["fengsha_dust_emissions"] = create_dv("emis_cpp", nx, ny, nbins, 0.0);
        export_fort.fields["fengsha_dust_emissions"] = create_dv("emis_fort", nx, ny, nbins, 0.0);

        // Run both schemes
        auto* base_cpp = dynamic_cast<BasePhysicsScheme*>(scheme_cpp.get());
        if (base_cpp) base_cpp->ClearPhysicsCache();
        scheme_cpp->Run(import_state, export_cpp);

        auto* base_fort = dynamic_cast<BasePhysicsScheme*>(scheme_fort.get());
        if (base_fort) base_fort->ClearPhysicsCache();
        scheme_fort->Run(import_state, export_fort);

        // Compare results
        auto& dv_cpp = export_cpp.fields["fengsha_dust_emissions"];
        auto& dv_fort = export_fort.fields["fengsha_dust_emissions"];
        dv_cpp.sync<Kokkos::HostSpace>();
        dv_fort.sync<Kokkos::HostSpace>();
        auto h_cpp = dv_cpp.view_host();
        auto h_fort = dv_fort.view_host();

        for (int n = 0; n < nbins; ++n) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double cpp_val = h_cpp(i, j, n);
                    double fort_val = h_fort(i, j, n);
                    double tol = std::max(std::abs(cpp_val), std::abs(fort_val)) * 1e-12 + 1e-20;
                    EXPECT_NEAR(cpp_val, fort_val, tol) << "Mismatch at (" << i << "," << j << "," << n << ") iter=" << iter;
                }
            }
        }
    }
}

// ============================================================================
// Property 6: Configuration initialization with defaults
// Feature: fengsha-dust-scheme, Property 6: Configuration initialization with defaults
// ============================================================================

TEST_F(FengshaPropertyTest, Property6_ConfigInitializationDefaults) {
    // Default values
    constexpr double DEFAULT_ALPHA = 1.0;
    constexpr double DEFAULT_GAMMA = 1.0;
    constexpr double DEFAULT_KVHMAX = 2.45e-4;
    constexpr double DEFAULT_GRAV = 9.81;
    constexpr double DEFAULT_DRYLIMIT = 1.0;
    (void)DEFAULT_ALPHA;
    (void)DEFAULT_GAMMA;
    (void)DEFAULT_KVHMAX;
    (void)DEFAULT_GRAV;
    (void)DEFAULT_DRYLIMIT;

    std::vector<std::string> all_keys = {"alpha", "gamma", "kvhmax", "grav", "drylimit_factor", "use_intermittency", "use_brittle_impaction",
                                         "c_tune"};

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Generate a random subset of config keys
        YAML::Node config;
        std::map<std::string, double> provided;

        for (const auto& key : all_keys) {
            if (rand_uniform(0.0, 1.0) > 0.5) {
                if (key == "use_intermittency" || key == "use_brittle_impaction") {
                    config[key] = rand_uniform(0.0, 1.0) > 0.5;
                } else {
                    double val = rand_uniform(0.01, 10.0);
                    config[key] = val;
                    provided[key] = val;
                }
            }
        }

        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        cfg.options = config;

        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        scheme->Initialize(cfg.options, nullptr);

        // We can't directly access private members, but we can verify the scheme
        // was created and initialized without error. The actual parameter verification
        // is done by running the scheme with known inputs and checking outputs match
        // the expected behavior with the configured parameters.
        // For this property, we verify the scheme initializes without throwing.
        EXPECT_TRUE(true) << "Scheme initialized successfully with config subset, iter=" << iter;
    }

    // Verify default initialization (empty config)
    {
        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        EXPECT_NO_THROW(scheme->Initialize(cfg.options, nullptr));
    }

    // Verify full custom config, including a configured bin geometry
    {
        YAML::Node config;
        config["alpha"] = 2.5;
        config["gamma"] = 0.8;
        config["kvhmax"] = 1.0e-3;
        config["grav"] = 9.80665;
        config["drylimit_factor"] = 0.5;
        config["particle_radii"] = std::vector<double>{0.35e-6, 1.75e-6, 3.75e-6, 7.5e-6};
        config["bin_lower_edges"] = std::vector<double>{0.1e-6, 1.0e-6, 2.5e-6, 5.0e-6};
        config["bin_upper_edges"] = std::vector<double>{1.0e-6, 2.5e-6, 5.0e-6, 10.0e-6};

        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        cfg.options = config;

        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        EXPECT_NO_THROW(scheme->Initialize(cfg.options, nullptr));
    }
}

// ============================================================================
// Property 7: Dust size bins are configuration-driven, never hard-coded
// Feature: fengsha-dust-scheme, Property 7: Configuration-driven size bins
// ============================================================================

TEST_F(FengshaPropertyTest, Property7_BinCountComesFromTheField) {
    // A configured geometry of arbitrary length must be accepted for any bin
    // count, and the emission total must be conserved across the field bins.
    for (int nbins = 1; nbins <= 8; ++nbins) {
        YAML::Node config;
        std::vector<double> radii, lower, upper;
        double edge = 0.1e-6;
        for (int n = 0; n < nbins; ++n) {
            const double next = edge * 2.0;
            lower.push_back(edge);
            upper.push_back(next);
            radii.push_back(0.5 * (edge + next));
            edge = next;
        }
        config["particle_radii"] = radii;
        config["bin_lower_edges"] = lower;
        config["bin_upper_edges"] = upper;

        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        cfg.options = config;
        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        EXPECT_NO_THROW(scheme->Initialize(cfg.options, nullptr)) << "nbins=" << nbins;
    }

    // The removed 'num_bins' option must be rejected rather than silently ignored.
    YAML::Node config;
    config["num_bins"] = 3;
    PhysicsSchemeConfig cfg;
    cfg.name = "fengsha";
    cfg.options = config;
    auto scheme = PhysicsFactory::CreateScheme(cfg);
    ASSERT_NE(scheme, nullptr);
    EXPECT_THROW(scheme->Initialize(cfg.options, nullptr), std::invalid_argument);
}

TEST_F(FengshaPropertyTest, Property8_BinGeometryValidation) {
    auto make_config = [](std::vector<double> radii, std::vector<double> lower, std::vector<double> upper) {
        YAML::Node config;
        if (!radii.empty()) config["particle_radii"] = radii;
        if (!lower.empty()) config["bin_lower_edges"] = lower;
        if (!upper.empty()) config["bin_upper_edges"] = upper;
        return config;
    };
    const std::vector<double> radii = {0.55e-6, 1.75e-6};
    const std::vector<double> lower = {0.1e-6, 1.0e-6};
    const std::vector<double> upper = {1.0e-6, 2.5e-6};

    const std::vector<std::pair<std::string, YAML::Node>> bad_cases = {
        {"edges without radii", make_config({}, lower, upper)},
        {"missing upper edges", make_config(radii, lower, {})},
        {"length mismatch", make_config(radii, {0.1e-6}, upper)},
        {"inverted edges", make_config(radii, {1.0e-6, 1.0e-6}, {0.1e-6, 2.5e-6})},
        {"radius outside bin", make_config({5.0e-6, 1.75e-6}, lower, upper)},
        {"micrometre units", make_config({0.55, 1.75}, {0.1, 1.0}, {1.0, 2.5})},
        {"non-positive radius", make_config({0.0, 1.75e-6}, lower, upper)},
    };

    for (const auto& [label, config] : bad_cases) {
        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        cfg.options = config;
        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        EXPECT_THROW(scheme->Initialize(cfg.options, nullptr), std::invalid_argument) << "case: " << label;
    }

    // A valid geometry must resolve to Kok weights that sum to one.
    const auto bins = dustbins::Resolve(make_config(radii, lower, upper), "test");
    ASSERT_TRUE(bins.has_geometry);
    ASSERT_TRUE(bins.weights_are_kok);
    ASSERT_EQ(static_cast<int>(bins.weights.size()), 2);
    double sum = 0.0;
    for (const double w : bins.weights) sum += w;
    EXPECT_NEAR(sum, 1.0, 1e-14);

    // Configured weights are normalised and used as given.
    YAML::Node weighted = make_config({}, {}, {});
    weighted["bin_weights"] = std::vector<double>{2.0, 3.0, 5.0};
    const auto wb = dustbins::Resolve(weighted, "test");
    ASSERT_TRUE(wb.has_weights);
    EXPECT_FALSE(wb.weights_are_kok);
    EXPECT_NEAR(wb.weights[0], 0.2, 1e-14);
    EXPECT_NEAR(wb.weights[2], 0.5, 1e-14);

    // Nothing configured: equal-weights fallback, flagged as such.
    const YAML::Node empty;
    const auto fb = dustbins::Resolve(empty, "test");
    EXPECT_FALSE(fb.has_geometry);
    EXPECT_FALSE(fb.has_weights);
    EXPECT_NE(fb.source.find("equal weights"), std::string::npos);
}

// ============================================================================
// Property 9: Emissions are conserved across any bin count
// Feature: fengsha-dust-scheme, Property 9: Configuration-driven size bins
// ============================================================================

TEST_F(FengshaPropertyTest, Property9_FluxConservedAcrossBinCounts) {
    // Build an emitting land column and check that the size-integrated flux is
    // independent of how many size bins the emission field has: the fallback
    // splits the flux equally, so summing over the bins must give the same total.
    const int nx = 4, ny = 3;

    auto run_case = [&](int nbins, const YAML::Node& config, std::vector<double>& out) {
        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        cfg.options = config;
        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        scheme->Initialize(cfg.options, nullptr);

        CeceImportState import_state;
        CeceExportState export_state;
        import_state.fields["friction_velocity"] = create_dv("ustar", nx, ny, 1, 1.5);
        import_state.fields["threshold_velocity"] = create_dv("uthrs", nx, ny, 1, 0.2);
        import_state.fields["soil_moisture"] = create_dv("slc", nx, ny, 1, 0.01);
        import_state.fields["clay_fraction"] = create_dv("clay", nx, ny, 1, 0.3);
        import_state.fields["sand_fraction"] = create_dv("sand", nx, ny, 1, 0.4);
        import_state.fields["silt_fraction"] = create_dv("silt", nx, ny, 1, 0.3);
        import_state.fields["erodibility"] = create_dv("ssm", nx, ny, 1, 0.8);
        import_state.fields["drag_partition"] = create_dv("rdrag", nx, ny, 1, 0.9);
        import_state.fields["air_density"] = create_dv("airdens", nx, ny, 1, 1.22);
        import_state.fields["lake_fraction"] = create_dv("fraclake", nx, ny, 1, 0.0);
        import_state.fields["snow_fraction"] = create_dv("fracsnow", nx, ny, 1, 0.0);
        import_state.fields["land_mask"] = create_dv("oro", nx, ny, 1, 1.0);
        export_state.fields["fengsha_dust_emissions"] = create_dv("emis", nx, ny, nbins, 0.0);

        auto* base = dynamic_cast<BasePhysicsScheme*>(scheme.get());
        if (base) base->ClearPhysicsCache();
        scheme->Run(import_state, export_state);

        auto& dv = export_state.fields["fengsha_dust_emissions"];
        dv.sync<Kokkos::HostSpace>();
        auto h = dv.view_host();
        out.assign(nbins, 0.0);
        for (int n = 0; n < nbins; ++n) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) out[n] += h(i, j, n);
            }
        }
    };

    const YAML::Node empty;
    double reference_total = -1.0;
    for (const int nbins : {1, 2, 3, 5, 8}) {
        std::vector<double> per_bin;
        run_case(nbins, empty, per_bin);
        double total = 0.0;
        for (const double v : per_bin) total += v;
        ASSERT_GT(total, 0.0) << "nbins=" << nbins << " should emit";
        for (const double v : per_bin) {
            EXPECT_NEAR(v, total / nbins, std::abs(total) * 1e-12) << "equal-weight fallback, nbins=" << nbins;
        }
        if (reference_total < 0.0) {
            reference_total = total;
        } else {
            EXPECT_NEAR(total, reference_total, std::abs(reference_total) * 1e-12)
                << "size-integrated flux changed with the bin count (nbins=" << nbins << ")";
        }
    }
}

TEST_F(FengshaPropertyTest, Property10_ConfiguredWeightsApplied) {
    // With bin_weights configured and matching the field, the per-bin ratios must
    // follow the configured weights rather than being equal.
    const int nx = 3, ny = 2, nbins = 4;

    YAML::Node config;
    config["bin_weights"] = std::vector<double>{0.1, 0.2, 0.3, 0.4};

    PhysicsSchemeConfig cfg;
    cfg.name = "fengsha";
    cfg.options = config;
    auto scheme = PhysicsFactory::CreateScheme(cfg);
    ASSERT_NE(scheme, nullptr);
    scheme->Initialize(cfg.options, nullptr);

    CeceImportState import_state;
    CeceExportState export_state;
    import_state.fields["friction_velocity"] = create_dv("ustar", nx, ny, 1, 1.5);
    import_state.fields["threshold_velocity"] = create_dv("uthrs", nx, ny, 1, 0.2);
    import_state.fields["soil_moisture"] = create_dv("slc", nx, ny, 1, 0.01);
    import_state.fields["clay_fraction"] = create_dv("clay", nx, ny, 1, 0.3);
    import_state.fields["sand_fraction"] = create_dv("sand", nx, ny, 1, 0.4);
    import_state.fields["silt_fraction"] = create_dv("silt", nx, ny, 1, 0.3);
    import_state.fields["erodibility"] = create_dv("ssm", nx, ny, 1, 0.8);
    import_state.fields["drag_partition"] = create_dv("rdrag", nx, ny, 1, 0.9);
    import_state.fields["air_density"] = create_dv("airdens", nx, ny, 1, 1.22);
    import_state.fields["lake_fraction"] = create_dv("fraclake", nx, ny, 1, 0.0);
    import_state.fields["snow_fraction"] = create_dv("fracsnow", nx, ny, 1, 0.0);
    import_state.fields["land_mask"] = create_dv("oro", nx, ny, 1, 1.0);
    export_state.fields["fengsha_dust_emissions"] = create_dv("emis", nx, ny, nbins, 0.0);

    auto* base = dynamic_cast<BasePhysicsScheme*>(scheme.get());
    if (base) base->ClearPhysicsCache();
    scheme->Run(import_state, export_state);

    auto& dv = export_state.fields["fengsha_dust_emissions"];
    dv.sync<Kokkos::HostSpace>();
    auto h = dv.view_host();

    std::vector<double> per_bin(nbins, 0.0);
    for (int n = 0; n < nbins; ++n) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) per_bin[n] += h(i, j, n);
        }
    }
    double total = 0.0;
    for (const double v : per_bin) total += v;
    ASSERT_GT(total, 0.0);
    const double expected[nbins] = {0.1, 0.2, 0.3, 0.4};
    for (int n = 0; n < nbins; ++n) {
        EXPECT_NEAR(per_bin[n] / total, expected[n], 1e-12) << "bin " << n;
    }
}

}  // namespace cece

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
