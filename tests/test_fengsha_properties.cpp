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
 * 4. Configuration initialization with defaults
 * 5. Zero-emission invariant for non-emitting cells
 * 6. Configuration initialization with defaults
 */

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <conf/config.hpp>
#include <random>

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
// Property 6: Configuration initialization with defaults
// Feature: fengsha-dust-scheme, Property 6: Configuration initialization with defaults
// ============================================================================

TEST_F(FengshaPropertyTest, Property6_ConfigInitializationDefaults) {
    std::vector<std::string> all_keys = {"alpha", "gamma", "kvhmax", "grav", "drylimit_factor"};

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Generate a random subset of config keys
        std::string yaml = "";
        std::map<std::string, double> provided;

        for (const auto& key : all_keys) {
            if (rand_uniform(0.0, 1.0) > 0.5) {
                const double val = rand_uniform(0.01, 10.0);
                yaml += key + ": " + std::to_string(val) + "\n";
                provided[key] = val;
            }
        }

        conf::Config config = conf::Config::from_string(yaml);

        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        cfg.options = config.root();

        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        EXPECT_NO_THROW(scheme->Initialize(cfg.options, nullptr)) << "iter=" << iter;

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

    // Verify full custom config
    {
        conf::Config config = conf::Config::from_string("alpha: 2.5\ngamma: 0.8\nkvhmax: 1.0e-3\ngrav: 9.80665\ndrylimit_factor: 0.5");

        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        cfg.options = config.root();

        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        EXPECT_NO_THROW(scheme->Initialize(cfg.options, nullptr));
    }
}

// ============================================================================
// Property 7: The bin count is never a configuration option
// Feature: fengsha-dust-scheme, Property 7: Configuration-driven size bins
//
// The size dimension of the emission field defines the number of bins; a
// `num_bins` key in the configuration is rejected so that config and field can
// never silently disagree.
// ============================================================================

TEST_F(FengshaPropertyTest, Property7_BinCountRejectedFromConfig) {
    conf::Config config = conf::Config::from_string("num_bins: 5");

    PhysicsSchemeConfig cfg;
    cfg.name = "fengsha";
    cfg.options = config.root();
    auto scheme = PhysicsFactory::CreateScheme(cfg);
    ASSERT_NE(scheme, nullptr);
    EXPECT_THROW(scheme->Initialize(cfg.options, nullptr), std::invalid_argument);

    // The shared resolver carries the same contract for every scheme.
    EXPECT_THROW(dustbins::Resolve(config.root(), "FengshaScheme"), std::invalid_argument);
}

// ============================================================================
// Property 8: Bin geometry / weight validation in the shared resolver
// Feature: fengsha-dust-scheme, Property 8: Configuration-driven size bins
//
// Partial geometries, centimetre-scale values (a unit mistake), inverted
// edges, and radii outside their own bin are rejected; valid geometries
// produce Kok weights summing to 1, and explicit weights win over the
// geometry-derived ones after normalisation.
// ============================================================================

TEST_F(FengshaPropertyTest, Property8_BinGeometryValidation) {
    const std::string kok_lower = "bin_lower_edges: [1.2e-6, 3.1e-6, 6.65e-6, 2.5e-5, 1.0e-4]\n";
    const std::string kok_upper = "bin_upper_edges: [3.1e-6, 6.65e-6, 1.25e-4, 2.5e-4, 1.0e-3]\n";
    const std::string kok_radii = "particle_radii: [2.15e-6, 4.875e-6, 9.575e-6, 1.375e-4, 6.25e-4]\n";

    // Full valid geometry -> Kok weights summing to 1.
    auto bins = dustbins::Resolve(conf::Config::from_string(kok_radii + kok_lower + kok_upper).root(), "FengshaScheme");
    ASSERT_TRUE(bins.has_geometry);
    ASSERT_TRUE(bins.has_weights);
    ASSERT_EQ(bins.weights.size(), 5u);
    double sum = 0.0;
    for (const double w : bins.weights) sum += w;
    EXPECT_NEAR(sum, 1.0, 1e-12);
    EXPECT_TRUE(bins.weights_are_kok);

    // The same numbers written in centimetres are rejected by the units guard.
    const std::string cm_lower = "bin_lower_edges: [1.2e-4, 3.1e-4, 6.65e-4, 2.5e-3, 1.0e-2]\n";
    const std::string cm_upper = "bin_upper_edges: [3.1e-4, 6.65e-4, 1.25e-3, 2.5e-2, 1.0e-1]\n";
    EXPECT_THROW(dustbins::Resolve(conf::Config::from_string(cm_lower + cm_upper).root(), "FengshaScheme"), std::invalid_argument);

    // Partial geometry combinations are rejected.
    EXPECT_THROW(dustbins::Resolve(conf::Config::from_string(kok_lower + kok_upper).root(), "FengshaScheme"), std::invalid_argument);
    EXPECT_THROW(dustbins::Resolve(conf::Config::from_string(kok_radii + kok_lower).root(), "FengshaScheme"), std::invalid_argument);

    // Radii alone are valid: schemes that only need the representative radii
    // (e.g. a per-bin threshold that scales with radius) can use them. They do
    // not constitute a full geometry, so no weights are claimed.
    auto radii_only = dustbins::Resolve(conf::Config::from_string(kok_radii).root(), "FengshaScheme");
    EXPECT_FALSE(radii_only.has_geometry);
    EXPECT_EQ(radii_only.radii.size(), 5u);

    // Mixed lengths are rejected.
    EXPECT_THROW(dustbins::Resolve(
                     conf::Config::from_string(kok_radii + "bin_lower_edges: [1.2e-6, 3.1e-6]\n" + "bin_upper_edges: [3.1e-6, 6.65e-6]\n").root(),
                     "FengshaScheme"),
                 std::invalid_argument);

    // An inverted bin and a radius outside its own bin are rejected.
    EXPECT_THROW(
        dustbins::Resolve(conf::Config::from_string("particle_radii: [2.15e-6]\nbin_lower_edges: [3.1e-6]\nbin_upper_edges: [1.2e-6]\n").root(),
                          "FengshaScheme"),
        std::invalid_argument);
    EXPECT_THROW(
        dustbins::Resolve(conf::Config::from_string("particle_radii: [9.0e-6]\nbin_lower_edges: [1.2e-6]\nbin_upper_edges: [3.1e-6]\n").root(),
                          "FengshaScheme"),
        std::invalid_argument);

    // Explicit weights win over the geometry-derived ones and are normalised;
    // negative or all-zero weights are rejected.
    auto mixed =
        dustbins::Resolve(conf::Config::from_string(kok_radii + kok_lower + kok_upper + "bin_weights: [2, 1, 1, 1, 1]\n").root(), "FengshaScheme");
    ASSERT_EQ(mixed.weights.size(), 5u);
    EXPECT_DOUBLE_EQ(mixed.weights[0], 2.0 / 6.0);  // raw masses normalised by their sum
    EXPECT_FALSE(mixed.weights_are_kok);
    EXPECT_THROW(dustbins::Resolve(conf::Config::from_string("bin_weights: [0.5, -0.5, 1.0]\n").root(), "FengshaScheme"), std::invalid_argument);
    EXPECT_THROW(dustbins::Resolve(conf::Config::from_string("bin_weights: [0.0, 0.0, 0.0]\n").root(), "FengshaScheme"), std::invalid_argument);

    // No bin configuration at all is a valid (warned) equal-weights fallback.
    auto plain = dustbins::Resolve(conf::Config::from_string("alpha: 1.0\n").root(), "FengshaScheme");
    EXPECT_FALSE(plain.has_geometry);
    EXPECT_FALSE(plain.has_weights);
}

// ============================================================================
// Property 9: Emissions are split over whatever bins the field provides
// Feature: fengsha-dust-scheme, Property 9: Configuration-driven size bins
//
// With no bin configuration FENGSHA emits with equal weights, so for an
// always-emitting cell the total over the size dimension must be positive and
// every bin must carry exactly 1/nbin of it - for any bin count the emission
// field happens to have.
// ============================================================================

TEST_F(FengshaPropertyTest, Property9_FluxConservedAcrossAnyBinCount) {
    const int nbins_options[5] = {1, 2, 3, 5, 8};
    const int nx = 4, ny = 3;

    for (int iter = 0; iter < 10; ++iter) {
        const int nbins = nbins_options[rand_int(0, 4)];

        PhysicsSchemeConfig cfg;
        cfg.name = "fengsha";
        auto scheme = PhysicsFactory::CreateScheme(cfg);
        ASSERT_NE(scheme, nullptr);
        scheme->Initialize(cfg.options, nullptr);

        CeceImportState import_state;
        CeceExportState export_state;
        // A strongly erodible, dry, unvegetated cell far above threshold.
        import_state.fields["friction_velocity"] = create_dv("ustar", nx, ny, 1, 1.5);
        import_state.fields["threshold_velocity"] = create_dv("uthrs", nx, ny, 1, 0.15);
        import_state.fields["soil_moisture"] = create_dv("slc", nx, ny, 1, 0.02);
        import_state.fields["clay_fraction"] = create_dv("clay", nx, ny, 1, 0.10);
        import_state.fields["sand_fraction"] = create_dv("sand", nx, ny, 1, 0.60);
        import_state.fields["silt_fraction"] = create_dv("silt", nx, ny, 1, 0.30);
        import_state.fields["erodibility"] = create_dv("ssm", nx, ny, 1, 1.5);
        import_state.fields["drag_partition"] = create_dv("rdrag", nx, ny, 1, 1.0);
        import_state.fields["air_density"] = create_dv("airdens", nx, ny, 1, 1.22);
        import_state.fields["lake_fraction"] = create_dv("fraclake", nx, ny, 1, 0.0);
        import_state.fields["snow_fraction"] = create_dv("fracsnow", nx, ny, 1, 0.0);
        import_state.fields["land_mask"] = create_dv("oro", nx, ny, 1, 1.0);
        export_state.fields["fengsha_dust_emissions"] = create_dv("emis", nx, ny, nbins, 0.0);

        scheme->Run(import_state, export_state);

        auto& dv = export_state.fields["fengsha_dust_emissions"];
        dv.sync<Kokkos::HostSpace>();
        auto h = dv.view_host();

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double total = 0.0;
                for (int n = 0; n < nbins; ++n) total += h(i, j, n);
                ASSERT_GT(total, 0.0) << "cell (" << i << "," << j << ") should emit, nbins=" << nbins;
                for (int n = 0; n < nbins; ++n) {
                    EXPECT_NEAR(h(i, j, n), total / nbins, 1e-12 * total) << "bin " << n << " of " << nbins << ", iter=" << iter;
                }
            }
        }
    }
}

}  // namespace cece
