/**
 * @file test_stream_key_properties.cpp
 * @brief Property-based tests for CeceDriverOrchestrator::StreamKey.
 *
 * Feature: regrid-per-stream, Property 1: StreamKey concatenation correctness
 *
 * StreamKey(cfg) returns exactly cfg.input_file_path + "|" + cfg.mapalgo. It is
 * the composite Stream_Identity_Key used to re-key the regrid_plans_,
 * amio_handles_, and file_nt_cache_ caches so that variables belonging to the
 * same stream (same input file + mapping algorithm) share one plan, one open
 * handle set, and one record count.
 *
 * Properties tested:
 *   - Concatenation: StreamKey(cfg) == cfg.input_file_path + "|" + cfg.mapalgo.
 *   - Determinism / equality: two configs with the same input_file_path and
 *     mapalgo produce the same key regardless of any other field.
 *   - Injectivity on (path, algo): two configs differing in input_file_path
 *     or mapalgo produce different keys.
 *
 * **Validates: Requirements 1.1, 1.2, 1.3, 8.1, 9.4**
 */

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <string>

#include "cece/cece_driver_facade.hpp"

namespace cece {

// ============================================================================
// Test-only friend shim.
//
// StreamKey is a private static member of CeceDriverOrchestrator. This struct
// is declared a friend inside the class (see
// include/cece/cece_driver_facade.hpp, Task 6.1) so the tests below can invoke
// the private static helper without altering its signature, logic, or
// visibility. It does not touch any production code path.
// ============================================================================
struct StreamKeyTestAccess {
    static std::string Key(const StreamConfig& cfg) { return CeceDriverOrchestrator::StreamKey(cfg); }
};

namespace {

// Generate an arbitrary StreamConfig. Every field is populated with arbitrary
// (but constrained-to-sane) values so the tests exercise the key derivation
// against noise in the non-key fields. input_file_path and mapalgo are drawn
// from a broad space (including empty strings and characters such as '|' and
// path separators) to stress the concatenation formula.
rc::Gen<StreamConfig> genStreamConfig() {
    return rc::gen::apply(
        [](std::string input_file_path, std::string input_var_name, std::string mapalgo, std::string cadence, std::string tintalgo,
           std::string data_model, bool data_model_explicit, int amio_worker_threads, int amio_staging_buffer_count) {
            StreamConfig cfg;
            cfg.input_file_path = std::move(input_file_path);
            cfg.input_var_name = std::move(input_var_name);
            cfg.mapalgo = std::move(mapalgo);
            cfg.cadence = std::move(cadence);
            cfg.tintalgo = std::move(tintalgo);
            cfg.data_model = std::move(data_model);
            cfg.data_model_explicit = data_model_explicit;
            cfg.amio_worker_threads = amio_worker_threads;
            cfg.amio_staging_buffer_count = amio_staging_buffer_count;
            return cfg;
        },
        rc::gen::arbitrary<std::string>(), rc::gen::arbitrary<std::string>(), rc::gen::arbitrary<std::string>(),
        rc::gen::arbitrary<std::string>(), rc::gen::arbitrary<std::string>(), rc::gen::arbitrary<std::string>(),
        rc::gen::arbitrary<bool>(), rc::gen::inRange(1, 65), rc::gen::inRange(1, 65));
}

}  // namespace

// ============================================================================
// Property 1a: Concatenation correctness
// Feature: regrid-per-stream, Property 1: StreamKey concatenation correctness
// **Validates: Requirements 1.1, 9.4**
//
// For any StreamConfig, StreamKey(cfg) equals exactly
// cfg.input_file_path + "|" + cfg.mapalgo.
// ============================================================================
RC_GTEST_PROP(StreamKeyProperty, Property1_Concatenation, ()) {
    const StreamConfig cfg = *genStreamConfig();
    const std::string expected = cfg.input_file_path + "|" + cfg.mapalgo;
    RC_ASSERT(StreamKeyTestAccess::Key(cfg) == expected);
}

// ============================================================================
// Property 1b: Determinism / equality when path + algo match
// Feature: regrid-per-stream, Property 1: StreamKey concatenation correctness
// **Validates: Requirements 1.2, 8.1**
//
// Two configs sharing input_file_path and mapalgo produce the same key,
// regardless of any other field.
// ============================================================================
RC_GTEST_PROP(StreamKeyProperty, Property1_EqualWhenPathAndAlgoMatch, ()) {
    const StreamConfig a = *genStreamConfig();

    // b copies a's key-relevant fields but varies everything else arbitrarily.
    StreamConfig b = *genStreamConfig();
    b.input_file_path = a.input_file_path;
    b.mapalgo = a.mapalgo;

    RC_ASSERT(StreamKeyTestAccess::Key(a) == StreamKeyTestAccess::Key(b));
}

// ============================================================================
// Property 1c: Determinism on identical configs
// Feature: regrid-per-stream, Property 1: StreamKey concatenation correctness
// **Validates: Requirements 8.1**
//
// StreamKey is a pure function: the same config produces the same key on
// repeated evaluation (models the every-rank-agrees invariant).
// ============================================================================
RC_GTEST_PROP(StreamKeyProperty, Property1_DeterministicOnIdenticalConfig, ()) {
    const StreamConfig a = *genStreamConfig();
    RC_ASSERT(StreamKeyTestAccess::Key(a) == StreamKeyTestAccess::Key(a));
}

// ============================================================================
// Property 1d: Inequality when path or algo differs
// Feature: regrid-per-stream, Property 1: StreamKey concatenation correctness
// **Validates: Requirements 1.3, 9.4**
//
// Two configs differing in input_file_path or mapalgo produce different keys.
// Because "|" cannot appear in either field here (the generator draws from a
// broad space, so we guard with RC_PRE that neither key field contains "|"),
// the concatenation is unambiguous and the mapping (path, algo) -> key is
// injective.
// ============================================================================
RC_GTEST_PROP(StreamKeyProperty, Property1_UnequalWhenPathOrAlgoDiffers, ()) {
    StreamConfig a = *genStreamConfig();
    StreamConfig b = *genStreamConfig();

    // The separator ban only needs to hold to keep concatenation unambiguous.
    // Discard cases where a key field contains the reserved separator.
    RC_PRE(a.input_file_path.find('|') == std::string::npos);
    RC_PRE(a.mapalgo.find('|') == std::string::npos);
    RC_PRE(b.input_file_path.find('|') == std::string::npos);
    RC_PRE(b.mapalgo.find('|') == std::string::npos);

    // Only meaningful when the (path, algo) pairs actually differ.
    RC_PRE(a.input_file_path != b.input_file_path || a.mapalgo != b.mapalgo);

    RC_ASSERT(StreamKeyTestAccess::Key(a) != StreamKeyTestAccess::Key(b));
}

}  // namespace cece
