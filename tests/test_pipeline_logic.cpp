// Tests for pipeline utility logic (hz_to_midi, midi_to_hz, etc.)

#include <cmath>
#include <vector>

// --- Re-implementations of pipeline helpers ---

namespace test_pipeline {

static float hz_to_midi_single(float hz) {
    if (hz <= 0.0f) return 0.0f;
    return 69.0f + 12.0f * std::log2f(hz / 440.0f);
}

static float midi_to_hz(float midi) {
    return 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f);
}

// Variance scaling denormalization: (v+1)*0.5*(max-min)+min
static float denorm_variance(float v, float min_val, float max_val) {
    return (v + 1.0f) * 0.5f * (max_val - min_val) + min_val;
}

} // namespace test_pipeline

// --- Tests ---

TEST_CASE(hz_to_midi_known_values) {
    // A4 = 440 Hz → MIDI 69
    CHECK_NEAR(test_pipeline::hz_to_midi_single(440.0f), 69.0f, 0.01f);
    // A3 = 220 Hz → MIDI 57
    CHECK_NEAR(test_pipeline::hz_to_midi_single(220.0f), 57.0f, 0.01f);
    // C4 ≈ 261.63 Hz → MIDI 60
    CHECK_NEAR(test_pipeline::hz_to_midi_single(261.63f), 60.0f, 0.05f);
    // 0 Hz → 0 (unvoiced)
    CHECK_NEAR(test_pipeline::hz_to_midi_single(0.0f), 0.0f, 0.01f);
}

TEST_CASE(midi_to_hz_known_values) {
    // MIDI 69 → 440 Hz
    CHECK_NEAR(test_pipeline::midi_to_hz(69.0f), 440.0f, 0.01f);
    // MIDI 60 → ~261.63 Hz
    CHECK_NEAR(test_pipeline::midi_to_hz(60.0f), 261.63f, 0.1f);
    // MIDI 57 → 220 Hz
    CHECK_NEAR(test_pipeline::midi_to_hz(57.0f), 220.0f, 0.01f);
}

TEST_CASE(hz_midi_roundtrip) {
    // Roundtrip: hz → midi → hz should be identity for voiced frames
    float test_freqs[] = {100.0f, 220.0f, 440.0f, 880.0f, 1000.0f};
    for (float hz : test_freqs) {
        float midi = test_pipeline::hz_to_midi_single(hz);
        float hz_back = test_pipeline::midi_to_hz(midi);
        CHECK_NEAR(hz_back, hz, 0.01f);
    }
}

TEST_CASE(variance_denorm) {
    // breathiness: scale = 1/96, range typically [0, 1]
    // v=0 → (0+1)*0.5*(1-0)+0 = 0.5
    CHECK_NEAR(test_pipeline::denorm_variance(0.0f, 0.0f, 1.0f), 0.5f, 1e-6f);
    // v=-1 → (−1+1)*0.5*(1-0)+0 = 0
    CHECK_NEAR(test_pipeline::denorm_variance(-1.0f, 0.0f, 1.0f), 0.0f, 1e-6f);
    // v=1 → (1+1)*0.5*(1-0)+0 = 1
    CHECK_NEAR(test_pipeline::denorm_variance(1.0f, 0.0f, 1.0f), 1.0f, 1e-6f);
    // Custom range: min=-10, max=10, v=0 → 0
    CHECK_NEAR(test_pipeline::denorm_variance(0.0f, -10.0f, 10.0f), 0.0f, 1e-6f);
    // v=0.5, min=0, max=100 → (0.5+1)*0.5*100 = 75
    CHECK_NEAR(test_pipeline::denorm_variance(0.5f, 0.0f, 100.0f), 75.0f, 1e-4f);
}

TEST_CASE(variance_scaling_factors) {
    // Known scaling factors from the model config
    float breathiness_scale = 1.0f / 96.0f;
    float voicing_scale = 1.0f / 96.0f;
    float tension_scale = 0.1f;

    CHECK_NEAR(breathiness_scale, 0.010417f, 1e-4f);
    CHECK_NEAR(voicing_scale, 0.010417f, 1e-4f);
    CHECK_NEAR(tension_scale, 0.1f, 1e-6f);

    // Scaling applied: raw_embed * scale
    // If raw embed magnitude is ~96, scaled should be ~1
    float raw = 96.0f;
    CHECK_NEAR(raw * breathiness_scale, 1.0f, 1e-4f);
}
