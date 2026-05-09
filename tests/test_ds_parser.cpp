// Tests for ds_parser.cpp logic
// We test the internal helper algorithms by re-implementing them here
// (since they are static in ds_parser.cpp). This tests the logic, not the file I/O.

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <cstdint>

// --- Re-implementations of static helpers from ds_parser.cpp for testing ---

namespace test_dsp {

static std::vector<int32_t> frames_from_seconds(const std::vector<float> & durations, int sr, int hop) {
    const double timestep = (double)hop / (double)sr;
    std::vector<int32_t> frames(durations.size());
    double accum = 0.0;
    int32_t prev_boundary = 0;
    for (size_t i = 0; i < durations.size(); ++i) {
        accum += (double)durations[i];
        int32_t boundary = (int32_t)std::lround(accum / timestep + 0.5);
        frames[i] = boundary - prev_boundary;
        if (frames[i] < 0) frames[i] = 0;
        prev_boundary = boundary;
    }
    return frames;
}

static std::vector<int32_t> build_mel2x_from_frames(const std::vector<int32_t> & frames) {
    int total = 0;
    for (int32_t v : frames) total += std::max<int32_t>(v, 0);
    std::vector<int32_t> out;
    out.reserve((size_t)total);
    for (size_t i = 0; i < frames.size(); ++i) {
        const int32_t len = std::max<int32_t>(frames[i], 0);
        for (int32_t k = 0; k < len; ++k) out.push_back((int32_t)i + 1);
    }
    return out;
}

static float note_to_midi(const std::string & note) {
    if (note == "rest" || note == "REST" || note == "Rest") return -1.0f;
    static const int name_to_semitone[] = {
        9, 11, 0, 2, 4, 5, 7  // A B C D E F G
    };
    if (note.empty()) return -1.0f;
    char letter = (char)std::toupper((unsigned char)note[0]);
    if (letter < 'A' || letter > 'G') return -1.0f;
    int base = name_to_semitone[letter - 'A'];
    size_t pos = 1;
    if (pos < note.size() && note[pos] == '#') { base += 1; pos++; }
    else if (pos < note.size() && note[pos] == 'b') { base -= 1; pos++; }
    int octave = 0;
    bool negative = false;
    if (pos < note.size() && note[pos] == '-') { negative = true; pos++; }
    while (pos < note.size() && note[pos] >= '0' && note[pos] <= '9') {
        octave = octave * 10 + (note[pos] - '0');
        pos++;
    }
    if (negative) octave = -octave;
    return (float)((octave + 1) * 12 + base);
}

static std::vector<float> resample_align_curve(const std::vector<float> & points,
                                               float original_timestep,
                                               float target_timestep,
                                               int align_length) {
    if (points.empty()) return std::vector<float>(align_length, 0.0f);
    if (points.size() == 1) return std::vector<float>(align_length, points[0]);

    const float t_max = (float)(points.size() - 1) * original_timestep;
    std::vector<float> curve;
    curve.reserve(align_length);
    for (float t = 0.0f; t < t_max; t += target_timestep) {
        float idx_f = t / original_timestep;
        int idx0 = (int)idx_f;
        int idx1 = idx0 + 1;
        if (idx1 >= (int)points.size()) idx1 = (int)points.size() - 1;
        float frac = idx_f - (float)idx0;
        curve.push_back(points[idx0] * (1.0f - frac) + points[idx1] * frac);
    }
    if ((int)curve.size() < align_length) {
        float last = curve.empty() ? 0.0f : curve.back();
        curve.resize(align_length, last);
    } else if ((int)curve.size() > align_length) {
        curve.resize(align_length);
    }
    return curve;
}

} // namespace test_dsp

// --- Tests ---

TEST_CASE(note_to_midi_basic) {
    // A4 = 69
    CHECK_NEAR(test_dsp::note_to_midi("A4"), 69.0f, 0.01f);
    // C4 = 60
    CHECK_NEAR(test_dsp::note_to_midi("C4"), 60.0f, 0.01f);
    // C#4 = 61
    CHECK_NEAR(test_dsp::note_to_midi("C#4"), 61.0f, 0.01f);
    // Bb3 = 58
    CHECK_NEAR(test_dsp::note_to_midi("Bb3"), 58.0f, 0.01f);
    // rest → -1
    CHECK_NEAR(test_dsp::note_to_midi("rest"), -1.0f, 0.01f);
    CHECK_NEAR(test_dsp::note_to_midi("REST"), -1.0f, 0.01f);
    // Empty → -1
    CHECK_NEAR(test_dsp::note_to_midi(""), -1.0f, 0.01f);
}

TEST_CASE(note_to_midi_edge_cases) {
    // C-1 = 0 (MIDI note 0)
    CHECK_NEAR(test_dsp::note_to_midi("C-1"), 0.0f, 0.01f);
    // G9 = 127
    CHECK_NEAR(test_dsp::note_to_midi("G9"), 127.0f, 0.01f);
}

TEST_CASE(frames_from_seconds_basic) {
    // 3 phonemes each 0.1s at sr=44100, hop=512
    // timestep = 512/44100 ≈ 0.01161s
    std::vector<float> dur = {0.1f, 0.1f, 0.1f};
    auto frames = test_dsp::frames_from_seconds(dur, 44100, 512);
    CHECK_EQ((int)frames.size(), 3);
    // Each should be ~8-9 frames (0.1 / 0.01161 ≈ 8.6)
    int total = 0;
    for (auto f : frames) {
        CHECK(f >= 7 && f <= 10);
        total += f;
    }
    // Total should be ~26 frames for 0.3s
    CHECK(total >= 24 && total <= 28);
}

TEST_CASE(frames_from_seconds_zero_dur) {
    std::vector<float> dur = {0.0f, 0.1f};
    auto frames = test_dsp::frames_from_seconds(dur, 44100, 512);
    CHECK_EQ((int)frames.size(), 2);
    // With the +0.5 rounding bias, a 0-duration phoneme gets boundary=1, so frame=1
    CHECK(frames[0] <= 1);
    CHECK(frames[1] > 0);
}

TEST_CASE(build_mel2x_basic) {
    std::vector<int32_t> frames = {3, 2, 4};
    auto mel2x = test_dsp::build_mel2x_from_frames(frames);
    // Total length = 3+2+4 = 9
    CHECK_EQ((int)mel2x.size(), 9);
    // First 3 should be 1, next 2 should be 2, last 4 should be 3
    CHECK_EQ(mel2x[0], 1); CHECK_EQ(mel2x[1], 1); CHECK_EQ(mel2x[2], 1);
    CHECK_EQ(mel2x[3], 2); CHECK_EQ(mel2x[4], 2);
    CHECK_EQ(mel2x[5], 3); CHECK_EQ(mel2x[6], 3); CHECK_EQ(mel2x[7], 3); CHECK_EQ(mel2x[8], 3);
}

TEST_CASE(build_mel2x_empty) {
    std::vector<int32_t> frames = {};
    auto mel2x = test_dsp::build_mel2x_from_frames(frames);
    CHECK_EQ((int)mel2x.size(), 0);
}

TEST_CASE(resample_align_curve_identity) {
    // Same timestep → should return input (padded/truncated to align_length)
    std::vector<float> pts = {1.0f, 2.0f, 3.0f, 4.0f};
    auto out = test_dsp::resample_align_curve(pts, 1.0f, 1.0f, 4);
    CHECK_EQ((int)out.size(), 4);
    // First 3 should match (t=0,1,2 → idx 0,1,2); t=3 is >= t_max so not generated
    CHECK_NEAR(out[0], 1.0f, 0.01f);
    CHECK_NEAR(out[1], 2.0f, 0.01f);
    CHECK_NEAR(out[2], 3.0f, 0.01f);
    // 4th is padded with last value
    CHECK_NEAR(out[3], 3.0f, 0.01f);
}

TEST_CASE(resample_align_curve_empty) {
    std::vector<float> pts = {};
    auto out = test_dsp::resample_align_curve(pts, 1.0f, 1.0f, 5);
    CHECK_EQ((int)out.size(), 5);
    for (auto v : out) CHECK_NEAR(v, 0.0f, 0.01f);
}

TEST_CASE(resample_align_curve_single) {
    std::vector<float> pts = {42.0f};
    auto out = test_dsp::resample_align_curve(pts, 1.0f, 0.5f, 3);
    CHECK_EQ((int)out.size(), 3);
    for (auto v : out) CHECK_NEAR(v, 42.0f, 0.01f);
}
