#include "ds_parser.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace dsp {

// Helper: .ds files sometimes store numeric values as strings
static float json_float(const json & j) {
    if (j.is_number()) return j.get<float>();
    if (j.is_string()) return std::stof(j.get<std::string>());
    return 0.0f;
}
static double json_double(const json & j) {
    if (j.is_number()) return j.get<double>();
    if (j.is_string()) return std::stod(j.get<std::string>());
    return 0.0;
}

// ---------------------------------------------------------------------------
// Phoneme / speaker map loading
// ---------------------------------------------------------------------------

std::unordered_map<std::string, int> load_phoneme_map(const std::string & path,
                                                      const std::string & default_lang) {
    std::unordered_map<std::string, int> out;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[err] cannot open phoneme map: %s\n", path.c_str());
        return out;
    }
    json j;
    try { j = json::parse(f); } catch (const json::exception & e) {
        fprintf(stderr, "[err] phoneme map JSON parse: %s\n", e.what());
        return out;
    }
    for (auto & [k, v] : j.items()) {
        int id = v.get<int>();
        out[k] = id;
        // Also add lowercase variant
        std::string lower = k;
        for (char & c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower != k) out.emplace(lower, id);
        // Expand "lang/phone" → "phone" for the default language
        auto slash = k.find('/');
        if (slash != std::string::npos) {
            std::string lang_part = k.substr(0, slash);
            std::string bare = k.substr(slash + 1);
            if (lang_part == default_lang) {
                out.emplace(bare, id);
                // Also lowercase bare
                std::string bare_lower = bare;
                for (char & c : bare_lower) c = (char)std::tolower((unsigned char)c);
                if (bare_lower != bare) out.emplace(bare_lower, id);
            }
        }
    }
    return out;
}

std::unordered_map<std::string, int> load_spk_map(const std::string & path) {
    std::unordered_map<std::string, int> out;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[err] cannot open spk_map: %s\n", path.c_str());
        return out;
    }
    json j;
    try { j = json::parse(f); } catch (const json::exception & e) {
        fprintf(stderr, "[err] spk_map JSON parse: %s\n", e.what());
        return out;
    }
    for (auto & [k, v] : j.items()) {
        out[k] = v.get<int>();
    }
    return out;
}

// ---------------------------------------------------------------------------
// Helper algorithms (ported from sing_ggml.py)
// ---------------------------------------------------------------------------

static std::vector<std::string> split_string(const std::string & s) {
    std::vector<std::string> parts;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) parts.push_back(tok);
    return parts;
}

static std::vector<float> split_floats(const std::string & s) {
    std::vector<float> out;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) out.push_back(std::stof(tok));
    return out;
}

static std::vector<int> split_ints(const std::string & s) {
    std::vector<int> out;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) out.push_back(std::stoi(tok));
    return out;
}

// frames_from_seconds: convert duration array (seconds) to frame counts
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

// resample_align_curve: linear interpolation from one time grid to another
static std::vector<float> resample_align_curve(const std::vector<float> & points,
                                               float original_timestep,
                                               float target_timestep,
                                               int align_length) {
    if (points.empty()) return std::vector<float>(align_length, 0.0f);
    if (points.size() == 1) return std::vector<float>(align_length, points[0]);

    const float t_max = (float)(points.size() - 1) * original_timestep;
    // Generate output by linear interpolation
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
    // Pad/truncate to align_length
    if ((int)curve.size() < align_length) {
        float last = curve.empty() ? 0.0f : curve.back();
        curve.resize(align_length, last);
    } else if ((int)curve.size() > align_length) {
        curve.resize(align_length);
    }
    return curve;
}

// note_to_midi: parse note string like "C4", "D#3", "rest" → MIDI number
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

// hz_to_midi with unvoiced interpolation
static std::vector<float> hz_to_midi(const std::vector<float> & f0_hz) {
    const int T = (int)f0_hz.size();
    std::vector<float> midi(T, 0.0f);
    // Find voiced indices
    std::vector<int> voiced_idx;
    std::vector<float> voiced_log2;
    for (int i = 0; i < T; ++i) {
        if (f0_hz[i] > 0.0f) {
            voiced_idx.push_back(i);
            voiced_log2.push_back(std::log2f(f0_hz[i]));
        }
    }
    if (voiced_idx.empty()) return midi;

    // Interpolate log2(hz) for unvoiced frames
    std::vector<float> log2_hz(T, 0.0f);
    for (size_t k = 0; k < voiced_idx.size(); ++k) {
        log2_hz[voiced_idx[k]] = voiced_log2[k];
    }
    // Linear interpolation between voiced points for unvoiced frames
    for (int i = 0; i < T; ++i) {
        if (f0_hz[i] > 0.0f) continue;
        // Find surrounding voiced frames
        int left = -1, right = -1;
        // Binary search left
        auto it = std::lower_bound(voiced_idx.begin(), voiced_idx.end(), i);
        if (it != voiced_idx.begin()) {
            left = *std::prev(it);
        }
        if (it != voiced_idx.end()) {
            right = *it;
        }
        if (left < 0 && right < 0) continue;
        if (left < 0) log2_hz[i] = log2_hz[right];
        else if (right < 0) log2_hz[i] = log2_hz[left];
        else {
            float frac = (float)(i - left) / (float)(right - left);
            log2_hz[i] = log2_hz[left] * (1.0f - frac) + log2_hz[right] * frac;
        }
    }
    // Convert to MIDI
    for (int i = 0; i < T; ++i) {
        float hz = std::pow(2.0f, log2_hz[i]);
        if (hz > 1e-6f) {
            midi[i] = 69.0f + 12.0f * std::log2f(std::max(hz, 1e-6f) / 440.0f);
        }
    }
    return midi;
}

// ---------------------------------------------------------------------------
// Main .ds parser
// ---------------------------------------------------------------------------

static int encode_phone(const std::unordered_map<std::string, int> & map, const std::string & phone) {
    auto it = map.find(phone);
    if (it != map.end()) return it->second;
    // Try lowercase
    std::string lower = phone;
    for (char & c : lower) c = (char)std::tolower((unsigned char)c);
    it = map.find(lower);
    if (it != map.end()) return it->second;
    return -1;
}

std::vector<DSSegment> parse_ds_file(const std::string & ds_path,
                                     const std::unordered_map<std::string, int> & phoneme_map,
                                     int sr, int hop) {
    std::ifstream f(ds_path);
    if (!f) {
        fprintf(stderr, "[err] cannot open .ds file: %s\n", ds_path.c_str());
        return {};
    }
    json j;
    try { j = json::parse(f); } catch (const json::exception & e) {
        fprintf(stderr, "[err] .ds JSON parse: %s\n", e.what());
        return {};
    }
    if (!j.is_array()) {
        fprintf(stderr, "[err] .ds file is not a segment array\n");
        return {};
    }

    const float timestep = (float)hop / (float)sr;
    std::vector<DSSegment> segments;

    for (size_t seg_idx = 0; seg_idx < j.size(); ++seg_idx) {
        const auto & seg = j[seg_idx];
        DSSegment out;

        // offset
        if (seg.contains("offset")) out.offset = json_double(seg["offset"]);

        // ---- phonemes & durations ----
        if (!seg.contains("ph_seq") || !seg.contains("ph_dur")) {
            fprintf(stderr, "[err] segment %zu missing ph_seq/ph_dur\n", seg_idx);
            return {};
        }
        auto phones = split_string(seg["ph_seq"].get<std::string>());
        auto ph_dur_s = split_floats(seg["ph_dur"].get<std::string>());
        if (phones.size() != ph_dur_s.size()) {
            fprintf(stderr, "[err] segment %zu: phone/dur size mismatch (%zu vs %zu)\n",
                    seg_idx, phones.size(), ph_dur_s.size());
            return {};
        }
        const int L = (int)phones.size();

        // Store raw phoneme names (for re-encoding with alternate phoneme maps)
        out.ph_names = phones;

        // Encode phonemes
        out.tokens.resize(L);
        for (int i = 0; i < L; ++i) {
            int id = encode_phone(phoneme_map, phones[i]);
            if (id < 0) {
                fprintf(stderr, "[err] segment %zu: unknown phoneme '%s'\n", seg_idx, phones[i].c_str());
                return {};
            }
            out.tokens[i] = id;
        }

        // Compute frame durations
        auto ph_frames = frames_from_seconds(ph_dur_s, sr, hop);
        int T = 0;
        for (int32_t fr : ph_frames) T += fr;

        // Store per-phone frame durations
        out.ph_dur.resize(L);
        for (int i = 0; i < L; ++i) out.ph_dur[i] = (float)ph_frames[i];

        // mel2ph
        out.mel2ph.reserve(T);
        for (int i = 0; i < L; ++i) {
            for (int32_t k = 0; k < ph_frames[i]; ++k) {
                out.mel2ph.push_back(i + 1);  // 1-based
            }
        }

        // ---- ph2word & word_dur ----
        if (!seg.contains("ph_num")) {
            fprintf(stderr, "[err] segment %zu missing ph_num\n", seg_idx);
            return {};
        }
        auto ph_num = split_ints(seg["ph_num"].get<std::string>());
        int ph_num_sum = 0;
        for (int n : ph_num) ph_num_sum += n;
        if (ph_num_sum != L) {
            fprintf(stderr, "[err] segment %zu: ph_num sum %d != phones %d\n", seg_idx, ph_num_sum, L);
            return {};
        }
        const int W = (int)ph_num.size();
        out.ph2word.resize(L);
        int ph_idx = 0;
        for (int w = 0; w < W; ++w) {
            for (int k = 0; k < ph_num[w]; ++k) {
                out.ph2word[ph_idx++] = w + 1;  // 1-based
            }
        }
        // word_dur: sum ph_frames per word
        out.word_dur.assign(W, 0.0f);
        ph_idx = 0;
        for (int w = 0; w < W; ++w) {
            for (int k = 0; k < ph_num[w]; ++k) {
                out.word_dur[w] += (float)ph_frames[ph_idx++];
            }
        }

        // ---- Note-level info (always parse if present, for pitch model) ----
        if (seg.contains("note_seq")) {
            auto notes = split_string(seg["note_seq"].get<std::string>());
            const int N = (int)notes.size();

            out.note_midi.resize(N);
            out.note_rest.resize(N);
            for (int n = 0; n < N; ++n) {
                float midi = note_to_midi(notes[n]);
                out.note_midi[n] = midi;
                out.note_rest[n] = (midi < 0.0f) ? 1 : 0;
            }

            // note_dur: duration per note (in seconds)
            if (seg.contains("note_dur") && !seg["note_dur"].is_null()) {
                auto note_dur_s = split_floats(seg["note_dur"].get<std::string>());
                if ((int)note_dur_s.size() != N) {
                    fprintf(stderr, "[err] segment %zu: note_dur size %zu != note_seq %d\n",
                            seg_idx, note_dur_s.size(), N);
                    return {};
                }
                auto note_frames = frames_from_seconds(note_dur_s, sr, hop);
                out.note_dur_frames.resize(N);
                for (int n = 0; n < N; ++n) out.note_dur_frames[n] = note_frames[n];

                // mel2note: expand note durations to frame-level index [T]
                out.mel2note.resize(T, 0);
                int frame_pos = 0;
                for (int n = 0; n < N && frame_pos < T; ++n) {
                    for (int32_t fr = 0; fr < note_frames[n] && frame_pos < T; ++fr) {
                        out.mel2note[frame_pos++] = n + 1;  // 1-based
                    }
                }
                // Fill remaining with last note
                for (; frame_pos < T; ++frame_pos) {
                    out.mel2note[frame_pos] = N;
                }
            } else if (N == W) {
                // Fallback: notes = words, use word durations
                out.note_dur_frames.resize(N);
                for (int n = 0; n < N; ++n) {
                    out.note_dur_frames[n] = (int32_t)out.word_dur[n];
                }
                out.mel2note.resize(T);
                int frame_pos = 0;
                ph_idx = 0;
                for (int w = 0; w < W; ++w) {
                    for (int k = 0; k < ph_num[w]; ++k) {
                        for (int32_t fr = 0; fr < ph_frames[ph_idx]; ++fr) {
                            out.mel2note[frame_pos++] = w + 1;
                        }
                        ph_idx++;
                    }
                }
            }

            // base_pitch: expand note_midi to frame level via mel2note
            out.base_pitch.resize(T);
            for (int i = 0; i < T; ++i) {
                int ni = out.mel2note[i] - 1;  // 0-based
                if (ni >= 0 && ni < N && out.note_midi[ni] > 0.0f) {
                    out.base_pitch[i] = out.note_midi[ni];
                } else {
                    out.base_pitch[i] = 0.0f;
                }
            }
        }

        // ---- F0 ----
        if (seg.contains("f0_seq") && !seg["f0_seq"].is_null()) {
            auto f0_src = split_floats(seg["f0_seq"].get<std::string>());
            float f0_step = 0.005f;
            if (seg.contains("f0_timestep") && !seg["f0_timestep"].is_null())
                f0_step = json_float(seg["f0_timestep"]);
            out.f0_hz = resample_align_curve(f0_src, f0_step, timestep, T);
        } else if (!out.base_pitch.empty()) {
            // Synthesize f0 from note sequence (staircase, no expression)
            out.f0_hz.resize(T);
            for (int i = 0; i < T; ++i) {
                if (out.base_pitch[i] > 0.0f) {
                    out.f0_hz[i] = 440.0f * std::pow(2.0f, (out.base_pitch[i] - 69.0f) / 12.0f);
                } else {
                    out.f0_hz[i] = 0.0f;
                }
            }
        } else {
            fprintf(stderr, "[err] segment %zu: no f0_seq or note_seq\n", seg_idx);
            return {};
        }

        // ---- pitch_midi (from f0_hz) ----
        out.pitch_midi = hz_to_midi(out.f0_hz);

        // ---- languages (all zeros) ----
        out.languages.assign(L, 0);

        // ---- Variance curves ----
        auto resample_var = [&](const char * name, const char * ts_name) -> std::vector<float> {
            if (!seg.contains(name) || seg[name].is_null()) return {};
            if (!seg.contains(ts_name) || seg[ts_name].is_null()) return {};
            auto vals = split_floats(seg[name].get<std::string>());
            float step = json_float(seg[ts_name]);
            return resample_align_curve(vals, step, timestep, T);
        };

        out.breathiness = resample_var("breathiness", "breathiness_timestep");
        out.voicing = resample_var("voicing", "voicing_timestep");
        out.tension = resample_var("tension", "tension_timestep");
        out.energy = resample_var("energy", "energy_timestep");

        // Gender → key_shift
        if (seg.contains("gender") && !seg["gender"].is_null()) {
            const float shift_min = -5.0f, shift_max = 5.0f;
            if (seg["gender"].is_number()) {
                float gv = json_float(seg["gender"]);
                float ks = gv * (gv >= 0.0f ? shift_max : std::abs(shift_min));
                out.gender.assign(T, std::clamp(ks, shift_min, shift_max));
            } else {
                auto vals = split_floats(seg["gender"].get<std::string>());
                float step = seg.contains("gender_timestep") ? json_float(seg["gender_timestep"]) : timestep;
                auto seq = resample_align_curve(vals, step, timestep, T);
                out.gender.resize(T);
                for (int i = 0; i < T; ++i) {
                    float s = seq[i];
                    float ks = s * (s >= 0.0f ? shift_max : std::abs(shift_min));
                    out.gender[i] = std::clamp(ks, shift_min, shift_max);
                }
            }
        }

        // Velocity → speed
        if (seg.contains("velocity") && !seg["velocity"].is_null()) {
            if (seg.contains("velocity_timestep") && !seg["velocity_timestep"].is_null()) {
                auto vals = split_floats(seg["velocity"].get<std::string>());
                float step = json_float(seg["velocity_timestep"]);
                auto seq = resample_align_curve(vals, step, timestep, T);
                out.velocity.resize(T);
                for (int i = 0; i < T; ++i)
                    out.velocity[i] = std::clamp(seq[i], 0.5f, 2.0f);
            }
        }

        segments.push_back(std::move(out));
    }

    return segments;
}

} // namespace dsp
