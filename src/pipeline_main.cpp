#include "diffsinger.h"
#include "ds_parser.h"
#include "pitch.h"
#include "runtime_backend.h"
#include "variance.h"
#include "vocoder.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static inline void ds_setenv(const char * key, const char * val) {
#if defined(_WIN32)
    _putenv_s(key, val);
#else
    setenv(key, val, 1);
#endif
}

using SteadyClock = std::chrono::steady_clock;

static double elapsed_s(SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

struct SegmentTiming {
    bool pitch_run = false;
    bool variance_run = false;
    bool stitch_run = false;
    bool write_run = false;
    double pitch_s = 0.0;
    double variance_s = 0.0;
    double acoustic_s = 0.0;
    double vocoder_s = 0.0;
    double stitch_s = 0.0;
    double write_s = 0.0;
    double total_s = 0.0;
};

static void print_segment_timing(size_t index, size_t total, const SegmentTiming & t) {
    fprintf(stderr, "[time] segment %zu/%zu total=%.3fs", index, total, t.total_s);
    if (t.pitch_run) fprintf(stderr, " pitch=%.3fs", t.pitch_s);
    else             fprintf(stderr, " pitch=skip");
    if (t.variance_run) fprintf(stderr, " variance=%.3fs", t.variance_s);
    else                fprintf(stderr, " variance=skip");
    fprintf(stderr, " acoustic=%.3fs vocoder=%.3fs", t.acoustic_s, t.vocoder_s);
    if (t.stitch_run) fprintf(stderr, " stitch=%.3fs", t.stitch_s);
    if (t.write_run)  fprintf(stderr, " write=%.3fs", t.write_s);
    fprintf(stderr, "\n");
}

struct Segment {
    std::string tokens;
    std::string ph2word;
    std::string word_dur;
    std::string mel2ph;
    std::string f0_hz;
    std::string pitch_midi;
    std::string lang;
    std::string variance_dir;
    std::string wav_out;
};

static void usage() {
    fprintf(stderr,
        "usage: diffsinger_pipeline --variance-model <variance.gguf>\n"
        "                           --acoustic-model <acoustic.gguf>\n"
        "                           --vocoder-model <vocoder.gguf>\n"
        "                           --ds <file.ds> --phonemes <phonemes.json>\n"
        "                           --out <output.wav>\n"
        "                           [--pitch-model <pitch.gguf>] [--auto-duration] [--auto-pitch]\n"
        "                           [--pitch-phonemes <pitch_phonemes.json>]\n"
        "                           [--spk-id N | --spk-name NAME --spk-map MAP]\n"
        "                           [--seed N] [--steps N] [--algorithm euler|midpoint|rk4]\n"
        "                           [--precision f32|f16] [--mel-min X] [--mel-max X]\n"
        "                           [--noise-scale X] [--backend cpu|gpu|auto|cuda[:N]|vulkan[:N]]\n"
        "                           [--threads N|auto]\n"
        "                           [--variance-backend cpu|gpu|auto|cuda[:N]|vulkan[:N]]\n"
        "                           [--acoustic-backend cpu|gpu|auto|cuda[:N]|vulkan[:N]]\n"
        "                           [--vocoder-backend cpu|gpu|auto|cuda[:N]|vulkan[:N]]\n"
        "                           [--pitch-backend cpu|gpu|auto|cuda[:N]|vulkan[:N]]\n"
        "                           [--predict-all-variances]\n"
        "\n"
        "Legacy TSV mode:\n"
        "  diffsinger_pipeline ... --manifest <segments.tsv>\n"
        "\n"
        "manifest columns: tokens ph2word word_dur mel2ph f0_hz pitch_midi lang variance_dir wav_out\n");
}

static bool exists_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return (bool)f;
}

template<typename T>
static bool read_all(const std::string & path, std::vector<T> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "[err] open failed: %s\n", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    if (sz < 0 || (sz % (std::streamsize)sizeof(T)) != 0) {
        fprintf(stderr, "[err] bad binary size: %s\n", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize((size_t)sz / sizeof(T));
    if (!out.empty() && !f.read(reinterpret_cast<char *>(out.data()), sz)) {
        fprintf(stderr, "[err] read failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

static bool write_f32(const std::string & path, const std::vector<float> & data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[err] write failed: %s\n", path.c_str());
        return false;
    }
    if (!data.empty()) {
        f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)data.size() * 4);
    }
    return (bool)f;
}

static std::vector<Segment> read_manifest(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[err] open manifest failed: %s\n", path.c_str());
        return {};
    }
    std::vector<Segment> out;
    std::string line;
    int line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, '\t')) cols.push_back(col);
        if (cols.size() != 9) {
            fprintf(stderr, "[err] manifest line %d: expected 9 columns, got %zu\n", line_no, cols.size());
            return {};
        }
        out.push_back({cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8]});
    }
    return out;
}

static bool read_variance_fs2(const Segment & s, int spk_id, dsv::FS2ConditionInputs & in) {
    in.spk_id = spk_id;
    if (!read_all<int32_t>(s.tokens, in.tokens)) return false;
    if (!read_all<int32_t>(s.ph2word, in.ph2word)) return false;
    if (!read_all<float>(s.word_dur, in.word_dur)) return false;
    if (!read_all<int32_t>(s.mel2ph, in.mel2ph)) return false;
    if (exists_file(s.lang) && !read_all<int32_t>(s.lang, in.languages)) return false;
    in.phones = (int)in.tokens.size();
    in.frames = (int)in.mel2ph.size();
    return true;
}

static bool write_missing_variances(const dsv::VarianceSampleOutputs & out,
                                    const std::vector<bool> & missing,
                                    const std::string & variance_dir) {
    for (size_t i = 0; i < out.targets.size(); ++i) {
        if (i >= missing.size() || !missing[i]) continue;
        const std::string path = variance_dir + "/" + out.targets[i].name + ".bin";
        if (!write_f32(path, out.values[i])) return false;
    }
    return true;
}

static bool ensure_variance_curves(const dsv::Model & model,
                                   const Segment & s,
                                   int spk_id,
                                   uint32_t seed,
                                   bool predict_all_variances) {
    std::vector<bool> missing;
    bool any_missing = false;
    for (const auto & target : model.cfg.variance_targets) {
        const bool miss = predict_all_variances || !exists_file(s.variance_dir + "/" + target.name + ".bin");
        missing.push_back(miss);
        any_missing = any_missing || miss;
    }
    if (!any_missing) return true;

    dsv::FS2ConditionInputs fs2_in;
    if (!read_variance_fs2(s, spk_id, fs2_in)) return false;

    std::vector<float> pitch;
    if (!read_all<float>(s.pitch_midi, pitch)) return false;
    if ((int)pitch.size() != fs2_in.frames) {
        fprintf(stderr, "[err] pitch size mismatch for %s: got %zu expected %d\n",
                s.pitch_midi.c_str(), pitch.size(), fs2_in.frames);
        return false;
    }

    dsv::ConditionOutputs fs2_out;
    if (!dsv::run_fs2_condition(model, fs2_in, fs2_out)) return false;

    dsv::VarianceConditionInputs vc_in;
    vc_in.frames = fs2_in.frames;
    vc_in.fs2_condition = std::move(fs2_out.condition);
    vc_in.pitch = std::move(pitch);
    dsv::ConditionOutputs vc_out;
    if (!dsv::compose_variance_condition(model, vc_in, vc_out)) return false;

    dsv::VarianceSampleInputs sample_in;
    sample_in.frames = fs2_in.frames;
    sample_in.seed = seed;
    sample_in.condition = std::move(vc_out.condition);
    dsv::VarianceSampleOutputs sample_out;
    sample_in.existing_values.assign(model.cfg.variance_targets.size(), std::vector<float>((size_t)fs2_in.frames, 0.0f));
    sample_in.retake_masks.assign(model.cfg.variance_targets.size(), std::vector<int32_t>((size_t)fs2_in.frames, 1));
    for (size_t i = 0; i < model.cfg.variance_targets.size(); ++i) {
        if (i >= missing.size() || missing[i]) continue;
        const std::string path = s.variance_dir + "/" + model.cfg.variance_targets[i].name + ".bin";
        if (!read_all<float>(path, sample_in.existing_values[i])) return false;
        if ((int)sample_in.existing_values[i].size() != fs2_in.frames) {
            fprintf(stderr, "[err] variance curve size mismatch: %s\n", path.c_str());
            return false;
        }
        std::fill(sample_in.retake_masks[i].begin(), sample_in.retake_masks[i].end(), 0);
    }
    if (!dsv::sample_variances(model, sample_in, sample_out)) return false;
    return write_missing_variances(sample_out, missing, s.variance_dir);
}

static bool read_optional_curve(const std::string & path, int frames, std::vector<float> & dst) {
    if (!exists_file(path)) return true;
    if (!read_all<float>(path, dst)) return false;
    if ((int)dst.size() != frames) {
        fprintf(stderr, "[err] curve size mismatch: %s got %zu expected %d\n", path.c_str(), dst.size(), frames);
        return false;
    }
    return true;
}

static bool run_segment(const ds::Model & acoustic,
                        const nsv::Model & vocoder,
                        const Segment & s,
                        int spk_id,
                        uint32_t seed,
                        int steps,
                        bool has_mel_min,
                        float mel_min,
                        bool has_mel_max,
                        float mel_max,
                        float noise_scale,
                        SegmentTiming * timing = nullptr) {
    ds::InferenceInputs ac_in;
    if (!read_all<int32_t>(s.tokens, ac_in.tokens)) return false;
    if (!read_all<int32_t>(s.mel2ph, ac_in.mel2ph)) return false;
    if (!read_all<float>(s.f0_hz, ac_in.f0)) return false;
    if (exists_file(s.lang) && !read_all<int32_t>(s.lang, ac_in.languages)) return false;
    const int frames = (int)ac_in.mel2ph.size();
    if ((int)ac_in.f0.size() != frames) {
        fprintf(stderr, "[err] f0 size mismatch: got %zu expected %d\n", ac_in.f0.size(), frames);
        return false;
    }

    if (acoustic.has_lang_embed && ac_in.languages.empty()) ac_in.languages.assign(ac_in.tokens.size(), 0);
    if (acoustic.has_variance) {
        ac_in.breathiness.assign(frames, 0.0f);
        ac_in.voicing.assign(frames, 1.0f);
        ac_in.tension.assign(frames, 0.0f);
        ac_in.energy.assign(frames, 0.0f);
        if (!read_optional_curve(s.variance_dir + "/breathiness.bin", frames, ac_in.breathiness)) return false;
        if (!read_optional_curve(s.variance_dir + "/voicing.bin", frames, ac_in.voicing)) return false;
        if (!read_optional_curve(s.variance_dir + "/tension.bin", frames, ac_in.tension)) return false;
        if (!read_optional_curve(s.variance_dir + "/energy.bin", frames, ac_in.energy)) return false;
    }
    if (acoustic.has_key_shift) {
        ac_in.gender.assign(frames, 0.0f);
        if (!read_optional_curve(s.variance_dir + "/gender.bin", frames, ac_in.gender)) return false;
    }
    if (acoustic.has_speed) {
        ac_in.velocity.assign(frames, 1.0f);
        if (!read_optional_curve(s.variance_dir + "/velocity.bin", frames, ac_in.velocity)) return false;
    }
    ac_in.spk_id = spk_id;
    ac_in.seed = seed;
    ac_in.sampling_steps_override = steps;
    if (const char * alg = std::getenv("DSGGML_ALGORITHM"))
        ac_in.algorithm_override = alg;

    ds::InferenceOutputs mel;
    {
        auto t0 = SteadyClock::now();
        if (!ds::run_inference(acoustic, ac_in, mel)) return false;
        if (timing) timing->acoustic_s += elapsed_s(t0);
    }

    nsv::InferenceInputs voc_in;
    voc_in.frames = frames;
    voc_in.mel = std::move(mel.data);
    voc_in.seed = seed;
    voc_in.noise_scale = noise_scale;
    for (float & v : voc_in.mel) {
        if (has_mel_min) v = std::max(v, mel_min);
        if (has_mel_max) v = std::min(v, mel_max);
    }
    voc_in.f0 = std::move(ac_in.f0);
    nsv::InferenceOutputs wav;
    {
        auto t0 = SteadyClock::now();
        if (!nsv::run_vocoder(vocoder, voc_in, wav)) return false;
        if (timing) timing->vocoder_s += elapsed_s(t0);
    }
    {
        auto t0 = SteadyClock::now();
        if (!write_f32(s.wav_out, wav.wav)) return false;
        if (timing) {
            timing->write_s += elapsed_s(t0);
            timing->write_run = true;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Memory-based segment runner (for --ds mode, no temp files)
// ---------------------------------------------------------------------------

static bool run_segment_mem(const ds::Model & acoustic,
                            const nsv::Model & vocoder,
                            const dsp::DSSegment & seg,
                            int spk_id,
                            uint32_t seed,
                            int steps,
                            bool has_mel_min, float mel_min,
                            bool has_mel_max, float mel_max,
                            float noise_scale,
                            std::vector<float> & out_wav,
                            SegmentTiming * timing = nullptr) {
    ds::InferenceInputs ac_in;
    ac_in.tokens = seg.tokens;
    ac_in.mel2ph = seg.mel2ph;
    ac_in.f0 = seg.f0_hz;
    ac_in.languages = seg.languages;
    const int frames = (int)ac_in.mel2ph.size();

    if (acoustic.has_lang_embed && ac_in.languages.empty())
        ac_in.languages.assign(ac_in.tokens.size(), 0);
    if (acoustic.has_variance) {
        ac_in.breathiness = seg.breathiness.empty() ? std::vector<float>(frames, 0.0f) : seg.breathiness;
        ac_in.voicing     = seg.voicing.empty()     ? std::vector<float>(frames, 1.0f) : seg.voicing;
        ac_in.tension     = seg.tension.empty()     ? std::vector<float>(frames, 0.0f) : seg.tension;
        ac_in.energy      = seg.energy.empty()      ? std::vector<float>(frames, 0.0f) : seg.energy;
    }
    if (acoustic.has_key_shift) {
        ac_in.gender = seg.gender.empty() ? std::vector<float>(frames, 0.0f) : seg.gender;
    }
    if (acoustic.has_speed) {
        ac_in.velocity = seg.velocity.empty() ? std::vector<float>(frames, 1.0f) : seg.velocity;
    }
    ac_in.spk_id = spk_id;
    ac_in.seed = seed;
    ac_in.sampling_steps_override = steps;
    if (const char * alg = std::getenv("DSGGML_ALGORITHM"))
        ac_in.algorithm_override = alg;

    ds::InferenceOutputs mel;
    {
        auto t0 = SteadyClock::now();
        if (!ds::run_inference(acoustic, ac_in, mel)) return false;
        if (timing) timing->acoustic_s += elapsed_s(t0);
    }

    nsv::InferenceInputs voc_in;
    voc_in.frames = frames;
    voc_in.mel = std::move(mel.data);
    voc_in.seed = seed;
    voc_in.noise_scale = noise_scale;
    for (float & v : voc_in.mel) {
        if (has_mel_min) v = std::max(v, mel_min);
        if (has_mel_max) v = std::min(v, mel_max);
    }
    voc_in.f0 = std::move(ac_in.f0);
    nsv::InferenceOutputs wav;
    {
        auto t0 = SteadyClock::now();
        if (!nsv::run_vocoder(vocoder, voc_in, wav)) return false;
        if (timing) timing->vocoder_s += elapsed_s(t0);
    }
    out_wav = std::move(wav.wav);
    return true;
}

static bool ensure_variance_mem(const dsv::Model & model,
                                dsp::DSSegment & seg,
                                int spk_id,
                                uint32_t seed,
                                bool predict_all) {
    // Check which targets are missing
    const int T = (int)seg.mel2ph.size();
    std::vector<bool> missing;
    bool any_missing = false;
    for (const auto & target : model.cfg.variance_targets) {
        bool miss = predict_all;
        if (!miss) {
            if (target.name == "breathiness") miss = seg.breathiness.empty();
            else if (target.name == "voicing") miss = seg.voicing.empty();
            else if (target.name == "tension") miss = seg.tension.empty();
            else if (target.name == "energy") miss = seg.energy.empty();
            else miss = true;
        }
        missing.push_back(miss);
        any_missing = any_missing || miss;
    }
    if (!any_missing) return true;

    // Run variance FS2 condition
    dsv::FS2ConditionInputs fs2_in;
    fs2_in.spk_id = spk_id;
    fs2_in.tokens = seg.tokens;
    fs2_in.ph2word = seg.ph2word;
    fs2_in.word_dur = seg.word_dur;
    fs2_in.mel2ph = seg.mel2ph;
    fs2_in.languages = seg.languages;
    fs2_in.phones = (int)seg.tokens.size();
    fs2_in.frames = T;

    dsv::ConditionOutputs fs2_out;
    if (!dsv::run_fs2_condition(model, fs2_in, fs2_out)) return false;

    dsv::VarianceConditionInputs vc_in;
    vc_in.frames = T;
    vc_in.fs2_condition = std::move(fs2_out.condition);
    vc_in.pitch = seg.pitch_midi;
    dsv::ConditionOutputs vc_out;
    if (!dsv::compose_variance_condition(model, vc_in, vc_out)) return false;

    dsv::VarianceSampleInputs sample_in;
    sample_in.frames = T;
    sample_in.seed = seed;
    sample_in.condition = std::move(vc_out.condition);
    sample_in.existing_values.assign(model.cfg.variance_targets.size(), std::vector<float>((size_t)T, 0.0f));
    sample_in.retake_masks.assign(model.cfg.variance_targets.size(), std::vector<int32_t>((size_t)T, 1));
    for (size_t i = 0; i < model.cfg.variance_targets.size(); ++i) {
        if (i >= missing.size() || missing[i]) continue;
        const std::string & name = model.cfg.variance_targets[i].name;
        const std::vector<float> * src = nullptr;
        if (name == "breathiness") src = &seg.breathiness;
        else if (name == "voicing") src = &seg.voicing;
        else if (name == "tension") src = &seg.tension;
        else if (name == "energy") src = &seg.energy;
        if (!src || (int)src->size() != T) continue;
        sample_in.existing_values[i] = *src;
        std::fill(sample_in.retake_masks[i].begin(), sample_in.retake_masks[i].end(), 0);
    }
    dsv::VarianceSampleOutputs sample_out;
    if (!dsv::sample_variances(model, sample_in, sample_out)) return false;

    // Fill missing variance curves back into the segment
    for (size_t i = 0; i < sample_out.targets.size(); ++i) {
        if (i >= missing.size() || !missing[i]) continue;
        const std::string & name = sample_out.targets[i].name;
        if (name == "breathiness") seg.breathiness = std::move(sample_out.values[i]);
        else if (name == "voicing") seg.voicing = std::move(sample_out.values[i]);
        else if (name == "tension") seg.tension = std::move(sample_out.values[i]);
        else if (name == "energy") seg.energy = std::move(sample_out.values[i]);
    }
    return true;
}

static bool needs_variance_mem(const dsv::Model & model,
                               const dsp::DSSegment & seg,
                               bool predict_all) {
    if (predict_all) return !model.cfg.variance_targets.empty();
    for (const auto & target : model.cfg.variance_targets) {
        const std::string & name = target.name;
        if (name == "breathiness" && seg.breathiness.empty()) return true;
        if (name == "voicing" && seg.voicing.empty()) return true;
        if (name == "tension" && seg.tension.empty()) return true;
        if (name == "energy" && seg.energy.empty()) return true;
        if (name != "breathiness" && name != "voicing" &&
            name != "tension" && name != "energy") return true;
    }
    return false;
}

static int target_segment_frames(const dsp::DSSegment & seg) {
    if (!seg.base_pitch.empty()) return (int)seg.base_pitch.size();
    if (!seg.f0_hz.empty()) return (int)seg.f0_hz.size();
    if (!seg.pitch_midi.empty()) return (int)seg.pitch_midi.size();
    int total = 0;
    for (float v : seg.word_dur) total += (int)std::lround(v);
    return total;
}

static std::vector<float> mel2ph_to_ph_dur(const std::vector<int32_t> & mel2ph, int phones) {
    std::vector<float> ph_dur((size_t)phones, 0.0f);
    for (int32_t p : mel2ph) {
        if (p > 0 && p <= phones) ph_dur[(size_t)(p - 1)] += 1.0f;
    }
    return ph_dur;
}

static void align_duration_segment(dsp::DSSegment & seg) {
    const int target = target_segment_frames(seg);
    if (seg.mel2ph.empty() || target <= 0) return;
    if ((int)seg.mel2ph.size() < target) {
        const int32_t pad = seg.mel2ph.back();
        seg.mel2ph.resize((size_t)target, pad);
    } else if ((int)seg.mel2ph.size() > target) {
        seg.mel2ph.resize((size_t)target);
    }
    seg.ph_dur = mel2ph_to_ph_dur(seg.mel2ph, (int)seg.tokens.size());
}

static bool ensure_duration_mem(const dsv::Model & model,
                                dsp::DSSegment & seg,
                                int spk_id) {
    if (!seg.ph_dur.empty() && !seg.mel2ph.empty()) return true;
    dsv::DurationInputs in;
    in.phones = (int)seg.tokens.size();
    in.spk_id = spk_id;
    in.tokens = seg.tokens;
    in.ph2word = seg.ph2word;
    in.word_dur = seg.word_dur;
    in.midi = seg.midi;
    in.languages = seg.languages;
    dsv::DurationOutputs out;
    if (!dsv::predict_durations(model, in, out)) return false;
    seg.ph_dur = std::move(out.ph_dur);
    seg.mel2ph = std::move(out.mel2ph);
    align_duration_segment(seg);
    return true;
}

// ---------------------------------------------------------------------------
// WAV writing & stitching
// ---------------------------------------------------------------------------

static void cross_fade(std::vector<float> & result, const std::vector<float> & b, int idx) {
    if (idx < 0) idx = 0;
    const int a_len = (int)result.size();
    const int b_len = (int)b.size();
    if (idx >= a_len) {
        // Gap or append
        result.resize(idx + b_len, 0.0f);
        std::copy(b.begin(), b.end(), result.begin() + idx);
        return;
    }
    const int fade_len = std::min(a_len - idx, b_len);
    const int new_len = std::max(a_len, idx + b_len);
    result.resize(new_len, 0.0f);
    // Crossfade overlap region
    for (int i = 0; i < fade_len; ++i) {
        float k = (float)i / (float)fade_len;
        result[idx + i] = (1.0f - k) * result[idx + i] + k * b[i];
    }
    // Copy remainder of b
    for (int i = fade_len; i < b_len; ++i) {
        result[idx + i] = b[i];
    }
}

static bool write_wav(const std::string & path, const std::vector<float> & data, int sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[err] cannot write wav: %s\n", path.c_str());
        return false;
    }
    const int num_samples = (int)data.size();
    const int byte_rate = sr * 2;  // 16-bit mono
    const int data_size = num_samples * 2;
    const int file_size = 36 + data_size;

    // WAV header
    f.write("RIFF", 4);
    auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
    auto write_u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };
    write_u32((uint32_t)file_size);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    write_u32(16);        // chunk size
    write_u16(1);         // PCM
    write_u16(1);         // mono
    write_u32((uint32_t)sr);
    write_u32((uint32_t)byte_rate);
    write_u16(2);         // block align
    write_u16(16);        // bits per sample
    f.write("data", 4);
    write_u32((uint32_t)data_size);

    // Convert float to int16
    for (int i = 0; i < num_samples; ++i) {
        float v = std::clamp(data[i], -1.0f, 1.0f);
        int16_t sample = (int16_t)std::lround(v * 32767.0f);
        f.write(reinterpret_cast<const char *>(&sample), 2);
    }
    return (bool)f;
}

int main(int argc, char ** argv) {
    const auto program_t0 = SteadyClock::now();
    std::string variance_path;
    std::string acoustic_path;
    std::string vocoder_path;
    std::string pitch_model_path;
    std::string pitch_phonemes_path;
    std::string manifest_path;
    std::string ds_path;
    std::string phonemes_path;
    std::string out_path;
    std::string spk_name;
    std::string spk_map_path;
    std::string ds_lang = "zh";
    int spk_id = -1;
    uint32_t seed = 1234;
    int steps = -1;
    bool has_mel_min = false;
    bool has_mel_max = false;
    float mel_min = 0.0f;
    float mel_max = 0.0f;
    float noise_scale = 1.0f;
    bool predict_all_variances = false;
    bool auto_duration = false;
    bool auto_pitch = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--variance-model") variance_path = next();
        else if (a == "--acoustic-model") acoustic_path = next();
        else if (a == "--vocoder-model") vocoder_path = next();
        else if (a == "--pitch-model") pitch_model_path = next();
        else if (a == "--pitch-phonemes") pitch_phonemes_path = next();
        else if (a == "--auto-duration") auto_duration = true;
        else if (a == "--auto-pitch") auto_pitch = true;
        else if (a == "--manifest") manifest_path = next();
        else if (a == "--ds") ds_path = next();
        else if (a == "--phonemes") phonemes_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--spk-id") spk_id = std::atoi(next().c_str());
        else if (a == "--spk-name") spk_name = next();
        else if (a == "--spk-map") spk_map_path = next();
        else if (a == "--lang") ds_lang = next();
        else if (a == "--seed") seed = (uint32_t)std::atoll(next().c_str());
        else if (a == "--steps") steps = std::atoi(next().c_str());
        else if (a == "--algorithm") {
            ds_setenv("DSGGML_ALGORITHM", next().c_str());
        }
        else if (a == "--mel-min") { mel_min = (float)std::atof(next().c_str()); has_mel_min = true; }
        else if (a == "--mel-max") { mel_max = (float)std::atof(next().c_str()); has_mel_max = true; }
        else if (a == "--noise-scale") noise_scale = (float)std::atof(next().c_str());
        else if (a == "--backend") dsrt::set_backend_mode(next().c_str());
        else if (a == "--threads") dsrt::set_runtime_threads(next().c_str());
        else if (a == "--variance-backend") {
            std::string mode = next();
            ds_setenv("DSGGML_BACKEND_VARIANCE", mode.c_str());
        }
        else if (a == "--acoustic-backend") {
            std::string mode = next();
            ds_setenv("DSGGML_BACKEND_ACOUSTIC", mode.c_str());
        }
        else if (a == "--vocoder-backend") {
            std::string mode = next();
            ds_setenv("DSGGML_BACKEND_VOCODER", mode.c_str());
        }
        else if (a == "--pitch-backend") {
            std::string mode = next();
            ds_setenv("DSGGML_BACKEND_PITCH", mode.c_str());
        }
        else if (a == "--predict-all-variances") predict_all_variances = true;
        else if (a == "--precision") {
            std::string mode = next();
            if (mode == "f16" || mode == "fp16") dsrt::set_precision(dsrt::Precision::F16);
            else if (mode == "f32" || mode == "fp32") dsrt::set_precision(dsrt::Precision::F32);
            else { fprintf(stderr, "unknown precision: %s\n", mode.c_str()); return 2; }
        }
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            fprintf(stderr, "unknown: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (variance_path.empty() || acoustic_path.empty() || vocoder_path.empty()) {
        usage();
        return 2;
    }

    // Resolve speaker ID
    if (spk_id < 0) {
        if (!spk_name.empty()) {
            if (spk_map_path.empty()) {
                fprintf(stderr, "[err] --spk-name requires --spk-map\n");
                return 2;
            }
            auto spk_map = dsp::load_spk_map(spk_map_path);
            if (spk_map.empty()) return 1;
            auto it = spk_map.find(spk_name);
            if (it == spk_map.end()) {
                fprintf(stderr, "[err] unknown speaker '%s'. Available:", spk_name.c_str());
                for (const auto & [k, _] : spk_map) fprintf(stderr, " %s", k.c_str());
                fprintf(stderr, "\n");
                return 1;
            }
            spk_id = it->second;
        } else {
            spk_id = 0;
        }
    }

    // ---- DS mode ----
    if (!ds_path.empty()) {
        if (phonemes_path.empty()) {
            fprintf(stderr, "[err] --ds mode requires --phonemes <phonemes.json>\n");
            return 2;
        }
        if (out_path.empty()) {
            fprintf(stderr, "[err] --ds mode requires --out <output.wav>\n");
            return 2;
        }

        auto phoneme_map = dsp::load_phoneme_map(phonemes_path, ds_lang);
        if (phoneme_map.empty()) return 1;

        auto segments = dsp::parse_ds_file(ds_path, phoneme_map, 44100, 512);
        if (segments.empty()) return 1;

        // Load models
        // Diffusion models (variance, acoustic, pitch) MUST stay F32 — ODE accumulates errors.
        // Only vocoder (feed-forward) is safe for F16.
        auto user_precision = dsrt::get_precision();
        dsrt::set_precision(dsrt::Precision::F32);  // Force F32 for diffusion models

        dsv::Model variance;
        if (!variance.load(variance_path)) return 1;
        ds::Model acoustic;
        if (!acoustic.load(acoustic_path)) return 1;

        dsrt::set_precision(user_precision);  // Restore user setting for vocoder
        nsv::Model vocoder;
        if (!vocoder.load(vocoder_path)) return 1;
        dsrt::set_precision(dsrt::Precision::F32);  // Back to F32 for pitch model

        // Optional pitch model
        dsp_pitch::Model pitch_model;
        bool have_pitch_model = false;
        std::unordered_map<std::string, int> pitch_phoneme_map;
        if (!pitch_model_path.empty()) {
            if (!pitch_model.load(pitch_model_path)) return 1;
            have_pitch_model = true;
            // Load pitch-specific phoneme map if provided; otherwise reuse main phoneme map
            if (!pitch_phonemes_path.empty()) {
                pitch_phoneme_map = dsp::load_phoneme_map(pitch_phonemes_path, ds_lang);
                if (pitch_phoneme_map.empty()) return 1;
            }
        }

        // Process all segments
        std::vector<float> final_wav;
        int current_length = 0;
        double total_pitch_s = 0.0;
        double total_variance_s = 0.0;
        double total_acoustic_s = 0.0;
        double total_vocoder_s = 0.0;
        double total_stitch_s = 0.0;

        for (size_t i = 0; i < segments.size(); ++i) {
            SegmentTiming timing;
            const auto segment_t0 = SteadyClock::now();
            fprintf(stderr, "[pipeline] segment %zu/%zu (T=%d)\n",
                    i + 1, segments.size(), target_segment_frames(segments[i]));

            if (segments[i].ph_dur.empty() || segments[i].mel2ph.empty()) {
                if (!auto_duration) {
                    fprintf(stderr,
                            "[err] segment %zu is missing phone durations/alignment; rerun with --auto-duration\n",
                            i + 1);
                    return 1;
                }
                if (!variance.cfg.predict_dur) {
                    fprintf(stderr,
                            "[err] variance model does not export duration prediction, but segment %zu needs it\n",
                            i + 1);
                    return 1;
                }
                const auto dur_t0 = SteadyClock::now();
                if (!ensure_duration_mem(variance, segments[i], spk_id)) return 1;
                timing.variance_s += elapsed_s(dur_t0);
                timing.variance_run = true;
                fprintf(stderr, "[pipeline] duration model predicted mel2ph for segment %zu\n", i + 1);
            }

            // Run pitch model if available and needed
            if (have_pitch_model && (auto_pitch || segments[i].f0_hz.empty()) &&
                !segments[i].note_midi.empty()) {
                const auto pitch_t0 = SteadyClock::now();
                dsp_pitch::PitchInputs pin;
                pin.phones = (int)segments[i].tokens.size();
                pin.frames = (int)segments[i].mel2ph.size();
                pin.notes = (int)segments[i].note_midi.size();
                pin.spk_id = spk_id;
                pin.seed = seed;
                pin.ph2word = segments[i].ph2word;
                pin.ph_dur = segments[i].ph_dur;
                pin.mel2ph = segments[i].mel2ph;
                pin.languages = segments[i].languages;
                pin.note_midi = segments[i].note_midi;
                pin.note_rest = segments[i].note_rest;
                pin.note_dur = segments[i].note_dur_frames;
                pin.mel2note = segments[i].mel2note;
                pin.base_pitch = segments[i].base_pitch;

                // Re-encode tokens for pitch model's vocabulary if needed
                if (!pitch_phoneme_map.empty()) {
                    pin.tokens.resize(pin.phones);
                    for (int p = 0; p < pin.phones; ++p) {
                        auto it = pitch_phoneme_map.find(segments[i].ph_names[p]);
                        if (it != pitch_phoneme_map.end()) {
                            pin.tokens[p] = it->second;
                        } else {
                            // Try lowercase
                            std::string lower = segments[i].ph_names[p];
                            for (char & c : lower) c = (char)std::tolower((unsigned char)c);
                            auto it2 = pitch_phoneme_map.find(lower);
                            if (it2 != pitch_phoneme_map.end()) {
                                pin.tokens[p] = it2->second;
                            } else {
                                fprintf(stderr, "[warn] pitch phoneme unknown: '%s', using 0\n",
                                        segments[i].ph_names[p].c_str());
                                pin.tokens[p] = 0;
                            }
                        }
                    }
                } else {
                    pin.tokens = segments[i].tokens;
                }

                dsp_pitch::PitchOutputs pout;
                if (!dsp_pitch::run_pitch_inference(pitch_model, pin, pout)) return 1;
                timing.pitch_s += elapsed_s(pitch_t0);
                timing.pitch_run = true;
                segments[i].f0_hz = std::move(pout.f0_hz);
                segments[i].pitch_midi = std::move(pout.pitch_midi);
                if (!pout.voicing.empty()) {
                    segments[i].voicing = std::move(pout.voicing);
                }
                fprintf(stderr, "[pipeline] pitch model predicted F0 for segment %zu\n", i + 1);
            }

            {
                const bool needs_variance = needs_variance_mem(variance, segments[i],
                                                               predict_all_variances);
                const auto variance_t0 = SteadyClock::now();
                if (!ensure_variance_mem(variance, segments[i], spk_id, seed, predict_all_variances))
                    return 1;
                if (needs_variance) {
                    timing.variance_s += elapsed_s(variance_t0);
                    timing.variance_run = true;
                }
            }

            std::vector<float> seg_wav;
            if (!run_segment_mem(acoustic, vocoder, segments[i], spk_id, seed, steps,
                                 has_mel_min, mel_min, has_mel_max, mel_max, noise_scale, seg_wav,
                                 &timing))
                return 1;

            // Stitch with offset-based crossfade
            {
                const auto stitch_t0 = SteadyClock::now();
                int offset_samples = (int)std::lround(segments[i].offset * 44100.0);
                int silent_length = offset_samples - current_length;
                if (silent_length >= 0) {
                    if (silent_length > 0) {
                        final_wav.resize(final_wav.size() + silent_length, 0.0f);
                    }
                    final_wav.insert(final_wav.end(), seg_wav.begin(), seg_wav.end());
                } else {
                    cross_fade(final_wav, seg_wav, current_length + silent_length);
                }
                current_length = current_length + silent_length + (int)seg_wav.size();
                timing.stitch_s += elapsed_s(stitch_t0);
                timing.stitch_run = true;
            }

            timing.total_s = elapsed_s(segment_t0);
            total_pitch_s += timing.pitch_s;
            total_variance_s += timing.variance_s;
            total_acoustic_s += timing.acoustic_s;
            total_vocoder_s += timing.vocoder_s;
            total_stitch_s += timing.stitch_s;
            print_segment_timing(i + 1, segments.size(), timing);
        }

        const auto write_t0 = SteadyClock::now();
        if (!write_wav(out_path, final_wav, 44100)) return 1;
        const double write_s = elapsed_s(write_t0);
        const double total_s = elapsed_s(program_t0);
        fprintf(stderr, "[ok] wrote %s (%d segments, %.1fs audio)\n",
                out_path.c_str(), (int)segments.size(), (float)final_wav.size() / 44100.0f);
        fprintf(stderr,
                "[time] totals load+pipeline+write=%.3fs pitch=%.3fs variance=%.3fs acoustic=%.3fs vocoder=%.3fs stitch=%.3fs write=%.3fs\n",
                total_s, total_pitch_s, total_variance_s, total_acoustic_s,
                total_vocoder_s, total_stitch_s, write_s);
        return 0;
    }

    // ---- Legacy manifest mode ----
    if (manifest_path.empty()) {
        usage();
        return 2;
    }
    std::vector<Segment> segments = read_manifest(manifest_path);
    if (segments.empty()) {
        fprintf(stderr, "[err] manifest has no segments\n");
        return 1;
    }

    auto user_precision = dsrt::get_precision();
    dsrt::set_precision(dsrt::Precision::F32);
    dsv::Model variance;
    if (!variance.load(variance_path)) return 1;
    ds::Model acoustic;
    if (!acoustic.load(acoustic_path)) return 1;
    dsrt::set_precision(user_precision);
    nsv::Model vocoder;
    if (!vocoder.load(vocoder_path)) return 1;
    dsrt::set_precision(dsrt::Precision::F32);

    double total_variance_s = 0.0;
    double total_acoustic_s = 0.0;
    double total_vocoder_s = 0.0;
    double total_write_s = 0.0;
    for (size_t i = 0; i < segments.size(); ++i) {
        SegmentTiming timing;
        const auto segment_t0 = SteadyClock::now();
        fprintf(stderr, "[pipeline] segment %zu/%zu\n", i + 1, segments.size());
        {
            bool needs_variance = predict_all_variances;
            if (!needs_variance) {
                for (const auto & target : variance.cfg.variance_targets) {
                    if (!exists_file(segments[i].variance_dir + "/" + target.name + ".bin")) {
                        needs_variance = true;
                        break;
                    }
                }
            }
            const auto variance_t0 = SteadyClock::now();
            if (!ensure_variance_curves(variance, segments[i], spk_id, seed, predict_all_variances)) return 1;
            if (needs_variance) {
                timing.variance_s += elapsed_s(variance_t0);
                timing.variance_run = true;
            }
        }
        if (!run_segment(acoustic, vocoder, segments[i], spk_id, seed, steps,
                         has_mel_min, mel_min, has_mel_max, mel_max, noise_scale,
                         &timing)) return 1;
        fprintf(stderr, "[pipeline] wrote %s\n", segments[i].wav_out.c_str());
        timing.total_s = elapsed_s(segment_t0);
        total_variance_s += timing.variance_s;
        total_acoustic_s += timing.acoustic_s;
        total_vocoder_s += timing.vocoder_s;
        total_write_s += timing.write_s;
        print_segment_timing(i + 1, segments.size(), timing);
    }
    fprintf(stderr,
            "[time] totals load+pipeline=%.3fs variance=%.3fs acoustic=%.3fs vocoder=%.3fs write=%.3fs\n",
            elapsed_s(program_t0), total_variance_s, total_acoustic_s,
            total_vocoder_s, total_write_s);
    return 0;
}
