#include "variance.h"
#include "runtime_backend.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void usage() {
    fprintf(stderr,
        "usage: diffsinger_variance --model <variance.gguf> [--inspect]\n"
        "                           [--fs2-cond --tokens-bin <i32>]\n"
        "                           [--ph2word-bin <i32> --word-dur-bin <f32>]\n"
        "                           [--mel2ph-bin <i32> --phones <L> --frames <T>]\n"
        "                           [--lang-bin <i32>] [--spk-id <id>] --out <cond.f32>\n"
        "                           [--variance-cond --fs2-cond-bin <f32> --pitch-bin <f32> --out <cond.f32>]\n"
        "                           [--sample-variance --cond-bin <cond.f32> --frames <T>]\n"
        "                           [--sample-variance --fs2-cond-bin <f32> --pitch-bin <f32> --frames <T>]\n"
        "                           [--infer-variance --tokens-bin <i32> --mel2ph-bin <i32>]\n"
        "                           [--ph2word-bin <i32> --word-dur-bin <f32> --pitch-bin <f32>]\n"
        "                           [--out-breathiness <f32> --out-voicing <f32> --out-tension <f32>]\n"
        "                           [--out-energy <f32>] [--seed <u32>]\n"
        "                           [--backend cpu|gpu|auto]\n"
        "\n"
        "All modes are for the current word-duration variance checkpoint. Pitch must be\n"
        "provided as frame-level MIDI via --pitch-bin; this tool no longer predicts pitch.\n");
}

static bool read_f32_file(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "[err] open failed: %s\n", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    if (sz < 0 || (sz % 4) != 0) {
        fprintf(stderr, "[err] %s is not a float32 file\n", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize((size_t)sz / 4);
    if (!f.read(reinterpret_cast<char *>(out.data()), sz)) {
        fprintf(stderr, "[err] read failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

static bool read_i32_file(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "[err] open failed: %s\n", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    if (sz < 0 || (sz % 4) != 0) {
        fprintf(stderr, "[err] %s is not an int32 file\n", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize((size_t)sz / 4);
    if (!f.read(reinterpret_cast<char *>(out.data()), sz)) {
        fprintf(stderr, "[err] read failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

static bool write_f32_file(const std::string & path, const std::vector<float> & out) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[err] write failed: %s\n", path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char *>(out.data()), (std::streamsize)out.size() * 4);
    return (bool)f;
}

static void print_tensor_shape(const dsv::Model & model, const char * name) {
    ggml_tensor * t = model.get(name, true);
    if (!t) {
        fprintf(stderr, "  %-64s missing\n", name);
        return;
    }
    fprintf(stderr, "  %-64s type=%d ne=[%lld,%lld,%lld,%lld]\n",
            name, (int)t->type,
            (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3]);
}

static bool read_fs2_inputs(dsv::FS2ConditionInputs & in,
                            const std::string & tokens_path,
                            const std::string & ph2word_path,
                            const std::string & word_dur_path,
                            const std::string & mel2ph_path,
                            const std::string & lang_path) {
    if (!read_i32_file(tokens_path, in.tokens)) return false;
    if (!read_i32_file(ph2word_path, in.ph2word)) return false;
    if (!read_f32_file(word_dur_path, in.word_dur)) return false;
    if (!read_i32_file(mel2ph_path, in.mel2ph)) return false;
    if (!lang_path.empty() && !read_i32_file(lang_path, in.languages)) return false;
    if (in.phones == 0) in.phones = (int)in.tokens.size();
    if (in.frames == 0) in.frames = (int)in.mel2ph.size();
    return true;
}

static std::string output_path_for_target(const std::string & name,
                                          const std::string & energy,
                                          const std::string & breathiness,
                                          const std::string & voicing,
                                          const std::string & tension) {
    if (name == "energy") return energy;
    if (name == "breathiness") return breathiness;
    if (name == "voicing") return voicing;
    if (name == "tension") return tension;
    return {};
}

static bool write_variance_outputs(const dsv::VarianceSampleOutputs & out,
                                   const std::string & out_energy_path,
                                   const std::string & out_breathiness_path,
                                   const std::string & out_voicing_path,
                                   const std::string & out_tension_path) {
    for (size_t i = 0; i < out.targets.size(); ++i) {
        const std::string path = output_path_for_target(
            out.targets[i].name, out_energy_path, out_breathiness_path,
            out_voicing_path, out_tension_path);
        if (path.empty()) {
            fprintf(stderr, "[err] missing output path for variance target '%s'\n",
                    out.targets[i].name.c_str());
            return false;
        }
        if (!write_f32_file(path, out.values[i])) return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string tokens_path;
    std::string ph2word_path;
    std::string word_dur_path;
    std::string lang_path;
    std::string mel2ph_path;
    std::string fs2_cond_path;
    std::string pitch_path;
    std::string cond_path;
    std::string out_path;
    std::string out_energy_path;
    std::string out_breathiness_path;
    std::string out_voicing_path;
    std::string out_tension_path;
    int phones = 0;
    int frames = 0;
    int spk_id = -1;
    uint32_t seed = 1234;
    bool inspect = false;
    bool fs2_cond = false;
    bool variance_cond = false;
    bool sample_variance_flag = false;
    bool infer_variance = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--model") model_path = next();
        else if (a == "--inspect") inspect = true;
        else if (a == "--fs2-cond") fs2_cond = true;
        else if (a == "--variance-cond") variance_cond = true;
        else if (a == "--sample-variance") sample_variance_flag = true;
        else if (a == "--infer-variance") infer_variance = true;
        else if (a == "--tokens-bin") tokens_path = next();
        else if (a == "--ph2word-bin") ph2word_path = next();
        else if (a == "--word-dur-bin") word_dur_path = next();
        else if (a == "--lang-bin") lang_path = next();
        else if (a == "--mel2ph-bin") mel2ph_path = next();
        else if (a == "--fs2-cond-bin") fs2_cond_path = next();
        else if (a == "--pitch-bin") pitch_path = next();
        else if (a == "--phones") phones = std::atoi(next().c_str());
        else if (a == "--spk-id") spk_id = std::atoi(next().c_str());
        else if (a == "--seed") seed = (uint32_t)std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--backend") dsrt::set_backend_mode(next().c_str());
        else if (a == "--precision") {
            std::string m = next();
            if (m == "f16" || m == "fp16") dsrt::set_precision(dsrt::Precision::F16);
            else dsrt::set_precision(dsrt::Precision::F32);
        }
        else if (a == "--cond-bin") cond_path = next();
        else if (a == "--frames") frames = std::atoi(next().c_str());
        else if (a == "--out") out_path = next();
        else if (a == "--out-energy") out_energy_path = next();
        else if (a == "--out-breathiness") out_breathiness_path = next();
        else if (a == "--out-voicing") out_voicing_path = next();
        else if (a == "--out-tension") out_tension_path = next();
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            fprintf(stderr, "unknown: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (model_path.empty()) {
        usage();
        return 2;
    }

    dsv::Model model;
    if (!model.load(model_path)) return 1;

    if (inspect) {
        const dsv::Config & c = model.cfg;
        fprintf(stderr, "[inspect] enc_layers=%u heads=%u ffn=%s rope=%d lang=%d spk=%d\n",
                c.enc_layers, c.num_heads, c.ffn_act.c_str(), (int)c.use_rope,
                (int)c.use_lang_id, (int)c.use_spk_id);
        fprintf(stderr, "[inspect] predict: dur=%d energy=%d breathiness=%d voicing=%d tension=%d\n",
                (int)c.predict_dur, (int)c.predict_energy,
                (int)c.predict_breathiness, (int)c.predict_voicing, (int)c.predict_tension);
        print_tensor_shape(model, "fs2.txt_embed.weight");
        print_tensor_shape(model, "fs2.onset_embed.weight");
        print_tensor_shape(model, "fs2.word_dur_embed.weight");
        print_tensor_shape(model, "pitch_embed.weight");
        print_tensor_shape(model, "spk_embed.weight");
        print_tensor_shape(model, "var.vf.in.weight");
        print_tensor_shape(model, "var.vf.r.0.net.2.weight");
        if (!(fs2_cond || variance_cond || sample_variance_flag || infer_variance)) {
            return 0;
        }
    }

    auto compose_variance_from_files = [&]() -> std::vector<float> {
        dsv::VarianceConditionInputs in;
        in.frames = frames;
        if (!read_f32_file(fs2_cond_path, in.fs2_condition)) return {};
        if (!read_f32_file(pitch_path, in.pitch)) return {};
        if (in.frames == 0 && model.cfg.hidden_size > 0) {
            in.frames = (int)(in.fs2_condition.size() / model.cfg.hidden_size);
        }
        dsv::ConditionOutputs out;
        if (!dsv::compose_variance_condition(model, in, out)) return {};
        return out.condition;
    };

    if (fs2_cond) {
        if (tokens_path.empty() || ph2word_path.empty() || word_dur_path.empty() ||
            mel2ph_path.empty() || out_path.empty()) {
            fprintf(stderr, "[err] --fs2-cond requires tokens, ph2word, word_dur, mel2ph and --out\n");
            usage();
            return 2;
        }
        dsv::FS2ConditionInputs in;
        in.phones = phones;
        in.frames = frames;
        in.spk_id = spk_id;
        if (!read_fs2_inputs(in, tokens_path, ph2word_path, word_dur_path, mel2ph_path, lang_path)) return 1;
        dsv::ConditionOutputs out;
        if (!dsv::run_fs2_condition(model, in, out)) return 1;
        if (!write_f32_file(out_path, out.condition)) return 1;
        fprintf(stderr, "[ok] wrote %s frames=%d hidden=%d\n",
                out_path.c_str(), out.frames, out.hidden);
    }

    if (variance_cond) {
        if (fs2_cond_path.empty() || pitch_path.empty() || out_path.empty()) {
            fprintf(stderr, "[err] --variance-cond requires --fs2-cond-bin, --pitch-bin and --out\n");
            usage();
            return 2;
        }
        std::vector<float> condition = compose_variance_from_files();
        if (condition.empty()) return 1;
        if (!write_f32_file(out_path, condition)) return 1;
        int out_frames = frames;
        if (out_frames == 0 && model.cfg.hidden_size > 0) {
            out_frames = (int)(condition.size() / model.cfg.hidden_size);
        }
        fprintf(stderr, "[ok] wrote %s frames=%d hidden=%u\n",
                out_path.c_str(), out_frames, model.cfg.hidden_size);
    }

    if (sample_variance_flag) {
        dsv::VarianceSampleInputs in;
        in.frames = frames;
        in.seed = seed;
        if (!cond_path.empty()) {
            if (!read_f32_file(cond_path, in.condition)) return 1;
        } else if (!fs2_cond_path.empty() && !pitch_path.empty()) {
            in.condition = compose_variance_from_files();
            if (in.condition.empty()) return 1;
        } else {
            fprintf(stderr, "[err] --sample-variance requires --cond-bin or --fs2-cond-bin plus --pitch-bin\n");
            usage();
            return 2;
        }
        if (in.frames == 0 && model.cfg.hidden_size > 0) {
            in.frames = (int)(in.condition.size() / model.cfg.hidden_size);
        }
        dsv::VarianceSampleOutputs out;
        if (!dsv::sample_variances(model, in, out)) return 1;
        if (!write_variance_outputs(out, out_energy_path, out_breathiness_path,
                                    out_voicing_path, out_tension_path)) return 1;
        fprintf(stderr, "[ok] wrote variance curves frames=%d\n", out.frames);
    }

    if (infer_variance) {
        if (tokens_path.empty() || ph2word_path.empty() || word_dur_path.empty() ||
            mel2ph_path.empty() || pitch_path.empty()) {
            fprintf(stderr, "[err] --infer-variance requires tokens, ph2word, word_dur, mel2ph and pitch\n");
            usage();
            return 2;
        }

        dsv::FS2ConditionInputs fs2_in;
        fs2_in.phones = phones;
        fs2_in.frames = frames;
        fs2_in.spk_id = spk_id;
        if (!read_fs2_inputs(fs2_in, tokens_path, ph2word_path, word_dur_path, mel2ph_path, lang_path)) return 1;

        std::vector<float> pitch_abs;
        if (!read_f32_file(pitch_path, pitch_abs)) return 1;
        if ((int)pitch_abs.size() != fs2_in.frames) {
            fprintf(stderr, "[err] pitch size mismatch: got %zu expected %d\n",
                    pitch_abs.size(), fs2_in.frames);
            return 1;
        }

        dsv::ConditionOutputs fs2_out;
        if (!dsv::run_fs2_condition(model, fs2_in, fs2_out)) return 1;
        dsv::VarianceConditionInputs vc_in;
        vc_in.frames = fs2_in.frames;
        vc_in.fs2_condition = fs2_out.condition;
        vc_in.pitch = pitch_abs;
        dsv::ConditionOutputs vc_out;
        if (!dsv::compose_variance_condition(model, vc_in, vc_out)) return 1;

        dsv::VarianceSampleInputs vs_in;
        vs_in.frames = fs2_in.frames;
        vs_in.seed = seed;
        vs_in.condition = vc_out.condition;
        dsv::VarianceSampleOutputs vs_out;
        if (!dsv::sample_variances(model, vs_in, vs_out)) return 1;
        if (!write_variance_outputs(vs_out, out_energy_path, out_breathiness_path,
                                    out_voicing_path, out_tension_path)) return 1;
        fprintf(stderr, "[ok] wrote variance curves frames=%d\n", fs2_in.frames);
        return 0;
    }

    return 0;
}
