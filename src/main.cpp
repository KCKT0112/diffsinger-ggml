// CLI front-end for diffsinger_acoustic ggml runtime.
//
// Usage:
//   diffsinger_acoustic --model acoustic.gguf --out mel.bin
//   diffsinger_acoustic --model acoustic.gguf --out cond.bin --dump-cond
#include "diffsinger.h"
#include "runtime_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage() {
    fprintf(stderr,
        "usage: diffsinger_acoustic --model <gguf> --out <output.bin>\n"
        "  [--len-tokens N] [--frames-per-token N] [--f0 HZ]\n"
        "  [--mel2ph-bin P] raw int32 [T] 1-based phone index, overrides durations\n"
        "  [--depth D]      shallow diffusion depth in [0, max_depth] (default 1.0)\n"
        "  [--aux-mel PATH] raw float32 [T, mel_bins] aux decoder mel\n"
        "                   (required when use_variable_depth=true and t_start>0)\n"
        "  [--steps N]      override sampling_steps from gguf\n"
        "  [--algorithm A]  euler|midpoint|rk4 (default: from gguf)\n"
        "  [--seed N]       PRNG seed (default 1234)\n"
        "  [--spk-id N]     speaker id for multi-speaker acoustic checkpoints\n"
        "  [--backend cpu|gpu|auto]\n"
        "  [--noise-bin P]  raw float32 [T, mel_bins] fixed noise for x_t init\n"
        "                   (overrides --seed)\n"
        "  [--dump-cond]    output encoder cond [T,H] instead of mel\n");
}

int main(int argc, char ** argv) {
    std::string model_path, out_path = "mel.bin";
    int len_tokens = 16, frames_per_token = 8;
    float f0_hz = 220.0f;
    bool dump_cond = false;
    float depth = 1.0f;
    std::string aux_mel_path;
    int steps_override = -1;
    uint32_t seed = 1234;
    int spk_id = 0;
    std::string noise_bin_path;
    std::string f0_bin_path;       // raw float32 [T]; overrides --f0
    std::string tokens_bin_path;   // raw int32 [L]
    std::string dur_bin_path;      // raw int32 [L]
    std::string mel2ph_bin_path;   // raw int32 [T]
    std::string lang_bin_path;     // raw int32 [L]
    std::string variance_dir;      // dir containing voicing.bin/breathiness.bin/etc
    std::string algorithm;         // euler|midpoint|rk4 (empty => from gguf)

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (a == "--model")            model_path = next();
        else if (a == "--out")              out_path   = next();
        else if (a == "--len-tokens")       len_tokens = std::atoi(next().c_str());
        else if (a == "--frames-per-token") frames_per_token = std::atoi(next().c_str());
        else if (a == "--f0")               f0_hz = (float)std::atof(next().c_str());
        else if (a == "--depth")            depth = (float)std::atof(next().c_str());
        else if (a == "--aux-mel")          aux_mel_path = next();
        else if (a == "--steps")            steps_override = std::atoi(next().c_str());
        else if (a == "--seed")             seed = (uint32_t)std::atoll(next().c_str());
        else if (a == "--spk-id")           spk_id = std::atoi(next().c_str());
        else if (a == "--backend")          dsrt::set_backend_mode(next().c_str());
        else if (a == "--noise-bin")        noise_bin_path = next();
        else if (a == "--f0-bin")           f0_bin_path = next();
        else if (a == "--tokens-bin")       tokens_bin_path = next();
        else if (a == "--dur-bin")          dur_bin_path = next();
        else if (a == "--mel2ph-bin")       mel2ph_bin_path = next();
        else if (a == "--lang-bin")         lang_bin_path = next();
        else if (a == "--variance-dir")     variance_dir = next();
        else if (a == "--algorithm")        algorithm = next();
        else if (a == "--dump-cond")        dump_cond = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown: %s\n", a.c_str()); usage(); return 2; }
    }
    if (model_path.empty()) { usage(); return 2; }

    ds::Model model;
    if (!model.load(model_path)) return 1;

    int L = len_tokens;
    int T = L * frames_per_token;

    auto read_bin = [](const std::string & path, void * dst, size_t bytes) -> bool {
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) { perror(path.c_str()); return false; }
        size_t r = std::fread(dst, 1, bytes, f);
        std::fclose(f);
        if (r != bytes) {
            fprintf(stderr, "[err] %s: read %zu bytes, expected %zu\n",
                    path.c_str(), r, bytes);
            return false;
        }
        return true;
    };
    auto read_all_i32 = [](const std::string & path, std::vector<int32_t> & out) -> bool {
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) { perror(path.c_str()); return false; }
        if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return false; }
        long sz = std::ftell(f);
        if (sz < 0 || (sz % 4) != 0) {
            fprintf(stderr, "[err] %s: not an int32 file\n", path.c_str());
            std::fclose(f);
            return false;
        }
        std::rewind(f);
        out.resize((size_t)sz / 4);
        size_t r = std::fread(out.data(), sizeof(int32_t), out.size(), f);
        std::fclose(f);
        if (r != out.size()) {
            fprintf(stderr, "[err] %s: read %zu int32, expected %zu\n", path.c_str(), r, out.size());
            return false;
        }
        return true;
    };

    ds::InferenceInputs in;
    if (!tokens_bin_path.empty()) {
        if (!read_all_i32(tokens_bin_path, in.tokens)) return 1;
        L = (int)in.tokens.size();
    } else {
        in.tokens.resize(L);
        for (int i = 0; i < L; ++i)
            in.tokens[i] = 1 + (i % ((int)model.cfg.vocab_size - 1));
    }
    in.durations.resize(L);
    if (!dur_bin_path.empty()) {
        if (!read_bin(dur_bin_path, in.durations.data(), L * 4)) return 1;
        // recompute T from durations
        T = 0;
        for (int d : in.durations) T += d;
    } else {
        for (int i = 0; i < L; ++i) in.durations[i] = frames_per_token;
    }
    if (!mel2ph_bin_path.empty()) {
        if (!read_all_i32(mel2ph_bin_path, in.mel2ph)) return 1;
        T = (int)in.mel2ph.size();
        in.durations.assign((size_t)L, 0);
        for (int idx : in.mel2ph) {
            if (idx > 0 && idx <= L) in.durations[(size_t)idx - 1]++;
        }
    }
    in.f0.resize(T);
    if (!f0_bin_path.empty()) {
        if (!read_bin(f0_bin_path, in.f0.data(), T * 4)) return 1;
    } else {
        for (int i = 0; i < T; ++i) in.f0[i] = f0_hz;
    }

    // Match the reference script: languages = zeros, variances = zeros/ones
    if (model.has_lang_embed) {
        in.languages.assign(L, 0);
        if (!lang_bin_path.empty()) {
            if (!read_bin(lang_bin_path, in.languages.data(), L * 4)) return 1;
        }
    }
    if (model.has_variance) {
        in.breathiness.assign(T, 0.f);
        in.voicing.assign(T, 1.f);
        in.tension.assign(T, 0.f);
        in.energy.assign(T, 0.f);
        if (!variance_dir.empty()) {
            std::string br = variance_dir + "/breathiness.bin";
            std::string vo = variance_dir + "/voicing.bin";
            std::string te = variance_dir + "/tension.bin";
            std::string en = variance_dir + "/energy.bin";
            FILE * f;
            if ((f = std::fopen(br.c_str(), "rb"))) { std::fread(in.breathiness.data(), 4, T, f); std::fclose(f); }
            if ((f = std::fopen(vo.c_str(), "rb"))) { std::fread(in.voicing.data(),     4, T, f); std::fclose(f); }
            if ((f = std::fopen(te.c_str(), "rb"))) { std::fread(in.tension.data(),     4, T, f); std::fclose(f); }
            if ((f = std::fopen(en.c_str(), "rb"))) { std::fread(in.energy.data(),      4, T, f); std::fclose(f); }
        }
    }
    if (model.has_key_shift) in.gender.assign(T, 0.f);
    if (model.has_speed)     in.velocity.assign(T, 1.f);
    if (!variance_dir.empty()) {
        FILE * f;
        if (model.has_key_shift) {
            std::string ge = variance_dir + "/gender.bin";
            if ((f = std::fopen(ge.c_str(), "rb"))) {
                std::fread(in.gender.data(), 4, T, f);
                std::fclose(f);
            }
        }
        if (model.has_speed) {
            std::string ve = variance_dir + "/velocity.bin";
            if ((f = std::fopen(ve.c_str(), "rb"))) {
                std::fread(in.velocity.data(), 4, T, f);
                std::fclose(f);
            }
        }
    }

    in.mode = dump_cond ? ds::OutputMode::Cond : ds::OutputMode::Mel;
    in.depth = depth;
    in.sampling_steps_override = steps_override;
    in.algorithm_override = algorithm;
    in.seed = seed;
    in.spk_id = spk_id;

    if (!aux_mel_path.empty()) {
        FILE * f = std::fopen(aux_mel_path.c_str(), "rb");
        if (!f) { perror("aux-mel"); return 1; }
        in.aux_mel.resize((size_t)T * model.cfg.mel_bins);
        size_t r = std::fread(in.aux_mel.data(), sizeof(float), in.aux_mel.size(), f);
        std::fclose(f);
        if (r != in.aux_mel.size()) {
            fprintf(stderr, "[err] aux-mel: read %zu floats, expected %zu\n",
                    r, in.aux_mel.size());
            return 1;
        }
    }
    if (!noise_bin_path.empty()) {
        // Reuse the existing DSDIAG_NOISE pathway by setting the env var so the
        // runtime picks it up without further API surface.
#if defined(_WIN32)
        _putenv_s("DSDIAG_NOISE", noise_bin_path.c_str());
#else
        setenv("DSDIAG_NOISE", noise_bin_path.c_str(), 1);
#endif
    }

    ds::InferenceOutputs out;
    if (!ds::run_inference(model, in, out)) return 1;

    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) { perror("fopen"); return 1; }
    fwrite(out.data.data(), sizeof(float), out.data.size(), f);
    fclose(f);

    fprintf(stderr, "[ok] wrote %s : float32 [T=%d, dim=%d]  (%zu bytes)\n",
            out_path.c_str(), out.T, out.dim, out.data.size() * sizeof(float));
    return 0;
}
