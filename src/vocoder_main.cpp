#include "vocoder.h"
#include "runtime_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

static void usage() {
    fprintf(stderr,
        "usage: diffsinger_vocoder --model <vocoder.gguf> [--inspect]\n"
        "                         [--mel-bin <mel.f32> --f0-bin <f0.f32> --frames <T> --out <wav|f32>]\n"
        "                         [--mel-base10] [--mel-min X] [--mel-max X]\n"
        "                         [--seed N] [--noise-scale X] [--backend cpu|gpu|auto|cuda[:N]]\n"
        "\n"
        "mel-bin is raw float32 [T, num_mels] row-major. f0-bin is raw float32 [T] in Hz.\n"
        "Outputs PCM16 WAV when --out ends with .wav, otherwise raw float32 samples.\n");
}

static bool ends_with(const std::string & s, const char * suffix) {
    std::string e(suffix);
    return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
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

static void put_u16(std::ofstream & f, uint16_t v) {
    char b[2] = { char(v & 255), char((v >> 8) & 255) };
    f.write(b, 2);
}

static void put_u32(std::ofstream & f, uint32_t v) {
    char b[4] = {
        char(v & 255),
        char((v >> 8) & 255),
        char((v >> 16) & 255),
        char((v >> 24) & 255),
    };
    f.write(b, 4);
}

static bool write_wav_i16(const std::string & path, const std::vector<float> & wav, uint32_t sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[err] write failed: %s\n", path.c_str());
        return false;
    }
    const uint16_t channels = 1;
    const uint16_t bits = 16;
    const uint32_t data_bytes = (uint32_t)wav.size() * channels * (bits / 8);
    f.write("RIFF", 4);
    put_u32(f, 36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    put_u32(f, 16);
    put_u16(f, 1);
    put_u16(f, channels);
    put_u32(f, sr);
    put_u32(f, sr * channels * (bits / 8));
    put_u16(f, channels * (bits / 8));
    put_u16(f, bits);
    f.write("data", 4);
    put_u32(f, data_bytes);
    for (float x : wav) {
        if (!std::isfinite(x)) x = 0.0f;
        x = std::max(-1.0f, std::min(1.0f, x));
        int v = (int)std::lrintf(x * 32767.0f);
        put_u16(f, (uint16_t)(int16_t)v);
    }
    return (bool)f;
}

static bool write_f32_file(const std::string & path, const std::vector<float> & wav) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[err] write failed: %s\n", path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char *>(wav.data()), (std::streamsize)wav.size() * 4);
    return (bool)f;
}

static void print_tensor_shape(const nsv::Model & model, const char * name) {
    ggml_tensor * t = model.get(name, true);
    if (!t) {
        fprintf(stderr, "  %-40s missing\n", name);
        return;
    }
    fprintf(stderr, "  %-40s type=%d ne=[%lld,%lld,%lld,%lld]\n",
            name, (int)t->type,
            (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3]);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string mel_path;
    std::string f0_path;
    std::string out_path;
    int frames = 0;
    bool inspect = false;
    bool mel_base10 = false;
    bool has_mel_min = false;
    bool has_mel_max = false;
    float mel_min = 0.0f;
    float mel_max = 0.0f;
    uint32_t seed = 1234;
    float noise_scale = 1.0f;

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
        else if (a == "--mel-bin") mel_path = next();
        else if (a == "--f0-bin") f0_path = next();
        else if (a == "--frames") frames = std::atoi(next().c_str());
        else if (a == "--out") out_path = next();
        else if (a == "--inspect") inspect = true;
        else if (a == "--mel-base10") mel_base10 = true;
        else if (a == "--mel-min") { mel_min = (float)std::atof(next().c_str()); has_mel_min = true; }
        else if (a == "--mel-max") { mel_max = (float)std::atof(next().c_str()); has_mel_max = true; }
        else if (a == "--seed") seed = (uint32_t)std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--noise-scale") noise_scale = (float)std::atof(next().c_str());
        else if (a == "--backend") dsrt::set_backend_mode(next().c_str());
        else if (a == "--precision") {
            std::string m = next();
            if (m == "f16" || m == "fp16") dsrt::set_precision(dsrt::Precision::F16);
            else dsrt::set_precision(dsrt::Precision::F32);
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

    if (model_path.empty()) {
        usage();
        return 2;
    }

    nsv::Model model;
    if (!model.load(model_path)) return 1;

    if (inspect) {
        print_tensor_shape(model, "conv_pre.weight");
        print_tensor_shape(model, "ups.0.weight");
        print_tensor_shape(model, "ups.4.weight");
        print_tensor_shape(model, "source_conv.weight");
        print_tensor_shape(model, "resblocks.0.convs1.0.weight");
        print_tensor_shape(model, "resblocks.14.convs2.2.weight");
        print_tensor_shape(model, "conv_post.weight");
    }

    if (!mel_path.empty() || !f0_path.empty() || !out_path.empty()) {
        if (mel_path.empty() || f0_path.empty() || out_path.empty()) {
            fprintf(stderr, "[err] --mel-bin, --f0-bin and --out must be provided together\n");
            usage();
            return 2;
        }
        nsv::InferenceInputs in;
        in.frames = frames;
        in.mel_base10 = mel_base10;
        in.seed = seed;
        in.noise_scale = noise_scale;
        if (!read_f32_file(mel_path, in.mel)) return 1;
        if (has_mel_min || has_mel_max) {
            for (float & v : in.mel) {
                if (has_mel_min) v = std::max(v, mel_min);
                if (has_mel_max) v = std::min(v, mel_max);
            }
        }
        if (!read_f32_file(f0_path, in.f0)) return 1;
        if (in.frames == 0) in.frames = (int)in.f0.size();
        if (in.frames <= 0) {
            fprintf(stderr, "[err] --frames must be positive\n");
            return 2;
        }
        const size_t expect_mel = (size_t)in.frames * model.cfg.num_mels;
        if (in.mel.size() != expect_mel || in.f0.size() != (size_t)in.frames) {
            fprintf(stderr, "[err] expected mel=%zu floats and f0=%d floats, got mel=%zu f0=%zu\n",
                    expect_mel, in.frames, in.mel.size(), in.f0.size());
            return 2;
        }
        nsv::InferenceOutputs out;
        if (!nsv::run_vocoder(model, in, out)) return 1;
        bool ok = ends_with(out_path, ".wav")
            ? write_wav_i16(out_path, out.wav, model.cfg.sampling_rate)
            : write_f32_file(out_path, out.wav);
        if (!ok) return 1;
        fprintf(stderr, "[ok] wrote %s samples=%d sr=%u\n",
                out_path.c_str(), out.samples, model.cfg.sampling_rate);
    }

    return 0;
}
