#include "diffsinger.h"
#include "runtime_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ds {

Model::~Model() {
    if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
    if (ctx_w)          ggml_free(ctx_w);
    if (backend)        ggml_backend_free(backend);
}

ggml_tensor * Model::get(const std::string & name, bool optional) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        if (optional) return nullptr;
        fprintf(stderr, "[fatal] missing tensor '%s' in GGUF\n", name.c_str());
        std::abort();
    }
    return it->second;
}

static std::string get_str(gguf_context * gc, const char * key, const char * def = "") {
    int64_t id = gguf_find_key(gc, key);
    return id < 0 ? std::string(def) : std::string(gguf_get_val_str(gc, id));
}
static uint32_t get_u32(gguf_context * gc, const char * key, uint32_t def = 0) {
    int64_t id = gguf_find_key(gc, key);
    return id < 0 ? def : gguf_get_val_u32(gc, id);
}
static float get_f32(gguf_context * gc, const char * key, float def = 0.0f) {
    int64_t id = gguf_find_key(gc, key);
    return id < 0 ? def : gguf_get_val_f32(gc, id);
}
static bool get_bool(gguf_context * gc, const char * key, bool def = false) {
    int64_t id = gguf_find_key(gc, key);
    return id < 0 ? def : gguf_get_val_bool(gc, id);
}

bool Model::load(const std::string & path) {
    backend = dsrt::init_backend("acoustic");
    if (!backend) return false;

    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp { /*no_alloc=*/true, /*ctx=*/ &ctx_meta };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gp);
    if (!gctx) {
        fprintf(stderr, "[fatal] gguf_init_from_file failed: %s\n", path.c_str());
        return false;
    }

    cfg.hidden_size           = get_u32(gctx, "diffsinger-acoustic.hidden_size", 256);
    cfg.enc_layers            = get_u32(gctx, "diffsinger-acoustic.enc_layers", 4);
    cfg.num_heads             = get_u32(gctx, "diffsinger-acoustic.num_heads", 2);
    cfg.enc_ffn_kernel_size   = get_u32(gctx, "diffsinger-acoustic.enc_ffn_kernel_size", 3);
    cfg.ffn_act               = get_str(gctx, "diffsinger-acoustic.ffn_act", "gelu");
    cfg.use_rope              = get_bool(gctx,"diffsinger-acoustic.use_rope", true);
    cfg.rope_interleaved      = get_bool(gctx,"diffsinger-acoustic.rope_interleaved", false);
    cfg.use_mix_ln            = get_bool(gctx,"diffsinger-acoustic.use_mix_ln", false);
    cfg.mix_ln_layers.clear();
    for (uint32_t i = 0; i < cfg.enc_layers; ++i) {
        char key[96];
        snprintf(key, sizeof key, "diffsinger-acoustic.mix_ln_layer.%u", i);
        int64_t id = gguf_find_key(gctx, key);
        if (id >= 0) cfg.mix_ln_layers.push_back(gguf_get_val_u32(gctx, id));
    }
    cfg.use_stretch_embed     = get_bool(gctx,"diffsinger-acoustic.use_stretch_embed", false);
    cfg.use_spk_id            = get_bool(gctx,"diffsinger-acoustic.use_spk_id", false);
    cfg.num_spk               = get_u32(gctx, "diffsinger-acoustic.num_spk", 0);
    cfg.use_variance_scaling  = get_bool(gctx,"diffsinger-acoustic.use_variance_scaling", false);
    cfg.use_energy_embed      = get_bool(gctx,"diffsinger-acoustic.use_energy_embed", false);
    cfg.use_breathiness_embed = get_bool(gctx,"diffsinger-acoustic.use_breathiness_embed", false);
    cfg.use_voicing_embed     = get_bool(gctx,"diffsinger-acoustic.use_voicing_embed", false);
    cfg.use_tension_embed     = get_bool(gctx,"diffsinger-acoustic.use_tension_embed", false);
    cfg.use_key_shift_embed   = get_bool(gctx,"diffsinger-acoustic.use_key_shift_embed", false);
    cfg.use_speed_embed       = get_bool(gctx,"diffsinger-acoustic.use_speed_embed", false);
    cfg.vocab_size            = get_u32(gctx, "diffsinger-acoustic.vocab_size", 100);
    cfg.mel_bins              = get_u32(gctx, "diffsinger-acoustic.mel_bins", 128);

    cfg.diffusion_type        = get_str(gctx, "diffsinger-acoustic.diffusion_type", "reflow");
    cfg.time_scale_factor     = get_u32(gctx, "diffsinger-acoustic.time_scale_factor", 1000);
    cfg.sampling_algorithm    = get_str(gctx, "diffsinger-acoustic.sampling_algorithm", "euler");
    cfg.sampling_steps        = get_u32(gctx, "diffsinger-acoustic.sampling_steps", 20);
    cfg.t_start_infer         = get_f32(gctx, "diffsinger-acoustic.t_start_infer", 0.0f);
    cfg.use_variable_depth    = get_bool(gctx,"diffsinger-acoustic.use_variable_depth", false);
    cfg.max_depth             = get_f32(gctx, "diffsinger-acoustic.max_depth", 1.0f);

    cfg.backbone_type         = get_str(gctx, "diffsinger-acoustic.backbone_type", "lynxnet2");
    cfg.bb_num_channels       = get_u32(gctx, "diffsinger-acoustic.backbone.num_channels", 768);
    cfg.bb_num_layers         = get_u32(gctx, "diffsinger-acoustic.backbone.num_layers", 6);
    cfg.bb_kernel_size        = get_u32(gctx, "diffsinger-acoustic.backbone.kernel_size", 31);
    cfg.bb_expansion_factor   = get_u32(gctx, "diffsinger-acoustic.backbone.expansion_factor", 1);
    cfg.bb_glu_type           = get_str(gctx, "diffsinger-acoustic.backbone.glu_type", "atanglu");
    cfg.bb_use_conditioner_cache = get_bool(gctx,"diffsinger-acoustic.backbone.use_conditioner_cache", true);

    cfg.has_aux_decoder  = get_bool(gctx,"diffsinger-acoustic.has_aux_decoder", false);
    cfg.aux_num_channels = get_u32(gctx, "diffsinger-acoustic.aux.num_channels", 512);
    cfg.aux_num_layers   = get_u32(gctx, "diffsinger-acoustic.aux.num_layers", 6);
    cfg.aux_kernel_size  = get_u32(gctx, "diffsinger-acoustic.aux.kernel_size", 7);

    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    {
        ggml_init_params ip {
            ggml_tensor_overhead() * (size_t)(n_tensors + 16),
            nullptr, /*no_alloc*/ true,
        };
        ctx_w = ggml_init(ip);
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * src = ggml_get_tensor(ctx_meta, name);
        if (!src) continue;
        ggml_tensor * dst = ggml_dup_tensor(ctx_w, src);
        ggml_set_name(dst, name);
        tensors[name] = dst;
    }

    weights_buffer = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (!weights_buffer) {
        fprintf(stderr, "[fatal] backend buffer alloc failed\n");
        gguf_free(gctx);
        ggml_free(ctx_meta);
        return false;
    }

    {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "[fatal] reopen %s failed\n", path.c_str());
            return false;
        }
        const size_t data_off = gguf_get_data_offset(gctx);
        std::vector<uint8_t> buf;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            ggml_tensor * dst = tensors[name];
            size_t off = data_off + gguf_get_tensor_offset(gctx, i);
            size_t sz  = ggml_nbytes(dst);
            buf.resize(sz);
            fseek(f, (long)off, SEEK_SET);
            if (fread(buf.data(), 1, sz, f) != sz) {
                fprintf(stderr, "[fatal] read tensor '%s' failed\n", name);
                fclose(f);
                return false;
            }
            ggml_backend_tensor_set(dst, buf.data(), 0, sz);
        }
        fclose(f);
    }

    // Mirror spec_min/spec_max to host memory for fast denorm in the sampler.
    auto pull = [&](const char * name, std::vector<float> & out) {
        ggml_tensor * t = get(name);
        out.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    };
    pull("spec_min", spec_min);
    pull("spec_max", spec_max);

    has_lang_embed = tensors.count("fs2.lang_embed.weight") > 0;
    has_key_shift  = tensors.count("fs2.key_shift_embed.weight") > 0;
    has_speed      = tensors.count("fs2.speed_embed.weight") > 0;
    has_spk_embed  = tensors.count("fs2.spk_embed.weight") > 0;
    has_variance   = (tensors.count("fs2.breathiness.weight") > 0 ||
                      tensors.count("fs2.voicing.weight") > 0 ||
                      tensors.count("fs2.tension.weight") > 0 ||
                      tensors.count("fs2.energy.weight") > 0);

    gguf_free(gctx);
    ggml_free(ctx_meta);

    fprintf(stderr,
            "[load] %s  enc=%u/H=%u/heads=%u  bb=%s/L=%u/C=%u  mel=%u  steps=%u  diff=%s/%s\n",
            path.c_str(),
            cfg.enc_layers, cfg.hidden_size, cfg.num_heads,
            cfg.backbone_type.c_str(), cfg.bb_num_layers, cfg.bb_num_channels,
            cfg.mel_bins, cfg.sampling_steps,
            cfg.diffusion_type.c_str(), cfg.sampling_algorithm.c_str());
    fprintf(stderr,
            "[load] %zu tensors, %zu MB on backend\n",
            tensors.size(),
            ggml_backend_buffer_get_size(weights_buffer) / (1024 * 1024));
    return true;
}

} // namespace ds
