"""
Convert the current DiffSinger acoustic checkpoint (Rectified Flow + LYNXNet2)
to a single GGUF file consumable by the ggml C++ runtime in this directory.

Usage:
    python convert_ckpt_to_gguf.py \
        --ckpt path/to/model.ckpt \
        --config path/to/config.yaml \
        --out  acoustic.gguf

Supported scope:
 - Diffusion type:        reflow
 - Backbone:              lynxnet2
 - Acoustic checkpoints with the tensor naming used by DiffSinger acoustic task
 - Mix layer norm, stretch embedding, speaker/variance/key-shift/speed embeds,
   and ConvNeXt aux decoder used by the 251228 zhibin club acoustic checkpoint
 - n_feats == 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf  # provided by the `gguf` PyPI package
except ImportError:  # pragma: no cover
    print("ERROR: `pip install gguf pyyaml numpy torch` first.", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "hidden_size": 256,
    "enc_layers": 4,
    "num_heads": 2,
    "enc_ffn_kernel_size": 3,
    "ffn_act": "gelu",
    "use_pos_embed": True,
    "use_rope": True,
    "rope_interleaved": False,
    "rel_pos": True,
    "vocab_size": 100,             # set from dictionary in real run
    "audio_num_mel_bins": 128,
    "spec_min": [-12.0],
    "spec_max": [0.0],
    "diffusion_type": "reflow",
    "time_scale_factor": 1000,
    "sampling_algorithm": "euler",
    "sampling_steps": 20,
    "T_start_infer": 0.0,
    "use_shallow_diffusion": False,
    "backbone_type": "lynxnet2",
    "backbone_args": {
        "num_channels": 768,
        "num_layers": 6,
        "kernel_size": 31,
        "expansion_factor": 1,
        "glu_type": "atanglu",
        "use_conditioner_cache": True,
        "dropout_rate": 0.0,
    },
}


def load_yaml_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # walk base_config chain (DiffSinger style)
    base = cfg.get("base_config")
    if base:
        if isinstance(base, str):
            base = [base]
        merged = {}
        for b in base:
            bp = (path.parent / b).resolve() if not Path(b).is_absolute() else Path(b)
            if not bp.exists():
                # configs are relative to repo root
                bp = (Path(__file__).resolve().parents[2] / b)
            merged.update(load_yaml_config(bp))
        merged.update({k: v for k, v in cfg.items() if k != "base_config"})
        cfg = merged
    return cfg


def merge_config(user_cfg: dict | None) -> dict:
    cfg = {**DEFAULT_CONFIG}
    if user_cfg:
        cfg.update(user_cfg)
        if "backbone_args" in user_cfg:
            cfg["backbone_args"] = {
                **DEFAULT_CONFIG["backbone_args"],
                **user_cfg["backbone_args"],
            }
    return cfg


# ---------------------------------------------------------------------------
# state_dict helpers
# ---------------------------------------------------------------------------

def _strip_prefix(sd: dict, prefix: str) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def load_state_dict(ckpt_path: Path) -> dict:
    import torch
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    # DiffSinger Lightning checkpoints prefix everything with "model."
    if any(k.startswith("model.") for k in sd):
        sd = _strip_prefix(sd, "model.")
    # numpy view
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def validate_acoustic_state_dict(sd: dict):
    required = [
        "fs2.txt_embed.weight",
        "fs2.dur_embed.weight",
        "fs2.dur_embed.bias",
        "fs2.pitch_embed.weight",
        "fs2.pitch_embed.bias",
        "diffusion.velocity_fn.input_projection.weight",
        "diffusion.velocity_fn.output_projection.weight",
    ]
    missing = [k for k in required if k not in sd]
    if not missing:
        return

    looks_like_variance = any(k.startswith("variance_predictor.") for k in sd)
    if looks_like_variance:
        raise ValueError(
            "This checkpoint looks like a variance checkpoint, not an "
            "acoustic checkpoint. Missing acoustic tensors include: "
            + ", ".join(missing[:4])
        )
    raise ValueError("Missing acoustic tensors: " + ", ".join(missing))


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------

ARCH = "diffsinger-acoustic"


class Writer:
    def __init__(self, path: Path, cfg: dict, quant_mode: str = "f32"):
        self.cfg = cfg
        self.gw = gguf.GGUFWriter(str(path), ARCH)
        self.quant_mode = quant_mode
        self._add_metadata()

    def _add_metadata(self):
        c = self.cfg
        gw = self.gw
        # --- general ---
        gw.add_string("general.architecture", ARCH)
        gw.add_string("general.name", "DiffSinger Acoustic")
        # --- model dims ---
        gw.add_uint32(f"{ARCH}.hidden_size", c["hidden_size"])
        gw.add_uint32(f"{ARCH}.enc_layers", c["enc_layers"])
        gw.add_uint32(f"{ARCH}.num_heads", c["num_heads"])
        gw.add_uint32(f"{ARCH}.enc_ffn_kernel_size", c["enc_ffn_kernel_size"])
        gw.add_string(f"{ARCH}.ffn_act", c["ffn_act"])
        gw.add_bool(f"{ARCH}.use_rope", c["use_rope"])
        gw.add_bool(f"{ARCH}.rope_interleaved", c["rope_interleaved"])
        gw.add_bool(f"{ARCH}.use_mix_ln", bool(c.get("use_mix_ln", False)))
        for i, layer in enumerate(c.get("mix_ln_layer", [])):
            gw.add_uint32(f"{ARCH}.mix_ln_layer.{i}", int(layer))
        gw.add_bool(f"{ARCH}.use_stretch_embed", bool(c.get("use_stretch_embed", False)))
        gw.add_bool(f"{ARCH}.use_spk_id", bool(c.get("use_spk_id", False)))
        gw.add_uint32(f"{ARCH}.num_spk", int(c.get("num_spk", 0)))
        gw.add_bool(f"{ARCH}.use_variance_scaling", bool(c.get("use_variance_scaling", False)))
        for name in ("energy", "breathiness", "voicing", "tension"):
            gw.add_bool(f"{ARCH}.use_{name}_embed", bool(c.get(f"use_{name}_embed", False)))
        gw.add_bool(f"{ARCH}.use_key_shift_embed", bool(c.get("use_key_shift_embed", False)))
        gw.add_bool(f"{ARCH}.use_speed_embed", bool(c.get("use_speed_embed", False)))
        gw.add_uint32(f"{ARCH}.vocab_size", c["vocab_size"])
        gw.add_uint32(f"{ARCH}.mel_bins", c["audio_num_mel_bins"])
        # --- diffusion ---
        gw.add_string(f"{ARCH}.diffusion_type", c["diffusion_type"])
        gw.add_uint32(f"{ARCH}.time_scale_factor", c["time_scale_factor"])
        gw.add_string(f"{ARCH}.sampling_algorithm", c["sampling_algorithm"])
        gw.add_uint32(f"{ARCH}.sampling_steps", c["sampling_steps"])
        gw.add_float32(f"{ARCH}.t_start_infer", float(c.get("T_start_infer", 0.0)))
        gw.add_bool(f"{ARCH}.use_variable_depth", bool(c.get("use_variable_depth", False)))
        gw.add_float32(f"{ARCH}.max_depth", float(c.get("max_depth", 1.0)))
        # spec normalization
        smin = np.array(c["spec_min"], dtype=np.float32)
        smax = np.array(c["spec_max"], dtype=np.float32)
        if smin.size == 1:
            smin = np.full(c["audio_num_mel_bins"], float(smin[0]), dtype=np.float32)
        if smax.size == 1:
            smax = np.full(c["audio_num_mel_bins"], float(smax[0]), dtype=np.float32)
        # --- backbone ---
        bb = c["backbone_args"]
        gw.add_string(f"{ARCH}.backbone_type", c["backbone_type"])
        gw.add_uint32(f"{ARCH}.backbone.num_channels", bb["num_channels"])
        gw.add_uint32(f"{ARCH}.backbone.num_layers", bb["num_layers"])
        gw.add_uint32(f"{ARCH}.backbone.kernel_size", bb.get("kernel_size", 3))
        gw.add_uint32(f"{ARCH}.backbone.expansion_factor", bb.get("expansion_factor", 1))
        gw.add_string(f"{ARCH}.backbone.glu_type", bb.get("glu_type", "atanglu"))
        gw.add_bool(f"{ARCH}.backbone.use_conditioner_cache", bool(bb.get("use_conditioner_cache", False)))

        # store spec_min / spec_max as tensors so we can read them at inference time
        self._pending_extra = {"spec_min": smin, "spec_max": smax}

    def add_tensor(self, name: str, data: np.ndarray):
        """Write a tensor. Large weight matrices may be stored in F16/BF16
        to shrink the file; small/numerically-sensitive tensors (biases, LN,
        embeddings, spec_min/max) stay F32. Quantization controlled by
        `Writer.quant_mode`:
            - "f32":  everything stays F32 (lossless, largest output)
            - "f16":  Conv kernels & Linear weights -> F16 (~130MB; large
                      relative error on lynxnet velocity_fn — only safe for
                      encoder/aux subnets in practice)
            - "bf16": same but BF16 — preserves F32 dynamic range, much
                      better behavior in deep diffusion loops
        """
        from gguf import GGMLQuantizationType
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        data = np.ascontiguousarray(data)

        mode = getattr(self, "quant_mode", "f32")
        do_quant = False
        if mode in ("f16", "bf16"):
            if data.ndim >= 2 and data.size >= 4096 and \
               name not in ("spec_min", "spec_max"):
                do_quant = True
            if name.endswith("txt_embed.weight") or name.endswith("lang_embed.weight"):
                do_quant = False
            if name == "fs2.enc.pe":
                do_quant = False

        if do_quant and mode == "f16":
            data16 = data.astype(np.float16)
            self.gw.add_tensor(name, data16, raw_dtype=GGMLQuantizationType.F16)
        elif do_quant and mode == "bf16":
            # numpy lacks bfloat16; emulate by zeroing low 16 bits of float32.
            u32 = data.view(np.uint32)
            bf16_u16 = (u32 >> 16).astype(np.uint16)
            self.gw.add_tensor(name, bf16_u16, raw_dtype=GGMLQuantizationType.BF16)
        else:
            self.gw.add_tensor(name, data)

    def finalize(self):
        for k, v in self._pending_extra.items():
            self.add_tensor(k, v)
        self.gw.write_header_to_file()
        self.gw.write_kv_data_to_file()
        self.gw.write_tensors_to_file()
        self.gw.close()


# ---------------------------------------------------------------------------
# Tensor mapping: PyTorch state_dict name -> GGUF tensor name + permutation
# ---------------------------------------------------------------------------
# We use stable names in the GGUF that the C++ side will look up directly.
#
# Conventions:
#   linear/embed weights:    saved as PyTorch order [out, in]
#   conv1d weights:          saved as [out, in/groups, k]  (PyTorch native)
#   layernorm:               weight + bias, both [dim]
#
# All tensors are written as float32 to keep the C++ loader simple.


def _emit_fastspeech2_encoder(w: Writer, sd: dict, cfg: dict) -> set[str]:
    used: set[str] = set()
    H = cfg["hidden_size"]
    NL = cfg["enc_layers"]

    # text embed
    name = "fs2.txt_embed.weight"
    w.add_tensor("fs2.txt_embed.weight", sd[name])
    used.add(name)

    # dur embed (Linear 1->H)
    w.add_tensor("fs2.dur_embed.weight", sd["fs2.dur_embed.weight"])
    w.add_tensor("fs2.dur_embed.bias",   sd["fs2.dur_embed.bias"])
    used.update({"fs2.dur_embed.weight", "fs2.dur_embed.bias"})

    # pitch embed (Linear 1->H)
    w.add_tensor("fs2.pitch_embed.weight", sd["fs2.pitch_embed.weight"])
    w.add_tensor("fs2.pitch_embed.bias",   sd["fs2.pitch_embed.bias"])
    used.update({"fs2.pitch_embed.weight", "fs2.pitch_embed.bias"})

    # ---- optional embeddings: copy if present so the C++ side can read them
    # using `model.get(name, optional=True)`.
    optional_linear = {
        "fs2.key_shift_embed": "fs2.key_shift_embed",
        "fs2.speed_embed":     "fs2.speed_embed",
        "fs2.variance_embeds.breathiness": "fs2.breathiness",
        "fs2.variance_embeds.voicing":     "fs2.voicing",
        "fs2.variance_embeds.tension":     "fs2.tension",
        "fs2.variance_embeds.energy":      "fs2.energy",
    }
    for src_prefix, dst_prefix in optional_linear.items():
        for suf in ("weight", "bias"):
            k = f"{src_prefix}.{suf}"
            if k in sd:
                w.add_tensor(f"{dst_prefix}.{suf}", sd[k])
                used.add(k)

    # language embed lookup table (optional)
    if "fs2.lang_embed.weight" in sd:
        w.add_tensor("fs2.lang_embed.weight", sd["fs2.lang_embed.weight"])
        used.add("fs2.lang_embed.weight")

    if "fs2.spk_embed.weight" in sd:
        w.add_tensor("fs2.spk_embed.weight", sd["fs2.spk_embed.weight"])
        used.add("fs2.spk_embed.weight")

    for src, dst in [
        ("fs2.stretch_embed.1.weight", "fs2.stretch_embed.1.weight"),
        ("fs2.stretch_embed.1.bias", "fs2.stretch_embed.1.bias"),
        ("fs2.stretch_embed.3.weight", "fs2.stretch_embed.3.weight"),
        ("fs2.stretch_embed.3.bias", "fs2.stretch_embed.3.bias"),
        ("fs2.stretch_embed_rnn.weight_ih_l0", "fs2.stretch_rnn.weight_ih"),
        ("fs2.stretch_embed_rnn.weight_hh_l0", "fs2.stretch_rnn.weight_hh"),
        ("fs2.stretch_embed_rnn.bias_ih_l0", "fs2.stretch_rnn.bias_ih"),
        ("fs2.stretch_embed_rnn.bias_hh_l0", "fs2.stretch_rnn.bias_hh"),
    ]:
        if src in sd:
            w.add_tensor(dst, sd[src])
            used.add(src)

    # transformer layers
    for i in range(NL):
        base = f"fs2.encoder.layers.{i}.op"

        for ln_idx in (1, 2):
            affine_w = f"{base}.layer_norm{ln_idx}.affine.weight"
            affine_b = f"{base}.layer_norm{ln_idx}.affine.bias"
            if affine_w in sd:
                w.add_tensor(f"fs2.enc.{i}.ln{ln_idx}.affine.weight", sd[affine_w])
                w.add_tensor(f"fs2.enc.{i}.ln{ln_idx}.affine.bias", sd[affine_b])
                used.update({affine_w, affine_b})
            else:
                for suf in ("weight", "bias"):
                    k = f"{base}.layer_norm{ln_idx}.{suf}"
                    w.add_tensor(f"fs2.enc.{i}.ln{ln_idx}.{suf}", sd[k])
                    used.add(k)

        # attention (RoPE branch -> in_proj / out_proj; native MHA branch -> in_proj_weight / out_proj.weight)
        # Accept either spelling used by PyTorch checkpoints.
        in_proj_keys = (f"{base}.self_attn.in_proj.weight",
                        f"{base}.self_attn.in_proj_weight")
        wk = next((k for k in in_proj_keys if k in sd), None)
        if wk is None:
            raise KeyError(in_proj_keys[0])
        w.add_tensor(f"fs2.enc.{i}.attn.in_proj.weight", sd[wk]); used.add(wk)
        ok = f"{base}.self_attn.out_proj.weight"
        w.add_tensor(f"fs2.enc.{i}.attn.out_proj.weight", sd[ok]); used.add(ok)
        # bias for out_proj is optional (some configs disable bias)
        for opt_b in (f"{base}.self_attn.out_proj.bias",):
            if opt_b in sd:
                w.add_tensor(f"fs2.enc.{i}.attn.out_proj.bias", sd[opt_b]); used.add(opt_b)
        for opt_b in (f"{base}.self_attn.in_proj.bias", f"{base}.self_attn.in_proj_bias"):
            if opt_b in sd:
                w.add_tensor(f"fs2.enc.{i}.attn.in_proj.bias", sd[opt_b]); used.add(opt_b)

        # ffn: ffn_1 conv (H -> 4H or 8H), ffn_2 linear (4H -> H)
        f1w = f"{base}.ffn.ffn_1.weight"; f1b = f"{base}.ffn.ffn_1.bias"
        f2w = f"{base}.ffn.ffn_2.weight"; f2b = f"{base}.ffn.ffn_2.bias"
        w.add_tensor(f"fs2.enc.{i}.ffn1.weight", sd[f1w]); used.add(f1w)
        w.add_tensor(f"fs2.enc.{i}.ffn1.bias",   sd[f1b]); used.add(f1b)
        w.add_tensor(f"fs2.enc.{i}.ffn2.weight", sd[f2w]); used.add(f2w)
        w.add_tensor(f"fs2.enc.{i}.ffn2.bias",   sd[f2b]); used.add(f2b)

    # final layer norm
    w.add_tensor("fs2.enc.final_ln.weight", sd["fs2.encoder.layer_norm.weight"]); used.add("fs2.encoder.layer_norm.weight")
    w.add_tensor("fs2.enc.final_ln.bias",   sd["fs2.encoder.layer_norm.bias"]);   used.add("fs2.encoder.layer_norm.bias")

    return used


def _emit_lynxnet2(w: Writer, sd: dict, cfg: dict) -> set[str]:
    used: set[str] = set()
    bb = cfg["backbone_args"]
    NL = bb["num_layers"]
    base = "diffusion.velocity_fn"

    for src, dst in [
        (f"{base}.input_projection.weight",  "bb.in.weight"),
        (f"{base}.input_projection.bias",    "bb.in.bias"),
        (f"{base}.conditioner_projection.weight", "bb.cond.weight"),
        (f"{base}.conditioner_projection.bias",   "bb.cond.bias"),
        (f"{base}.output_projection.weight", "bb.out.weight"),
        (f"{base}.output_projection.bias",   "bb.out.bias"),
        (f"{base}.norm.weight", "bb.norm.weight"),
        (f"{base}.norm.bias",   "bb.norm.bias"),
        (f"{base}.diffusion_embedding.1.weight", "bb.diff.1.weight"),
        (f"{base}.diffusion_embedding.1.bias",   "bb.diff.1.bias"),
        (f"{base}.diffusion_embedding.3.weight", "bb.diff.3.weight"),
        (f"{base}.diffusion_embedding.3.bias",   "bb.diff.3.bias"),
    ]:
        w.add_tensor(dst, sd[src]); used.add(src)

    for i in range(NL):
        rb = f"{base}.residual_layers.{i}"
        for src, dst in [
            (f"{rb}.net.0.weight", f"bb.r.{i}.net.0.weight"),
            (f"{rb}.net.0.bias",   f"bb.r.{i}.net.0.bias"),
            (f"{rb}.net.2.weight", f"bb.r.{i}.net.2.weight"),
            (f"{rb}.net.2.bias",   f"bb.r.{i}.net.2.bias"),
            (f"{rb}.net.4.weight", f"bb.r.{i}.net.4.weight"),
            (f"{rb}.net.4.bias",   f"bb.r.{i}.net.4.bias"),
            (f"{rb}.net.6.weight", f"bb.r.{i}.net.6.weight"),
            (f"{rb}.net.6.bias",   f"bb.r.{i}.net.6.bias"),
            (f"{rb}.net.8.weight", f"bb.r.{i}.net.8.weight"),
            (f"{rb}.net.8.bias",   f"bb.r.{i}.net.8.bias"),
        ]:
            w.add_tensor(dst, sd[src]); used.add(src)
    return used


def _emit_aux_decoder(w: Writer, sd: dict, cfg: dict) -> set[str]:
    """Optional ConvNeXt aux decoder (used by shallow diffusion as x_end).
    Returns empty set if no aux decoder weights are present.
    Auto-detects num_layers / num_channels / kernel_size from inconv weight."""
    used: set[str] = set()
    in_w_key = "aux_decoder.decoder.inconv.weight"
    if in_w_key not in sd:
        return used
    in_w = sd[in_w_key]                # [C, in_dims=H, K]
    C  = int(in_w.shape[0])
    K  = int(in_w.shape[2])
    # detect num_layers: scan for conv.{i}.dwconv.weight
    NL = 0
    while f"aux_decoder.decoder.conv.{NL}.dwconv.weight" in sd:
        NL += 1
    cfg_aux = cfg.setdefault("aux_decoder", {})
    cfg_aux["num_channels"] = C
    cfg_aux["num_layers"]   = NL
    cfg_aux["kernel_size"]  = K

    for src, dst in [
        ("aux_decoder.decoder.inconv.weight",  "aux.input_proj.weight"),
        ("aux_decoder.decoder.inconv.bias",    "aux.input_proj.bias"),
        ("aux_decoder.decoder.outconv.weight", "aux.output_proj.weight"),
        ("aux_decoder.decoder.outconv.bias",   "aux.output_proj.bias"),
    ]:
        w.add_tensor(dst, sd[src]); used.add(src)

    for i in range(NL):
        rb = f"aux_decoder.decoder.conv.{i}"
        for src, dst in [
            (f"{rb}.dwconv.weight",  f"aux.layer.{i}.dw.weight"),
            (f"{rb}.dwconv.bias",    f"aux.layer.{i}.dw.bias"),
            (f"{rb}.norm.weight",    f"aux.layer.{i}.ln.weight"),
            (f"{rb}.norm.bias",      f"aux.layer.{i}.ln.bias"),
            (f"{rb}.pwconv1.weight", f"aux.layer.{i}.pw1.weight"),
            (f"{rb}.pwconv1.bias",   f"aux.layer.{i}.pw1.bias"),
            (f"{rb}.pwconv2.weight", f"aux.layer.{i}.pw2.weight"),
            (f"{rb}.pwconv2.bias",   f"aux.layer.{i}.pw2.bias"),
        ]:
            w.add_tensor(dst, sd[src]); used.add(src)
        gamma_key = f"{rb}.gamma"
        if gamma_key in sd:
            w.add_tensor(f"aux.layer.{i}.gamma", sd[gamma_key])
            used.add(gamma_key)
    # Emit aux_decoder metadata directly via the underlying gguf writer.
    w.gw.add_bool("diffsinger-acoustic.has_aux_decoder", True)
    w.gw.add_uint32("diffsinger-acoustic.aux.num_channels", C)
    w.gw.add_uint32("diffsinger-acoustic.aux.num_layers",   NL)
    w.gw.add_uint32("diffsinger-acoustic.aux.kernel_size",  K)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quant", choices=["f32", "f16", "bf16"], default="f32",
                    help="weight precision. F16 may have large rel error on "
                         "deep loops; BF16 keeps F32 dynamic range.")
    args = ap.parse_args()

    if not args.ckpt or not args.config:
        ap.error("--ckpt and --config are required")
    user_cfg = load_yaml_config(args.config)
    cfg = merge_config(user_cfg)
    sd = load_state_dict(args.ckpt)
    validate_acoustic_state_dict(sd)
    cfg["vocab_size"] = int(sd["fs2.txt_embed.weight"].shape[0])

    bt = cfg["backbone_type"]
    if bt != "lynxnet2":
        raise ValueError(f"Unsupported backbone_type: {bt}; this ggml runtime is scoped to lynxnet2")
    if cfg["diffusion_type"] != "reflow":
        raise ValueError("Only diffusion_type=reflow is supported.")

    w = Writer(args.out, cfg, quant_mode=args.quant)
    used = _emit_fastspeech2_encoder(w, sd, cfg)
    used |= _emit_lynxnet2(w, sd, cfg)
    used |= _emit_aux_decoder(w, sd, cfg)
    w.finalize()

    unused = sorted(set(sd.keys()) - used)
    if unused:
        print(f"[info] {len(unused)} state_dict tensors were ignored "
              f"(speaker / variance / aux decoder / etc.)")
        for k in unused[:10]:
            print(f"        - {k}")
        if len(unused) > 10:
            print(f"        ... (+{len(unused)-10} more)")

    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
