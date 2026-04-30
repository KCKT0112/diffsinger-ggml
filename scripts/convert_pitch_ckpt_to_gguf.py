"""
Convert the DiffSinger pitch checkpoint (260104_zhibin_club_pitch) to GGUF.

This checkpoint contains:
- FastSpeech2 encoder (phoneme encoder with ph_dur_embed)
- MelodyEncoder (note_midi + note_dur → transformer → out_proj)
- delta_pitch_embed, pitch_retake_embed
- PitchPredictor (lynxnet2 reflow, repeat_bins=96)
- Optional VariancePredictor (voicing, lynxnet2 reflow, total_repeat_bins=72)
- pitch_embed (for variance condition)
- Speaker embedding, language embedding

Tensor name compression: PyTorch names often exceed GGUF's 64-byte limit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gguf
except ImportError:
    print("ERROR: `pip install gguf pyyaml numpy torch` first.", file=sys.stderr)
    raise


ARCH = "diffsinger-pitch"


def load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = cfg.get("base_config")
    if base:
        if isinstance(base, str):
            base = [base]
        merged: dict[str, Any] = {}
        for b in base:
            bp = (path.parent / b).resolve() if not Path(b).is_absolute() else Path(b)
            if not bp.exists():
                bp = Path(__file__).resolve().parents[2] / b
            merged.update(load_yaml_config(bp))
        merged.update({k: v for k, v in cfg.items() if k != "base_config"})
        cfg = merged
    return cfg


def load_state_dict(ckpt_path: Path) -> dict[str, np.ndarray]:
    import torch

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    out = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v.detach().cpu().numpy()
    return out


def require_pitch_checkpoint(sd: dict[str, np.ndarray]):
    required = [
        "fs2.txt_embed.weight",
        "fs2.ph_dur_embed.weight",
        "fs2.ph_dur_embed.bias",
        "fs2.encoder.layer_norm.weight",
        "pitch_retake_embed.weight",
        "pitch_predictor.velocity_fn.input_projection.weight",
        "spk_embed.weight",
    ]
    missing = [k for k in required if k not in sd]
    if missing:
        raise ValueError("Missing pitch checkpoint tensors: " + ", ".join(missing))


# Tensors to skip (RoPE cached freqs are recomputed at runtime)
SKIP_PATTERNS = [
    "rotary_embed.cached_freqs",
    "rotary_embed.inv_freq",
]


def should_skip(name: str) -> bool:
    return any(pat in name for pat in SKIP_PATTERNS)


def short_name(name: str) -> str:
    replacements = [
        ("fs2.encoder.layer_norm.", "fs2.enc.final_ln."),
        ("fs2.encoder.layers.", "fs2.enc."),
        ("melody_encoder.encoder.layer_norm.", "mel.enc.final_ln."),
        ("melody_encoder.encoder.layers.", "mel.enc."),
        ("melody_encoder.note_midi_embed.", "mel.midi_embed."),
        ("melody_encoder.note_dur_embed.", "mel.dur_embed."),
        ("melody_encoder.out_proj.", "mel.out_proj."),
        ("pitch_predictor.velocity_fn.", "pitch.vf."),
        ("variance_predictor.velocity_fn.", "var.vf."),
        ("variance_embeds.", "var_emb."),
        ("residual_layers.", "r."),
        ("input_projection.", "in."),
        ("output_projection.", "out."),
        ("conditioner_projection.", "cond."),
        ("diffusion_embedding.", "diff."),
        (".op.layer_norm", ".ln"),
        (".op.self_attn.", ".attn."),
        (".op.ffn.ffn_", ".ffn"),
    ]
    out = name
    for src, dst in replacements:
        out = out.replace(src, dst)
    if len(out.encode("utf-8")) >= 64:
        raise ValueError(f"Short tensor name is still too long: {name} -> {out}")
    return out


def add_scalar_metadata(gw: gguf.GGUFWriter, cfg: dict[str, Any], sd: dict[str, np.ndarray]):
    gw.add_string("general.name", "DiffSinger Pitch")
    gw.add_uint32(f"{ARCH}.hidden_size", int(cfg["hidden_size"]))
    gw.add_uint32(f"{ARCH}.enc_layers", int(cfg["enc_layers"]))
    gw.add_uint32(f"{ARCH}.num_heads", int(cfg["num_heads"]))
    gw.add_uint32(f"{ARCH}.enc_ffn_kernel_size", int(cfg["enc_ffn_kernel_size"]))
    gw.add_string(f"{ARCH}.ffn_act", str(cfg["ffn_act"]))
    gw.add_bool(f"{ARCH}.use_rope", bool(cfg.get("use_rope", False)))
    gw.add_bool(f"{ARCH}.rope_interleaved", bool(cfg.get("rope_interleaved", True)))
    gw.add_bool(f"{ARCH}.use_lang_id", bool(cfg.get("use_lang_id", False)))
    gw.add_bool(f"{ARCH}.use_spk_id", bool(cfg.get("use_spk_id", False)))
    gw.add_bool(f"{ARCH}.use_variance_scaling", bool(cfg.get("use_variance_scaling", False)))
    gw.add_bool(f"{ARCH}.use_melody_encoder", bool(cfg.get("use_melody_encoder", False)))
    gw.add_string(f"{ARCH}.diffusion_type", str(cfg.get("diffusion_type", "reflow")))
    gw.add_uint32(f"{ARCH}.time_scale_factor", int(cfg.get("time_scale_factor", 1000)))
    gw.add_string(f"{ARCH}.sampling_algorithm", str(cfg.get("sampling_algorithm", "euler")))
    gw.add_uint32(f"{ARCH}.sampling_steps", int(cfg.get("sampling_steps", 20)))
    gw.add_uint32(f"{ARCH}.vocab_size", int(sd["fs2.txt_embed.weight"].shape[0]))
    gw.add_uint32(f"{ARCH}.num_spk", int(sd["spk_embed.weight"].shape[0]))
    gw.add_uint32(f"{ARCH}.num_lang", int(sd.get("fs2.lang_embed.weight", np.zeros((0,))).shape[0]))

    # Melody encoder params
    if cfg.get("use_melody_encoder", False):
        mel_args = cfg.get("melody_encoder_args", {})
        mel_hidden = mel_args.get("hidden_size", cfg.get("hidden_size", 128))
        mel_layers = mel_args.get("enc_layers", cfg.get("enc_layers", 6))
        gw.add_uint32(f"{ARCH}.melody.hidden_size", int(mel_hidden))
        gw.add_uint32(f"{ARCH}.melody.enc_layers", int(mel_layers))
        # Melody encoder inherits num_heads, ffn_kernel from top-level unless overridden
        gw.add_uint32(f"{ARCH}.melody.num_heads",
                      int(mel_args.get("num_heads", cfg.get("num_heads", 2))))
        gw.add_uint32(f"{ARCH}.melody.enc_ffn_kernel_size",
                      int(mel_args.get("enc_ffn_kernel_size", cfg.get("enc_ffn_kernel_size", 3))))

    # Pitch flow backbone
    pitch_args = cfg.get("pitch_prediction_args", {})
    pitch_bb = pitch_args.get("backbone_args", {})
    gw.add_uint32(f"{ARCH}.pitch.repeat_bins", int(pitch_args.get("repeat_bins", 96)))
    gw.add_float32(f"{ARCH}.pitch.pitd_norm_min", float(pitch_args.get("pitd_norm_min", -8.0)))
    gw.add_float32(f"{ARCH}.pitch.pitd_norm_max", float(pitch_args.get("pitd_norm_max", 8.0)))
    gw.add_float32(f"{ARCH}.pitch.pitd_clip_min", float(pitch_args.get("pitd_clip_min", -12.0)))
    gw.add_float32(f"{ARCH}.pitch.pitd_clip_max", float(pitch_args.get("pitd_clip_max", 12.0)))
    gw.add_string(f"{ARCH}.pitch.backbone_type",
                  str(pitch_args.get("backbone_type", cfg.get("backbone_type", "lynxnet2"))))
    gw.add_uint32(f"{ARCH}.pitch.backbone.num_channels", int(pitch_bb.get("num_channels", 512)))
    gw.add_uint32(f"{ARCH}.pitch.backbone.num_layers", int(pitch_bb.get("num_layers", 6)))
    # Infer kernel size from weights if not in config
    pitch_kernel = int(pitch_bb.get(
        "kernel_size",
        sd["pitch_predictor.velocity_fn.residual_layers.0.net.2.weight"].shape[-1],
    ))
    gw.add_uint32(f"{ARCH}.pitch.backbone.kernel_size", pitch_kernel)
    gw.add_string(f"{ARCH}.pitch.backbone.glu_type",
                  str(pitch_bb.get("glu_type", "atanglu")))
    gw.add_bool(f"{ARCH}.pitch.backbone.use_conditioner_cache",
                bool(pitch_bb.get("use_conditioner_cache", False)))

    # Variance (voicing) prediction
    predict_voicing = bool(cfg.get("predict_voicing", False))
    gw.add_bool(f"{ARCH}.predict_voicing", predict_voicing)
    if predict_voicing:
        var_args = cfg.get("variances_prediction_args", {})
        var_bb = var_args.get("backbone_args", {})
        gw.add_uint32(f"{ARCH}.variance.total_repeat_bins",
                      int(var_args.get("total_repeat_bins", 72)))
        gw.add_string(f"{ARCH}.variance.backbone_type",
                      str(var_args.get("backbone_type", cfg.get("backbone_type", "lynxnet2"))))
        gw.add_uint32(f"{ARCH}.variance.backbone.num_channels",
                      int(var_bb.get("num_channels", 512)))
        gw.add_uint32(f"{ARCH}.variance.backbone.num_layers",
                      int(var_bb.get("num_layers", 6)))
        variance_kernel = int(var_bb.get(
            "kernel_size",
            sd["variance_predictor.velocity_fn.residual_layers.0.net.2.weight"].shape[-1],
        ))
        gw.add_uint32(f"{ARCH}.variance.backbone.kernel_size", variance_kernel)
        gw.add_string(f"{ARCH}.variance.backbone.glu_type",
                      str(var_bb.get("glu_type", "atanglu")))
        gw.add_bool(f"{ARCH}.variance.backbone.use_conditioner_cache",
                    bool(var_bb.get("use_conditioner_cache", False)))
        # Voicing norm/clip ranges
        gw.add_float32(f"{ARCH}.variance.voicing_norm_min",
                       float(cfg.get("voicing_db_min", -96.0)))
        gw.add_float32(f"{ARCH}.variance.voicing_norm_max",
                       float(cfg.get("voicing_db_max", -12.0)))
        gw.add_float32(f"{ARCH}.variance.voicing_clip_min",
                       float(cfg.get("voicing_db_min", -96.0)))
        gw.add_float32(f"{ARCH}.variance.voicing_clip_max", 0.0)


def add_tensor(gw: gguf.GGUFWriter, name: str, data: np.ndarray):
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    gw.add_tensor(short_name(name), np.ascontiguousarray(data))


def main():
    ap = argparse.ArgumentParser(description="Convert DiffSinger pitch checkpoint to GGUF")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to .ckpt file")
    ap.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    ap.add_argument("--out", type=Path, required=True, help="Output .gguf path")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    sd = load_state_dict(args.ckpt)
    require_pitch_checkpoint(sd)

    if cfg.get("diffusion_type") != "reflow":
        raise ValueError("Only diffusion_type=reflow is supported.")

    # Filter out RoPE cached tensors
    sd_filtered = {k: v for k, v in sd.items() if not should_skip(k)}

    gw = gguf.GGUFWriter(str(args.out), ARCH)
    add_scalar_metadata(gw, cfg, sd)
    for name in sorted(sd_filtered):
        add_tensor(gw, name, sd_filtered[name])
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    print(f"[ok] wrote {args.out}")
    print(f"[ok] tensors: {len(sd_filtered)} (skipped {len(sd) - len(sd_filtered)} RoPE cache tensors)")


if __name__ == "__main__":
    main()
