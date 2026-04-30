"""
Convert the DiffSinger variance checkpoint used by this project to GGUF.

The current 251228_zhibin_club_variance_mix-ln checkpoint contains the
word-duration FastSpeech2 variance encoder, speaker embeddings, pitch
conditioning, and the rectified-flow predictor for breathiness/voicing/tension.
This converter maps stripped PyTorch tensor names to short stable GGUF names.
ggml's C GGUF reader rejects tensor names >= 64 bytes, while many PyTorch names
in this checkpoint are longer than that. Example mappings:

    fs2.encoder.layers.0.op.self_attn.in_proj.weight -> fs2.enc.0.attn.in_proj.weight
    variance_predictor.velocity_fn.output_projection.weight -> var.vf.out.weight
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gguf
except ImportError:  # pragma: no cover
    print("ERROR: `pip install gguf pyyaml numpy torch` first.", file=sys.stderr)
    raise


ARCH = "diffsinger-variance"


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


def require_variance_checkpoint(sd: dict[str, np.ndarray]):
    required = [
        "fs2.txt_embed.weight",
        "fs2.onset_embed.weight",
        "fs2.word_dur_embed.weight",
        "fs2.word_dur_embed.bias",
        "fs2.encoder.layer_norm.weight",
        "pitch_embed.weight",
        "pitch_embed.bias",
        "spk_embed.weight",
        "variance_predictor.velocity_fn.input_projection.weight",
    ]
    missing = [k for k in required if k not in sd]
    if missing:
        raise ValueError("Missing variance checkpoint tensors: " + ", ".join(missing))


def short_name(name: str) -> str:
    replacements = [
        ("fs2.encoder.layer_norm.", "fs2.enc.final_ln."),
        ("fs2.encoder.layers.", "fs2.enc."),
        ("variance_predictor.velocity_fn.", "var.vf."),
        ("residual_layers.", "r."),
        ("input_projection.", "in."),
        ("output_projection.", "out."),
        ("conditioner_projection.", "cond."),
        ("diffusion_embedding.", "diff."),
        ("variance_embeds.", "var_emb."),
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
    gw.add_string("general.name", "DiffSinger Variance")
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
    gw.add_string(f"{ARCH}.diffusion_type", str(cfg.get("diffusion_type", "reflow")))
    gw.add_uint32(f"{ARCH}.time_scale_factor", int(cfg.get("time_scale_factor", 1000)))
    gw.add_string(f"{ARCH}.sampling_algorithm", str(cfg.get("sampling_algorithm", "euler")))
    gw.add_uint32(f"{ARCH}.sampling_steps", int(cfg.get("sampling_steps", 20)))
    gw.add_uint32(f"{ARCH}.vocab_size", int(sd["fs2.txt_embed.weight"].shape[0]))
    gw.add_uint32(f"{ARCH}.num_spk", int(sd["spk_embed.weight"].shape[0]))
    gw.add_uint32(f"{ARCH}.num_lang", int(sd.get("fs2.lang_embed.weight", np.zeros((0,))).shape[0]))

    gw.add_bool(f"{ARCH}.predict_dur", bool(cfg.get("predict_dur", False)))
    for name in ("energy", "breathiness", "voicing", "tension"):
        gw.add_bool(f"{ARCH}.predict_{name}", bool(cfg.get(f"predict_{name}", False)))

    var_args = cfg.get("variances_prediction_args", {})
    var_bb = var_args.get("backbone_args", {})
    predicted_variances = [
        name for name in ("energy", "breathiness", "voicing", "tension")
        if bool(cfg.get(f"predict_{name}", False))
    ]
    repeat_bins = int(var_args.get("total_repeat_bins", 0))
    if predicted_variances:
        repeat_bins //= len(predicted_variances)
    gw.add_string(f"{ARCH}.variance.backbone_type", str(var_args.get("backbone_type", cfg.get("backbone_type", ""))))
    gw.add_uint32(f"{ARCH}.variance.repeat_bins", repeat_bins)
    gw.add_uint32(f"{ARCH}.variance.total_repeat_bins", int(var_args.get("total_repeat_bins", 0)))
    ranges = {
        "energy": (
            float(cfg.get("energy_db_min", -96.0)),
            float(cfg.get("energy_db_max", 0.0)),
            float(cfg.get("energy_db_min", -96.0)),
            0.0,
        ),
        "breathiness": (
            float(cfg.get("breathiness_db_min", -96.0)),
            float(cfg.get("breathiness_db_max", -20.0)),
            float(cfg.get("breathiness_db_min", -96.0)),
            0.0,
        ),
        "voicing": (
            float(cfg.get("voicing_db_min", -96.0)),
            float(cfg.get("voicing_db_max", -12.0)),
            float(cfg.get("voicing_db_min", -96.0)),
            0.0,
        ),
        "tension": (
            float(cfg.get("tension_logit_min", -10.0)),
            float(cfg.get("tension_logit_max", 10.0)),
            float(cfg.get("tension_logit_min", -10.0)),
            float(cfg.get("tension_logit_max", 10.0)),
        ),
    }
    gw.add_uint32(f"{ARCH}.variance.target_count", len(predicted_variances))
    for idx, name in enumerate(predicted_variances):
        norm_min, norm_max, clip_min, clip_max = ranges[name]
        gw.add_string(f"{ARCH}.variance.target.{idx}.name", name)
        gw.add_float32(f"{ARCH}.variance.target.{idx}.norm_min", norm_min)
        gw.add_float32(f"{ARCH}.variance.target.{idx}.norm_max", norm_max)
        gw.add_float32(f"{ARCH}.variance.target.{idx}.clip_min", clip_min)
        gw.add_float32(f"{ARCH}.variance.target.{idx}.clip_max", clip_max)
    gw.add_uint32(f"{ARCH}.variance.backbone.num_channels", int(var_bb.get("num_channels", 0)))
    gw.add_uint32(f"{ARCH}.variance.backbone.num_layers", int(var_bb.get("num_layers", 0)))
    variance_kernel = int(var_bb.get(
        "kernel_size",
        sd["variance_predictor.velocity_fn.residual_layers.0.net.2.weight"].shape[-1],
    ))
    gw.add_uint32(f"{ARCH}.variance.backbone.kernel_size", variance_kernel)
    gw.add_uint32(f"{ARCH}.variance.backbone.expansion_factor", int(var_bb.get("expansion_factor", 1)))
    gw.add_string(f"{ARCH}.variance.backbone.glu_type", str(var_bb.get("glu_type", "swiglu")))
    gw.add_bool(f"{ARCH}.variance.backbone.use_conditioner_cache", bool(var_bb.get("use_conditioner_cache", False)))

def add_tensor(gw: gguf.GGUFWriter, name: str, data: np.ndarray):
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    gw.add_tensor(short_name(name), np.ascontiguousarray(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    sd = load_state_dict(args.ckpt)
    require_variance_checkpoint(sd)

    if cfg.get("diffusion_type") != "reflow":
        raise ValueError("Only diffusion_type=reflow is supported.")
    for key in ("variances_prediction_args",):
        bt = cfg.get(key, {}).get("backbone_type", cfg.get("backbone_type"))
        if bt != "lynxnet2":
            raise ValueError(f"{key}.backbone_type must be lynxnet2 for this converter, got {bt!r}")

    gw = gguf.GGUFWriter(str(args.out), ARCH)
    add_scalar_metadata(gw, cfg, sd)
    for name in sorted(sd):
        add_tensor(gw, name, sd[name])
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    print(f"[ok] wrote {args.out}")
    print(f"[ok] tensors: {len(sd)}")


if __name__ == "__main__":
    main()
