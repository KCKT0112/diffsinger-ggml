#!/usr/bin/env python3
"""Dump PyTorch reference for one DS segment to bisect ggml/PyTorch divergence.

Outputs (raw float32, row-major) into --out-dir:
  tokens.bin       int32  [T_txt]
  mel2ph.bin       int32  [T]
  f0_hz.bin        float  [T]
  variance/{breathiness,voicing,tension,energy}.bin float [T] (if used)
  variance/key_shift.bin float [T]   (zero unless gender curve)
  variance/velocity.bin  float [T]   (default 1.0)
  cond_pt.bin      float  [T, H]      (after fs2.encoder)
  cond_pt_full.bin float  [T, H]      (after stretch+spk+pitch+variance embeds)
  aux_mel_pt.bin   float  [T, M]      (aux_decoder denorm output)
  mel_pt.bin       float  [T, M]      (final reflow mel)
  meta.json        sizes, hidden, seed, etc.

Run from repo root:
  python ggml_acoustic/scripts/dump_pt_reference.py \
      --ckpt-dir ckpt/251228_zhibin_club_acoustic_mix-ln \
      --ds samples/06_不谓侠.ds --segment 0 \
      --out-dir /tmp/dsref --spk-id 0 --steps 20 --seed 1234
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load_hparams(ckpt_dir: Path) -> dict:
    cfg_path = ckpt_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # rewrite dictionaries to local copies so hparams.workdir isn't required
    cfg["work_dir"] = str(ckpt_dir)
    return cfg


def build_phoneme_dict(cfg: dict, ckpt_dir: Path):
    from utils.phoneme_utils import PhonemeDictionary
    dicts = {}
    for lang in cfg["dictionaries"]:
        p = ckpt_dir / f"dictionary-{lang}.txt"
        if not p.exists():
            p = REPO / cfg["dictionaries"][lang]
        dicts[lang] = p
    return PhonemeDictionary(
        dictionaries=dicts,
        extra_phonemes=cfg.get("extra_phonemes"),
        merged_groups=cfg.get("merged_phoneme_groups"),
    )


def parse_ds_segment(seg: dict, pd, lang: str, sr: int, hop: int):
    phones = seg["ph_seq"].split()
    ph_dur_s = [float(x) for x in seg["ph_dur"].split()]
    timestep = hop / sr
    accum = np.cumsum(np.asarray(ph_dur_s, dtype=np.float64))
    boundaries = np.rint(accum / timestep + 0.5).astype(np.int32)
    ph_frames = np.diff(boundaries, prepend=np.array([0], dtype=np.int32)).astype(np.int32)
    tokens = np.asarray([pd.encode_one(p, lang=lang) for p in phones], dtype=np.int64)
    mel2ph = np.repeat(np.arange(1, len(tokens) + 1, dtype=np.int64), ph_frames)

    if "f0_seq" in seg:
        f0_src = np.asarray([float(x) for x in seg["f0_seq"].split()], dtype=np.float32)
        step = float(seg.get("f0_timestep", 0.005))
        T = int(mel2ph.size)
        # resample to frame rate
        if f0_src.size == 0:
            f0 = np.zeros(T, dtype=np.float32)
        elif f0_src.size == 1:
            f0 = np.full(T, f0_src[0], dtype=np.float32)
        else:
            t_max = (f0_src.size - 1) * step
            curve = np.interp(np.arange(0, t_max, hop / sr),
                              step * np.arange(f0_src.size), f0_src).astype(np.float32)
            if curve.size < T:
                curve = np.concatenate((curve, np.full(T - curve.size, curve[-1], dtype=np.float32)))
            else:
                curve = curve[:T]
            f0 = curve
    else:
        raise SystemExit("segment has no f0_seq; not supported")

    return tokens, mel2ph, ph_frames, f0


def resample_curve(seg: dict, key: str, T: int, sr: int, hop: int) -> Optional[np.ndarray]:
    if key not in seg or seg.get(key) is None:
        return None
    ts_key = f"{key}_timestep"
    if ts_key not in seg:
        return None
    values = np.asarray([float(x) for x in str(seg[key]).split()], dtype=np.float32)
    step = float(seg[ts_key])
    if values.size == 0:
        return np.zeros(T, dtype=np.float32)
    if values.size == 1:
        return np.full(T, values[0], dtype=np.float32)
    t_max = (values.size - 1) * step
    curve = np.interp(np.arange(0, t_max, hop / sr),
                      step * np.arange(values.size), values).astype(np.float32)
    if curve.size < T:
        curve = np.concatenate((curve, np.full(T - curve.size, curve[-1], dtype=np.float32)))
    return curve[:T].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--spk-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "variance").mkdir(exist_ok=True)

    cfg = load_hparams(args.ckpt_dir)
    sr = int(cfg.get("audio_sample_rate", 44100))
    hop = int(cfg.get("hop_size", 512))

    # Late imports so utils.hparams is bound to our cfg.
    from utils.hparams import hparams as hp
    hp.update(cfg)
    hp["work_dir"] = str(args.ckpt_dir)
    hp["infer"] = True
    hp["sampling_algorithm"] = cfg.get("sampling_algorithm", "euler")
    hp["sampling_steps"] = args.steps
    pd = build_phoneme_dict(cfg, args.ckpt_dir)
    vocab_size = len(pd)

    from modules.toplevel import DiffSingerAcoustic
    model = DiffSingerAcoustic(vocab_size=vocab_size, out_dims=int(cfg["audio_num_mel_bins"]))
    ckpt_files = sorted(args.ckpt_dir.glob("model_ckpt_steps_*.ckpt"))
    if not ckpt_files:
        raise SystemExit(f"no model_ckpt_steps_*.ckpt in {args.ckpt_dir}")
    ckpt_path = ckpt_files[-1]
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("state_dict", sd)
    # state keys are 'model.fs2.txt_embed.weight' style; strip 'model.' prefix.
    cleaned = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[ckpt] {ckpt_path.name} missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print(" missing:", missing[:5])
    if unexpected[:5]:
        print(" unexpected:", unexpected[:5])
    model.eval()

    data = json.loads(args.ds.read_text(encoding="utf-8"))
    seg = data[args.segment]
    tokens, mel2ph, ph_frames, f0 = parse_ds_segment(seg, pd, args.lang, sr, hop)
    T = int(mel2ph.size)
    H = int(cfg["hidden_size"])
    M = int(cfg["audio_num_mel_bins"])

    breathiness = resample_curve(seg, "breathiness", T, sr, hop)
    voicing = resample_curve(seg, "voicing", T, sr, hop)
    tension = resample_curve(seg, "tension", T, sr, hop)
    energy = resample_curve(seg, "energy", T, sr, hop)
    # Pad missing variances with zeros so the same inputs can be sent to ggml.
    if cfg.get("use_breathiness_embed") and breathiness is None:
        print("[ref] breathiness missing in DS, using zeros"); breathiness = np.zeros(T, dtype=np.float32)
    if cfg.get("use_voicing_embed") and voicing is None:
        print("[ref] voicing missing in DS, using zeros"); voicing = np.zeros(T, dtype=np.float32)
    if cfg.get("use_tension_embed") and tension is None:
        print("[ref] tension missing in DS, using zeros"); tension = np.zeros(T, dtype=np.float32)
    if cfg.get("use_energy_embed") and energy is None:
        print("[ref] energy missing in DS, using zeros"); energy = np.zeros(T, dtype=np.float32)
    # Save score data
    tokens.astype(np.int32).tofile(out / "tokens.bin")
    mel2ph.astype(np.int32).tofile(out / "mel2ph.bin")
    f0.astype(np.float32).tofile(out / "f0_hz.bin")

    def save_var(name, arr):
        if arr is None:
            return
        arr.astype(np.float32).tofile(out / "variance" / f"{name}.bin")

    save_var("breathiness", breathiness)
    save_var("voicing", voicing)
    save_var("tension", tension)
    save_var("energy", energy)

    key_shift = np.zeros(T, dtype=np.float32)
    speed = np.ones(T, dtype=np.float32)
    save_var("key_shift", key_shift)
    save_var("velocity", speed)

    # Run the encoder by hand to capture intermediates.
    txt_tokens = torch.from_numpy(tokens.astype(np.int64))[None]
    mel2ph_t = torch.from_numpy(mel2ph.astype(np.int64))[None]
    f0_t = torch.from_numpy(f0.astype(np.float32))[None]
    spk_id_t = torch.tensor([args.spk_id], dtype=torch.long)

    extra = {}
    if breathiness is not None:
        extra["breathiness"] = torch.from_numpy(breathiness.astype(np.float32))[None]
    if voicing is not None:
        extra["voicing"] = torch.from_numpy(voicing.astype(np.float32))[None]
    if tension is not None:
        extra["tension"] = torch.from_numpy(tension.astype(np.float32))[None]
    if energy is not None:
        extra["energy"] = torch.from_numpy(energy.astype(np.float32))[None]

    with torch.no_grad():
        condition = model.fs2(
            txt_tokens, mel2ph_t, f0_t,
            key_shift=torch.from_numpy(key_shift.astype(np.float32))[None] if cfg.get("use_key_shift_embed") else None,
            speed=torch.from_numpy(speed.astype(np.float32))[None] if cfg.get("use_speed_embed") else None,
            spk_embed_id=spk_id_t if cfg.get("use_spk_id") else None,
            languages=None,
            **extra,
        )
        cond_full = condition[0].cpu().numpy().astype(np.float32)  # [T, H]
        cond_full.tofile(out / "cond_pt_full.bin")

        # Aux mel
        aux_mel = None
        if model.use_shallow_diffusion:
            aux = model.aux_decoder(condition, infer=True)
            aux_mel = aux[0].cpu().numpy().astype(np.float32)
            aux_mel.tofile(out / "aux_mel_pt.bin")

        # Reflow with deterministic injected noise so PT and ggml runs are
        # bit-identical at the noise input.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        noise = torch.randn(1, 1, M, T)  # [B, F, M, T] — matches reflow.inference layout
        # Save in [T, M] row-major like ggml expects via DSDIAG_NOISE.
        noise[0, 0].cpu().numpy().T.astype(np.float32).tofile(out / "noise.bin")
        # Monkey-patch torch.randn so reflow.inference picks up our noise tensor.
        orig_randn = torch.randn
        def _patched_randn(*shape, **kw):
            if shape == (1, 1, M, T):
                return noise.clone()
            return orig_randn(*shape, **kw)
        torch.randn = _patched_randn
        try:
            mel = model.diffusion(condition,
                                  src_spec=torch.from_numpy(aux_mel)[None] if aux_mel is not None else None,
                                  infer=True)
        finally:
            torch.randn = orig_randn
        mel_np = mel[0].cpu().numpy().astype(np.float32)
        mel_np.tofile(out / "mel_pt.bin")

    meta = {
        "T": T, "H": H, "M": M,
        "vocab_size": vocab_size,
        "phone_count": int(tokens.size),
        "spk_id": args.spk_id,
        "seed": args.seed,
        "steps": args.steps,
        "ph_dur_frames": ph_frames.tolist(),
        "ckpt": str(ckpt_path),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] wrote PT reference to {out}")
    print(f"  T={T} H={H} M={M} phones={tokens.size}")
    if aux_mel is not None:
        print(f"  aux_mel range [{aux_mel.min():.3f}, {aux_mel.max():.3f}]")
    print(f"  mel range [{mel_np.min():.3f}, {mel_np.max():.3f}]")


if __name__ == "__main__":
    main()
