#!/usr/bin/env python3
"""Dump PyTorch reference for the variance model on one DS segment.

Outputs (raw float32) into --out-dir:
  variance_pt/{breathiness,voicing,tension,energy}.bin   (whichever the model predicts)
  variance_pt/noise.bin   the deterministic noise injected into the variance reflow

Use the matching --noise-bin via DSDIAG_VAR_NOISE if/when added to the C++ runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def note_to_midi_int(note: str) -> int:
    if note.lower() == "rest":
        return 0
    names = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    base = names[note[0].upper()]
    pos = 1
    if pos < len(note) and note[pos] in "#b":
        base += 1 if note[pos] == "#" else -1
        pos += 1
    octave = int(note[pos:])
    return (octave + 1) * 12 + base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="variance checkpoint dir (with config.yaml + dictionary-*.txt)")
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--ref-dir", type=Path, required=True,
                    help="reference dir with tokens.bin, mel2ph.bin, f0_hz.bin produced by dump_pt_reference.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--spk-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "variance_pt").mkdir(exist_ok=True)

    cfg_path = args.ckpt_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["work_dir"] = str(args.ckpt_dir)

    from utils.hparams import hparams as hp
    hp.update(cfg)
    hp["work_dir"] = str(args.ckpt_dir)
    hp["infer"] = True
    hp["sampling_algorithm"] = cfg.get("sampling_algorithm", "euler")
    hp["sampling_steps"] = cfg.get("sampling_steps", 20)

    from utils.phoneme_utils import PhonemeDictionary
    dicts = {lang: args.ckpt_dir / f"dictionary-{lang}.txt" for lang in cfg["dictionaries"]}
    pd = PhonemeDictionary(
        dictionaries=dicts,
        extra_phonemes=cfg.get("extra_phonemes"),
        merged_groups=cfg.get("merged_phoneme_groups"),
    )
    vocab_size = len(pd)

    from modules.toplevel import DiffSingerVariance
    model = DiffSingerVariance(vocab_size=vocab_size)
    ckpt_files = sorted(args.ckpt_dir.glob("model_ckpt_steps_*.ckpt"))
    if not ckpt_files:
        raise SystemExit(f"no checkpoint in {args.ckpt_dir}")
    ckpt_path = ckpt_files[-1]
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("state_dict", sd)
    cleaned = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[ckpt] {ckpt_path.name} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    seg = json.loads(args.ds.read_text(encoding="utf-8"))[args.segment]
    phones = seg["ph_seq"].split()
    ph_num = [int(x) for x in seg["ph_num"].split()]
    note_seq = seg["note_seq"].split()
    note_dur_s = [float(x) for x in seg["note_dur"].split()]

    # Phone-level midi: for each phone, midi of its note (0 if rest)
    phone_note = np.repeat(np.arange(len(note_seq), dtype=np.int64),
                           np.asarray(ph_num, dtype=np.int64))
    midi = np.asarray([note_to_midi_int(note_seq[i]) for i in phone_note], dtype=np.int64)
    # Phone -> word (1-based)
    ph2word = np.repeat(np.arange(1, len(ph_num) + 1, dtype=np.int64),
                        np.asarray(ph_num, dtype=np.int64))

    # Reuse tokens and mel2ph from acoustic ref (built with the SAME phoneme dict)
    tokens = np.fromfile(args.ref_dir / "tokens.bin", dtype=np.int32).astype(np.int64)
    mel2ph = np.fromfile(args.ref_dir / "mel2ph.bin", dtype=np.int32).astype(np.int64)
    f0_hz = np.fromfile(args.ref_dir / "f0_hz.bin", dtype=np.float32)
    T = int(mel2ph.size)
    L = int(tokens.size)
    assert len(phones) == L, f"phone count mismatch {len(phones)} vs {L}"

    # word_dur in frames per word
    sr = int(cfg.get("audio_sample_rate", 44100))
    hop = int(cfg.get("hop_size", 512))
    timestep = hop / sr
    ph_dur_s = [float(x) for x in seg["ph_dur"].split()]
    accum = np.cumsum(np.asarray(ph_dur_s, dtype=np.float64))
    boundaries = np.rint(accum / timestep + 0.5).astype(np.int64)
    ph_frames = np.diff(boundaries, prepend=np.array([0], dtype=np.int64))
    word_dur = np.zeros(len(ph_num), dtype=np.int64)
    np.add.at(word_dur, ph2word - 1, ph_frames.astype(np.int64))

    # base pitch from f0
    voiced = f0_hz > 0
    pitch = np.zeros(T, dtype=np.float32)
    pitch[voiced] = 69.0 + 12.0 * np.log2(np.maximum(f0_hz[voiced], 1e-6) / 440.0)
    if not voiced.all() and voiced.any():
        idx = np.flatnonzero(~voiced)
        ref_idx = np.flatnonzero(voiced)
        pitch[~voiced] = np.interp(idx, ref_idx, pitch[voiced])

    # Tensor inputs
    txt_t = torch.from_numpy(tokens.astype(np.int64))[None]
    midi_t = torch.from_numpy(midi.astype(np.int64))[None]
    ph2word_t = torch.from_numpy(ph2word.astype(np.int64))[None]
    ph_dur_t = torch.from_numpy(ph_frames.astype(np.int64))[None]
    word_dur_t = torch.from_numpy(word_dur.astype(np.int64))[None]
    mel2ph_t = torch.from_numpy(mel2ph.astype(np.int64))[None]
    pitch_t = torch.from_numpy(pitch.astype(np.float32))[None]
    base_pitch_t = pitch_t.clone()
    spk_id_t = torch.tensor([args.spk_id], dtype=torch.long)

    # Need to inject deterministic noise into reflow.
    # MultiVarianceRectifiedFlow.inference uses torch.randn(b, num_feats, M, T_cond_or_T_x)
    # where the last two dims come from RepetitiveRectifiedFlow.norm_spec(unsqueeze repeat).
    # The actual shape is determined by var_cond[..., -1] = T frames and out_dims = repeat_bins.
    # We monkey-patch torch.randn to capture the requested shape, save it, and reuse.
    orig_randn = torch.randn
    saved = {}
    def _patched_randn(*shape, **kw):
        out = orig_randn(*shape, **kw)
        saved.setdefault('noise', out.clone())
        return out
    torch.randn = _patched_randn
    try:
        with torch.no_grad():
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            dur_pred, pitch_pred, var_pred = model(
                txt_tokens=txt_t,
                midi=midi_t,
                ph2word=ph2word_t,
                ph_dur=ph_dur_t,
                word_dur=word_dur_t,
                mel2ph=mel2ph_t,
                base_pitch=base_pitch_t,
                pitch=pitch_t,
                spk_id=spk_id_t if hp.get("use_spk_id") else None,
                infer=True,
            )
    finally:
        torch.randn = orig_randn

    # Save predicted variances
    if isinstance(var_pred, dict):
        for name, t in var_pred.items():
            arr = t[0].cpu().numpy().astype(np.float32)
            arr.tofile(out / "variance_pt" / f"{name}.bin")
            print(f"  {name}: T={arr.size} range=[{arr.min():.3f},{arr.max():.3f}] mean={arr.mean():.3f}")
    if 'noise' in saved:
        nb = saved['noise'].cpu().numpy().astype(np.float32)
        print(f"  variance noise shape: {nb.shape}")
        nb.tofile(out / "variance_pt" / "noise.bin")

    print(f"[ok] wrote variance PT reference to {out / 'variance_pt'}")


if __name__ == "__main__":
    main()
