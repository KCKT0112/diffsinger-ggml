#!/usr/bin/env python3
"""Compare PT NSF-HiFiGAN vocoder vs ggml vocoder on the same mel input.

Inputs (raw float32 [T, M]):
  --mel mel.bin   the mel that both vocoders consume

Outputs:
  --pt-out wav   PT-rendered waveform (float32, sr from ckpt)
  --ggml-out wav ggml-rendered waveform (run separately and pass)

Reports per-frequency-band power difference between PT and ggml waveforms.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder-ckpt", type=Path, required=True,
                    help="path to pc-nsf-hifigan.ckpt or .pt")
    ap.add_argument("--mel", type=Path, required=True)
    ap.add_argument("--f0", type=Path, required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mel-base", default="e", help="'e' or '10' (DiffSinger ckpts use 'e')")
    ap.add_argument("--noise-scale", type=float, default=0.0,
                    help="multiply noise_sigma by this; 0 to make deterministic")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from utils.hparams import hparams as hp
    hp["vocoder_ckpt"] = str(args.vocoder_ckpt)
    hp["audio_sample_rate"] = 44100
    hp["audio_num_mel_bins"] = args.M
    hp["fft_size"] = 2048
    hp["win_size"] = 2048
    hp["hop_size"] = 512
    hp["fmin"] = 40
    hp["fmax"] = 16000
    hp["mel_base"] = args.mel_base

    from modules.vocoders.nsf_hifigan import NsfHifiGAN
    voc = NsfHifiGAN()
    if args.noise_scale != 1.0:
        voc.model.noise_sigma = (voc.model.noise_sigma or 0.0) * float(args.noise_scale)
        print(f"[noise] noise_sigma override -> {voc.model.noise_sigma}")
    voc.to_device(torch.device("cpu"))

    mel = np.fromfile(args.mel, dtype=np.float32).reshape(args.T, args.M)
    f0 = np.fromfile(args.f0, dtype=np.float32)
    assert f0.size == args.T, f"f0 size {f0.size} != T {args.T}"

    torch.manual_seed(1234)
    wav = voc.spec2wav(mel, f0=f0)
    wav.astype(np.float32).tofile(args.out)
    sr = voc.h.sampling_rate
    print(f"[ok] PT vocoder -> {args.out}  N={wav.size} sr={sr} range=[{wav.min():.3f},{wav.max():.3f}]")


if __name__ == "__main__":
    main()
