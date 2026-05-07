# DiffSinger ggml Runtime

Standalone C++17/ggml inference for DiffSinger singing voice synthesis.
Supports variance, pitch, acoustic, and vocoder models with CPU, Metal, and CUDA backends.

## Assets

Converted GGUF files are expected under `assets/`:

```text
zhibin_variance.gguf     breathiness/voicing/tension variance model
zhibin_acoustic.gguf     acoustic RectifiedFlow + LYNXNet2 model
zhibin_pitch.gguf        pitch prediction (melody encoder + reflow)
pc_nsf_hifigan.gguf      pitch-controllable NSF-HiFiGAN vocoder
phonemes.json            phoneme-to-id map (acoustic/variance, 151 entries)
pitch_phonemes.json      phoneme-to-id map (pitch model, 74 vocab with merges)
```

Regenerate from checkpoints:

```bash
# Variance
python3 scripts/convert_variance_ckpt_to_gguf.py \
  --ckpt ../ckpt/251228_zhibin_club_variance_mix-ln/model_ckpt_steps_32000.ckpt \
  --config ../ckpt/251228_zhibin_club_variance_mix-ln/config.yaml \
  --out assets/zhibin_variance.gguf

# Acoustic
python3 scripts/convert_ckpt_to_gguf.py \
  --ckpt ../ckpt/251228_zhibin_club_acoustic_mix-ln/model_ckpt_steps_64000.ckpt \
  --config ../ckpt/251228_zhibin_club_acoustic_mix-ln/config.yaml \
  --out assets/zhibin_acoustic.gguf

# Pitch
python3 scripts/convert_pitch_ckpt_to_gguf.py \
  --ckpt ../ckpt/260104_zhibin_club_pitch/model_ckpt_steps_62000.ckpt \
  --config ../ckpt/260104_zhibin_club_pitch/config.yaml \
  --out assets/zhibin_pitch.gguf

# Vocoder
python3 scripts/convert_nsf_hifigan_ckpt_to_gguf.py \
  --ckpt ../ckpt/260104_zhibin_club_pitch/pc-nsf-hifigan.ckpt \
  --out assets/pc_nsf_hifigan.gguf
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CUDA builds require an NVIDIA CUDA Toolkit visible to CMake:

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
```

The same option can be used without presets:

```bash
cmake -S . -B build/cuda-release -DCMAKE_BUILD_TYPE=Release -DDSGGML_CUDA=ON
cmake --build build/cuda-release -j
```

## Full Pipeline (`diffsinger_pipeline`)

The fused C++ pipeline loads all models once and processes `.ds` files end-to-end:

```bash
./build/diffsinger_pipeline \
  --variance-model assets/zhibin_variance.gguf \
  --acoustic-model assets/zhibin_acoustic.gguf \
  --vocoder-model assets/pc_nsf_hifigan.gguf \
  --ds ../samples/06_不谓侠.ds \
  --phonemes assets/phonemes.json \
  --spk-name zhibin-base --spk-map ../ckpt/251228_zhibin_club_acoustic_mix-ln/spk_map.json \
  --steps 5 --algorithm midpoint \
  --out output.wav
```

### Pitch Model (auto F0 prediction)

When a pitch model is provided, the pipeline predicts expressive F0 from note sequences
instead of using the `.ds` file's `f0_seq`:

```bash
./build/diffsinger_pipeline \
  --pitch-model assets/zhibin_pitch.gguf \
  --pitch-phonemes assets/pitch_phonemes.json \
  --variance-model assets/zhibin_variance.gguf \
  --acoustic-model assets/zhibin_acoustic.gguf \
  --vocoder-model assets/pc_nsf_hifigan.gguf \
  --ds ../samples/06_不谓侠.ds \
  --phonemes assets/phonemes.json \
  --spk-name zhibin-base --spk-map ../ckpt/260104_zhibin_club_pitch/spk_map.json \
  --auto-pitch \
  --steps 5 --algorithm midpoint \
  --out output_autopitch.wav
```

- `--pitch-model` loads the pitch GGUF (optional; without it F0 comes from the .ds file)
- `--pitch-phonemes` provides the pitch model's phoneme map (required when pitch model has different vocabulary)
- `--auto-pitch` forces pitch prediction even when the .ds file has `f0_seq`

### Pipeline Options

| Flag | Description |
|------|-------------|
| `--steps N` | ODE sampling steps (default: model's training steps) |
| `--algorithm euler\|midpoint\|rk4` | Sampler (midpoint recommended) |
| `--seed N` | RNG seed for diffusion noise |
| `--spk-id N` / `--spk-name NAME` | Speaker selection |
| `--spk-map PATH` | Speaker name→id JSON (required with `--spk-name`) |
| `--predict-all-variances` | Overwrite existing variance curves from .ds |
| `--precision f32\|f16` | Weight precision (default: f32, see below) |
| `--backend cpu\|gpu\|auto\|cuda[:N]` | Global compute backend |
| `--variance-backend cpu\|gpu\|auto\|cuda[:N]` | Per-component backend override |
| `--acoustic-backend cpu\|gpu\|auto\|cuda[:N]` | |
| `--vocoder-backend cpu\|gpu\|auto\|cuda[:N]` | |
| `--pitch-backend cpu\|gpu\|auto\|cuda[:N]` | |

### Precision (`--precision`)

`--precision f16` converts F32 weight matrices to F16 at load time, halving model memory.

**Important:** Diffusion/reflow models (acoustic, variance, pitch) are precision-sensitive.
ODE sampling accumulates F16 rounding errors over multiple steps, destroying output quality.
The pipeline automatically keeps diffusion model weights at F32 and only applies F16 to
the vocoder (a single-pass feed-forward network where F16 is lossless, corr > 0.99999).

In practice, vocoder-only F16 saves ~27 MB — the benefit is modest since the vocoder is
already the smallest and fastest component. The GGUF files should remain F32 for distribution.

### Backend Notes

- **CPU** (default): stable, fast on Apple Silicon (~1.8s per segment with midpoint-5)
- **GPU** (Metal/CUDA): all ops use ggml's registered GPU backend when available.
- **CUDA**: build with `-DDSGGML_CUDA=ON` or the `cuda-release` preset, then run with
  `--backend cuda` for device 0 or `--backend cuda:N` for device N. `--backend gpu`
  also selects the first registered discrete GPU. CUDA builds apply local ggml
  patches for long-output `IM2COL` and ConvTranspose1D, so the NSF-HiFiGAN
  vocoder runs on CUDA too.
- **Metal**: all ops are Metal-native, but ~38s cold JIT compilation cost.
  Only worthwhile for long-running servers that amortize the JIT.
  Recommended: keep vocoder on CPU regardless.

### Recommended Settings

Production quality with good speed:
```
--steps 5 --algorithm midpoint
```

## Python Wrapper (`sing_ggml.py`)

Higher-level Python script that invokes the C++ tools:

```bash
# Quick demo (generates a tiny test score)
python3 scripts/sing_ggml.py --demo --steps 2 --out /tmp/demo.wav

# Single segment from .ds file
python3 scripts/sing_ggml.py \
  --ds-file ../samples/03_撒娇八连.ds --ds-segment 0 \
  --spk-name zhibin-pop --steps 5 --out /tmp/sample.wav

# Full .ds file (all segments, offset-stitched)
python3 scripts/sing_ggml.py \
  --ds-file ../samples/06_不谓侠.ds --ds-all \
  --spk-name zhibin-pop --backend cpu --out /tmp/full.wav
```

## Individual Tools

Inspect model metadata:
```bash
./build/diffsinger_variance --model assets/zhibin_variance.gguf --inspect
./build/diffsinger_vocoder --model assets/pc_nsf_hifigan.gguf --inspect
```

Run acoustic from pre-computed inputs:
```bash
./build/diffsinger_acoustic \
  --model assets/zhibin_acoustic.gguf \
  --len-tokens <L> \
  --tokens-bin tokens.i32 \
  --mel2ph-bin mel2ph.i32 \
  --f0-bin f0_hz.f32 \
  --variance-dir variance_dir \
  --spk-id 0 \
  --steps 5 --algorithm midpoint \
  --out mel.f32
```

Run vocoder from mel:
```bash
./build/diffsinger_vocoder \
  --model assets/pc_nsf_hifigan.gguf \
  --mel-bin mel.f32 \
  --f0-bin f0_hz.f32 \
  --frames <T> \
  --out out.wav
```

## Performance (Apple Silicon M4, CPU, F32, 4 threads)

| Component | T=333 frames | Notes |
|-----------|-------------|-------|
| Pitch (midpoint-5) | ~2.5s | FS2 encoder + melody encoder + reflow |
| Variance (midpoint-5) | ~0.8s | breathiness + voicing + tension |
| Acoustic (midpoint-5) | ~1.8s | shallow diffusion with aux decoder |
| Vocoder | ~0.3s | NSF-HiFiGAN |
| **Total per segment** | **~5.4s** | For ~3.9s of audio (T=333) |

Set thread count: `DSGGML_THREADS=N` (default: 4).

## Scripts

| Script | Purpose |
|--------|---------|
| `convert_ckpt_to_gguf.py` | Acoustic checkpoint → GGUF |
| `convert_variance_ckpt_to_gguf.py` | Variance checkpoint → GGUF |
| `convert_pitch_ckpt_to_gguf.py` | Pitch checkpoint → GGUF |
| `convert_nsf_hifigan_ckpt_to_gguf.py` | NSF-HiFiGAN vocoder → GGUF |
| `sing_ggml.py` | Python pipeline wrapper |
| `dump_pt_reference.py` | Diagnostic: dump PyTorch acoustic reference tensors |
| `dump_pt_variance.py` | Diagnostic: dump PyTorch variance reference + noise |
| `dump_pt_vocoder.py` | Diagnostic: compare PT vs ggml vocoder output |

## Diagnostic Environment Variables

For debugging/validation against PyTorch:

| Variable | Target | Description |
|----------|--------|-------------|
| `DSDIAG_NOISE` | acoustic | Load fixed noise tensor |
| `DSDIAG_COND` | acoustic | Load fixed encoder condition |
| `DSDIAG_TAP` | acoustic | Replace velocity output with named intermediate |
| `DSDIAG_VEL_OUT` / `DSDIAG_VEL_STEP` | acoustic | Dump velocity at step k |
| `DSDIAG_TRAJ_DIR` | acoustic | Dump x_t trajectory |
| `DSDIAG_AUX_OUT` / `DSDIAG_AUX_TAP` | acoustic | Dump aux decoder intermediates |
| `DSDIAG_VAR_NOISE` | variance | Load fixed variance noise |
| `DSDIAG_VAR_RAW_OUT` | variance | Dump raw post-flow output |
| `DSDIAG_PITCH_NOISE` | pitch | Load fixed pitch noise |
| `DSGGML_THREADS` | all | Override CPU thread count |
| `DSGGML_BACKEND_*` | all | Per-component backend (VARIANCE/ACOUSTIC/VOCODER/PITCH), supports `cpu`, `gpu`, `auto`, and `cuda[:N]` |
