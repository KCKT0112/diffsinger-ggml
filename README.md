# DiffSinger ggml Runtime

This directory is the ggml-only inference path for the local DiffSinger
checkpoints.

## Assets

Converted GGUF files are expected under `assets/`:

```text
zhibin_variance.gguf     breathiness/voicing/tension variance model
zhibin_acoustic.gguf     acoustic RectifiedFlow + LYNXNet2 model
pc_nsf_hifigan.gguf      pitch-controllable NSF-HiFiGAN vocoder
```

Regenerate them from checkpoints with:

```bash
python3 scripts/convert_variance_ckpt_to_gguf.py \
  --ckpt ../ckpt/251228_zhibin_club_variance_mix-ln/model_ckpt_steps_32000.ckpt \
  --config ../ckpt/251228_zhibin_club_variance_mix-ln/config.yaml \
  --out assets/zhibin_variance.gguf

python3 scripts/convert_ckpt_to_gguf.py \
  --ckpt ../ckpt/251228_zhibin_club_acoustic_mix-ln/model_ckpt_steps_64000.ckpt \
  --config ../ckpt/251228_zhibin_club_acoustic_mix-ln/config.yaml \
  --out assets/zhibin_acoustic.gguf

python3 scripts/convert_nsf_hifigan_ckpt_to_gguf.py \
  --ckpt ../ckpt/260104_zhibin_club_pitch/pc-nsf-hifigan.ckpt \
  --out assets/pc_nsf_hifigan.gguf
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## End-To-End Synthesis

`scripts/sing_ggml.py` runs:

```text
variance -> breathiness/voicing/tension
acoustic -> mel
vocoder -> wav
```

Smoke test with a tiny generated score:

```bash
python3 scripts/sing_ggml.py --demo --steps 2 --out /tmp/ds_ggml_demo.wav
```

Run one segment from a DiffSinger `.ds` file. This path uses the `ph_dur`,
`ph_num`, and `f0_seq` stored in the DS file, predicts missing variance curves,
then runs acoustic + vocoder:

```bash
python3 scripts/sing_ggml.py \
  --ds-file ../samples/03_撒娇八连.ds \
  --ds-segment 0 \
  --spk-name zhibin-pop \
  --steps 2 \
  --out /tmp/ds_sample03.wav
```

Run and offset-stitch the whole `.ds` file:

```bash
python3 scripts/sing_ggml.py \
  --ds-file ../samples/06_不谓侠.ds \
  --ds-all \
  --spk-name zhibin-pop \
  --backend cpu \
  --out /tmp/ds_06_full.wav
```

For `.ds` input, `build/diffsinger_pipeline` loads variance, acoustic, and
vocoder GGUF files once and processes the generated segment manifest.
Because these checkpoints are multi-speaker and the DS files do not identify a
speaker, DS inference requires `--spk-name` or `--spk-id`.

`--backend cpu|gpu|auto` is forwarded to the C++ runtime. `cpu` is the stable
default; `gpu` requires a registered ggml GPU backend and fails loudly if one
cannot be initialized, while `auto` falls back to CPU. The vocoder now applies
the checkpoint `noise_sigma`; use `--vocoder-noise-scale 0` only for
deterministic numerical comparisons.
By default, the variance stage only fills curves missing from a segment; use
`--predict-all-variances` to overwrite existing manual variance curves for A/B
checks against a forced-prediction workflow.

Real score inputs use raw binary files:

```bash
python3 scripts/sing_ggml.py \
  --tokens-bin tokens.i32 \
  --ph2word-bin ph2word.i32 \
  --word-dur-bin word_dur.f32 \
  --mel2ph-bin mel2ph.i32 \
  --f0-bin f0_hz.f32 \
  --lang-bin lang.i32 \
  --phones <L> \
  --spk-id 0 \
  --steps 20 \
  --out out.wav
```

`ph2word` and `mel2ph` are 1-based index maps with 0 reserved for padding,
matching DiffSinger. If `--pitch-bin` is omitted, the script derives frame-level
MIDI pitch from `--f0-bin`. If `--lang-bin` is omitted, the script writes
all-zero language ids.

## Individual Tools

```bash
./build/diffsinger_variance --model assets/zhibin_variance.gguf --inspect
./build/diffsinger_vocoder --model assets/pc_nsf_hifigan.gguf --inspect
```

Acoustic can emit mel directly from score-aligned phone inputs:

```bash
./build/diffsinger_acoustic \
  --model assets/zhibin_acoustic.gguf \
  --len-tokens <L> \
  --tokens-bin tokens.i32 \
  --mel2ph-bin mel2ph.i32 \
  --f0-bin f0_hz.f32 \
  --variance-dir variance_dir \
  --spk-id 0 \
  --out mel.f32
```

The vocoder expects raw float32 mel `[T, 128]` and f0 Hz `[T]`:

```bash
./build/diffsinger_vocoder \
  --model assets/pc_nsf_hifigan.gguf \
  --mel-bin mel.f32 \
  --f0-bin f0_hz.f32 \
  --frames <T> \
  --out out.wav
```
