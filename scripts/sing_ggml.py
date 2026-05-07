#!/usr/bin/env python3
"""Run the ggml-only DiffSinger variance -> acoustic -> vocoder pipeline.

The score input format is the raw binary format used by the C++ tools:
  tokens:     int32 [phones]
  ph2word:    int32 [phones], 1-based word index
  word_dur:   float32 [words], frame durations
  mel2ph:     int32 [frames], 1-based phone index
  f0_hz:      float32 [frames]
  pitch_midi: float32 [frames], optional; derived from f0_hz when omitted
"""
from __future__ import annotations

import argparse
import json
import math
import wave
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(f"missing {what}: {path}")
    return path


def read_ds_segments(ds_path: Path) -> list[dict]:
    data = json.loads(ds_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{ds_path} is not a segment-list DS file")
    return data


def write_wav(path: Path, wav_data: np.ndarray, sr: int) -> None:
    audio = np.nan_to_num(wav_data.astype(np.float32), copy=False)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.rint(audio * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int) -> np.ndarray:
    idx = max(0, idx)
    if idx >= a.shape[0]:
        return np.concatenate((a, np.zeros(idx - a.shape[0], dtype=np.float32), b))
    fade_len = min(a.shape[0] - idx, b.shape[0])
    result = np.zeros(max(a.shape[0], idx + b.shape[0]), dtype=np.float32)
    result[:idx] = a[:idx]
    if fade_len > 0:
        k = np.linspace(0.0, 1.0, num=fade_len, endpoint=True, dtype=np.float32)
        result[idx:idx + fade_len] = (1.0 - k) * a[idx:idx + fade_len] + k * b[:fade_len]
    if idx + fade_len < a.shape[0]:
        result[idx + fade_len:a.shape[0]] = a[idx + fade_len:]
    if fade_len < b.shape[0]:
        result[idx + fade_len:idx + b.shape[0]] = b[fade_len:]
    return result


def write_demo_score(work: Path) -> dict[str, Path]:
    tokens = np.array([1, 2], dtype=np.int32)
    ph2word = np.array([1, 2], dtype=np.int32)
    word_dur = np.array([2.0, 2.0], dtype=np.float32)
    mel2ph = np.array([1, 1, 2, 2], dtype=np.int32)
    pitch_midi = np.array([60.0, 60.0, 62.0, 62.0], dtype=np.float32)
    f0_hz = np.where(pitch_midi > 0.0, 440.0 * np.power(2.0, (pitch_midi - 69.0) / 12.0), 0.0).astype(np.float32)
    lang = np.array([0, 0], dtype=np.int32)
    data = {
        "tokens": tokens,
        "ph2word": ph2word,
        "word_dur": word_dur,
        "mel2ph": mel2ph,
        "pitch_midi": pitch_midi,
        "f0_hz": f0_hz,
        "lang": lang,
    }
    paths: dict[str, Path] = {}
    for name, arr in data.items():
        path = work / f"{name}.bin"
        arr.tofile(path)
        paths[name] = path
    return paths


def midi_to_hz_file(pitch_midi_path: Path, f0_hz_path: Path) -> None:
    midi = np.fromfile(pitch_midi_path, dtype=np.float32)
    hz = np.where(midi > 0.0, 440.0 * np.power(2.0, (midi - 69.0) / 12.0), 0.0)
    hz.astype(np.float32).tofile(f0_hz_path)


def hz_to_midi_file(f0_hz_path: Path, pitch_midi_path: Path) -> None:
    hz = np.fromfile(f0_hz_path, dtype=np.float32)
    voiced = hz > 0.0
    if np.any(voiced) and not np.all(voiced):
        log_hz = np.zeros_like(hz, dtype=np.float32)
        log_hz[voiced] = np.log2(hz[voiced])
        log_hz[~voiced] = np.interp(np.flatnonzero(~voiced), np.flatnonzero(voiced), log_hz[voiced])
        hz = np.power(2.0, log_hz).astype(np.float32)
    midi = np.where(hz > 0.0, 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0), 0.0)
    midi.astype(np.float32).tofile(pitch_midi_path)


def load_name_map(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def load_exported_phoneme_map(path: Path, lang: str) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} is not a phoneme-id map")
    phone_to_id = {str(k): int(v) for k, v in raw.items()}
    out: dict[str, int] = {}
    for phone, idx in phone_to_id.items():
        out[phone] = idx
        out[phone.lower()] = idx
        if "/" in phone:
            key_lang, bare = phone.split("/", 1)
            if key_lang == lang:
                out[bare] = idx
                out[bare.lower()] = idx
    return out


def resolve_spk_id(spk_id: int | None,
                   spk_name: str | None,
                   spk_map_path: Path,
                   require_explicit: bool = False) -> int:
    if spk_name is None:
        if spk_id is not None:
            return spk_id
        if require_explicit:
            raise SystemExit("multi-speaker DS inference requires --spk-name or --spk-id")
        return 0
    spk_map = load_name_map(require(spk_map_path, "speaker map"))
    if spk_name not in spk_map:
        choices = ", ".join(sorted(spk_map))
        raise SystemExit(f"unknown speaker '{spk_name}'. Available speakers: {choices}")
    return spk_map[spk_name]


def load_phoneme_map(config_path: Path, dictionary_dir: Path, lang: str) -> dict[str, int]:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    all_phonemes = {"AP", "SP"}
    for ph in cfg.get("extra_phonemes", []) or []:
        all_phonemes.add(ph)
    dictionaries = cfg.get("dictionaries") or {"default": "dictionary.txt"}
    multi_lang = len(dictionaries) > 1
    for dict_lang in dictionaries:
        dict_path = dictionary_dir / f"dictionary-{dict_lang}.txt"
        if not dict_path.exists():
            dict_path = REPO / str(dictionaries[dict_lang])
        with dict_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                _, phones = line.split("\t", maxsplit=1)
                for ph in phones.split():
                    all_phonemes.add(f"{dict_lang}/{ph}" if multi_lang and ph not in all_phonemes else ph)
    phone_to_id: dict[str, int] = {}
    for idx, ph in enumerate(sorted(all_phonemes), start=1):
        phone_to_id[ph] = idx

    out = dict(phone_to_id)
    for key, idx in phone_to_id.items():
        if "/" in key:
            key_lang, bare = key.split("/", 1)
            if key_lang == lang:
                out[bare] = idx
    return out


def encode_phone(phone_map: dict[str, int], phone: str, normalized: set[tuple[str, str]] | None = None) -> int:
    if phone in phone_map:
        return phone_map[phone]
    lower = phone.lower()
    if lower in phone_map:
        if normalized is not None:
            normalized.add((phone, lower))
        return phone_map[lower]
    raise KeyError(phone)


def note_to_midi(note: str) -> float:
    if note.lower() == "rest":
        return -1.0
    names = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    base = names[note[0].upper()]
    pos = 1
    if pos < len(note) and note[pos] in "#b":
        base += 1 if note[pos] == "#" else -1
        pos += 1
    octave = int(note[pos:])
    return float((octave + 1) * 12 + base)


def resample_align_curve(points: np.ndarray,
                         original_timestep: float,
                         target_timestep: float,
                         align_length: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros(align_length, dtype=np.float32)
    if points.size == 1:
        return np.full(align_length, points[0], dtype=points.dtype)
    t_max = (points.size - 1) * original_timestep
    curve = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(points.size),
        points,
    ).astype(points.dtype)
    delta = align_length - curve.size
    if delta < 0:
        curve = curve[:align_length]
    elif delta > 0:
        curve = np.concatenate((curve, np.full(delta, curve[-1], dtype=curve.dtype)), axis=0)
    return curve


def frames_from_seconds(values: list[float], sr: int, hop: int) -> np.ndarray:
    timestep = hop / sr
    accum = np.cumsum(np.asarray(values, dtype=np.float64))
    boundaries = np.rint(accum / timestep + 0.5).astype(np.int32)
    frames = np.diff(boundaries, prepend=np.array([0], dtype=np.int32)).astype(np.int32)
    if np.any(frames < 0):
        raise SystemExit("negative duration after DS frame discretization")
    return frames


def write_ds_segment(ds_path: Path,
                     segment_index: int,
                     work: Path,
                     phonemes_path: Path,
                     config_path: Path,
                     dictionary_dir: Path,
                     lang: str,
                     sr: int,
                     hop: int) -> tuple[dict[str, Path], int, int]:
    data = read_ds_segments(ds_path)
    if segment_index < 0 or segment_index >= len(data):
        raise SystemExit(f"segment index {segment_index} out of range 0..{len(data)-1}")
    seg = data[segment_index]
    for key in ("ph_seq", "ph_dur"):
        if key not in seg:
            raise SystemExit(f"segment {segment_index} missing {key}; duration-predictor DS is not supported yet")

    phones = seg["ph_seq"].split()
    ph_dur_s = [float(x) for x in seg["ph_dur"].split()]
    if len(phones) != len(ph_dur_s):
        raise SystemExit(f"phone/duration length mismatch: {len(phones)} vs {len(ph_dur_s)}")
    if phonemes_path.exists():
        phone_map = load_exported_phoneme_map(phonemes_path, lang)
    else:
        phone_map = load_phoneme_map(config_path, dictionary_dir, lang)
    normalized_phones: set[tuple[str, str]] = set()
    try:
        tokens = np.asarray([encode_phone(phone_map, p, normalized_phones) for p in phones], dtype=np.int32)
    except KeyError as exc:
        raise SystemExit(f"unknown phoneme in DS segment: {exc.args[0]}") from exc
    if normalized_phones:
        pairs = ", ".join(f"{src}->{dst}" for src, dst in sorted(normalized_phones))
        print(f"[ds] segment {segment_index}: normalized phoneme case: {pairs}", flush=True)
    ph_frames = frames_from_seconds(ph_dur_s, sr, hop)
    mel2ph = np.repeat(np.arange(1, len(tokens) + 1, dtype=np.int32), ph_frames)
    T = int(mel2ph.size)
    if "ph_num" not in seg:
        raise SystemExit(f"segment {segment_index} missing ph_num; word-mode variance requires it")
    ph_num = np.asarray([int(x) for x in seg["ph_num"].split()], dtype=np.int32)
    if int(ph_num.sum()) != len(tokens):
        raise SystemExit(f"ph_num sum mismatch: {int(ph_num.sum())} vs {len(tokens)} phones")
    ph2word = np.repeat(np.arange(1, ph_num.size + 1, dtype=np.int32), ph_num)
    word_dur = np.zeros(ph_num.size, dtype=np.float32)
    np.add.at(word_dur, ph2word - 1, ph_frames.astype(np.float32))

    if "f0_seq" in seg:
        f0_src = np.asarray([float(x) for x in seg["f0_seq"].split()], dtype=np.float32)
        step = float(seg.get("f0_timestep", 0.005))
        f0 = resample_align_curve(f0_src, step, hop / sr, T).astype(np.float32)
    else:
        ph_num = [int(x) for x in seg["ph_num"].split()]
        note_seq = seg["note_seq"].split()
        if len(ph_num) != len(note_seq) or sum(ph_num) != len(tokens):
            raise SystemExit("cannot synthesize f0 from notes: ph_num/note_seq mismatch")
        note_midi = np.asarray([note_to_midi(n) for n in note_seq], dtype=np.float32)
        phone_note = np.repeat(np.arange(len(note_seq), dtype=np.int32), np.asarray(ph_num, dtype=np.int32))
        frame_note = np.repeat(phone_note, ph_frames)
        midi = note_midi[frame_note]
        f0 = np.where(midi > 0, 440.0 * np.power(2.0, (midi - 69.0) / 12.0), 0.0).astype(np.float32)

    lang_ids = np.zeros(tokens.size, dtype=np.int32)
    variance_dir = work / "variance"
    variance_dir.mkdir(exist_ok=True)

    def write_curve_if_present(name: str) -> None:
        if name not in seg or seg.get(name) is None:
            return
        timestep_key = f"{name}_timestep"
        if timestep_key not in seg or seg.get(timestep_key) is None:
            raise SystemExit(f"segment {segment_index} has {name} but no {timestep_key}")
        values = np.asarray([float(x) for x in str(seg[name]).split()], dtype=np.float32)
        curve = resample_align_curve(values, float(seg[timestep_key]), hop / sr, T).astype(np.float32)
        curve.tofile(variance_dir / f"{name}.bin")

    for name in ("energy", "breathiness", "voicing", "tension"):
        write_curve_if_present(name)

    if "gender" in seg and seg.get("gender") is not None:
        shift_min, shift_max = -5.0, 5.0
        gender = seg["gender"]
        if isinstance(gender, (int, float, bool)):
            key_shift = float(gender) * (shift_max if float(gender) >= 0 else abs(shift_min))
            np.full(T, key_shift, dtype=np.float32).tofile(variance_dir / "gender.bin")
        else:
            values = np.asarray([float(x) for x in str(gender).split()], dtype=np.float32)
            seq = resample_align_curve(values, float(seg["gender_timestep"]), hop / sr, T)
            mask = seq >= 0
            key_shift = seq * (mask * shift_max + (1 - mask) * abs(shift_min))
            np.clip(key_shift, shift_min, shift_max).astype(np.float32).tofile(variance_dir / "gender.bin")

    if "velocity" in seg and seg.get("velocity") is not None:
        values = np.asarray([float(x) for x in str(seg["velocity"]).split()], dtype=np.float32)
        speed = resample_align_curve(values, float(seg["velocity_timestep"]), hop / sr, T)
        np.clip(speed, 0.5, 2.0).astype(np.float32).tofile(variance_dir / "velocity.bin")

    paths = {
        "tokens": work / "tokens.bin",
        "mel2ph": work / "mel2ph.bin",
        "ph2word": work / "ph2word.bin",
        "word_dur": work / "word_dur.bin",
        "f0_hz": work / "f0_hz.bin",
        "pitch_midi": work / "pitch_midi.bin",
        "lang": work / "lang.bin",
        "variance_dir": variance_dir,
    }
    tokens.tofile(paths["tokens"])
    mel2ph.astype(np.int32).tofile(paths["mel2ph"])
    ph2word.astype(np.int32).tofile(paths["ph2word"])
    word_dur.astype(np.float32).tofile(paths["word_dur"])
    f0.tofile(paths["f0_hz"])
    hz_to_midi_file(paths["f0_hz"], paths["pitch_midi"])
    lang_ids.tofile(paths["lang"])
    return paths, len(tokens), T


def prepare_ds_pipeline_manifest(args: argparse.Namespace,
                                 ds_path: Path,
                                 segment_indices: list[int],
                                 work: Path) -> tuple[Path, list[Path]]:
    manifest = work / "segments.tsv"
    wav_paths: list[Path] = []
    with manifest.open("w", encoding="utf-8") as f:
        for idx in segment_indices:
            seg_work = work / f"seg_{idx:03d}"
            seg_work.mkdir(parents=True, exist_ok=True)
            paths, _phones, _frames = write_ds_segment(
                ds_path,
                idx,
                seg_work,
                require(args.phonemes, "phonemes map"),
                require(args.config, "config"),
                require(args.dictionary_dir, "dictionary dir"),
                args.ds_lang,
                44100,
                512,
            )
            wav_path = seg_work / "wav.f32"
            wav_paths.append(wav_path)
            cols = [
                paths["tokens"],
                paths["ph2word"],
                paths["word_dur"],
                paths["mel2ph"],
                paths["f0_hz"],
                paths["pitch_midi"],
                paths["lang"],
                paths["variance_dir"],
                wav_path,
            ]
            f.write("\t".join(str(c) for c in cols) + "\n")
    return manifest, wav_paths


def run_ds_pipeline(args: argparse.Namespace,
                    manifest: Path,
                    spk_id: int) -> None:
    pipeline = require(ROOT / "build/diffsinger_pipeline", "diffsinger_pipeline")
    cmd = [
        str(pipeline),
        "--variance-model", str(args.variance_model),
        "--acoustic-model", str(args.acoustic_model),
        "--vocoder-model", str(args.vocoder_model),
        "--manifest", str(manifest),
        "--spk-id", str(spk_id),
        "--seed", str(args.seed),
        "--backend", args.backend,
        "--noise-scale", str(args.vocoder_noise_scale),
    ]
    if args.steps > 0:
        cmd += ["--steps", str(args.steps)]
    if args.vocoder_mel_clamp:
        cmd += ["--mel-min", str(args.vocoder_mel_min), "--mel-max", str(args.vocoder_mel_max)]
    if args.predict_all_variances:
        cmd += ["--predict-all-variances"]
    run(cmd)


def stitch_ds_segments(segments: list[dict], wav_paths: list[Path], out_path: Path) -> None:
    result = np.zeros(0, dtype=np.float32)
    current_length = 0
    for segment, wav_path in zip(segments, wav_paths):
        waveform = np.fromfile(wav_path, dtype=np.float32)
        offset_samples = round(float(segment.get("offset", 0.0)) * 44100)
        silent_length = offset_samples - current_length
        if silent_length >= 0:
            if silent_length > 0:
                result = np.concatenate((result, np.zeros(silent_length, dtype=np.float32)))
            result = np.concatenate((result, waveform))
        else:
            result = cross_fade(result, waveform, current_length + silent_length)
        current_length = current_length + silent_length + waveform.shape[0]
    write_wav(out_path, result, 44100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variance-model", type=Path, default=ROOT / "assets/zhibin_variance.gguf")
    ap.add_argument("--acoustic-model", type=Path, default=ROOT / "assets/zhibin_acoustic.gguf")
    ap.add_argument("--vocoder-model", type=Path, default=ROOT / "assets/pc_nsf_hifigan.gguf")
    ap.add_argument("--tokens-bin", type=Path)
    ap.add_argument("--ph2word-bin", type=Path)
    ap.add_argument("--word-dur-bin", type=Path)
    ap.add_argument("--mel2ph-bin", type=Path)
    ap.add_argument("--f0-bin", type=Path)
    ap.add_argument("--pitch-bin", type=Path, help="float32 frame-level MIDI pitch for variance")
    ap.add_argument("--lang-bin", type=Path)
    ap.add_argument("--phones", type=int, default=0)
    ap.add_argument("--spk-id", type=int)
    ap.add_argument("--spk-name", help="speaker name from spk_map.json; overrides --spk-id")
    ap.add_argument("--spk-map", type=Path, default=REPO / "ckpt/251228_zhibin_club_acoustic_mix-ln/spk_map.json")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--backend", default="cpu",
                    help="ggml backend for C++ tools: cpu, gpu, auto, cuda, or cuda:N")
    ap.add_argument("--steps", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=Path("out.wav"))
    ap.add_argument("--vocoder-mel-min", type=float, default=-6.0)
    ap.add_argument("--vocoder-mel-max", type=float, default=1.5)
    ap.add_argument("--vocoder-mel-clamp", dest="vocoder_mel_clamp", action="store_true",
                    help="clip mel before vocoder; off by default to match DiffSinger inference")
    ap.add_argument("--no-vocoder-mel-clamp", dest="vocoder_mel_clamp", action="store_false")
    ap.add_argument("--vocoder-noise-scale", type=float, default=1.0,
                    help="scale NSF-HiFiGAN conv_pre noise_sigma; use 0 for deterministic vocoder alignment")
    ap.add_argument("--predict-all-variances", action="store_true",
                    help="predict all variance targets instead of only filling curves missing from the DS segment")
    ap.set_defaults(vocoder_mel_clamp=False)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--demo", action="store_true", help="run a tiny generated score")
    ap.add_argument("--ds-file", type=Path, help="DiffSinger .ds file")
    ap.add_argument("--ds-segment", type=int, default=0,
                    help="segment index to synthesize; use -1 to synthesize all segments")
    ap.add_argument("--ds-all", action="store_true", help="synthesize and offset-stitch all DS segments")
    ap.add_argument("--ds-lang", default="zh")
    ap.add_argument("--config", type=Path, default=REPO / "ckpt/251228_zhibin_club_acoustic_mix-ln/config.yaml")
    ap.add_argument("--dictionary-dir", type=Path, default=REPO / "ckpt/251228_zhibin_club_acoustic_mix-ln")
    ap.add_argument("--phonemes", type=Path, default=ROOT / "assets/phonemes.json",
                    help="exported phoneme-id map; this is authoritative for deployment checkpoints")
    args = ap.parse_args()

    require(args.variance_model, "variance model")
    require(args.acoustic_model, "acoustic model")
    require(args.vocoder_model, "vocoder model")
    spk_id = resolve_spk_id(
        args.spk_id,
        args.spk_name,
        args.spk_map,
        require_explicit=args.ds_file is not None,
    )

    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None and args.keep_work:
        work = Path(tempfile.mkdtemp(prefix="ds_ggml_"))
    elif args.work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="ds_ggml_")
        work = Path(tmp_ctx.name)
    else:
        work = args.work_dir
        work.mkdir(parents=True, exist_ok=True)

    if args.demo:
        paths = write_demo_score(work)
        tokens = paths["tokens"]
        ph2word = paths["ph2word"]
        word_dur = paths["word_dur"]
        mel2ph = paths["mel2ph"]
        f0_hz = paths["f0_hz"]
        pitch_midi = paths["pitch_midi"]
        lang = paths["lang"]
        phones = 2
    elif args.ds_file is not None:
        ds_path = require(args.ds_file, "ds file")
        if args.ds_all or args.ds_segment < 0:
            segments = read_ds_segments(ds_path)
            manifest, wav_paths = prepare_ds_pipeline_manifest(args, ds_path, list(range(len(segments))), work)
            run_ds_pipeline(args, manifest, spk_id)
            stitch_ds_segments(segments, wav_paths, args.out)
        else:
            manifest, wav_paths = prepare_ds_pipeline_manifest(args, ds_path, [args.ds_segment], work)
            run_ds_pipeline(args, manifest, spk_id)
            write_wav(args.out, np.fromfile(wav_paths[0], dtype=np.float32), 44100)
        print(f"[ok] wrote {args.out}")
        print(f"[work] {work}")
        return 0
    else:
        needed = {
            "tokens": args.tokens_bin,
            "ph2word": args.ph2word_bin,
            "word_dur": args.word_dur_bin,
            "mel2ph": args.mel2ph_bin,
        }
        if args.f0_bin is None and args.pitch_bin is None:
            needed["f0_or_pitch"] = None
        missing = [k for k, v in needed.items() if v is None]
        if missing:
            raise SystemExit("missing score inputs: " + ", ".join(missing))
        tokens = require(args.tokens_bin, "tokens")  # type: ignore[arg-type]
        ph2word = require(args.ph2word_bin, "ph2word")  # type: ignore[arg-type]
        word_dur = require(args.word_dur_bin, "word_dur")  # type: ignore[arg-type]
        mel2ph = require(args.mel2ph_bin, "mel2ph")  # type: ignore[arg-type]
        f0_hz = require(args.f0_bin, "f0") if args.f0_bin is not None else work / "f0_hz.bin"
        pitch_midi = require(args.pitch_bin, "pitch") if args.pitch_bin is not None else work / "pitch_midi.bin"
        if args.f0_bin is None:
            midi_to_hz_file(pitch_midi, f0_hz)
        elif args.pitch_bin is None:
            hz_to_midi_file(f0_hz, pitch_midi)
        if args.lang_bin is not None:
            lang = require(args.lang_bin, "lang")
        else:
            lang = work / "lang.bin"
            np.zeros(np.fromfile(tokens, dtype=np.int32).size, dtype=np.int32).tofile(lang)
        phones = args.phones
        if phones <= 0:
            phones = int(np.fromfile(tokens, dtype=np.int32).size)

    mel = work / "mel.bin"
    variance_dir = work / "variance"
    variance_dir.mkdir(exist_ok=True)
    generated = {
        "breathiness": work / "breathiness.bin",
        "voicing": work / "voicing.bin",
        "tension": work / "tension.bin",
    }

    var_cmd = [
        str(ROOT / "build/diffsinger_variance"),
        "--model", str(args.variance_model),
        "--backend", args.backend,
        "--infer-variance",
        "--tokens-bin", str(tokens),
        "--ph2word-bin", str(ph2word),
        "--word-dur-bin", str(word_dur),
        "--mel2ph-bin", str(mel2ph),
        "--lang-bin", str(lang),
        "--pitch-bin", str(pitch_midi),
        "--out-breathiness", str(generated["breathiness"]),
        "--out-voicing", str(generated["voicing"]),
        "--out-tension", str(generated["tension"]),
        "--spk-id", str(spk_id),
        "--seed", str(args.seed),
    ]
    if phones > 0:
        var_cmd += ["--phones", str(phones)]
    run(var_cmd)

    for name, path in generated.items():
        shutil.copyfile(path, variance_dir / f"{name}.bin")

    acoustic_cmd = [
        str(ROOT / "build/diffsinger_acoustic"),
        "--model", str(args.acoustic_model),
        "--backend", args.backend,
        "--len-tokens", str(phones),
        "--tokens-bin", str(tokens),
        "--mel2ph-bin", str(mel2ph),
        "--lang-bin", str(lang),
        "--f0-bin", str(f0_hz),
        "--variance-dir", str(variance_dir),
        "--spk-id", str(spk_id),
        "--seed", str(args.seed),
        "--out", str(mel),
    ]
    if args.steps > 0:
        acoustic_cmd += ["--steps", str(args.steps)]
    run(acoustic_cmd)

    frames = np.fromfile(mel2ph, dtype=np.int32).size
    vocoder_cmd = [
        str(ROOT / "build/diffsinger_vocoder"),
        "--model", str(args.vocoder_model),
        "--backend", args.backend,
        "--mel-bin", str(mel),
        "--f0-bin", str(f0_hz),
        "--frames", str(frames),
        "--out", str(args.out),
        "--seed", str(args.seed),
        "--noise-scale", str(args.vocoder_noise_scale),
    ]
    if args.vocoder_mel_clamp:
        vocoder_cmd += ["--mel-min", str(args.vocoder_mel_min), "--mel-max", str(args.vocoder_mel_max)]
    run(vocoder_cmd)

    print(f"[ok] wrote {args.out}")
    print(f"[work] {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
