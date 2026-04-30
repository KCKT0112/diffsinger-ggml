"""
Convert the pitch-controllable NSF-HiFiGAN generator checkpoint to GGUF.

The checkpoint stores weight-normalized convolutions as weight_g/weight_v.
This converter folds those pairs into plain convolution weights so the C++
runtime only needs Conv1d/ConvTranspose1d weights plus biases.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:  # pragma: no cover
    print("ERROR: `pip install gguf numpy torch` first.", file=sys.stderr)
    raise


ARCH = "nsf-hifigan"


def load_generator(ckpt_path: Path) -> dict[str, np.ndarray]:
    import torch

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "generator" not in obj:
        raise ValueError("Expected checkpoint with top-level 'generator' key")
    return {k: v.detach().cpu().numpy() for k, v in obj["generator"].items()}


def fold_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    # torch.nn.utils.weight_norm default dim=0:
    # w = g * v / norm(v, dim=all_except_0)
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v.astype(np.float64) * v.astype(np.float64), axis=axes, keepdims=True))
    return (v * (g / norm).astype(v.dtype)).astype(np.float32)


def converted_tensors(sd: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    used: set[str] = set()

    for name in sorted(sd):
        if name.endswith(".weight_g"):
            base = name[:-len(".weight_g")]
            g_key = f"{base}.weight_g"
            v_key = f"{base}.weight_v"
            if v_key not in sd:
                raise ValueError(f"Missing {v_key} for weight norm tensor {g_key}")
            out[f"{base}.weight"] = fold_weight_norm(sd[g_key], sd[v_key])
            used.update({g_key, v_key})

    for name, value in sd.items():
        if name in used or name.endswith(".weight_v"):
            continue
        out[name] = value.astype(np.float32) if value.dtype != np.float32 else value

    return out


def add_metadata(gw: gguf.GGUFWriter, tensors: dict[str, np.ndarray]):
    up_ids = sorted(
        int(k.split(".")[1])
        for k in tensors
        if k.startswith("ups.") and k.endswith(".weight")
    )
    rb_ids = sorted(
        int(k.split(".")[1])
        for k in tensors
        if k.startswith("resblocks.") and ".convs1.0.weight" in k
    )

    upsample_kernel_sizes = [int(tensors[f"ups.{i}.weight"].shape[-1]) for i in up_ids]
    # This vocoder is trained for hop_size=512 with kernels [16,16,4,4,4].
    # The matching HiFiGAN rates are [8,8,2,2,2].
    if upsample_kernel_sizes == [16, 16, 4, 4, 4]:
        upsample_rates = [8, 8, 2, 2, 2]
    else:
        raise ValueError(f"Unknown upsample kernels: {upsample_kernel_sizes}")

    resblock_kernel_sizes = []
    if rb_ids:
        kernels = []
        for rb in rb_ids[:3]:
            kernels.append(int(tensors[f"resblocks.{rb}.convs1.0.weight"].shape[-1]))
        resblock_kernel_sizes = kernels

    gw.add_string("general.name", "PC NSF-HiFiGAN")
    gw.add_uint32(f"{ARCH}.num_mels", int(tensors["conv_pre.weight"].shape[1]))
    gw.add_uint32(f"{ARCH}.upsample_initial_channel", int(tensors["conv_pre.weight"].shape[0]))
    gw.add_uint32(f"{ARCH}.num_upsamples", len(up_ids))
    gw.add_uint32(f"{ARCH}.num_resblocks", len(rb_ids))
    gw.add_bool(f"{ARCH}.mini_nsf", "source_conv.weight" in tensors)
    gw.add_float32(f"{ARCH}.noise_sigma", 0.01)
    gw.add_uint32(f"{ARCH}.sampling_rate", 44100)
    gw.add_uint32(f"{ARCH}.hop_size", int(np.prod(upsample_rates)))
    for i, value in enumerate(upsample_rates):
        gw.add_uint32(f"{ARCH}.upsample_rates.{i}", value)
    for i, value in enumerate(upsample_kernel_sizes):
        gw.add_uint32(f"{ARCH}.upsample_kernel_sizes.{i}", value)
    for i, value in enumerate(resblock_kernel_sizes):
        gw.add_uint32(f"{ARCH}.resblock_kernel_sizes.{i}", value)
    # ResBlock1 default in modules/nsf_hifigan/models.py.
    for i, value in enumerate([1, 3, 5]):
        gw.add_uint32(f"{ARCH}.resblock_dilations.{i}", value)


def add_tensor(gw: gguf.GGUFWriter, name: str, data: np.ndarray):
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    gw.add_tensor(name, np.ascontiguousarray(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sd = load_generator(args.ckpt)
    tensors = converted_tensors(sd)

    gw = gguf.GGUFWriter(str(args.out), ARCH)
    add_metadata(gw, tensors)
    for name in sorted(tensors):
        add_tensor(gw, name, tensors[name])
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    print(f"[ok] wrote {args.out}")
    print(f"[ok] tensors: {len(tensors)}")


if __name__ == "__main__":
    main()
