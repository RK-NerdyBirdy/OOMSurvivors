#!/usr/bin/env python3
"""
Standalone evaluation script — KLA AI Hackathon submission.
Team: OOM Survivors

Usage:
    python run.py <input_dir> <output_dir> [--gt_dir <gt_dir>]

Behavior:
  * Reads every .npy file in <input_dir> (degraded / low-res images).
  * Restores each one with the trained NAFNet super-resolution model.
  * Writes one restored .npy file per input to <output_dir>.
  * If Ground Truth is available, computes and prints PSNR, SSIM, and LPIPS.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make the local `src` package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import build_model, is_legacy_state_dict, remap_legacy_state_dict  # noqa: E402
from src.tta import tta_forward   # noqa: E402

SCALE = 2

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "weights" / "best_nafnet.pt"
FALLBACK_WEIGHTS = BASE_DIR / "artifacts" / "best_nafnet.pt"


def pad_to_multiple(x: torch.Tensor, m: int):
    """Reflect-pad the last two dims of x up to the next multiple of m."""
    if m <= 1:
        return x, (0, 0)
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (ph, pw)


def load_npy(path: Path):
    """Load a .npy file and return (2D float32 array, original_ndim)."""
    arr = np.load(path).astype(np.float32)
    orig_ndim = arr.ndim
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    elif arr.ndim != 2:
        raise ValueError(
            f"Unexpected array shape {arr.shape} for {path.name}; expected (H,W) or (H,W,1)"
        )
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr, orig_ndim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=str, help="Directory containing degraded .npy images")
    ap.add_argument("output_dir", type=str, help="Directory to write restored .npy images")
    ap.add_argument("--gt_dir", type=str, default=None, help="Optional directory containing GT .npy images for metrics")
    ap.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    ap.add_argument("--weights", type=str, default=None, help="Checkpoint path")
    ap.add_argument("--dim", type=int, default=48, help="Model width (default: 48)")
    ap.add_argument("--levels", type=int, default=2, help="U-Net depth (default: 2)")
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8], help="TTA passes")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        print(f"ERROR: input directory not found: {in_dir}")
        return 2

    # Check for Ground Truth directory
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir is None:
        potential_gt = in_dir.parent / "GT"
        if potential_gt.is_dir():
            gt_dir = potential_gt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted(in_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {in_dir}")
        return 2

    # Determine weights path
    if args.weights:
        active_weights_path = Path(args.weights)
    else:
        active_weights_path = WEIGHTS_PATH if WEIGHTS_PATH.exists() else FALLBACK_WEIGHTS

    if not active_weights_path.exists():
        print(f"ERROR: Weights checkpoint not found at {active_weights_path}")
        return 2

    ckpt = torch.load(active_weights_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Check for legacy names or infer configurations
    dim = args.dim
    levels = args.levels
    blocks, mblocks = 2, 2

    if is_legacy_state_dict(state_dict):
        state_dict = remap_legacy_state_dict(state_dict)
        levels, blocks, mblocks = 1, 2, 4
        print("Legacy checkpoint detected -> levels=1, blocks=2, middle_blocks=4")

    non_local = any(".to_kv." in k for k in state_dict)
    if non_local:
        mblocks = max(mblocks, sum(1 for k in state_dict if k.startswith("middle.") and k.endswith(".conv1.weight")))

    # Build model
    model = build_model("nafnet", scale=SCALE, dim=dim, levels=levels,
                        blocks=blocks, middle_blocks=mblocks,
                        non_local=non_local).to(device).eval()

    model.load_state_dict(state_dict, strict=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Loaded weights: {active_weights_path} (dim={dim}, levels={levels}, {n_par/1e6:.2f}M params)")

    # Prepare optional metric trackers
    calculate_metrics = False
    if gt_dir and gt_dir.is_dir():
        try:
            from skimage.metrics import peak_signal_noise_ratio as psnr_metric
            from skimage.metrics import structural_similarity as ssim_metric
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad = False
            calculate_metrics = True
            print(f"Ground Truth directory found at {gt_dir}. Metric evaluation enabled.")
        except ImportError:
            print("Dependencies for metrics (scikit-image, lpips) missing. Skipping metric calculation.")

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    evaluated_count = 0

    # Group files by dimension for consistent batch sizes
    t_start = time.perf_counter()
    cache: dict[Path, tuple[np.ndarray, int]] = {}
    by_shape: dict[tuple, list[Path]] = defaultdict(list)
    for f in files:
        arr, orig_ndim = load_npy(f)
        cache[f] = (arr, orig_ndim)
        by_shape[arr.shape].append(f)

    n_done = 0
    with torch.no_grad():
        for shape, group in by_shape.items():
            for i in range(0, len(group), args.batch_size):
                chunk = group[i:i + args.batch_size]
                batch_np = np.stack([cache[f][0] for f in chunk])[:, None, :, :]  # (B, 1, H, W)
                x = torch.from_numpy(batch_np).to(device, non_blocking=True)

                xp, (ph, pw) = pad_to_multiple(x, 2 ** levels)
                y = tta_forward(model, xp, args.tta, clamp=None) if args.tta > 1 else model(xp)
                if ph or pw:
                    y = y[..., : y.shape[-2] - ph * SCALE, : y.shape[-1] - pw * SCALE]

                y = torch.clamp(y, 0.0, 1.0)
                y_np = y.float().cpu().numpy()[:, 0]  # (B, H, W)
                y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=0.0)

                for f, out_arr in zip(chunk, y_np):
                    _, orig_ndim = cache[f]
                    save_arr = out_arr[..., None] if orig_ndim == 3 else out_arr
                    np.save(out_dir / f.name, save_arr.astype(np.float32))
                    n_done += 1

                    # Optional metric calculation
                    if calculate_metrics:
                        gt_path = gt_dir / f.name
                        if gt_path.exists():
                            gt_arr, _ = load_npy(gt_path)
                            total_psnr += psnr_metric(gt_arr, out_arr, data_range=1.0)
                            total_ssim += ssim_metric(gt_arr, out_arr, data_range=1.0)

                            pred_t = torch.from_numpy(out_arr).unsqueeze(0).unsqueeze(0).to(device)
                            gt_t = torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0).to(device)
                            p_norm = pred_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
                            g_norm = gt_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
                            total_lpips += float(lpips_fn(p_norm, g_norm).item())
                            evaluated_count += 1

    elapsed = time.perf_counter() - t_start
    print(f"\nRestored {n_done}/{len(files)} images -> {out_dir}")
    print(f"device={device}  elapsed={elapsed:.3f}s  ({1000 * elapsed / max(n_done, 1):.2f} ms/img)")

    if calculate_metrics and evaluated_count > 0:
        print("\n" + "=" * 55)
        print(f"METRICS OVER {evaluated_count} GROUND TRUTH IMAGES")
        print("=" * 55)
        print(f"PSNR : {total_psnr / evaluated_count:.4f} dB")
        print(f"SSIM : {total_ssim / evaluated_count:.4f}")
        print(f"LPIPS: {total_lpips / evaluated_count:.4f}")
        print("=" * 55)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())