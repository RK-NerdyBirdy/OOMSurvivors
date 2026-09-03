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

Performance notes (device-aware, degrades gracefully on non-CUDA / older GPUs):
  * TF32 matmul + cuDNN benchmark enabled on CUDA (no-op elsewhere).
  * Autocast uses bf16 on GPUs that support it (Ampere/Hopper, e.g. H100/A100),
    falls back to fp16 on older GPUs (e.g. T4/V100), and is skipped on CPU.
  * channels_last memory format for the conv-heavy NAFNet body on CUDA.
  * torch.compile is attempted opportunistically and falls back to eager
    execution at runtime if compilation/execution fails for any reason —
    this must run unedited on whatever box grades it, so failure of an
    optimization should never fail the run.
  * File loading (disk I/O + numpy preprocessing) for the *next* batch is
    overlapped with GPU inference on the *current* batch via a background
    thread, since prior profiling showed disk I/O — not compute — dominates
    wall time, and that gap only widens on faster GPUs like the H100.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
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

_END = object()  # prefetch queue sentinel


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


def peek_shape(path: Path) -> tuple:
    """Return an array's effective (H, W) shape without reading its pixel
    data — used only to group files for batching. Mirrors the channel
    squeeze that load_npy performs, so grouping stays consistent with it."""
    arr = np.load(path, mmap_mode="r")
    shape = arr.shape
    del arr
    return shape[:2] if len(shape) == 3 else shape


class BatchPrefetcher:
    """Loads and stacks batches on a background thread so disk I/O and CPU
    preprocessing for batch N+1 overlap with GPU inference on batch N."""

    def __init__(self, by_shape: dict, batch_size: int, pin_memory: bool, depth: int = 2):
        self._chunks: list[list[Path]] = []
        for _, paths in by_shape.items():
            for i in range(0, len(paths), batch_size):
                self._chunks.append(paths[i:i + batch_size])
        self._pin_memory = pin_memory
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            for chunk in self._chunks:
                arrs, ndims = [], []
                for f in chunk:
                    arr, orig_ndim = load_npy(f)
                    arrs.append(arr)
                    ndims.append(orig_ndim)
                batch_np = np.stack(arrs)[:, None, :, :]  # (B, 1, H, W)
                t = torch.from_numpy(batch_np)
                if self._pin_memory:
                    t = t.pin_memory()
                self._q.put((chunk, t, ndims))
        except Exception as e:  # surface loader errors on the main thread
            self._q.put(e)
        finally:
            self._q.put(_END)

    def __len__(self):
        return len(self._chunks)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is _END:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item


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
    ap.add_argument("--compile", type=int, default=1, choices=[0, 1],
                     help="Attempt torch.compile on CUDA, with automatic eager fallback (default: 1)")
    ap.add_argument("--channels_last", type=int, default=1, choices=[0, 1],
                     help="Use channels_last memory format on CUDA (default: 1)")
    ap.add_argument("--prefetch", type=int, default=2,
                     help="Batches to prefetch ahead on a background thread (default: 2)")
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
    use_cuda = device.type == "cuda"
    use_channels_last = use_cuda and bool(args.channels_last)

    amp_dtype = None
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        try:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except AttributeError:
            amp_dtype = torch.float16

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext())

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

    try:
        ckpt = torch.load(active_weights_path, map_location=device, weights_only=False)
    except TypeError:
        # older torch versions don't expose the weights_only kwarg
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
                        non_local=non_local).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"Loaded weights: {active_weights_path} (dim={dim}, levels={levels}, {n_par/1e6:.2f}M params)")

    # Best-effort torch.compile with a permanent eager fallback on first failure
    eager_model = model
    run_model = model
    compile_failed = False
    if use_cuda and args.compile and hasattr(torch, "compile"):
        try:
            run_model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile setup failed, using eager model ({e})")
            run_model = eager_model

    amp_label = f"{amp_dtype}".replace("torch.", "") if use_cuda else "fp32 (cpu)"
    print(f"Device: {device} | autocast: {amp_label} | channels_last: {use_channels_last} | "
          f"compile: {use_cuda and args.compile == 1} | prefetch depth: {args.prefetch}")

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

    # Cheap metadata-only pass to group files by shape (no pixel data read yet;
    # actual loads happen inside BatchPrefetcher, overlapped with GPU compute).
    by_shape: dict[tuple, list[Path]] = defaultdict(list)
    for f in files:
        by_shape[peek_shape(f)].append(f)

    prefetcher = BatchPrefetcher(by_shape, args.batch_size, pin_memory=use_cuda, depth=args.prefetch)

    n_done = 0
    t_start = time.perf_counter()
    with torch.no_grad():
        for chunk, batch_cpu, ndims in prefetcher:
            x = batch_cpu.to(device, non_blocking=use_cuda)
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            xp, (ph, pw) = pad_to_multiple(x, 2 ** levels)
            if use_channels_last:
                xp = xp.contiguous(memory_format=torch.channels_last)

            try:
                with autocast_ctx:
                    y = tta_forward(run_model, xp, args.tta, clamp=None) if args.tta > 1 else run_model(xp)
            except Exception as e:
                if not compile_failed and run_model is not eager_model:
                    print(f"torch.compile failed at runtime, falling back to eager model ({e})")
                    run_model = eager_model
                    compile_failed = True
                    with autocast_ctx:
                        y = tta_forward(run_model, xp, args.tta, clamp=None) if args.tta > 1 else run_model(xp)
                else:
                    raise

            if ph or pw:
                y = y[..., : y.shape[-2] - ph * SCALE, : y.shape[-1] - pw * SCALE]

            y = torch.clamp(y.float(), 0.0, 1.0)
            y_np = y.cpu().numpy()[:, 0]  # (B, H, W)
            y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=0.0)

            for f, out_arr, orig_ndim in zip(chunk, y_np, ndims):
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