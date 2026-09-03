#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import build_model

def pad_to_multiple(x: torch.Tensor, m: int):
    if m <= 1: return x, (0, 0)
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (ph, pw)

def load_npy(path: Path):
    arr = np.load(path).astype(np.float32)
    orig_ndim = arr.ndim
    if arr.ndim == 3: arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), orig_ndim

# --- 1. Optimized Background Data Loader ---
class SEMDataset(Dataset):
    def __init__(self, file_list, gt_dir=None, levels=2):
        self.files = file_list
        self.gt_dir = gt_dir
        self.levels = levels
        self.pad_mult = 2 ** levels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        lr_arr, orig_ndim = load_npy(f)
        lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0) # (1, H, W)
        lr_padded, (ph, pw) = pad_to_multiple(lr_tensor, self.pad_mult)
        
        gt_arr = np.array([], dtype=np.float32)
        if self.gt_dir:
            gt_path = self.gt_dir / f.name
            if gt_path.exists():
                gt_arr, _ = load_npy(gt_path)

        return lr_padded, torch.from_numpy(gt_arr), str(f.name), orig_ndim, ph, pw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str)
    ap.add_argument("output_dir", type=str)
    ap.add_argument("--gt_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=8) # Increased default for batching!
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compile", action="store_true", help="Enable torch.compile for H100")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(in_dir.glob("*.npy"))
    
    # Initialize Vanilla Model
    model = build_model("nafnet", scale=2, dim=48, levels=2, non_local=False).to(device)
    weights_path = Path("weights/best_nafnet.pt")
    if not weights_path.exists(): weights_path = Path("artifacts/best_nafnet.pt")
        
    ckpt = torch.load(weights_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt.get("model", ckpt).items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Hardware acceleration
    if args.compile and hasattr(torch, "compile"):
        print("🔥 Optimizing model with torch.compile()...")
        model = torch.compile(model)

    # Setup Metrics
    calc_metrics = False
    if gt_dir and gt_dir.is_dir():
        from skimage.metrics import peak_signal_noise_ratio as psnr_metric
        from skimage.metrics import structural_similarity as ssim_metric
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters(): p.requires_grad = False
        calc_metrics = True
        total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
        eval_count = 0

    dataset = SEMDataset(files, gt_dir, levels=2)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True # Keeps memory in fast pinned RAM for GPU transfers
    )

    # Inference Loop
    t_e2e_start = time.perf_counter()
    forward_times = []
    
    # Warmup
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            _ = model(torch.zeros(1, 1, 128, 128, device=device))
            if device.type == "cuda": torch.cuda.synchronize()

    with torch.no_grad():
        for lr_padded, gt_batch, fnames, orig_ndims, phs, pws in dataloader:
            lr_padded = lr_padded.to(device, non_blocking=True)
            
            # Pure forward pass timing
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # Automatic Mixed Precision for T4 & H100
            with torch.autocast("cuda", dtype=torch.float16):
                y = model(lr_padded)
                
            if device.type == "cuda": torch.cuda.synchronize()
            # Append per-image time by dividing batch time by batch size
            forward_times.append(((time.perf_counter() - t0) * 1000.0) / lr_padded.shape[0])

            y = torch.clamp(y, 0.0, 1.0)
            
            for i in range(lr_padded.shape[0]):
                ph, pw = phs[i].item(), pws[i].item()
                out_tensor = y[i]
                if ph or pw: 
                    out_tensor = out_tensor[..., : out_tensor.shape[-2] - ph * 2, : out_tensor.shape[-1] - pw * 2]
                
                out_arr = out_tensor.cpu().numpy()[0]
                save_arr = out_arr[..., None] if orig_ndims[i].item() == 3 else out_arr
                np.save(out_dir / fnames[i], save_arr.astype(np.float32))

                if calc_metrics and gt_batch[i].numel() > 0:
                    gt_arr = gt_batch[i].numpy()
                    total_psnr += psnr_metric(gt_arr, out_arr, data_range=1.0)
                    total_ssim += ssim_metric(gt_arr, out_arr, data_range=1.0)
                    p_norm = torch.from_numpy(out_arr)[None, None, ...].to(device).repeat(1, 3, 1, 1) * 2 - 1
                    g_norm = torch.from_numpy(gt_arr)[None, None, ...].to(device).repeat(1, 3, 1, 1) * 2 - 1
                    total_lpips += float(lpips_fn(p_norm, g_norm).item())
                    eval_count += 1

    t_e2e_total = time.perf_counter() - t_e2e_start
    
    print("\n" + "=" * 60)
    print("      HIGH-THROUGHPUT INFERENCE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Images Processed : {len(files)}")
    print(f"Total E2E Runtime      : {t_e2e_total:.3f} s")
    print(f"Avg E2E Throughput     : {t_e2e_total / len(files) * 1000.0:.2f} ms / image")
    print(f"Pure Forward Pass Time : {np.mean(forward_times):.2f} ± {np.std(forward_times):.2f} ms / image")
    print("-" * 60)
    if calc_metrics and eval_count > 0:
        print(f"Evaluated GT Pairs     : {eval_count}")
        print(f"Average PSNR (↑)       : {total_psnr / eval_count:.4f} dB")
        print(f"Average SSIM (↑)       : {total_ssim / eval_count:.4f}")
        print(f"Average LPIPS (↓)      : {total_lpips / eval_count:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()