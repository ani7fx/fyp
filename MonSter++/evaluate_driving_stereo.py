import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / "core"))

import stereo_datasets as datasets
from monster import Monster, autocast
from utils.utils import InputPadder


def parse_args():
    parser = argparse.ArgumentParser(description="Run MonSter++ baseline on a DrivingStereo subset.")
    parser.add_argument("--restore_ckpt", type=Path, default=PROJECT_DIR / "checkpoints" / "KITTI_large.pth")
    parser.add_argument("--depth_anything_ckpt", type=Path, default=PROJECT_DIR / "checkpoints" / "depth_anything_v2_vitl.pth")
    parser.add_argument("--driving_root", type=Path, default=PROJECT_DIR / "datasets" / "drivingstereo")
    parser.add_argument("--image_set", type=str, default="cloudy", choices=["sunny", "cloudy", "foggy", "rainy", "test"])
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of stereo pairs to evaluate.")
    parser.add_argument("--results_root", type=Path, default=PROJECT_DIR / "results" / "baseline_driving_stereo")
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--slow_fast_gru", action="store_true")
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--edge_thresholds", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--resume", action="store_true", help="Reuse existing saved predictions in results_root when available.")
    return parser.parse_args()


def load_model(args):
    model = torch.nn.DataParallel(Monster(args), device_ids=[0])
    checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]

    state_dict = {}
    for key, value in checkpoint.items():
        state_dict[key if key.startswith("module.") else f"module.{key}"] = value

    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()
    return model


def build_edge_mask(disparity: np.ndarray, valid_mask: np.ndarray, threshold: float):
    disp_for_grad = disparity.copy()
    disp_for_grad[~valid_mask] = 0.0
    sobel_x = cv2.Sobel(disp_for_grad, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(disp_for_grad, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_mag[~valid_mask] = 0.0
    max_grad = float(grad_mag[valid_mask].max()) if np.any(valid_mask) else 0.0
    grad_norm = grad_mag / max_grad if max_grad > 0 else grad_mag
    return (grad_norm >= threshold) & valid_mask


def compute_metrics(pred_disp: np.ndarray, gt_disp: np.ndarray, mask: np.ndarray):
    if not np.any(mask):
        return None, None
    err = np.abs(pred_disp - gt_disp)[mask]
    return float(err.mean()), float((err > 3.0).mean())


def save_outputs(base_dir: Path, stem: str, left_image_path: str, prediction: torch.Tensor):
    npy_dir = base_dir / "npy"
    png_dir = base_dir / "png"
    input_dir = base_dir / "input"
    for out_dir in [npy_dir, png_dir, input_dir]:
        out_dir.mkdir(parents=True, exist_ok=True)

    prediction_np = prediction.cpu().numpy().squeeze().astype(np.float32)
    np.save(npy_dir / f"{stem}.npy", prediction_np)
    plt.imsave(png_dir / f"{stem}.png", prediction_np, cmap="plasma")
    shutil.copy2(left_image_path, input_dir / f"{stem}{Path(left_image_path).suffix}")


def ensure_saved_outputs(base_dir: Path, stem: str, left_image_path: str, prediction_np: np.ndarray):
    npy_dir = base_dir / "npy"
    png_dir = base_dir / "png"
    input_dir = base_dir / "input"
    for out_dir in [npy_dir, png_dir, input_dir]:
        out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = npy_dir / f"{stem}.npy"
    png_path = png_dir / f"{stem}.png"
    input_path = input_dir / f"{stem}{Path(left_image_path).suffix}"

    if not npy_path.exists():
        np.save(npy_path, prediction_np.astype(np.float32))
    if not png_path.exists():
        plt.imsave(png_path, prediction_np.astype(np.float32), cmap="plasma")
    if not input_path.exists():
        shutil.copy2(left_image_path, input_path)


def main():
    args = parse_args()
    for path in [args.restore_ckpt, args.depth_anything_ckpt, args.driving_root]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    model = load_model(args)
    dataset = datasets.DrivingStereo({}, root=str(args.driving_root), image_set=args.image_set)
    total_items = len(dataset)
    num_items = min(total_items, args.limit) if args.limit > 0 else total_items
    results_root = args.results_root

    global_epe = []
    global_d1 = []
    runtimes = []
    edge_epe = {thr: [] for thr in args.edge_thresholds}
    edge_d1 = {thr: [] for thr in args.edge_thresholds}
    edge_pct_total = {thr: [] for thr in args.edge_thresholds}
    edge_pct_valid = {thr: [] for thr in args.edge_thresholds}

    for thr in args.edge_thresholds:
        (results_root / "edge_masks" / f"thr_{thr:.1f}").mkdir(parents=True, exist_ok=True)

    for idx in range(num_items):
        (left_path, _, _), image1, image2, flow_gt, valid_gt = dataset[idx]
        stem = Path(left_path).stem
        existing_npy = results_root / "npy" / f"{stem}.npy"

        gt_disp = flow_gt.numpy().squeeze().astype(np.float32)
        valid_mask = (valid_gt.numpy().astype(bool)) & (np.abs(gt_disp) < args.max_disp)
        runtime = 0.0

        if args.resume and existing_npy.exists():
            pred_disp = np.load(existing_npy).astype(np.float32)
            ensure_saved_outputs(results_root, stem, left_path, pred_disp)
        else:
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with torch.no_grad():
                with autocast(enabled=args.corr_implementation.endswith("_cuda")):
                    torch.cuda.synchronize()
                    start = time.time()
                    flow_pr = model(image1, image2, iters=args.valid_iters, test_mode=True)
                    torch.cuda.synchronize()
                    runtime = time.time() - start

            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
            pred_disp = flow_pr.numpy().squeeze().astype(np.float32)
            save_outputs(results_root, stem, left_path, flow_pr)

        epe, d1 = compute_metrics(pred_disp, gt_disp, valid_mask)
        global_epe.append(epe)
        global_d1.append(d1)
        if runtime > 0:
            runtimes.append(runtime)

        total_pixels = gt_disp.size
        valid_pixels = int(valid_mask.sum())

        for thr in args.edge_thresholds:
            edge_mask = build_edge_mask(gt_disp, valid_mask, thr)
            cv2.imwrite(
                str(results_root / "edge_masks" / f"thr_{thr:.1f}" / f"{stem}.png"),
                (edge_mask.astype(np.uint8) * 255),
            )
            edge_e, edge_d = compute_metrics(pred_disp, gt_disp, edge_mask)
            if edge_e is not None:
                edge_epe[thr].append(edge_e)
                edge_d1[thr].append(edge_d)
            edge_pct_total[thr].append(float(edge_mask.sum() / total_pixels))
            edge_pct_valid[thr].append(float(edge_mask.sum() / valid_pixels) if valid_pixels > 0 else 0.0)

        if idx < 5 or (idx + 1) % 25 == 0:
            runtime_msg = f"{runtime:.3f}s" if runtime > 0 else "cached"
            print(f"[{idx + 1}/{num_items}] {stem} EPE={epe:.4f} D1={d1 * 100:.3f}% runtime={runtime_msg}")

    metrics = {
        "dataset": "DrivingStereo",
        "image_set": args.image_set,
        "driving_root": str(args.driving_root.resolve()),
        "num_images_evaluated": num_items,
        "max_disp_threshold": args.max_disp,
        "valid_iters": args.valid_iters,
        "mean_epe": float(np.mean(global_epe)),
        "mean_d1": float(np.mean(global_d1)),
        "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else None,
        "median_runtime_sec": float(np.median(runtimes)) if runtimes else None,
        "edge_metrics": {},
    }

    for thr in args.edge_thresholds:
        metrics["edge_metrics"][f"sobel_{thr:.1f}"] = {
            "threshold": thr,
            "mean_edge_epe": float(np.mean(edge_epe[thr])) if edge_epe[thr] else None,
            "mean_edge_d1": float(np.mean(edge_d1[thr])) if edge_d1[thr] else None,
            "mean_edge_pixel_ratio_total": float(np.mean(edge_pct_total[thr])),
            "mean_edge_pixel_ratio_valid": float(np.mean(edge_pct_valid[thr])),
            "mask_dir": str((results_root / "edge_masks" / f"thr_{thr:.1f}").resolve()),
        }

    results_root.mkdir(parents=True, exist_ok=True)
    metrics_path = results_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Metric Summary")
    print(f"Global      EPE={metrics['mean_epe']:.4f}   D1={metrics['mean_d1'] * 100:.3f}%   images={num_items}")
    for thr in args.edge_thresholds:
        edge = metrics["edge_metrics"][f"sobel_{thr:.1f}"]
        print(
            f"Edge t={thr:.1f}  EPE={edge['mean_edge_epe']:.4f}   "
            f"D1={edge['mean_edge_d1'] * 100:.3f}%   "
            f"edge/all={edge['mean_edge_pixel_ratio_total'] * 100:.2f}%   "
            f"edge/valid={edge['mean_edge_pixel_ratio_valid'] * 100:.2f}%"
        )
    print(f"Saved results to: {results_root}")


if __name__ == "__main__":
    main()
