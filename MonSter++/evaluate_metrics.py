import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_disp_kitti(filename: Path):
    disp = cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp.astype(np.float32), valid


def build_edge_mask(disparity: np.ndarray, valid_mask: np.ndarray, threshold: float):
    disp_for_grad = disparity.copy()
    disp_for_grad[~valid_mask] = 0.0

    sobel_x = cv2.Sobel(disp_for_grad, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(disp_for_grad, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_mag[~valid_mask] = 0.0

    max_grad = float(grad_mag[valid_mask].max()) if np.any(valid_mask) else 0.0
    if max_grad > 0:
        grad_norm = grad_mag / max_grad
    else:
        grad_norm = grad_mag

    edge_mask = (grad_norm >= threshold) & valid_mask
    return edge_mask.astype(np.uint8), grad_norm


def compute_metrics(pred_disp: np.ndarray, gt_disp: np.ndarray, mask: np.ndarray):
    if not np.any(mask):
        return {"epe": None, "d1": None, "count": 0}

    errors = np.abs(pred_disp - gt_disp)
    subset = errors[mask]
    return {
        "epe": float(subset.mean()),
        "d1": float((subset > 3.0).mean()),
        "count": int(mask.sum()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate global and edge metrics for saved KITTI baseline predictions.")
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--pred_dir", type=Path, default=project_dir / "results" / "baseline" / "npy")
    parser.add_argument("--kitti_root", type=Path, default=project_dir / "datasets" / "data_scene_flow")
    parser.add_argument("--metrics_path", type=Path, default=project_dir / "results" / "baseline" / "metrics.json")
    parser.add_argument("--mask_root", type=Path, default=project_dir / "results" / "baseline" / "edge_masks")
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2])
    return parser.parse_args()


def main():
    args = parse_args()

    gt_dir = args.kitti_root / "training" / "disp_noc_0"
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    pred_files = sorted(args.pred_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction .npy files found in {args.pred_dir}")

    global_epe = []
    global_d1 = []
    edge_epe = {thr: [] for thr in args.thresholds}
    edge_d1 = {thr: [] for thr in args.thresholds}
    edge_pct_total = {thr: [] for thr in args.thresholds}
    edge_pct_valid = {thr: [] for thr in args.thresholds}

    for threshold in args.thresholds:
        (args.mask_root / f"thr_{threshold:.1f}").mkdir(parents=True, exist_ok=True)

    evaluated = 0
    for pred_file in pred_files:
        stem = pred_file.stem
        gt_file = gt_dir / f"{stem}.png"
        if not gt_file.exists():
            continue

        pred_disp = np.load(pred_file).astype(np.float32)
        gt_disp, valid_gt = read_disp_kitti(gt_file)
        eval_mask = valid_gt & (np.abs(gt_disp) < args.max_disp)

        if pred_disp.shape != gt_disp.shape:
            raise ValueError(f"Shape mismatch for {stem}: pred {pred_disp.shape}, gt {gt_disp.shape}")

        metrics = compute_metrics(pred_disp, gt_disp, eval_mask)
        global_epe.append(metrics["epe"])
        global_d1.append(metrics["d1"])

        total_pixels = gt_disp.size
        valid_pixels = int(eval_mask.sum())

        for threshold in args.thresholds:
            edge_mask, grad_norm = build_edge_mask(gt_disp, eval_mask, threshold)
            cv2.imwrite(
                str(args.mask_root / f"thr_{threshold:.1f}" / f"{stem}.png"),
                (edge_mask * 255).astype(np.uint8),
            )

            edge_metrics = compute_metrics(pred_disp, gt_disp, edge_mask.astype(bool))
            if edge_metrics["epe"] is not None:
                edge_epe[threshold].append(edge_metrics["epe"])
                edge_d1[threshold].append(edge_metrics["d1"])
            edge_pct_total[threshold].append(float(edge_mask.sum() / total_pixels))
            edge_pct_valid[threshold].append(float(edge_mask.sum() / valid_pixels) if valid_pixels > 0 else 0.0)

        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No matching predictions and ground-truth files were evaluated.")

    metrics_payload = {}
    if args.metrics_path.exists():
        metrics_payload = json.loads(args.metrics_path.read_text())

    metrics_payload["global_metrics_recomputed"] = {
        "num_images": evaluated,
        "mean_epe": float(np.mean(global_epe)),
        "mean_d1": float(np.mean(global_d1)),
    }

    edge_metrics_payload = {}
    for threshold in args.thresholds:
        edge_metrics_payload[f"sobel_{threshold:.1f}"] = {
            "threshold": threshold,
            "mean_edge_epe": float(np.mean(edge_epe[threshold])) if edge_epe[threshold] else None,
            "mean_edge_d1": float(np.mean(edge_d1[threshold])) if edge_d1[threshold] else None,
            "mean_edge_pixel_ratio_total": float(np.mean(edge_pct_total[threshold])),
            "mean_edge_pixel_ratio_valid": float(np.mean(edge_pct_valid[threshold])),
            "mask_dir": str((args.mask_root / f"thr_{threshold:.1f}").resolve()),
        }

    metrics_payload["edge_metrics"] = edge_metrics_payload
    args.metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    print("Metric Summary")
    print(f"Global      EPE={np.mean(global_epe):.4f}   D1={np.mean(global_d1) * 100:.3f}%   images={evaluated}")
    for threshold in args.thresholds:
        print(
            f"Edge t={threshold:.1f}  "
            f"EPE={np.mean(edge_epe[threshold]):.4f}   "
            f"D1={np.mean(edge_d1[threshold]) * 100:.3f}%   "
            f"edge/all={np.mean(edge_pct_total[threshold]) * 100:.2f}%   "
            f"edge/valid={np.mean(edge_pct_valid[threshold]) * 100:.2f}%"
        )
    print(f"Updated metrics written to: {args.metrics_path}")


if __name__ == "__main__":
    main()
