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
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / "core"))

import stereo_datasets as datasets
from monster import Monster, autocast
from refinement_module import ConfigurableRefinementModule, LargeRefinementModule, LightweightRefinementModule
from utils.utils import InputPadder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate frozen MonSter++ plus trained refinement module on DrivingStereo.")
    parser.add_argument("--restore_ckpt", type=Path, default=PROJECT_DIR / "checkpoints" / "Mix_all_large.pth")
    parser.add_argument("--refinement_ckpt", type=Path, default=PROJECT_DIR / "checkpoints" / "refinement_module_final.pth")
    parser.add_argument("--arch", type=str, default="light", choices=["light", "large", "custom"])
    parser.add_argument("--refinement_channels", nargs="+", type=int, default=None, help="Only used with --arch custom, e.g. 4 64 128 64 1")
    parser.add_argument("--depth_anything_ckpt", type=Path, default=PROJECT_DIR / "checkpoints" / "depth_anything_v2_vitl.pth")
    parser.add_argument("--driving_root", type=Path, default=PROJECT_DIR / "datasets" / "drivingstereo")
    parser.add_argument("--image_set", type=str, default="cloudy", choices=["sunny", "cloudy", "foggy", "rainy", "test", "all"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--results_root", type=Path, default=PROJECT_DIR / "results" / "finetuned_driving_stereo")
    parser.add_argument("--baseline_metrics_path", type=Path, default=PROJECT_DIR / "results" / "baseline_mix_all_driving_stereo" / "metrics.json")
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
    parser.add_argument("--max_disp", type=int, default=416)
    parser.add_argument("--edge_thresholds", nargs="+", type=float, default=[0.2])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Reuse existing saved base/refined predictions when available.")
    return parser.parse_args()


def load_monster(args):
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


def load_refinement_module(args):
    if args.arch == "light":
        module = LightweightRefinementModule().cuda()
    elif args.arch == "large":
        module = LargeRefinementModule().cuda()
    else:
        if args.refinement_channels is None:
            raise ValueError("--refinement_channels is required when --arch custom is used")
        module = ConfigurableRefinementModule(args.refinement_channels).cuda()

    ckpt_path = args.refinement_ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "refinement_module" in checkpoint:
        checkpoint = checkpoint["refinement_module"]
    module.load_state_dict(checkpoint, strict=True)
    module.eval()
    return module


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


def save_prediction_outputs(base_dir: Path, stem: str, left_image_path: str, prediction_np: np.ndarray):
    npy_dir = base_dir / "npy"
    png_dir = base_dir / "png"
    input_dir = base_dir / "input"
    for out_dir in [npy_dir, png_dir, input_dir]:
        out_dir.mkdir(parents=True, exist_ok=True)

    np.save(npy_dir / f"{stem}.npy", prediction_np.astype(np.float32))
    plt.imsave(png_dir / f"{stem}.png", prediction_np.astype(np.float32), cmap="plasma")
    input_dest = input_dir / f"{stem}{Path(left_image_path).suffix}"
    if not input_dest.exists():
        shutil.copy2(left_image_path, input_dest)


def add_overlay_text(axis, lines, color="white"):
    y = 0.97
    for line in lines:
        text = line["text"] if isinstance(line, dict) else str(line)
        text_color = line.get("color", color) if isinstance(line, dict) else color
        axis.text(
            0.02,
            y,
            text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=text_color,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": (0, 0, 0, 0.7), "edgecolor": "none"},
        )
        y -= 0.095


def save_panel(
    output_path: Path,
    left_path: str,
    base_disp: np.ndarray,
    refined_disp: np.ndarray,
    residual: np.ndarray,
    gt_disp: np.ndarray,
    base_epe: float,
    base_d1: float,
    refined_epe: float,
    refined_d1: float,
    base_edge_epe: float,
    refined_edge_epe: float,
):
    left_img = np.array(Image.open(left_path).convert("RGB"))
    residual_abs = np.max(np.abs(residual))
    residual_vmax = residual_abs if residual_abs > 0 else 1.0
    edge_delta = refined_edge_epe - base_edge_epe
    delta_color = "#16a34a" if edge_delta < 0 else "#dc2626"
    base_abs_err = np.abs(base_disp - gt_disp)
    refined_abs_err = np.abs(refined_disp - gt_disp)
    err_vmax = max(float(base_abs_err.max()), float(refined_abs_err.max()), 1e-6)

    fig = plt.figure(figsize=(24, 5.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.4], wspace=0.08)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(4)]
    error_gs = gs[0, 4].subgridspec(1, 2, wspace=0.04)
    error_axes = [fig.add_subplot(error_gs[0, 0]), fig.add_subplot(error_gs[0, 1])]

    axes[0].imshow(left_img)
    axes[0].set_title("Left RGB")
    axes[1].imshow(base_disp, cmap="plasma")
    axes[1].set_title("Baseline Disparity")
    axes[2].imshow(refined_disp, cmap="plasma")
    axes[2].set_title("Refined Disparity")
    axes[3].imshow(residual, cmap="inferno", vmin=-residual_vmax, vmax=residual_vmax)
    axes[3].set_title(f"Residual R (max|R|={residual_abs:.4f})")
    error_axes[0].imshow(base_abs_err, cmap="RdYlBu_r", vmin=0.0, vmax=err_vmax)
    error_axes[0].set_title("|D_base - GT|")
    error_axes[1].imshow(refined_abs_err, cmap="RdYlBu_r", vmin=0.0, vmax=err_vmax)
    error_axes[1].set_title("|D_refined - GT|")

    for axis in axes:
        axis.axis("off")
    for axis in error_axes:
        axis.axis("off")

    add_overlay_text(
        axes[1],
        [
            {"text": f"EPE: {base_epe:.2f} | D1: {base_d1 * 100:.2f}%"},
            {"text": f"Edge EPE: {base_edge_epe:.2f}"},
        ],
    )
    add_overlay_text(
        axes[2],
        [
            {"text": f"EPE: {refined_epe:.2f} | D1: {refined_d1 * 100:.2f}%"},
            {"text": f"Edge EPE: {refined_edge_epe:.2f}"},
            {"text": f"Delta Edge EPE: {edge_delta:+.2f}", "color": delta_color},
        ],
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(metric_lists, edge_thresholds, runtimes):
    summary = {
        "mean_epe": float(np.mean(metric_lists["global_epe"])),
        "mean_d1": float(np.mean(metric_lists["global_d1"])),
        "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else None,
        "median_runtime_sec": float(np.median(runtimes)) if runtimes else None,
        "edge_metrics": {},
    }
    for thr in edge_thresholds:
        summary["edge_metrics"][f"sobel_{thr:.1f}"] = {
            "threshold": thr,
            "mean_edge_epe": float(np.mean(metric_lists[f"edge_epe_{thr}"])),
            "mean_edge_d1": float(np.mean(metric_lists[f"edge_d1_{thr}"])),
            "mean_edge_pixel_ratio_total": float(np.mean(metric_lists[f"edge_ratio_total_{thr}"])),
            "mean_edge_pixel_ratio_valid": float(np.mean(metric_lists[f"edge_ratio_valid_{thr}"])),
        }
    return summary


def init_metric_lists(edge_thresholds):
    metric_lists = {"global_epe": [], "global_d1": []}
    for thr in edge_thresholds:
        metric_lists[f"edge_epe_{thr}"] = []
        metric_lists[f"edge_d1_{thr}"] = []
        metric_lists[f"edge_ratio_total_{thr}"] = []
        metric_lists[f"edge_ratio_valid_{thr}"] = []
    return metric_lists


def evaluate_condition(args, model, refinement_module, image_set: str, results_root: Path, baseline_metrics_path: Path | None):
    dataset = datasets.DrivingStereo({}, root=str(args.driving_root), image_set=image_set)
    num_items = min(len(dataset), args.limit) if args.limit > 0 else len(dataset)

    baseline_root = results_root / "base"
    refined_root = results_root / "refined"
    panels_root = results_root / "qualitative_top_improved"
    rank_threshold = args.edge_thresholds[-1]

    for thr in args.edge_thresholds:
        (baseline_root / "edge_masks" / f"thr_{thr:.1f}").mkdir(parents=True, exist_ok=True)
        (refined_root / "edge_masks" / f"thr_{thr:.1f}").mkdir(parents=True, exist_ok=True)

    base_lists = init_metric_lists(args.edge_thresholds)
    refined_lists = init_metric_lists(args.edge_thresholds)
    improvement_records = []
    runtimes = []

    print(f"\nEvaluating DrivingStereo {image_set} ({num_items} images)")

    for idx in range(num_items):
        (left_path, _, _), image1, image2, flow_gt, valid_gt = dataset[idx]
        stem = Path(left_path).stem
        base_npy_path = baseline_root / "npy" / f"{stem}.npy"
        refined_npy_path = refined_root / "npy" / f"{stem}.npy"

        runtime = None
        if args.resume and base_npy_path.exists() and refined_npy_path.exists():
            base_np = np.load(base_npy_path).astype(np.float32)
            refined_np = np.load(refined_npy_path).astype(np.float32)
            residual_np = refined_np - base_np
            save_prediction_outputs(baseline_root, stem, left_path, base_np)
            save_prediction_outputs(refined_root, stem, left_path, refined_np)
        else:
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, divis_by=32)
            image1_pad, image2_pad = padder.pad(image1, image2)

            with torch.no_grad():
                with autocast(enabled=args.corr_implementation.endswith("_cuda")):
                    torch.cuda.synchronize()
                    start = time.time()
                    base_pred = model(image1_pad, image2_pad, iters=args.valid_iters, test_mode=True)
                    torch.cuda.synchronize()
                    runtime = time.time() - start
                base_pred = padder.unpad(base_pred).cpu().squeeze(0)
                refined_pred, residual = refinement_module(base_pred.unsqueeze(0).cuda(), image1.cuda())
                refined_pred = refined_pred.cpu().squeeze(0)
                residual = residual.cpu().squeeze(0)

            base_np = base_pred.numpy().squeeze().astype(np.float32)
            refined_np = refined_pred.numpy().squeeze().astype(np.float32)
            residual_np = residual.numpy().squeeze().astype(np.float32)

            save_prediction_outputs(baseline_root, stem, left_path, base_np)
            save_prediction_outputs(refined_root, stem, left_path, refined_np)

        if runtime is not None:
            runtimes.append(runtime)

        gt_disp = flow_gt.numpy().squeeze().astype(np.float32)
        valid_mask = (valid_gt.numpy().astype(bool)) & (np.abs(gt_disp) < args.max_disp)

        base_epe, base_d1 = compute_metrics(base_np, gt_disp, valid_mask)
        refined_epe, refined_d1 = compute_metrics(refined_np, gt_disp, valid_mask)
        base_lists["global_epe"].append(base_epe)
        base_lists["global_d1"].append(base_d1)
        refined_lists["global_epe"].append(refined_epe)
        refined_lists["global_d1"].append(refined_d1)

        total_pixels = gt_disp.size
        valid_pixels = int(valid_mask.sum())
        record = {
            "stem": stem,
            "left_path": left_path,
            "base": base_np,
            "refined": refined_np,
            "residual": residual_np,
            "gt_disp": gt_disp,
            "base_epe": base_epe,
            "base_d1": base_d1,
            "refined_epe": refined_epe,
            "refined_d1": refined_d1,
        }

        for thr in args.edge_thresholds:
            edge_mask = build_edge_mask(gt_disp, valid_mask, thr)
            cv2.imwrite(str(baseline_root / "edge_masks" / f"thr_{thr:.1f}" / f"{stem}.png"), (edge_mask.astype(np.uint8) * 255))
            cv2.imwrite(str(refined_root / "edge_masks" / f"thr_{thr:.1f}" / f"{stem}.png"), (edge_mask.astype(np.uint8) * 255))

            base_edge_epe, base_edge_d1 = compute_metrics(base_np, gt_disp, edge_mask)
            refined_edge_epe, refined_edge_d1 = compute_metrics(refined_np, gt_disp, edge_mask)

            base_lists[f"edge_epe_{thr}"].append(base_edge_epe)
            base_lists[f"edge_d1_{thr}"].append(base_edge_d1)
            base_lists[f"edge_ratio_total_{thr}"].append(float(edge_mask.sum() / total_pixels))
            base_lists[f"edge_ratio_valid_{thr}"].append(float(edge_mask.sum() / valid_pixels) if valid_pixels > 0 else 0.0)

            refined_lists[f"edge_epe_{thr}"].append(refined_edge_epe)
            refined_lists[f"edge_d1_{thr}"].append(refined_edge_d1)
            refined_lists[f"edge_ratio_total_{thr}"].append(float(edge_mask.sum() / total_pixels))
            refined_lists[f"edge_ratio_valid_{thr}"].append(float(edge_mask.sum() / valid_pixels) if valid_pixels > 0 else 0.0)

            if abs(thr - rank_threshold) < 1e-6:
                record["edge_improvement"] = base_edge_epe - refined_edge_epe
                record["base_edge_epe"] = base_edge_epe
                record["refined_edge_epe"] = refined_edge_epe

        improvement_records.append(record)

        if idx < 5 or (idx + 1) % 25 == 0:
            runtime_msg = f"{runtime:.3f}s" if runtime is not None else "cached"
            print(
                f"[{image_set} {idx + 1}/{num_items}] {stem} "
                f"base_EPE={base_epe:.4f} refined_EPE={refined_epe:.4f} "
                f"base_D1={base_d1 * 100:.3f}% refined_D1={refined_d1 * 100:.3f}% "
                f"runtime={runtime_msg}"
            )

    base_summary = summarize_metrics(base_lists, args.edge_thresholds, runtimes)
    refined_summary = summarize_metrics(refined_lists, args.edge_thresholds, runtimes)

    top_records = sorted(improvement_records, key=lambda item: item.get("edge_improvement", -np.inf), reverse=True)[: args.top_k]
    for rank, record in enumerate(top_records, start=1):
        save_panel(
            panels_root / f"{rank:02d}_{record['stem']}.png",
            record["left_path"],
            record["base"],
            record["refined"],
            record["residual"],
            record["gt_disp"],
            record["base_epe"],
            record["base_d1"],
            record["refined_epe"],
            record["refined_d1"],
            record["base_edge_epe"],
            record["refined_edge_epe"],
        )

    baseline_consistency = None
    if baseline_metrics_path is not None and baseline_metrics_path.exists():
        previous = json.loads(baseline_metrics_path.read_text())
        baseline_consistency = {
            "previous_mean_epe": previous.get("mean_epe"),
            "previous_mean_d1": previous.get("mean_d1"),
            "recomputed_mean_epe": base_summary["mean_epe"],
            "recomputed_mean_d1": base_summary["mean_d1"],
            "epe_abs_diff": abs(base_summary["mean_epe"] - previous.get("mean_epe", base_summary["mean_epe"])),
            "d1_abs_diff": abs(base_summary["mean_d1"] - previous.get("mean_d1", base_summary["mean_d1"])),
        }

    metrics = {
        "dataset": "DrivingStereo",
        "image_set": image_set,
        "num_images_evaluated": num_items,
        "max_disp_threshold": args.max_disp,
        "valid_iters": args.valid_iters,
        "base_metrics": base_summary,
        "refined_metrics": refined_summary,
        "qualitative_dir": str(panels_root.resolve()),
        "baseline_consistency_check": baseline_consistency,
    }

    results_root.mkdir(parents=True, exist_ok=True)
    metrics_path = results_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {image_set} metrics to: {metrics_path}")

    return {
        "metrics": metrics,
        "base_lists": base_lists,
        "refined_lists": refined_lists,
        "runtimes": runtimes,
    }


def main():
    args = parse_args()
    for path in [args.restore_ckpt, args.refinement_ckpt, args.depth_anything_ckpt, args.driving_root]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    model = load_monster(args)
    refinement_module = load_refinement_module(args)
    image_sets = ["cloudy", "sunny", "rainy", "foggy"] if args.image_set == "all" else [args.image_set]
    all_results = {}
    overall_base_lists = init_metric_lists(args.edge_thresholds)
    overall_refined_lists = init_metric_lists(args.edge_thresholds)
    overall_runtimes = []
    total_images = 0

    for image_set in image_sets:
        condition_root = args.results_root / image_set if args.image_set == "all" else args.results_root
        baseline_metrics_path = args.baseline_metrics_path if image_set == "cloudy" else None
        result = evaluate_condition(args, model, refinement_module, image_set, condition_root, baseline_metrics_path)
        all_results[image_set] = result["metrics"]
        total_images += result["metrics"]["num_images_evaluated"]
        overall_runtimes.extend(result["runtimes"])
        for key, values in result["base_lists"].items():
            overall_base_lists[key].extend(values)
        for key, values in result["refined_lists"].items():
            overall_refined_lists[key].extend(values)

    overall_base_summary = summarize_metrics(overall_base_lists, args.edge_thresholds, overall_runtimes)
    overall_refined_summary = summarize_metrics(overall_refined_lists, args.edge_thresholds, overall_runtimes)
    edge_key = f"sobel_{args.edge_thresholds[-1]:.1f}"

    summary_payload = {
        "dataset": "DrivingStereo",
        "conditions": {},
        "overall": {
            "num_images_evaluated": total_images,
            "base_metrics": overall_base_summary,
            "refined_metrics": overall_refined_summary,
        },
    }
    for image_set, metrics in all_results.items():
        summary_payload["conditions"][image_set] = metrics

    summary_path = args.results_root / "all_weather_metrics.json"
    args.results_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print("\nAll-weather summary")
    print("Condition   Base EPE  Refined EPE  EPE Delta  Base D1(%)  Refined D1(%)  D1 Delta(%)")
    for image_set in image_sets:
        metrics = all_results[image_set]
        base_summary = metrics["base_metrics"]
        refined_summary = metrics["refined_metrics"]
        print(
            f"{image_set.capitalize():<10}"
            f"{base_summary['mean_epe']:<10.4f}"
            f"{refined_summary['mean_epe']:<13.4f}"
            f"{(refined_summary['mean_epe'] - base_summary['mean_epe']):<9.4f}"
            f"{base_summary['mean_d1'] * 100:<13.3f}"
            f"{refined_summary['mean_d1'] * 100:<17.3f}"
            f"{((refined_summary['mean_d1'] - base_summary['mean_d1']) * 100):<8.3f}"
        )
    print(
        f"{'Overall':<10}"
        f"{overall_base_summary['mean_epe']:<10.4f}"
        f"{overall_refined_summary['mean_epe']:<13.4f}"
        f"{(overall_refined_summary['mean_epe'] - overall_base_summary['mean_epe']):<9.4f}"
        f"{overall_base_summary['mean_d1'] * 100:<13.3f}"
        f"{overall_refined_summary['mean_d1'] * 100:<17.3f}"
        f"{((overall_refined_summary['mean_d1'] - overall_base_summary['mean_d1']) * 100):<8.3f}"
    )

    print(f"\nEdge metrics shown in JSON use threshold t={args.edge_thresholds[-1]:.1f} under key '{edge_key}'.")
    print(f"Saved all-weather metrics to: {summary_path}")


if __name__ == "__main__":
    main()
