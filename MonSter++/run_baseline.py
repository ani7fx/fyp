import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / "core"))

import stereo_datasets as datasets
from monster import Monster, autocast
from utils.utils import InputPadder


def parse_args():
    parser = argparse.ArgumentParser(description="Run MonSter++ baseline on KITTI 2015 training set.")
    parser.add_argument("--restore_ckpt", type=Path, default=PROJECT_DIR / "checkpoints" / "KITTI_large.pth")
    parser.add_argument("--depth_anything_ckpt", type=Path, default=PROJECT_DIR / "checkpoints" / "depth_anything_v2_vitl.pth")
    parser.add_argument("--kitti_root", type=Path, default=PROJECT_DIR / "datasets" / "data_scene_flow")
    parser.add_argument("--results_root", type=Path, default=PROJECT_DIR / "results" / "baseline")
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


def save_outputs(base_dir: Path, stem: str, left_image_path: str, prediction: torch.Tensor):
    npy_dir = base_dir / "npy"
    png_dir = base_dir / "png"
    input_dir = base_dir / "input"
    for out_dir in [npy_dir, png_dir, input_dir]:
        out_dir.mkdir(parents=True, exist_ok=True)

    prediction_np = prediction.cpu().numpy().squeeze().astype(np.float32)
    np.save(npy_dir / f"{stem}.npy", prediction_np)
    plt.imsave(png_dir / f"{stem}.png", prediction_np, cmap="plasma")
    shutil.copy2(left_image_path, input_dir / f"{stem}.png")


def main():
    args = parse_args()

    for required_path in [args.restore_ckpt, args.depth_anything_ckpt, args.kitti_root]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    model = load_model(args)
    dataset = datasets.KITTI_2015({}, root=str(args.kitti_root), image_set="training")

    epe_list = []
    d1_list = []
    runtimes = []

    for idx in range(len(dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = dataset[idx]
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
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < args.max_disp)
        epe_flat = epe.flatten()
        d1 = (epe_flat > 3.0)[val].float().mean().item()
        mean_epe = epe_flat[val].mean().item()

        stem = Path(imageL_file).stem
        save_outputs(args.results_root, stem, imageL_file, flow_pr)

        epe_list.append(mean_epe)
        d1_list.append(d1)
        runtimes.append(runtime)

        if idx < 5 or (idx + 1) % 25 == 0:
            print(
                f"[{idx + 1}/{len(dataset)}] {stem} "
                f"EPE={mean_epe:.4f} D1={d1:.4f} runtime={runtime:.3f}s"
            )

    metrics = {
        "dataset": "KITTI_2015_training",
        "kitti_root": str(args.kitti_root.resolve()),
        "num_images": len(dataset),
        "max_disp_threshold": args.max_disp,
        "valid_iters": args.valid_iters,
        "mean_epe": float(np.mean(epe_list)),
        "mean_d1": float(np.mean(d1_list)),
        "mean_runtime_sec": float(np.mean(runtimes)),
        "median_runtime_sec": float(np.median(runtimes)),
    }

    args.results_root.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
