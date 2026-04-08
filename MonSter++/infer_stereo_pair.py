import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / "core"))

from monster import Monster
from utils.utils import InputPadder


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run MonSter++ on a single stereo pair.")
    parser.add_argument(
        "--restore_ckpt",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "KITTI_large.pth",
        help="Path to the MonSter++ checkpoint (.pth).",
    )
    parser.add_argument(
        "--depth_anything_ckpt",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "depth_anything_v2_vitl.pth",
        help="Path to the Depth-Anything V2 backbone checkpoint (.pth).",
    )
    parser.add_argument("--left_image", type=Path, required=True, help="Path to the left image.")
    parser.add_argument("--right_image", type=Path, required=True, help="Path to the right image.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROJECT_DIR / "outputs" / "depth_map_plasma.png",
        help="Path to the output PNG visualization.",
    )
    parser.add_argument("--valid_iters", type=int, default=32, help="Number of refinement iterations.")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--slow_fast_gru", action="store_true")
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument(
        "--max_disp",
        type=int,
        default=192,
        help="Set to 192 for KITTI_large and 416 for Mix_all_large.",
    )
    return parser


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)


def load_monster_checkpoint(model: torch.nn.Module, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]

    state_dict = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        state_dict[new_key] = value

    model.load_state_dict(state_dict, strict=True)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MonSter++ inference on this script.")

    for path in [args.restore_ckpt, args.depth_anything_ckpt, args.left_image, args.right_image]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    device = torch.device("cuda")
    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = Monster(args).to(device)
    load_monster_checkpoint(model, args.restore_ckpt)
    model.eval()

    torch.cuda.empty_cache()

    image1 = load_image(args.left_image, device)
    image2 = load_image(args.right_image, device)
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    start_time = time.time()
    with torch.no_grad():
        prediction = model(image1, image2, iters=args.valid_iters, test_mode=True)
    elapsed = time.time() - start_time

    prediction = padder.unpad(prediction).detach().cpu().squeeze().numpy()

    plt.imsave(output_path, prediction, cmap="plasma")

    print(f"Saved depth map visualization to: {output_path}")
    print(f"Inference finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
