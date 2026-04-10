import argparse
import copy
import json
import math
import random
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from core.monster import Monster
from refinement_module import LightweightRefinementModule
import core.stereo_datasets as datasets


def charbonnier(x, epsilon=1e-3):
    return torch.sqrt(x**2 + epsilon**2)


def gradient_loss(pred, gt, valid_mask, epsilon=1e-3):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]

    valid_x = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
    valid_y = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]

    loss_dx = charbonnier(pred_dx - gt_dx, epsilon=epsilon)
    loss_dy = charbonnier(pred_dy - gt_dy, epsilon=epsilon)

    grad_x = loss_dx[valid_x].mean() if valid_x.any() else pred.new_tensor(0.0)
    grad_y = loss_dy[valid_y].mean() if valid_y.any() else pred.new_tensor(0.0)
    return (grad_x + grad_y) / 2


def edge_aware_loss(pred, gt, valid_mask, lambda_grad=0.1, epsilon=1e-3):
    valid_sum = valid_mask.float().sum().clamp_min(1.0)
    base = (charbonnier(pred - gt, epsilon=epsilon) * valid_mask.float()).sum() / valid_sum
    grad = gradient_loss(pred, gt, valid_mask, epsilon=epsilon)
    total = base + lambda_grad * grad
    return total, base, grad


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight refinement module on top of frozen MonSter++.")
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--restore_ckpt", type=Path, default=project_dir / "checkpoints" / "Mix_all_large.pth")
    parser.add_argument("--depth_anything_ckpt", type=Path, default=project_dir / "checkpoints" / "depth_anything_v2_vitl.pth")
    parser.add_argument("--kitti_root", type=Path, default=project_dir / "datasets" / "data_scene_flow")
    parser.add_argument("--save_dir", type=Path, default=project_dir / "checkpoints")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--train_iters", type=int, default=22)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pct_start", type=float, default=0.01)
    parser.add_argument("--image_height", type=int, default=320)
    parser.add_argument("--image_width", type=int, default=736)
    parser.add_argument("--lambda_grad", type=float, default=0.1)
    parser.add_argument("--loss_max_disp", type=float, default=192.0)
    parser.add_argument("--max_disp", type=int, default=416)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=655)
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--slow_fast_gru", action="store_true")
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument("--repeat_first_batch", action="store_true", help="Reuse the first batch every step; useful for dry runs.")
    return parser.parse_args()


def load_monster(args, device):
    model = Monster(args)
    checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]

    state_dict = {}
    for key, value in checkpoint.items():
        state_dict[key.replace("module.", "")] = value
    model.load_state_dict(state_dict, strict=True)

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    model.to(device)
    return model


def build_dataloader(args):
    aug_params = {
        "crop_size": [args.image_height, args.image_width],
        "min_scale": -0.2,
        "max_scale": 0.5,
        "do_flip": False,
        "yjitter": False,
        "saturation_range": [0.7, 1.3],
    }
    dataset = datasets.KITTI_2015(aug_params, root=str(args.kitti_root), image_set="training")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataset, loader


def get_valid_mask(disp_gt, valid, max_disp):
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    return ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)


def grad_status(module):
    stats = {}
    for name, param in module.named_parameters():
        if param.grad is None:
            stats[name] = None
        else:
            stats[name] = float(param.grad.detach().abs().mean().item())
    return stats


def save_checkpoint(path, refinement_module, optimizer, scheduler, step, best_loss):
    payload = {
        "refinement_module": refinement_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_loss": best_loss,
    }
    torch.save(payload, path)


def append_training_log(csv_path, step, total_loss, base_loss, grad_loss, learning_rate):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["step", "total_loss", "base_loss", "grad_loss", "learning_rate", "timestamp"])
        writer.writerow([
            step,
            f"{total_loss:.8f}",
            f"{base_loss:.8f}",
            f"{grad_loss:.8f}",
            f"{learning_rate:.10f}",
            datetime.now().isoformat(timespec="seconds"),
        ])


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this fine-tuning script.")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    dataset, dataloader = build_dataloader(args)
    model = load_monster(args, device)
    refinement_module = LightweightRefinementModule().to(device)

    optimizer = AdamW(refinement_module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(args.max_steps, 10),
        pct_start=args.pct_start,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    parameter_count = count_parameters(refinement_module)
    print(f"Refinement module trainable parameters: {parameter_count}")
    print(f"KITTI 2015 training samples available: {len(dataset)}")

    latest_checkpoint_path = args.save_dir / "latest.pth"
    training_log_path = args.save_dir / "training_log.csv"
    best_loss = math.inf
    start_step = 0

    if latest_checkpoint_path.exists():
        latest_checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        refinement_module.load_state_dict(latest_checkpoint["refinement_module"], strict=True)
        optimizer.load_state_dict(latest_checkpoint["optimizer"])
        scheduler.load_state_dict(latest_checkpoint["scheduler"])
        start_step = int(latest_checkpoint.get("step", 0))
        best_loss = float(latest_checkpoint.get("best_loss", math.inf))
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        print(f"Resuming from step {start_step} with best_loss={best_loss:.6f}")
    else:
        print("No latest checkpoint found. Starting fresh training run.")

    first_batch = None
    loader_iter = iter(dataloader)
    loss_history = []

    for step in range(start_step, args.max_steps):
        if args.repeat_first_batch and first_batch is not None:
            batch = first_batch
        else:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                batch = next(loader_iter)
            if args.repeat_first_batch and first_batch is None:
                first_batch = tuple(x.clone() if torch.is_tensor(x) else copy.deepcopy(x) for x in batch)

        _, left, right, disp_gt, valid = batch
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        disp_gt = disp_gt.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        refinement_module.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            _, disp_preds, _ = model(left, right, iters=args.train_iters)
            d_base = disp_preds[-1].detach()

        d_refined, residual = refinement_module(d_base, left)
        valid_mask = get_valid_mask(disp_gt, valid, args.loss_max_disp)
        total_loss, base_loss, grad_loss = edge_aware_loss(d_refined, disp_gt, valid_mask, lambda_grad=args.lambda_grad)
        total_loss.backward()

        monster_has_grads = any(param.grad is not None for param in model.parameters())
        refinement_grad_stats = grad_status(refinement_module)
        refinement_has_any_grad = any(value is not None and value > 0 for value in refinement_grad_stats.values())

        optimizer.step()
        scheduler.step()

        loss_value = float(total_loss.detach().item())
        base_value = float(base_loss.detach().item())
        grad_value = float(grad_loss.detach().item())
        loss_history.append(loss_value)

        if step == 0:
            print(f"Step 0 base loss: {base_value:.6f}")
            print(f"Step 0 grad loss: {grad_value:.6f}")
            print(f"Step 0 total loss: {loss_value:.6f}")
            print(f"Frozen MonSter++ has gradients: {monster_has_grads}")
            print(f"Refinement module has nonzero gradients: {refinement_has_any_grad}")
            print("Refinement grad means:")
            for name, value in refinement_grad_stats.items():
                status = "None" if value is None else f"{value:.8f}"
                print(f"  {name}: {status}")

        if (step + 1) % args.log_every == 0 or step == 0 or step == args.max_steps - 1:
            learning_rate = scheduler.get_last_lr()[0]
            print(
                f"step={step + 1}/{args.max_steps} "
                f"total_loss={loss_value:.6f} "
                f"base_loss={base_value:.6f} "
                f"grad_loss={grad_value:.6f} "
                f"lr={learning_rate:.8f}"
            )
            append_training_log(training_log_path, step + 1, loss_value, base_value, grad_value, learning_rate)

        if loss_value < best_loss:
            best_loss = loss_value
            save_checkpoint(args.save_dir / "refinement_module_best.pth", refinement_module, optimizer, scheduler, step + 1, best_loss)

        save_checkpoint(latest_checkpoint_path, refinement_module, optimizer, scheduler, step + 1, best_loss)

        if (step + 1) % args.save_every == 0:
            save_checkpoint(args.save_dir / f"refinement_module_step_{step + 1}.pth", refinement_module, optimizer, scheduler, step + 1, best_loss)

    final_summary = {
        "steps": args.max_steps,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "best_loss": best_loss,
        "refinement_parameters": parameter_count,
        "repeat_first_batch": args.repeat_first_batch,
    }
    summary_path = args.save_dir / "refinement_module_last_run.json"
    summary_path.write_text(json.dumps(final_summary, indent=2))
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
