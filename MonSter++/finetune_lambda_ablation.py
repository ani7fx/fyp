import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from core.monster import Monster
import core.stereo_datasets as datasets
from refinement_module import LightweightRefinementModule


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
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Sequential lambda_grad ablation for lightweight MonSter++ refinement.")
    parser.add_argument("--kitti_root", type=Path, default=project_dir / "datasets" / "data_scene_flow")
    parser.add_argument("--restore_ckpt", type=Path, default=project_dir / "checkpoints" / "Mix_all_large.pth")
    parser.add_argument("--depth_anything_ckpt", type=Path, default=project_dir / "checkpoints" / "depth_anything_v2_vitl.pth")
    parser.add_argument("--checkpoint_dir", type=Path, default=project_dir / "checkpoints")
    parser.add_argument("--log_dir", type=Path, default=project_dir / "logs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--dry_run_steps", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--train_iters", type=int, default=22)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pct_start", type=float, default=0.01)
    parser.add_argument("--image_height", type=int, default=320)
    parser.add_argument("--image_width", type=int, default=736)
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
    return parser.parse_args()


@dataclass(frozen=True)
class RunConfig:
    label: str
    lambda_grad: float
    checkpoint_name: str
    log_name: str


RUN_CONFIGS = [
    RunConfig(
        label="lambda005",
        lambda_grad=0.05,
        checkpoint_name="refinement_lambda005.pth",
        log_name="lambda005_log.csv",
    ),
    RunConfig(
        label="lambda020",
        lambda_grad=0.20,
        checkpoint_name="refinement_lambda020.pth",
        log_name="lambda020_log.csv",
    ),
]


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


def save_checkpoint(path, refinement_module, optimizer, scheduler, step, best_loss, lambda_grad):
    payload = {
        "refinement_module": refinement_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_loss": best_loss,
        "lambda_grad": lambda_grad,
    }
    torch.save(payload, path)


def append_training_log(csv_path, step, total_loss, base_loss, grad_loss, learning_rate):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["step", "total_loss", "base_loss", "grad_loss", "learning_rate", "timestamp"])
        writer.writerow(
            [
                step,
                f"{total_loss:.8f}",
                f"{base_loss:.8f}",
                f"{grad_loss:.8f}",
                f"{learning_rate:.10f}",
                datetime.now().isoformat(timespec="seconds"),
            ]
        )


def fetch_batch(loader_iter, dataloader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(dataloader)
        batch = next(loader_iter)
    return batch, loader_iter


def unpack_batch(batch, device):
    _, left, right, disp_gt, valid = batch
    left = left.to(device, non_blocking=True)
    right = right.to(device, non_blocking=True)
    disp_gt = disp_gt.to(device, non_blocking=True)
    valid = valid.to(device, non_blocking=True)
    return left, right, disp_gt, valid


def run_single_step(model, refinement_module, optimizer, scheduler, batch, device, args, lambda_grad):
    left, right, disp_gt, valid = unpack_batch(batch, device)
    refinement_module.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        _, disp_preds, _ = model(left, right, iters=args.train_iters)
        d_base = disp_preds[-1].detach()

    d_refined, _ = refinement_module(d_base, left)
    valid_mask = get_valid_mask(disp_gt, valid, args.loss_max_disp)
    total_loss, base_loss, grad_loss = edge_aware_loss(d_refined, disp_gt, valid_mask, lambda_grad=lambda_grad)
    total_loss.backward()

    monster_has_grads = any(param.grad is not None for param in model.parameters())
    refinement_grad_stats = grad_status(refinement_module)
    refinement_has_any_grad = any(value is not None and value > 0 for value in refinement_grad_stats.values())

    optimizer.step()
    scheduler.step()

    return {
        "total_loss": float(total_loss.detach().item()),
        "base_loss": float(base_loss.detach().item()),
        "grad_loss": float(grad_loss.detach().item()),
        "monster_has_grads": monster_has_grads,
        "refinement_has_any_grad": refinement_has_any_grad,
        "refinement_grad_stats": refinement_grad_stats,
        "learning_rate": float(scheduler.get_last_lr()[0]),
    }


def run_dry_run(model, dataloader, device, args, run_config):
    print("=" * 88)
    print(f"Dry run for {run_config.label} (lambda_grad={run_config.lambda_grad:.2f})")

    refinement_module = LightweightRefinementModule().to(device)
    optimizer = AdamW(refinement_module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(args.dry_run_steps, 10),
        pct_start=args.pct_start,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    loader_iter = iter(dataloader)
    first_batch, loader_iter = fetch_batch(loader_iter, dataloader)
    fixed_batch = tuple(x.clone() if torch.is_tensor(x) else copy.deepcopy(x) for x in first_batch)

    losses = []
    first_step_stats = None
    for step_idx in range(args.dry_run_steps):
        step_result = run_single_step(
            model=model,
            refinement_module=refinement_module,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=fixed_batch,
            device=device,
            args=args,
            lambda_grad=run_config.lambda_grad,
        )
        losses.append(step_result["total_loss"])
        if first_step_stats is None:
            first_step_stats = step_result

    print(f"Dry run losses: {[round(value, 6) for value in losses]}")
    print(f"Frozen MonSter++ has gradients: {first_step_stats['monster_has_grads']}")
    print(f"Refinement module has nonzero gradients: {first_step_stats['refinement_has_any_grad']}")

    if first_step_stats["monster_has_grads"]:
        raise RuntimeError(f"Dry run failed for {run_config.label}: gradients flowed into frozen MonSter++")
    if not first_step_stats["refinement_has_any_grad"]:
        raise RuntimeError(f"Dry run failed for {run_config.label}: no gradients flowed through refinement module")
    if losses[-1] > losses[0]:
        raise RuntimeError(
            f"Dry run failed for {run_config.label}: loss did not decrease on repeated-batch check "
            f"({losses[0]:.6f} -> {losses[-1]:.6f})"
        )

    print(
        f"Dry run passed for {run_config.label}: "
        f"initial_loss={losses[0]:.6f}, final_loss={losses[-1]:.6f}"
    )


def train_run(model, dataset, dataloader, device, args, run_config):
    print("\n" + "=" * 88)
    print(f"Starting full training for {run_config.label}")
    print(f"lambda_grad = {run_config.lambda_grad:.2f}")
    print(f"checkpoint = {args.checkpoint_dir / run_config.checkpoint_name}")
    print(f"log = {args.log_dir / run_config.log_name}")
    print("=" * 88)

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

    log_path = args.log_dir / run_config.log_name
    best_checkpoint_path = args.checkpoint_dir / run_config.checkpoint_name
    latest_checkpoint_path = args.checkpoint_dir / f"{run_config.label}_latest.pth"
    best_loss = math.inf
    loader_iter = iter(dataloader)
    loss_history = []

    for step in range(args.max_steps):
        batch, loader_iter = fetch_batch(loader_iter, dataloader)
        step_result = run_single_step(
            model=model,
            refinement_module=refinement_module,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=batch,
            device=device,
            args=args,
            lambda_grad=run_config.lambda_grad,
        )

        loss_value = step_result["total_loss"]
        base_value = step_result["base_loss"]
        grad_value = step_result["grad_loss"]
        learning_rate = step_result["learning_rate"]
        loss_history.append(loss_value)

        if step == 0:
            print(f"Step 0 base loss: {base_value:.6f}")
            print(f"Step 0 grad loss: {grad_value:.6f}")
            print(f"Step 0 total loss: {loss_value:.6f}")
            print(f"Frozen MonSter++ has gradients: {step_result['monster_has_grads']}")
            print(f"Refinement module has nonzero gradients: {step_result['refinement_has_any_grad']}")
            print("Refinement grad means:")
            for name, value in step_result["refinement_grad_stats"].items():
                status = "None" if value is None else f"{value:.8f}"
                print(f"  {name}: {status}")

        if (step + 1) % args.log_every == 0 or step == 0 or step == args.max_steps - 1:
            print(
                f"{run_config.label} step={step + 1}/{args.max_steps} "
                f"total_loss={loss_value:.6f} "
                f"base_loss={base_value:.6f} "
                f"grad_loss={grad_value:.6f} "
                f"lr={learning_rate:.8f}"
            )
            append_training_log(log_path, step + 1, loss_value, base_value, grad_value, learning_rate)

        if loss_value < best_loss:
            best_loss = loss_value
            save_checkpoint(
                best_checkpoint_path,
                refinement_module,
                optimizer,
                scheduler,
                step + 1,
                best_loss,
                run_config.lambda_grad,
            )

        save_checkpoint(
            latest_checkpoint_path,
            refinement_module,
            optimizer,
            scheduler,
            step + 1,
            best_loss,
            run_config.lambda_grad,
        )

        if (step + 1) % args.save_every == 0:
            save_checkpoint(
                args.checkpoint_dir / f"{run_config.label}_step{step + 1}.pth",
                refinement_module,
                optimizer,
                scheduler,
                step + 1,
                best_loss,
                run_config.lambda_grad,
            )

    run_summary = {
        "label": run_config.label,
        "lambda_grad": run_config.lambda_grad,
        "steps": args.max_steps,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "best_loss": best_loss,
        "refinement_parameters": parameter_count,
        "checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "training_log": str(log_path),
    }
    summary_path = args.checkpoint_dir / f"{run_config.label}_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))

    print("-" * 88)
    print(
        f"Completed {run_config.label}: "
        f"final_loss={run_summary['final_loss']:.6f}, "
        f"best_loss={run_summary['best_loss']:.6f}"
    )
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Summary saved to: {summary_path}")
    print("-" * 88)
    return run_summary


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this fine-tuning script.")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    dataset, dataloader = build_dataloader(args)
    model = load_monster(args, device)

    print("Preparing lambda ablation runs for the lightweight refinement module.")
    print(f"Frozen MonSter++ checkpoint: {args.restore_ckpt}")
    print(f"KITTI root: {args.kitti_root}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Log dir: {args.log_dir}")

    dry_run_summaries = {}
    for run_config in RUN_CONFIGS:
        run_dry_run(model, dataloader, device, args, run_config)
        dry_run_summaries[run_config.label] = "passed"

    print("\nAll dry runs passed. Starting full training jobs.\n")

    run_summaries = []
    for idx, run_config in enumerate(RUN_CONFIGS, start=1):
        set_seed(args.seed + idx)
        run_summary = train_run(model, dataset, dataloader, device, args, run_config)
        run_summaries.append(run_summary)

    final_report = {
        "runs": run_summaries,
        "dry_runs": dry_run_summaries,
    }
    final_report_path = args.checkpoint_dir / "lambda_ablation_summary.json"
    final_report_path.write_text(json.dumps(final_report, indent=2))

    print("\n" + "=" * 88)
    print("Lambda ablation complete")
    print("Final training losses:")
    print("Run          Lambda    Final Loss    Best Loss")
    for summary in run_summaries:
        print(
            f"{summary['label']:<12}"
            f"{summary['lambda_grad']:<10.2f}"
            f"{summary['final_loss']:<14.6f}"
            f"{summary['best_loss']:<12.6f}"
        )
    print(f"Saved combined summary to: {final_report_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
