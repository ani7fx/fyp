# Kubeflow Setup

This repo contains local modifications to MonSter++ for:
- KITTI 2015 baseline evaluation
- DrivingStereo baseline evaluation
- edge-aware metrics
- refinement-module fine-tuning with frozen MonSter++

## Recommended Workflow

1. Clone this repo in the Kubeflow notebook.
2. Create a Python environment.
3. Install dependencies.
4. Upload or mount datasets and checkpoints outside Git.
5. Update paths only if your mounted directories differ from the defaults below.

## Python Environment

Recommended:
- Python 3.10
- CUDA-compatible PyTorch 2.4.1

Example install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm scipy opencv-python scikit-image tensorboard matplotlib timm==0.6.13 accelerate==1.0.1 gradio_imageslider gradio==4.29.0 openexr pyexr imath h5py swanlab opt_einsum hydra-core omegaconf
```

Notes:
- `mmcv` is not required for the current MonSter++ inference / evaluation / refinement workflow here.
- `pytorch3d` is also not required for the KITTI and DrivingStereo workflows in this repo.

## Required Checkpoints

Place these files in:

```text
MonSter++/checkpoints/
```

Required:
- `Mix_all_large.pth`
- `KITTI_large.pth`
- `depth_anything_v2_vitl.pth`

## Dataset Layout

Do not commit datasets to Git. Mount or copy them separately.

### KITTI 2015

Expected root:

```text
MonSter++/datasets/data_scene_flow/
```

Expected structure:

```text
MonSter++/datasets/data_scene_flow/
  training/
    image_2/
    image_3/
    disp_noc_0/
  testing/
    image_2/
    image_3/
```

### DrivingStereo

Expected root:

```text
MonSter++/datasets/drivingstereo/
```

Expected structure for a weather subset such as `cloudy`:

```text
MonSter++/datasets/drivingstereo/cloudy/
  left-image-half-size/
  right-image-half-size/
  disparity-map-half-size/
```

Optional extra folders are fine:
- `left-image-full-size/`
- `right-image-full-size/`
- `depth-map-half-size/`
- `depth-map-full-size/`
- `disparity-map-full-size/`

## Useful Commands

### KITTI baseline artifacts

```bash
python MonSter++/run_baseline.py \
  --restore_ckpt MonSter++/checkpoints/KITTI_large.pth \
  --kitti_root MonSter++/datasets/data_scene_flow \
  --results_root MonSter++/results/baseline \
  --max_disp 192
```

### KITTI edge metrics on saved predictions

```bash
python MonSter++/evaluate_metrics.py \
  --pred_dir MonSter++/results/baseline/npy \
  --kitti_root MonSter++/datasets/data_scene_flow \
  --metrics_path MonSter++/results/baseline/metrics.json \
  --mask_root MonSter++/results/baseline/edge_masks \
  --max_disp 192
```

### DrivingStereo baseline with KITTI weights

```bash
python MonSter++/evaluate_driving_stereo.py \
  --driving_root MonSter++/datasets/drivingstereo \
  --image_set cloudy \
  --limit 500 \
  --results_root MonSter++/results/baseline_driving_stereo \
  --restore_ckpt MonSter++/checkpoints/KITTI_large.pth \
  --max_disp 192
```

### DrivingStereo baseline with mix_all weights

```bash
python MonSter++/evaluate_driving_stereo.py \
  --driving_root MonSter++/datasets/drivingstereo \
  --image_set cloudy \
  --limit 500 \
  --results_root MonSter++/results/baseline_mix_all_driving_stereo \
  --restore_ckpt MonSter++/checkpoints/Mix_all_large.pth \
  --max_disp 416
```

### Refinement-module fine-tuning

Full training:

```bash
python MonSter++/finetune_with_module.py \
  --restore_ckpt MonSter++/checkpoints/Mix_all_large.pth \
  --kitti_root MonSter++/datasets/data_scene_flow \
  --batch_size 12 \
  --max_steps 10000 \
  --lr 1e-4
```

Short dry run:

```bash
python MonSter++/finetune_with_module.py \
  --batch_size 1 \
  --max_steps 5 \
  --log_every 1 \
  --num_workers 0 \
  --repeat_first_batch
```

## Notes On Current Fine-Tuning Setup

- Base MonSter++ is frozen.
- Only the lightweight refinement module is trainable.
- The refinement module starts as identity because the final conv weights and bias are initialized to zero.
- The fine-tuning script saves:
  - `MonSter++/checkpoints/refinement_module_best.pth`
  - `MonSter++/checkpoints/refinement_module_step_*.pth`

## Suggested Cluster Practice

- Keep datasets and checkpoints on mounted persistent storage.
- Keep the repo clone lightweight and code-only.
- Save outputs and checkpoints to persistent storage, not ephemeral notebook storage.
- Record the exact commit hash used for each training run.
