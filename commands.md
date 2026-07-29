# 3DGUT Command Reference

This document contains the commands I commonly use while working with my modified version of 3DGUT.

---

# Training

## Kitchen dataset (standard)

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/frames \
    out_dir=runs \
    experiment_name=Kitchen
```

## Kitchen dataset (without extra metrics)

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/frames \
    out_dir=runs \
    experiment_name=Kitchen \
    compute_extra_metrics=False
```

## Kitchen dataset (50,000 iterations)

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/frames \
    out_dir=runs \
    experiment_name=Kitchen \
    compute_extra_metrics=False \
    n_iterations=50000
```

---

# Original Default Command (Repository)

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/mipnerf360/bonsai \
    out_dir=runs \
    experiment_name=bonsai_3dgut \
    dataset.downsample_factor=2
```

---

# Modified Commands

## Default training

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/frames \
    out_dir=runs \
    experiment_name=kitchen_opt0
```

## Without metrics

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/frames \
    out_dir=runs \
    experiment_name=kitchen_opt0 \
    compute_extra_metrics=False
```

## With GUI

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/frames \
    out_dir=runs \
    experiment_name=kitchen_opt0 \
    with_gui=True
```

---

# Export Mesh During Training

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/frames \
    out_dir=runs \
    experiment_name=kitchen_opt0 \
    compute_extra_metrics=False \
    export_ply.enabled=True \
    export_ply.path=runs/kitchen_opt0/mesh.ply
```

---

# Train on transformed_t2 and Export Mesh

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
    path=data/transformed_t2 \
    out_dir=runs \
    experiment_name=transformed_t2 \
    compute_extra_metrics=False \
    export_ply.enabled=True \
    export_ply.path=runs/kitchen_opt0/tra_t2.ply \
    n_iterations=10000
```

---

# Export PLY from a Checkpoint

```bash
PYTHONPATH=. python -m threedgrut.export_ply \
    --ckpt runs/ckpt_last.pt \
    --out scene.ply
```

---

# NeRF Synthetic (Lego)

```bash
python train.py --config-name apps/nerf_synthetic_3dgut.yaml \
    path=data/nerf_synthetic/lego \
    out_dir=runs \
    experiment_name=lego_3dgut
```






---

# Custom Pose Rendering

Render an image from an arbitrary camera pose using the trained 3DGUT model.

Script location:

```bash
render_custom_pose.py
```

Run from the project root:

```bash
cd ~/4D_reconstruction/3DGUT/src_code/3dgrut
```

---

## Basic Render

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt
```

---

## Choose Output Directory

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --out_dir my_render
```

---

## Use Front Camera Intrinsic (camera_0)

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --intrinsic_id 1
```

---

## Use Back Camera Intrinsic (camera_1)

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --intrinsic_id 2
```

---

## Move the Camera

### Move Right by 1 m

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --tx 1.0
```

### Move Up by 0.5 m

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --ty 0.5
```

### Move Forward by 2 m

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --tz 2.0
```

---

## Rotate the Camera

### Turn Right 45°

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --ry 45
```

### Turn Left 90°

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --ry -90
```

### Look Down 20°

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --rx 20
```

### Look Up 20°

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --rx -20
```

---

## Move and Rotate Together

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --tx 0.5 --ty 0.2 --tz 1.5 \
    --ry 90 \
    --name view01.png
```

---

## Render Using Back Fisheye Camera Looking Backward

```bash
python render_custom_pose.py \
    --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
    --intrinsic_id 2 \
    --ry 180 \
    --name back_view.png
```

---

## Create a 360° Orbit Sequence

```bash
mkdir -p orbit

for a in $(seq 0 10 350); do
  python render_custom_pose.py \
      --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
      --out_dir orbit \
      --tz 1.0 \
      --ry $a \
      --name frame_$(printf "%03d" $a).png
done
```

---

## Convert Orbit Frames to Video

```bash
ffmpeg -framerate 30 -i orbit/frame_%03d.png \
    -c:v libx264 -pix_fmt yuv420p orbit.mp4
```

---

## Parameter Summary

| Parameter | Description |
|---|---|
| `--checkpoint` | Path to checkpoint |
| `--out_dir` | Output directory |
| `--name` | Output image name |
| `--intrinsic_id` | `1` = front camera, `2` = back camera |
| `--tx --ty --tz` | Camera translation in meters |
| `--rx --ry --rz` | Camera rotation in degrees |

---

## Notes

- Keep `--intrinsic_id 1` or `2` to reuse the calibrated fisheye intrinsics.
- `--ry` rotates the virtual camera around the scene.
- `--tz` moves the camera forward/backward in the world coordinate system.
- Positive `ry` turns the camera to the right; negative turns it to the left.