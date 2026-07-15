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