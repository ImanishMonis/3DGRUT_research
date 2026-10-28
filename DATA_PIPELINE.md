# Gaussian Splatting Stage – Data & Usage Guide

This repo is the **Gaussian Splatting (3DGUT)** stage of the 4D construction-site monitoring pipeline:

```
Teammate 1: dynamic masks ──┐
                            ├─► Teammate 2: HLoc (sparse model + session alignment)
Hilti fisheye images ───────┘                     │
                                                  ▼
                         THIS REPO: train 3DGUT per session ─► render t1 poses in the t2 model
                                                  │
                                                  ▼
                              Teammate 4: change detection (t1 vs t2 renders)
```

---

## 1. Raw data (Hilti SLAM Challenge 2026)

Location: `/cache/s49imoni/SLAM_repo/fisheye_original/`

| Folder | Content |
|---|---|
| `<date>/` | Original frames + `timestamps.csv` (upside down) |
| `<date>_rotated/` | **Use this.** Same frames rotated 180° (upright), `.jpg` |
| `<date>_rotated_png/` | Same as above, `.png` |

Sessions: `2025-05-05`, `2025-07-07`, `2025-12-02`, all **floor 1**. Each has `cam0/` (front) and `cam1/` (back).

Official calibration: `/cache/s49imoni/SLAM_repo/hilti-trimble-slam-challenge-2026/config/hilti_openvins/kalibr_imucam_chain.yaml`. It has a pinhole-equidistant model (= COLMAP `OPENCV_FISHEYE`, f ≈ 465 px) for the **unrotated** images. For the 180°-rotated images in COLMAP pixel convention: `cx' = W − 0.5 − cx`, `cy' = H − 0.5 − cy`; f and k1..k4 stay the same.

- File names: `cam0_<frame_idx 6 digits>_<timestamp_ns>.jpg`, e.g. `cam0_001000_10029095740000.jpg`.
- Resolution: **1472 × 1440** (W × H, not square). Fisheye, about 200° FOV per lens, 30 Hz.
- Size: 4 000 – 8 300 frames **per camera** per session. That's far too many for HLoc/3DGUT, so they need subsampling.
- The operator's helmet is visible in every frame. It needs to be in the masks.

---

## 2. Input layout expected by this repo (one folder per session)

This is the format teammate 2's HLoc output (+ teammate 1's masks) must be in:

```
<session>/                      e.g. /cache/s49imoni/data/align_t1
├── images/
│   ├── cam0/<name>.jpg ...     # any sub-folder name / image format works
│   └── cam1/<name>.jpg ...
├── masks/
│   ├── cam0/<name>.png ...     # SAME relative path as the image, extension always .png
│   └── cam1/<name>.png ...
├── sparse/0/
│   ├── cameras.bin             # (or cameras.txt)
│   ├── images.bin              # (or images.txt)
│   └── points3D.bin            # initialises the Gaussians
└── render_mask.png             # optional, global fisheye-circle mask (fixed name) – see below
```

### Rules the loader ([dataset_colmap.py](threedgrut/datasets/dataset_colmap.py)) relies on

| What | Expected |
|---|---|
| Image names in `images.bin` | `images/<cam>/<name>` (HLoc style, **with** the `images/` prefix). The image is read from `<session>/<name>`. |
| Camera model | `OPENCV_FISHEYE` (also `PINHOLE`, `SIMPLE_PINHOLE`, `SIMPLE_RADIAL`). |
| Intrinsics resolution | Same as the images, or an integer multiple of it. |
| Mask | `<session>/masks/<name without "images/" and extension>.png`, grayscale, same H×W as the image. **255 = train on pixel, 0 = ignore.** It should also black out the area outside the fisheye circle (as the `align_t1` masks do). |
| Missing mask | No error – the image is silently trained **without** a mask. |
| Poses | Standard COLMAP: `qvec = (qw, qx, qy, qz)`, `tvec`, world→camera, OpenCV camera axes (x right, y down, z forward). |

### `render_mask.png` – what it actually does today

| Step | Masks used |
|---|---|
| Training ([trainer.py](threedgrut/trainer.py)) | **Only** the per-image masks. `render_mask.png` is **not** read. |
| Test-split eval / PSNR ([render.py](threedgrut/render.py) `render_all`) | **Only** `render_mask.png`. It *replaces* the per-image masks, not added to them. Without the file: no mask at all. |
| `render_ply.py` / `batch_render.py` | Neither. |

So the fisheye circle is only applied in training if it is **already baked into the per-image masks**. That is the case for `align_t1`: 0 % of pixels outside `render_mask.png` are kept in any mask, so the masks there = circle AND dynamic. For the new data, teammate 1's masks must include the (inverted, rotated) Hilti circle too, or the loader has to combine them.

---

## 3. Which frame is the "world"?

- **3DGUT doesn't define a world frame.** It uses the poses in `images.bin` and `points3D.bin` as they are. So the Gaussians live in whatever frame the HLoc model is in.
- **HLoc/COLMAP sets the frame arbitrarily.** A reconstruction has no natural origin, orientation or scale. COLMAP picks them, usually from the initial image pair, and with monocular images the scale has no units.
- **So the shared world frame comes from teammate 2's alignment step.** After alignment, t2's model is expressed in t1's frame (or both are in one common frame). This is the only reason a t1 pose renders the same place in the t2 Gaussian model. If the alignment is off (rotation, translation or **scale**), the renders are off by the same amount, and change detection will see "changes" that aren't real.
- **Alternative frame:** Hilti's ground truth (`hilti-trimble-slam-challenge-2026/groundtruth/floor_*_<date>_run_*.txt`) gives metric `cam0 → map` poses in a floorplan-aligned frame that all sessions share. The GT refers to the **unrotated** camera, so with the `_rotated` images you'd have to flip the camera x and y axes.

---

## 4. Training

```bash
conda activate 3dgrut
python train.py --config-name apps/colmap_3dgut.yaml \
    path=<session folder> \
    out_dir=/cache/s49imoni/runs \
    experiment_name=<name> \
    compute_extra_metrics=False \
    n_iterations=10000 \
    export_ply.enabled=True export_ply.path=/cache/s49imoni/runs/<name>.ply
```

Output: `<out_dir>/<experiment_name>/<session>-<DDMM_HHMMSS>/ckpt_last.pt` (+ `parsed.yaml`, metrics, optional `.ply`).

- `dataset.test_split_interval=8` (repo default): every 8th image is kept out of training and used to compute PSNR etc. `0` = train on all images (no quality numbers).
- The checkpoint stores the dataset `path`. **Rendering re-opens that dataset folder** (for the intrinsics), so don't move or delete it.

---

## 5. Rendering t1 poses in another session's model

### Pose file (`poses.json`)

A JSON list, one entry per view. Poses are COLMAP world→camera, in the **shared (aligned) world frame**. No script in this repo writes this file; it comes from outside.

```json
[
  {
    "image_path": "images/camera_1/0090.png",
    "image_id": 180,
    "rotation_qw_qx_qy_qz": [-0.0320, 0.0607, 0.9974, -0.0231],
    "translation_tx_ty_tz": [-3.4440, -0.3067, -1.9826]
  }
]
```

### Commands

```bash
# all poses in the file
python batch_render.py --json_path poses.json \
    --checkpoint <t2 ckpt_last.pt> \
    --out_dir batch_render_result/t2 [--start 0 --limit 20]

# a single pose
python render_ply.py --checkpoint <ckpt> --qw .. --qx .. --qy .. --qz .. \
    --tx .. --ty .. --tz .. --intrinsic_id 1 --out_dir <dir> --name 180.png
```

- Output: `<out_dir>/<image_id>.png`. Same resolution and fisheye intrinsics as the model's training camera.
- `--intrinsic_id` is the **COLMAP `camera_id`** in `cameras.bin`, not the folder. The HLoc models have one shared camera → always `1`.
- Pixels outside the fisheye circle are **not** masked in these renders. Apply `render_mask.png` (or the image mask) before comparing.

---

## 6. Script overview

| File | Purpose | Used in pipeline? |
|---|---|---|
| [train.py](train.py) | Train 3DGUT on a session folder. | yes |
| [batch_render.py](batch_render.py) | Loop over `poses.json`, call `render_ply.py` per entry. | yes |
| [render_ply.py](render_ply.py) | Render one image from a quaternion + translation (world→camera). | yes |
| [render_custom_pose.py](render_custom_pose.py) | Older tool: render from Euler angles + translation, taken as camera→world. Replaced by `render_ply.py`. | no |
| [apply_masks.py](apply_masks.py) | Active code copies one mask N times as `NNNN_mask.png`; the mask viewer is commented out. Not called anywhere. | no |
| [threedgrut/utils/camera.py](threedgrut/utils/camera.py) | `compute_max_angle`: fisheye FOV from the image circle. | yes |
| `python -m threedgrut.export_ply --ckpt <ckpt> --out scene.ply` | Export a checkpoint to `.ply`. | optional |

`data/timestep1`, `data/` and `/cache/s49imoni/data/align_t*` are **legacy** (old COLMAP / HLoc runs). `data/timestep1` uses COLMAP naming (no `images/` prefix) and won't load with the current code.

---

## 7. Issues & things to check

**Before using the new data**
1. **Fisheye FOV clipping (measured with Hilti's calibration).**
   - `compute_max_angle` uses the inscribed circle (~720 px → **~91°** half-FOV). The real image circle is ~795 px (**~110°**), clipped at top/bottom.
   - The 3DGUT tracer drops every Gaussian beyond `max_angle`, so **~12 % of the image (the outer ring) can never be rendered**. Unless the training masks exclude it, the loss still tries to fit that ring.
   - The original ncore version (farthest corner, 1034 px) can't work: Hilti's distortion polynomial r(θ) peaks at θ ≈ 115° / 799 px, so the corner radius is unreachable.
   - Fix options: (a) keep ~91° and mask out everything beyond that circle in the training masks; or (b) raise `max_angle` to ~105° (below the 115° peak) and mask only outside the true circle.
   - Hilti provides `config/hilti_openvins/mask_cam{0,1}.png`. These have **white = invalid** (the opposite of ours), are for the **unrotated** images, and keep pixels up to ~730 px (**~93°**).
   - **Decision (2026-10-28): keep `compute_max_angle` as it is for now** and check the first run on the new data. If the mask edge shows dark/blurry bands, switch to the option below.
   - **Fallback:** Hilti mask (inverted + rotated 180°, combined with the dynamic masks) and `max_angle = 110°`.
     - `max_angle` is only a cut-off, not a lens parameter. Rays inside the mask change by < 0.1° between 91° and 110°.
     - 110° stays below the 115° peak (recheck if HLoc refines k1..k4) and leaves a margin past the 93° mask edge.
     - The current 91° is actually *smaller* than the Hilti mask edge (93.7°).
     - The 93–110° ring is unsupervised, so renders must get the same mask before change detection.
2. **Rotation vs. official calibration and GT.** The `_rotated` images are turned 180°. Hilti's calibration and GT poses are for the unrotated camera. Fine if HLoc estimates its own intrinsics; if anyone reuses Hilti's, use the formula in section 1, and the camera x/y axes flip. Hilti's `mask_cam*.png` must be rotated too.
3. **Helmet in every frame.** It must be masked (or cropped), otherwise it ends up as floating Gaussians.
4. **Frame count.** 8 000–16 000 images per session (vs. 180 in `align_t1`). Agree on a subsampling rate with teammate 2.

**Code / workflow**

5. **`poses.json` has no record of where it came from.** Nothing in the repo writes it. The current file has the same names/IDs as `align_t1`, but it is **not** `align_t1`'s poses (mean 0.6 m, max 1.6 m apart even after best-fit alignment). It's probably the t2 session. For the new data, the file should state which session/model the poses come from.
6. **Eval PSNR/SSIM is inflated by masks.** `render_all` zeroes masked pixels in both prediction and GT, then averages over the *whole* image. The black pixels count as perfect matches (~25 % of the frame for the circle alone), so the comment "evaluation on valid rays only" isn't true. It also ignores the dynamic masks, so people in the test images count as errors.
7. **Masks are optional without warning.** A missing or misnamed mask (e.g. `NNNN_mask.png` instead of `NNNN.png`) silently trains on people/objects.
8. **`downsample_factor` > 1 is broken.** Intrinsics are read from `images_<N>/`, but training images always come from the full-res path in `images.bin` → size mismatch.
9. **`batch_render.py` is slow and leaves clutter.** It starts a new Python process and reloads the checkpoint per pose, and creates one run folder per pose (102 folders in `batch_render_result/t1/tmp/`).
10. **`--intrinsic_id 1/2 = front/back`** in `commands.md` only holds for the legacy 2-camera model. On HLoc models `2` gives a `KeyError`. `commands.md` also documents only the old `render_custom_pose.py`.
11. **Housekeeping.**
    - `batch_render_result/` (1.2 GB) is not in `.gitignore`.
    - `render_ply.py` starts with ~200 lines of commented-out code, and `compute_max_angle` prints debug output on every load.
    - `commands.md` and `commands.txt` duplicate each other.
    - README calls `apply_masks.py` "image masking preprocessing", which isn't what it does.
