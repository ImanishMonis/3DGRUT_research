import os
import argparse
import numpy as np
import torch
import torchvision

from threedgrut.render import Renderer
from threedgrut.datasets.protocols import Batch

def rot_x(a):
    a = np.deg2rad(a)
    return np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)],
    ], dtype=np.float32)


def rot_y(a):
    a = np.deg2rad(a)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [ 0,         1, 0        ],
        [-np.sin(a), 0, np.cos(a)],
    ], dtype=np.float32)


def rot_z(a):
    a = np.deg2rad(a)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1],
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="custom_render")
    parser.add_argument("--intrinsic_id", type=int, default=1)

    # Translation
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=0.0)
    parser.add_argument("--tz", type=float, default=0.0)

    # Rotation (degrees)
    parser.add_argument("--rx", type=float, default=0.0)
    parser.add_argument("--ry", type=float, default=0.0)
    parser.add_argument("--rz", type=float, default=0.0)

    parser.add_argument("--name", default="render.png")

    args = parser.parse_args()

    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        save_gt=False,
    )

    model = renderer.model
    dataset = renderer.dataset

    os.makedirs(args.out_dir, exist_ok=True)

    # Intrinsics
    camera_params_dict, rays_ori, rays_dir, camera_name, pixel_coords = dataset.intrinsics[args.intrinsic_id]

    # create dummy GT image 
    rgb_gt = torch.zeros_like(rays_ori)       # [1, H, W, 3]

    # Pose
    T = np.eye(4, dtype=np.float32)

    R = rot_z(args.rz) @ rot_y(args.ry) @ rot_x(args.rx)

    T[:3, :3] = R
    T[:3, 3] = np.array([args.tx, args.ty, args.tz], dtype=np.float32)

    sample = {
        "rgb_gt":rgb_gt.cuda(),
        "rays_ori": rays_ori.cuda(),
        "rays_dir": rays_dir.cuda(),
        "T_to_world": torch.from_numpy(T).unsqueeze(0).cuda(),
        f"intrinsics_{camera_name}": camera_params_dict,
        "camera_idx": torch.tensor([0], device="cuda"),
        "frame_idx": torch.tensor([0], device="cuda"),
        "pixel_coords": pixel_coords.cuda(),
    }


    with torch.no_grad():
        outputs = model(gpu_batch)

    pred = outputs["pred_rgb"].squeeze(0).permute(2, 0, 1).cpu()

    out_path = os.path.join(args.out_dir, args.name)
    torchvision.utils.save_image(pred, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()