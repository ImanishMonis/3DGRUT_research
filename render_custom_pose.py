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

def generate_pinhole_rays(height=800, width=800, fov_deg=60.0):
    # Calculate focal length from Field of View (horizontal)
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    # Create pixel grid (+X right, +Y down, +Z forward)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy"
    )

    # Compute ray directions in camera coordinate space
    x = (grid_x - cx) / fx
    y = (grid_y - cy) / fy
    z = torch.ones_like(x)

    dirs = torch.stack([x, y, z], dim=-1)
    rays_dir = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    rays_ori = torch.zeros_like(rays_dir)

    pixel_coords = torch.stack([grid_y, grid_x], dim=-1)

    camera_params_dict = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": width, "height": height, "type": "pinhole"
    }

    # Format with batch dimension [1, H, W, C]
    return (
        camera_params_dict,
        rays_ori.unsqueeze(0),
        rays_dir.unsqueeze(0),
        pixel_coords.unsqueeze(0)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="custom_render")
    parser.add_argument("--intrinsic_id", type=int, default=1)   ## comment out for pinhole cam projection


    # Perspective camera settings
    # parser.add_argument("--fov", type=float, default=60.0, help="Horizontal FOV in degrees")
    # parser.add_argument("--width", type=int, default=800)
    # parser.add_argument("--height", type=int, default=800)

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
    camera_params_dict, rays_ori, rays_dir, camera_name, pixel_coords = dataset.intrinsics[args.intrinsic_id]  ## for firsheye
    
    # orig_params, _, _, camera_name, _ = dataset.intrinsics[args.intrinsic_id]
    # camera_params_dict, rays_ori, rays_dir, pixel_coords = generate_pinhole_rays(height=args.height,width=args.width,fov_deg=args.fov)  ## for perspective
    
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
    gpu_batch = Batch(**sample)

    with torch.no_grad():
        outputs = model(gpu_batch)

    pred = outputs["pred_rgb"].squeeze(0).permute(2, 0, 1).cpu()

    out_path = os.path.join(args.out_dir, args.name)
    torchvision.utils.save_image(pred, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()