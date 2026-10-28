# import os
# import argparse
# import numpy as np
# import torch
# import torchvision

# from threedgrut.render import Renderer
# from threedgrut.datasets.protocols import Batch

# def rot_x(a):
#     a = np.deg2rad(a)
#     return np.array([
#         [1, 0, 0],
#         [0, np.cos(a), -np.sin(a)],
#         [0, np.sin(a),  np.cos(a)],
#     ], dtype=np.float32)


# def rot_y(a):
#     a = np.deg2rad(a)
#     return np.array([
#         [ np.cos(a), 0, np.sin(a)],
#         [ 0,         1, 0        ],
#         [-np.sin(a), 0, np.cos(a)],
#     ], dtype=np.float32)


# def rot_z(a):
#     a = np.deg2rad(a)
#     return np.array([
#         [np.cos(a), -np.sin(a), 0],
#         [np.sin(a),  np.cos(a), 0],
#         [0,          0,         1],
#     ], dtype=np.float32)


# def generate_pinhole_rays(height=800, width=800, fov_deg=60.0):
#     fov_rad = np.deg2rad(fov_deg)
#     fx = width / (2.0 * np.tan(fov_rad / 2.0))
#     fy = fx
#     cx = width / 2.0
#     cy = height / 2.0

#     grid_x, grid_y = torch.meshgrid(
#         torch.arange(width, dtype=torch.float32),
#         torch.arange(height, dtype=torch.float32),
#         indexing="xy"
#     )

#     x = (grid_x - cx) / fx
#     y = (grid_y - cy) / fy
#     z = torch.ones_like(x)

#     dirs = torch.stack([x, y, z], dim=-1)
#     rays_dir = dirs / torch.norm(dirs, dim=-1, keepdim=True)
#     rays_ori = torch.zeros_like(rays_dir)

#     pixel_coords = torch.stack([grid_y, grid_x], dim=-1)

#     pinhole_params = {
#         "fx": fx,
#         "fy": fy,
#         "cx": cx,
#         "cy": cy,
#         "width": width,
#         "height": height,
#         "resolution": [width, height],
#         "type": "pinhole"
#     }

#     return (
#         pinhole_params,
#         rays_ori.unsqueeze(0),
#         rays_dir.unsqueeze(0),
#         pixel_coords.unsqueeze(0)
#     )


# def extract_ref_pose(dataset, frame_idx):
#     """Safely extracts a camera pose matrix from dataset sample or dataset metadata."""
#     ref_sample = dataset[frame_idx]
#     possible_keys = ["T_to_world", "pose", "extrinsics", "T", "T_cw", "T_wc", "world_to_cam"]

#     T_ref = None
#     if isinstance(ref_sample, dict):
#         for k in possible_keys:
#             if k in ref_sample:
#                 T_ref = ref_sample[k]
#                 break
#     else:
#         for k in possible_keys:
#             if hasattr(ref_sample, k):
#                 T_ref = getattr(ref_sample, k)
#                 break

#     if T_ref is None:
#         if hasattr(dataset, "poses") and len(dataset.poses) > frame_idx:
#             T_ref = dataset.poses[frame_idx]
#         elif hasattr(dataset, "extrinsics") and len(dataset.extrinsics) > frame_idx:
#             T_ref = dataset.extrinsics[frame_idx]

#     if T_ref is None:
#         available_keys = list(ref_sample.keys()) if isinstance(ref_sample, dict) else dir(ref_sample)
#         raise KeyError(f"Could not find camera pose in dataset[{frame_idx}]. Available keys: {available_keys}")

#     if isinstance(T_ref, torch.Tensor):
#         T_ref = T_ref.detach().cpu().numpy()
#     else:
#         T_ref = np.array(T_ref, dtype=np.float32)

#     if T_ref.ndim == 3:
#         T_ref = T_ref[0]

#     if T_ref.shape == (3, 4):
#         T_ref = np.vstack([T_ref, [0.0, 0.0, 0.0, 1.0]])

#     return T_ref


# def main():
#     parser = argparse.ArgumentParser(description="Render 3DGRUT model using pinhole perspective projection.")

#     parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
#     parser.add_argument("--out_dir", default="custom_render", help="Output directory path")

#     # Perspective camera settings
#     parser.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV in degrees")
#     parser.add_argument("--width", type=int, default=800, help="Render width in pixels")
#     parser.add_argument("--height", type=int, default=800, help="Render height in pixels")
#     parser.add_argument("--frame_idx", type=int, default=0, help="Dataset frame index to borrow camera pose from")

#     # Offset relative to dataset camera pose
#     parser.add_argument("--tx", type=float, default=0.0)
#     parser.add_argument("--ty", type=float, default=0.0)
#     parser.add_argument("--tz", type=float, default=0.0)

#     # Rotation offset (degrees)
#     parser.add_argument("--rx", type=float, default=0.0)
#     parser.add_argument("--ry", type=float, default=0.0)
#     parser.add_argument("--rz", type=float, default=0.0)

#     parser.add_argument("--name", default="pinhole_render.png")

#     args = parser.parse_args()

#     renderer = Renderer.from_checkpoint(
#         checkpoint_path=args.checkpoint,
#         out_dir=args.out_dir,
#         save_gt=False,
#     )

#     model = renderer.model
#     dataset = renderer.dataset

#     os.makedirs(args.out_dir, exist_ok=True)

#     # Fetch reference pose securely
#     T_ref = extract_ref_pose(dataset, args.frame_idx)

#     # Apply relative translation and rotation offsets
#     T_offset = np.eye(4, dtype=np.float32)
#     T_offset[:3, :3] = rot_z(args.rz) @ rot_y(args.ry) @ rot_x(args.rx)
#     T_offset[:3, 3] = np.array([args.tx, args.ty, args.tz], dtype=np.float32)

#     T_to_world = T_ref @ T_offset

#     # Fetch dataset intrinsics key and native metadata
#     first_key = next(iter(dataset.intrinsics))
#     dataset_intrinsic_entry = dataset.intrinsics[first_key]
#     _, _, _, camera_name, _ = dataset_intrinsic_entry

#     if isinstance(dataset_intrinsic_entry[0], dict):
#         camera_params_dict = dataset_intrinsic_entry[0].copy()
#     else:
#         camera_params_dict = {}

#     # Generate synthetic pinhole rays
#     pinhole_params, rays_ori, rays_dir, pixel_coords = generate_pinhole_rays(
#         height=args.height,
#         width=args.width,
#         fov_deg=args.fov
#     )
#     camera_params_dict.update(pinhole_params)

#     # Ensure required tracer metadata is properly formatted
#     if "shutter_type" not in camera_params_dict:
#         camera_params_dict["shutter_type"] = "GLOBAL"
#     elif isinstance(camera_params_dict["shutter_type"], str):
#         camera_params_dict["shutter_type"] = camera_params_dict["shutter_type"].upper()

#     rgb_gt = torch.zeros_like(rays_ori)

#     sample = {
#         "rgb_gt": rgb_gt.cuda(),
#         "rays_ori": rays_ori.cuda(),
#         "rays_dir": rays_dir.cuda(),
#         "T_to_world": torch.from_numpy(T_to_world).unsqueeze(0).cuda(),
#         f"intrinsics_{camera_name}": camera_params_dict,
#         "camera_idx": torch.tensor([0], device="cuda"),
#         "frame_idx": torch.tensor([args.frame_idx], device="cuda"),
#         "pixel_coords": pixel_coords.cuda(),
#     }
#     gpu_batch = Batch(**sample)

#     with torch.no_grad():
#         outputs = model(gpu_batch)

#     pred = outputs["pred_rgb"].squeeze(0).permute(2, 0, 1).cpu()

#     out_path = os.path.join(args.out_dir, args.name)
#     torchvision.utils.save_image(pred, out_path)

#     print(f"Saved pinhole render: {out_path}")


# if __name__ == "__main__":
#     main()

import os
import argparse
import numpy as np
import torch
import torchvision

from threedgrut.render import Renderer
from threedgrut.datasets.protocols import Batch

def quat_to_rotmat(q):
    """Converts a scalar-first quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    # Normalize quaternion to ensure it's a valid rotation
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm > 0:
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
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
    parser.add_argument("--intrinsic_id", type=int, default=1)  # comment out for pinhole cam projection

    # Perspective camera settings
    # parser.add_argument("--fov", type=float, default=60.0, help="Horizontal FOV in degrees")
    # parser.add_argument("--width", type=int, default=800)
    # parser.add_argument("--height", type=int, default=800)

    # Translation
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=0.0)
    parser.add_argument("--tz", type=float, default=0.0)

    # Rotation (Quaternion [qw, qx, qy, qz])
    parser.add_argument("--qw", type=float, default=1.0)
    parser.add_argument("--qx", type=float, default=0.0)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--qz", type=float, default=0.0)

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
    print("Available intrinsic keys:", list(dataset.intrinsics.keys()))
    # Intrinsics
    camera_params_dict, rays_ori, rays_dir, camera_name, pixel_coords = dataset.intrinsics[args.intrinsic_id]  # for fisheye
    
    # orig_params, _, _, camera_name, _ = dataset.intrinsics[args.intrinsic_id]
    # camera_params_dict, rays_ori, rays_dir, pixel_coords = generate_pinhole_rays(height=args.height, width=args.width, fov_deg=args.fov)  # for perspective
    
    # create dummy GT image 
    rgb_gt = torch.zeros_like(rays_ori)       # [1, H, W, 3]

    # Pose
    T_w2c = np.eye(4, dtype=np.float32)

    # Convert quaternion to rotation matrix
    R_w2c = quat_to_rotmat([args.qw, args.qx, args.qy, args.qz])

    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = np.array([args.tx, args.ty, args.tz], dtype=np.float32)
    T_c2w = np.linalg.inv(T_w2c)

    sample = {
        "rgb_gt": rgb_gt.cuda(),
        "rays_ori": rays_ori.cuda(),
        "rays_dir": rays_dir.cuda(),
        "T_to_world": torch.from_numpy(T_c2w).unsqueeze(0).cuda(),
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