import numpy as np
import torch


def eval_poly_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Horner evaluation for polynomial:
        c0 + c1*x + c2*x^2 + ...
    coeffs: [c0, c1, c2, ...]
    """
    result = torch.zeros_like(x, dtype=torch.float64)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def compute_max_angle(
    resolution: np.ndarray,
    focal_length: np.ndarray,
    principal_point: np.ndarray,
    radial_coeffs: np.ndarray,
) -> float:
    """
    Estimate max angle from farthest image corner using OpenCV fisheye model.
    """

    # ----------------------------
    # 1. Image corners (pixel space)
    # ----------------------------
    w, h = resolution[0], resolution[1]

    corners = np.array(
        [
            [0, 0],
            [w, 0],
            [0, h],
            [w, h],
        ],
        dtype=np.float64,
    )

    # ----------------------------
    # 2. Normalize to camera frame
    # ----------------------------
    normalised = (corners - principal_point) / focal_length
    max_r = np.max(np.linalg.norm(normalised, axis=1))     # max distance from principle point to image frame 

    # ----------------------------
    # 3. Convert coefficients
    # ----------------------------
    k = torch.from_numpy(radial_coeffs.astype(np.float64))

    fw = torch.tensor(
        [1.0, k[0], k[1], k[2], k[3]],
        dtype=torch.float64,
    )

    dfw = torch.tensor(
        [1.0, 3.0 * k[0], 5.0 * k[1], 7.0 * k[2], 9.0 * k[3]],
        dtype=torch.float64,
    )
    # print("\n===== r(theta) curve =====")

    # theta_vals = np.linspace(0, np.pi, 20)

    # for t in theta_vals:
    #     t_tensor = torch.tensor(t, dtype=torch.float64)
    #     t2 = t_tensor * t_tensor

    #     r = t_tensor * eval_poly_horner(fw, t2)

    #     print(f"theta = {t:6.3f} rad ({np.rad2deg(t):6.1f}°) -> r = {r.item():7.4f}")

    # print(f"\nTarget max_r = {max_r:.4f}")
    # ----------------------------
    # 4. Newton-Raphson inversion
    # ----------------------------
    # theta = torch.tensor(max_r, dtype=torch.float64)
    theta = torch.tensor(max_r, dtype=torch.float64)
    # theta = torch.clamp(theta, 0, np.pi)
    # theta = torch.tensor(np.arctan(max_r), dtype=torch.float64)
    for _ in range(20):
        t2 = theta * theta

        r = theta * eval_poly_horner(fw, t2)
        dr = eval_poly_horner(dfw, t2)
        residual = r - max_r

        if torch.abs(residual) < 1e-12:
            break
        if torch.abs(dr) < 1e-12:
            break
        # theta_new = theta - (r - max_r) / dr
        theta = theta - (r - max_r) / dr
        theta = torch.clamp(theta,0,np.pi*0.7)
        # theta_new = torch.clamp(theta_new, 0.0, torch.pi)
        # if theta <= 0:                    ###
        #     theta = theta * 0.5      

        # if torch.abs(theta_new - theta) < 1e-10:
        #     theta = theta_new
        #     break
        # if torch.abs(r - max_r) < 1e-10:
        #     break

    
    print("max_r:", max_r)
    print("initial theta:", theta.item())
    print("FOV degrees:", np.rad2deg(2*theta.item()))

    return float(theta.item())

# def compute_max_angle(resolution, focal_length, principal_point, radial_coeffs,
#                        theta_max=np.pi * 0.65):  # physically sane upper bound for your lens
#     w, h = resolution[0], resolution[1]
#     corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
#     normalised = (corners - principal_point) / focal_length
#     # max_r = np.max(np.linalg.norm(normalised, axis=1))
#     cx, cy = principal_point[0], principal_point[1]
#     fx, fy = focal_length[0], focal_length[1]

#     if w <= h:
#         # circle tangent to left/right edges
#         radius_px = min(cx, w - cx)
#         max_r = radius_px / fx
#     else:
#         # circle tangent to top/bottom edges
#         radius_px = min(cy, h - cy)
#         max_r = radius_px / fy

#     k = torch.from_numpy(radial_coeffs.astype(np.float64))
#     fw  = torch.tensor([1.0, k[0], k[1], k[2], k[3]], dtype=torch.float64)
#     dfw = torch.tensor([1.0, 3*k[0], 5*k[1], 7*k[2], 9*k[3]], dtype=torch.float64)

#     def r_of(theta):
#         t2 = theta * theta
#         return theta * eval_poly_horner(fw, t2), eval_poly_horner(dfw, t2)

#     lo = torch.tensor(0.0, dtype=torch.float64)
#     hi = torch.tensor(theta_max, dtype=torch.float64)
#     r_hi, _ = r_of(hi)

#     if r_hi < max_r:
#         # max_r is unreachable within the lens's valid/physical domain —
#         # this means max_r itself is likely wrong (bad focal length /
#         # principal point / resolution), not a solver problem.
#         print(f"WARNING: max_r={max_r:.4f} exceeds r(theta_max)={r_hi.item():.4f}; "
#               f"clamping to theta_max. Check focal_length/principal_point.")
#         return float(hi.item())

#     theta = (lo + hi) / 2
#     for _ in range(60):
#         r, dr = r_of(theta)
#         resid = r - max_r
#         if torch.abs(resid) < 1e-12:
#             break
#         if resid < 0:
#             lo = theta
#         else:
#             hi = theta
#         newton_theta = theta - resid / dr if torch.abs(dr) > 1e-12 else None
#         if newton_theta is not None and lo < newton_theta < hi:
#             theta = newton_theta
#         else:
#             theta = (lo + hi) / 2

#     return float(theta.item())