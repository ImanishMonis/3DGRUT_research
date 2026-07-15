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
    max_r = np.max(np.linalg.norm(normalised, axis=1))

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

    # ----------------------------
    # 4. Newton-Raphson inversion
    # ----------------------------
    theta = torch.tensor(max_r, dtype=torch.float64)

    for _ in range(20):
        t2 = theta * theta

        r = theta * eval_poly_horner(fw, t2)
        dr = eval_poly_horner(dfw, t2)

        if torch.abs(dr) < 1e-12:
            break

        theta = theta - (r - max_r) / dr

        if theta <= 0:                    ###
            theta = theta * 0.5      

        if torch.abs(r - max_r) < 1e-10:
            break
    
    print("max_r:", max_r)
    print("initial theta:", theta.item())
    return float(theta.item())