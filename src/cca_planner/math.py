"""Batched rigid-body math implemented with Torch.

Every operation supports arbitrary leading dimensions (``...``), so the same code
serves a single sample (``[...]``), a batch (``[B, ...]``), or a nested batch
(``[B, J, ...]``) — the shape Isaac Lab asks for when one planner call must
produce trajectories for every environment at once.

Design notes
------------
* **SO(3)/SE(3) exponentials** use :func:`torch.linalg.matrix_exp` on the matrix
  representation. This is the *definition* of the group exponential, so it is
  exact and branch-free.
* **SO(3)/SE(3) logarithms** use the closed-form Modern-Robotics formula with
  ``torch.where`` fallbacks for the ``theta ~ 0`` and ``theta ~ pi``
  singularities (vectorised over the batch). These are the only branchy pieces.
* All ops are autograd-traceable; they are also valid run under ``torch.no_grad``
  when the planner is used purely as a reference-trajectory generator (the
  common Isaac Lab / RL case).

Shapes
------
Vectors are ``[..., 6]`` / ``[..., 3]`` and transforms are ``[..., 4, 4]``, matching
the conventional Modern-Robotics layout.
"""

from __future__ import annotations

import torch

# Tolerance below which a rotation angle is treated as zero (matches math_utils).
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# so(3) / so(3) <-> vector helpers
# --------------------------------------------------------------------------- #
def vec_to_so3(omg: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric (so(3)) matrix of a 3-vector. ``[..., 3] -> [..., 3, 3]``."""
    omg = omg[..., :3]
    o0 = omg[..., 0]
    o1 = omg[..., 1]
    o2 = omg[..., 2]
    z = torch.zeros_like(o0)
    row0 = torch.stack([z, -o2, o1], dim=-1)
    row1 = torch.stack([o2, z, -o0], dim=-1)
    row2 = torch.stack([-o1, o0, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def so3_to_vec(so3mat: torch.Tensor) -> torch.Tensor:
    """3-vector from a skew-symmetric matrix. ``[..., 3, 3] -> [..., 3]``."""
    return torch.stack([so3mat[..., 2, 1], so3mat[..., 0, 2], so3mat[..., 1, 0]], dim=-1)


# --------------------------------------------------------------------------- #
# se(3) / se(3) <-> vector helpers
# --------------------------------------------------------------------------- #
def vec_to_se3(v: torch.Tensor) -> torch.Tensor:
    """4x4 se(3) matrix from a 6-vector spatial velocity. ``[..., 6] -> [..., 4, 4]``."""
    se3 = torch.zeros(*v.shape[:-1], 4, 4, dtype=v.dtype, device=v.device)
    se3[..., :3, :3] = vec_to_so3(v[..., :3])
    se3[..., :3, 3] = v[..., 3:6]
    return se3


def se3_to_vec(se3mat: torch.Tensor) -> torch.Tensor:
    """6-vector spatial velocity from a 4x4 se(3) matrix. ``[..., 4, 4] -> [..., 6]``."""
    return torch.stack(
        [
            se3mat[..., 2, 1],
            se3mat[..., 0, 2],
            se3mat[..., 1, 0],
            se3mat[..., 0, 3],
            se3mat[..., 1, 3],
            se3mat[..., 2, 3],
        ],
        dim=-1,
    )


# --------------------------------------------------------------------------- #
# Group exponentials (branch-free via matrix_exp)
# --------------------------------------------------------------------------- #
def matrix_exp3(so3mat: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """SO(3) exponential via Rodrigues. ``[..., 3, 3] -> [..., 3, 3]``.

    Cheaper than ``torch.linalg.matrix_exp`` (no Pade scaling/squaring) and
    matches the reference's analytic formula. The identity is returned where the
    rotation angle is ~0.
    """
    omgtheta = so3_to_vec(so3mat)  # [..., 3]
    theta = omgtheta.norm(dim=-1)  # [...]
    small = theta.abs() < eps
    safe = torch.where(small, torch.ones_like(theta), theta)
    omgmat = so3mat / safe[..., None, None]
    I3 = torch.eye(3, dtype=so3mat.dtype, device=so3mat.device).expand(*so3mat.shape[:-2], 3, 3)
    rot = I3 + torch.sin(theta)[..., None, None] * omgmat + (1.0 - torch.cos(theta))[..., None, None] * (omgmat @ omgmat)
    return torch.where(small[..., None, None], I3, rot)


def matrix_exp6(se3mat: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """SE(3) exponential via the closed-form (Rodrigues) formula. ``[..., 4, 4] -> [..., 4, 4]``.

    This mirrors ``math_utils.matrix_exp6`` analytically (rather than calling the
    generic matrix exponential): it is markedly cheaper than
    ``torch.linalg.matrix_exp`` and matches the reference closely, which matters
    because the IK loop evaluates it many times per iteration (once per joint,
    twice in the closure correction). Accepts arbitrary leading dims, so the
    joint axis can be an extra batch dim (``[..., n, 4, 4]``) — used by the
    fused forward-kinematics / Jacobian below.
    """
    so3mat = se3mat[..., :3, :3]
    omgtheta = so3_to_vec(so3mat)  # [..., 3]
    theta = omgtheta.norm(dim=-1)  # [...]
    small = theta.abs() < eps
    safe = torch.where(small, torch.ones_like(theta), theta)
    omgmat = so3mat / safe[..., None, None]
    st = torch.sin(theta)
    ct = torch.cos(theta)
    I3 = torch.eye(3, dtype=se3mat.dtype, device=se3mat.device).expand(*so3mat.shape[:-2], 3, 3)
    omg2 = omgmat @ omgmat
    rot = I3 + st[..., None, None] * omgmat + (1.0 - ct)[..., None, None] * omg2
    p_mat = I3 * theta[..., None, None] + (1.0 - ct)[..., None, None] * omgmat + (theta - st)[..., None, None] * omg2
    v_lin = se3mat[..., :3, 3] / safe[..., None]  # [..., 3]
    p = (p_mat @ v_lin.unsqueeze(-1)).squeeze(-1)  # [..., 3]
    out = torch.eye(4, dtype=se3mat.dtype, device=se3mat.device).expand(*se3mat.shape).clone()
    out[..., :3, :3] = torch.where(small[..., None, None], I3, rot)
    out[..., :3, 3] = torch.where(small[..., None], se3mat[..., :3, 3], p)
    return out


# --------------------------------------------------------------------------- #
# Group logarithms (robust, vectorised)
# --------------------------------------------------------------------------- #
def so3_log_map(rot: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Rotation vector (axis * angle) from an SO(3) matrix. ``[..., 3, 3] -> [..., 3]``.

    General branch: ``theta / (2 sin theta) * vee(R - R^T)``.
    Near ``theta ~ 0`` the first-order ``vee / 2`` limit avoids an ill-conditioned
    division. Near ``theta ~ pi`` the matrix is symmetric (so the anti-symmetric
    part vanishes) and the axis is recovered from the diagonal, as in the Modern
    Robotics closed-form: the candidate with the largest ``1 + R_kk`` is selected
    element-wise across the batch.
    """
    rm = rot - rot.transpose(-2, -1)
    vee = torch.stack([rm[..., 2, 1], rm[..., 0, 2], rm[..., 1, 0]], dim=-1)  # [..., 3]

    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    # ``acos(cos)`` loses all small-angle resolution when cos rounds to one.
    # The skew part is first-order in theta, so atan2(||vee||/2, cos) remains
    # accurate in float32 down to the closure tolerances used by the planner.
    sin = 0.5 * vee.norm(dim=-1)
    theta = torch.atan2(sin, cos)  # [...]

    safe_sin = torch.where(sin > eps, sin, torch.ones_like(sin))
    log_general = (theta / (2.0 * safe_sin)).unsqueeze(-1) * vee  # [..., 3]
    # lim(theta -> 0) theta/(2 sin(theta)) * vee = vee/2.
    log_near_zero = 0.5 * vee

    # ---- near pi: recover the unit axis from the diagonal, batch-wise ----
    diag = torch.diagonal(rot, dim1=-2, dim2=-1)  # [..., 3]
    dp1 = 1.0 + diag  # [..., 3]

    d0, d1, d2 = dp1[..., 0], dp1[..., 1], dp1[..., 2]
    denom0 = torch.sqrt(torch.clamp(2.0 * d0, min=1e-20))
    denom1 = torch.sqrt(torch.clamp(2.0 * d1, min=1e-20))
    denom2 = torch.sqrt(torch.clamp(2.0 * d2, min=1e-20))
    # MR axis formulas at theta = pi (R is symmetric, so R_ij == R_ji):
    cand0 = torch.stack([d0, rot[..., 1, 0], rot[..., 2, 0]], dim=-1) / denom0.unsqueeze(-1)
    cand1 = torch.stack([rot[..., 0, 1], d1, rot[..., 2, 1]], dim=-1) / denom1.unsqueeze(-1)
    cand2 = torch.stack([rot[..., 0, 2], rot[..., 1, 2], d2], dim=-1) / denom2.unsqueeze(-1)

    k = dp1.argmax(dim=-1)  # [...]
    omg = torch.where(
        (k == 0).unsqueeze(-1), cand0, torch.where((k == 1).unsqueeze(-1), cand1, cand2)
    )
    log_pi = theta.unsqueeze(-1) * omg  # theta ~ pi

    near_pi = (sin <= eps) & (cos < 0.0)
    near_zero = (sin <= eps) & ~near_pi
    return torch.where(
        near_pi.unsqueeze(-1),
        log_pi,
        torch.where(near_zero.unsqueeze(-1), log_near_zero, log_general),
    )


def matrix_log6(transform: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """4x4 se(3) log of an SE(3) matrix. ``[..., 4, 4] -> [..., 4, 4]``."""
    rot = transform[..., :3, :3]
    p = transform[..., :3, 3]

    omgvec = so3_log_map(rot, eps)  # [..., 3]
    so3mat = vec_to_so3(omgvec)  # [..., 3, 3]

    # The norm of the already robust rotation vector is the rotation angle.
    # Recomputing it with acos(trace) reintroduces float32 quantisation near I.
    theta = omgvec.norm(dim=-1)  # [...]
    near_zero = theta.abs() < eps  # [...]

    eye3 = torch.eye(3, dtype=transform.dtype, device=transform.device)
    eye3 = eye3.expand(*rot.shape[:-2], 3, 3)
    safe_theta = torch.where(near_zero, torch.ones_like(theta), theta)
    theta2 = safe_theta.square()
    # G^-1 coefficient.  The direct expression subtracts two O(1/theta)
    # values and is catastrophically unstable in float32 for small rotations.
    # Its Taylor series is accurate throughout this small-angle branch.
    coef_series = 1.0 / 12.0 + theta2 / 720.0 + theta2.square() / 30240.0
    coef_general = (
        1.0 - 0.5 * safe_theta / torch.tan(0.5 * safe_theta)
    ) / theta2
    coef = torch.where(safe_theta < 0.05, coef_series, coef_general)  # [...]

    omg2 = so3mat @ so3mat
    p_general = eye3 - 0.5 * so3mat + coef.unsqueeze(-1).unsqueeze(-1) * omg2
    p_general = (p_general @ p.unsqueeze(-1)).squeeze(-1)  # [..., 3]

    p_log = torch.where(near_zero.unsqueeze(-1), p, p_general)

    se3 = torch.zeros(*transform.shape[:-2], 4, 4, dtype=transform.dtype, device=transform.device)
    se3[..., :3, :3] = so3mat
    se3[..., :3, 3] = p_log
    return se3


# --------------------------------------------------------------------------- #
# Adjoint / inverse transform
# --------------------------------------------------------------------------- #
def adjoint(htm: torch.Tensor) -> torch.Tensor:
    """6x6 adjoint of a homogeneous transform. ``[..., 4, 4] -> [..., 6, 6]``.

    Layout: ``[[R, 0], [skew(p) @ R, R]]``.
    """
    rot = htm[..., :3, :3]
    p = htm[..., :3, 3]
    bot_left = vec_to_so3(p) @ rot  # [..., 3, 3]
    z = torch.zeros_like(rot)
    top = torch.cat([rot, z], dim=-1)  # [..., 3, 6]
    bot = torch.cat([bot_left, rot], dim=-1)  # [..., 3, 6]
    return torch.cat([top, bot], dim=-2)  # [..., 6, 6]


def trans_inv(transform: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of an SE(3) transform. ``[..., 4, 4] -> [..., 4, 4]``."""
    rot = transform[..., :3, :3]
    p = transform[..., :3, 3]
    inv_rot = rot.transpose(-2, -1)
    out = torch.eye(4, dtype=transform.dtype, device=transform.device)
    out = out.expand(*transform.shape).clone()
    out[..., :3, :3] = inv_rot
    out[..., :3, 3] = -(inv_rot @ p.unsqueeze(-1)).squeeze(-1)
    return out


# --------------------------------------------------------------------------- #
# Forward kinematics / Jacobian (batch over leading dims, loop over joints)
# --------------------------------------------------------------------------- #
def fkin_space(m: torch.Tensor, slist: torch.Tensor, thetalist: torch.Tensor) -> torch.Tensor:
    """Space-form forward kinematics (product of exponentials). Returns ``[..., 4, 4]``.

    ``m`` may be a single ``[4, 4]`` or batched ``[..., 4, 4]``; ``slist`` is
    ``[..., 6, n]`` and ``thetalist`` is ``[..., n]``. All ``n`` joint
    exponentials are evaluated in ONE batched call (treating the joint axis as a
    batch dim); only the cumulative product is sequential over joints.
    """
    n = slist.shape[-1]
    v = (slist * thetalist[..., None, :]).transpose(-1, -2)  # [..., n, 6]
    exps = matrix_exp6(vec_to_se3(v))  # [..., n, 4, 4]
    transform = m
    for i in range(n - 1, -1, -1):
        transform = exps[..., i, :, :] @ transform
    return transform


def jacobian_space(slist: torch.Tensor, thetalist: torch.Tensor) -> torch.Tensor:
    """Space-form Jacobian. ``slist[..., 6, n]`` + ``thetalist[..., n]`` -> ``[..., 6, n]``.

    The ``n-1`` joint exponentials for the prefix products are evaluated in ONE
    batched call; the prefix products themselves stay sequential over joints.
    """
    n = slist.shape[-1]
    js = torch.zeros_like(slist)
    js[..., :, 0] = slist[..., :, 0]
    if n == 1:
        return js

    v = (slist[..., :, :-1] * thetalist[..., None, :-1]).transpose(-1, -2)  # [..., n-1, 6]
    exps = matrix_exp6(vec_to_se3(v))  # [..., n-1, 4, 4]

    transform = torch.eye(4, dtype=slist.dtype, device=slist.device).expand(*slist.shape[:-2], 4, 4)
    transform = transform.clone() if torch.is_grad_enabled() else transform
    for i in range(1, n):
        transform = transform @ exps[..., i - 1, :, :]
        adj = adjoint(transform)  # [..., 6, 6]
        js[..., :, i] = (adj @ slist[..., :, i].unsqueeze(-1)).squeeze(-1)
    return js


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def clamp_to_magnitude_minimum(mat: torch.Tensor, min_magnitude: float) -> torch.Tensor:
    """Clamp each element to at least ``min_magnitude`` in magnitude, sign-preserving."""
    signs = torch.sign(mat)
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    magnitudes = torch.clamp(mat.abs(), min=min_magnitude)
    return signs * magnitudes
