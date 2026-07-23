"""Rigid-body math utilities (Modern Robotics conventions).

The rotation exponential/logarithm (SO(3)) operations lean on
:mod:`scipy.spatial.transform`, while the SE(3)-specific formulas (matrix
exponential/logarithm of a twist, adjoint, forward kinematics and space
Jacobian) are implemented with NumPy.

All vectors are 1-D ``numpy.ndarray`` and all transforms are 4x4 homogeneous
matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Tolerance used to decide whether a scalar is effectively zero.
_NEAR_ZERO_TOL = 1e-6


def near_zero(value: float) -> bool:
    """Return ``True`` when ``value`` is small enough to be neglected."""
    return abs(value) < _NEAR_ZERO_TOL


def vec_to_so3(omg: np.ndarray) -> np.ndarray:
    """Skew-symmetric (so(3)) representation of a 3-vector."""
    omg = np.asarray(omg, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -omg[2], omg[1]],
            [omg[2], 0.0, -omg[0]],
            [-omg[1], omg[0], 0.0],
        ]
    )


def so3_to_vec(so3mat: np.ndarray) -> np.ndarray:
    """3-vector (angular velocity) from a 3x3 skew-symmetric matrix."""
    so3mat = np.asarray(so3mat, dtype=float)
    return np.array([so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])


def axis_ang3(expc3: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit rotation axis and rotation angle from exponential coordinates."""
    expc3 = np.asarray(expc3, dtype=float).reshape(3)
    theta = float(np.linalg.norm(expc3))
    omghat = expc3 / theta
    return omghat, theta


def matrix_exp3(so3mat: np.ndarray) -> np.ndarray:
    """SO(3) matrix achieved by rotating about ``omghat`` by ``theta``.

    The rotation is computed via ``scipy.spatial.transform.Rotation`` from the
    rotation vector (axis * angle) extracted from the so(3) matrix.
    """
    so3mat = np.asarray(so3mat, dtype=float)
    omgtheta = so3_to_vec(so3mat)
    if near_zero(float(np.linalg.norm(omgtheta))):
        return np.eye(3)
    return Rotation.from_rotvec(omgtheta).as_matrix()


def matrix_log3(rot: np.ndarray) -> np.ndarray:
    """so(3) representation of the exponential coordinates of a rotation matrix.

    Uses the closed-form Modern Robotics formula with explicit handling of the
    0 and pi rotation cases so that the identity maps to an exact zero matrix
    and avoids division by zero downstream.
    """
    rot = np.asarray(rot, dtype=float)
    acosinput = (np.trace(rot) - 1.0) / 2.0

    if acosinput >= 1.0:
        return np.zeros((3, 3))
    if acosinput <= -1.0:
        if not near_zero(1.0 + rot[2, 2]):
            omg = (1.0 / np.sqrt(2.0 * (1.0 + rot[2, 2]))) * np.array(
                [rot[0, 2], rot[1, 2], 1.0 + rot[2, 2]]
            )
        elif not near_zero(1.0 + rot[1, 1]):
            omg = (1.0 / np.sqrt(2.0 * (1.0 + rot[1, 1]))) * np.array(
                [rot[0, 1], 1.0 + rot[1, 1], rot[2, 1]]
            )
        else:
            omg = (1.0 / np.sqrt(2.0 * (1.0 + rot[0, 0]))) * np.array(
                [1.0 + rot[0, 0], rot[1, 0], rot[2, 0]]
            )
        return vec_to_so3(np.pi * omg)

    theta = float(np.arccos(acosinput))
    return (theta / (2.0 * np.sin(theta))) * (rot - rot.T)


def vec_to_se3(v: np.ndarray) -> np.ndarray:
    """4x4 se(3) matrix from a 6-vector spatial velocity."""
    v = np.asarray(v, dtype=float).reshape(6)
    se3mat = np.zeros((4, 4))
    se3mat[:3, :3] = vec_to_so3(v[:3])
    se3mat[:3, 3] = v[3:6]
    return se3mat


def se3_to_vec(se3mat: np.ndarray) -> np.ndarray:
    """6-vector spatial velocity from a 4x4 se(3) matrix."""
    se3mat = np.asarray(se3mat, dtype=float)
    return np.array(
        [
            se3mat[2, 1],
            se3mat[0, 2],
            se3mat[1, 0],
            se3mat[0, 3],
            se3mat[1, 3],
            se3mat[2, 3],
        ]
    )


def matrix_exp6(se3mat: np.ndarray) -> np.ndarray:
    """SE(3) matrix from a 4x4 se(3) representation of exponential coordinates."""
    se3mat = np.asarray(se3mat, dtype=float)
    so3mat = se3mat[:3, :3]
    omgtheta = so3_to_vec(so3mat)

    if near_zero(float(np.linalg.norm(omgtheta))):
        # Pure translation: rotation part is identity, translation taken as-is.
        out = np.eye(4)
        out[:3, 3] = se3mat[:3, 3]
        return out

    _, theta = axis_ang3(omgtheta)
    omgmat = so3mat / theta
    rot = matrix_exp3(so3mat)
    p = (
        np.eye(3) * theta
        + (1.0 - np.cos(theta)) * omgmat
        + (theta - np.sin(theta)) * (omgmat @ omgmat)
    ) @ (se3mat[:3, 3] / theta)
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = p
    return out


def matrix_log6(transform: np.ndarray) -> np.ndarray:
    """4x4 se(3) representation of exponential coordinates of an SE(3) matrix."""
    transform = np.asarray(transform, dtype=float)
    rot = transform[:3, :3]
    p = transform[:3, 3]
    omgmat = matrix_log3(rot)

    expmat = np.zeros((4, 4))
    if np.allclose(omgmat, 0.0):
        expmat[:3, 3] = p
        return expmat

    theta = float(np.arccos((np.trace(rot) - 1.0) / 2.0))
    eye3 = np.eye(3)
    expmat[:3, :3] = omgmat
    expmat[:3, 3] = (
        eye3
        - 0.5 * omgmat
        + (1.0 / theta - 1.0 / (2.0 * np.tan(0.5 * theta))) * (omgmat @ omgmat) / theta
    ) @ p
    return expmat


def adjoint(htm: np.ndarray) -> np.ndarray:
    """6x6 adjoint representation of a homogeneous transformation matrix."""
    htm = np.asarray(htm, dtype=float)
    rot = htm[:3, :3]
    translation = htm[:3, 3]
    bot_left = vec_to_so3(translation) @ rot
    adj = np.zeros((6, 6))
    adj[:3, :3] = rot
    adj[3:, 3:] = rot
    adj[3:, :3] = bot_left
    return adj


def trans_inv(transform: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transformation matrix (closed form for SE(3))."""
    transform = np.asarray(transform, dtype=float)
    rot = transform[:3, :3]
    p = transform[:3, 3]
    inv_rot = rot.T
    inv_p = -inv_rot @ p
    out = np.eye(4)
    out[:3, :3] = inv_rot
    out[:3, 3] = inv_p
    return out


def fkin_space(m: np.ndarray, slist: np.ndarray, thetalist: np.ndarray) -> np.ndarray:
    """Space-form forward kinematics using the product of exponentials formula."""
    m = np.asarray(m, dtype=float)
    slist = np.asarray(slist, dtype=float)
    thetalist = np.asarray(thetalist, dtype=float).reshape(-1)
    transform = m.copy()
    for i in range(thetalist.size - 1, -1, -1):
        exp_mat = matrix_exp6(vec_to_se3(slist[:, i] * thetalist[i]))
        transform = exp_mat @ transform
    return transform


def jacobian_space(slist: np.ndarray, thetalist: np.ndarray) -> np.ndarray:
    """Space-form Jacobian from space-form screw axes and joint angles."""
    slist = np.asarray(slist, dtype=float)
    thetalist = np.asarray(thetalist, dtype=float).reshape(-1)
    n_joints = thetalist.size
    js = np.zeros((6, n_joints))
    js[:, 0] = slist[:, 0]
    transform = np.eye(4)
    for i in range(1, n_joints):
        transform = transform @ matrix_exp6(vec_to_se3(slist[:, i - 1] * thetalist[i - 1]))
        js[:, i] = adjoint(transform) @ slist[:, i]
    return js


def clamp_to_magnitude_minimum(mat: np.ndarray, min_magnitude: float) -> np.ndarray:
    """Clamp each element to at least ``min_magnitude`` while preserving sign.

    Zero values are pushed to positive ``min_magnitude``.
    """
    mat = np.asarray(mat, dtype=float)
    signs = np.sign(mat)
    signs = np.where(signs == 0.0, 1.0, signs)
    magnitudes = np.maximum(np.abs(mat), min_magnitude)
    return signs * magnitudes
