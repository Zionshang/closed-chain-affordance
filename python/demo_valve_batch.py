"""Batched valve-turning with the x5 arm, visualized in viser.

Like ``demo_x5_urdf``'s affordance step but for a BATCH: a grid of x5 arms, each
turning a valve with different parameters (radius, turn angle). Each valve is
placed so the TCP (at the zero config) lies exactly on the rim —
``centre = EE0 - radius * grasp_dir`` — so the TCP orbits at the valve radius
and the drawn valve matches the planned motion exactly. All turns are planned in
ONE batched, pure-GPU call; viser draws every valve, every end-effector arc, and
animates every robot's mesh through its planned trajectory.

Requires the viewer extras (viser + yourdfpy + trimesh)::

    pip install -e ".[viewer]"

Standalone::

    python python/demo_valve_batch.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_chain_affordance as cca  # noqa: E402
import closed_chain_affordance.batched_math as bm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
X5_URDF = REPO_ROOT / "assets" / "robot" / "x5" / "urdf" / "x5.urdf"
X5_CONFIG = REPO_ROOT / "python" / "x5_urdf_config.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
N_ENVS = 128
GRID_COLS = 4

VALVE_AXIS = np.array([1.0, 0.0, 0.0])      # valve rotation axis (world +x)
GRASP_DIR = np.array([0.0, 0.0, 1.0])       # grasp from above (+z): TCP on top rim


# --------------------------------------------------------------------------- #
# Robot loading
# --------------------------------------------------------------------------- #
def _decimate(mesh, target_faces=1500):
    if len(mesh.faces) <= target_faces:
        return mesh
    step = max(1, len(mesh.faces) // target_faces)
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[::step], process=True)


def load_x5():
    robot = cca.build_robot_description_from_urdf(str(X5_URDF), str(X5_CONFIG), joint_states=np.zeros(6))
    urdf = yourdfpy.URDF.load(str(X5_URDF))
    geom_nodes = list(urdf.scene.graph.nodes_geometry)
    link_meshes = {}
    for n in geom_nodes:
        m = _decimate(trimesh.Trimesh(vertices=urdf.scene.geometry[n].vertices,
                                      faces=urdf.scene.geometry[n].faces))
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.tile([150, 152, 158, 255], (len(m.vertices), 1)))
        link_meshes[n] = m
    return robot, urdf, geom_nodes, link_meshes


def link_world_transforms(urdf, joint6, geom_nodes):
    cfg = np.zeros(len(urdf.actuated_joint_names)); cfg[:6] = joint6
    urdf.update_cfg(cfg)
    g = urdf.scene.graph
    return np.stack([g.get(n)[0] for n in geom_nodes])


def rotmat_to_wxyz(R):
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


# --------------------------------------------------------------------------- #
# Valve setup + two-stage batched plan (approach then turn)
# --------------------------------------------------------------------------- #
def build_valves(robot, n):
    """Per-env valves, geometrically consistent with the zero config: each valve
    centre is placed so the TCP (at zeros) lies exactly on the rim —
    ``centre = EE0 - radius * GRASP_DIR`` — so the planned turn orbits at the
    valve radius (no approach needed, and the drawn valve matches the motion)."""
    p0 = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
    ee0 = p0[:3, 3]
    rng = np.random.default_rng(0)
    radii = rng.uniform(0.05, 0.09, n)
    angles = rng.uniform(np.pi / 3, np.pi, n)
    centres = ee0[None, :] - radii[:, None] * GRASP_DIR  # TCP is `radius` above each centre
    return centres, radii, angles


def plan_batch(robot, centres, angles, density=12, max_itr=50):
    """Single batched AFFORDANCE turn per environment (robot already grasping at
    zeros)."""
    n = len(angles)
    js0 = torch.tensor(np.broadcast_to(robot.joint_states, (n, 6)).copy(), dtype=DTYPE, device=DEVICE)
    turn_screw = torch.tensor(
        np.stack([np.concatenate([VALVE_AXIS, np.cross(centres[i], VALVE_AXIS)]) for i in range(n)]),
        dtype=DTYPE, device=DEVICE)
    kw = dict(
        robot_slist=torch.tensor(np.asarray(robot.slist), dtype=DTYPE, device=DEVICE).expand(n, 6, 6).contiguous(),
        robot_m=torch.tensor(np.asarray(robot.M), dtype=DTYPE, device=DEVICE).expand(n, 4, 4).contiguous(),
        joint_states=js0,
        motion_type=cca.MotionType.AFFORDANCE,
        affordance_screw=turn_screw,
        goal_affordance=torch.tensor(angles, dtype=DTYPE, device=DEVICE),
        trajectory_density=density,
        vir_screw_order=cca.VirtualScrewOrder.XYZ,
    )
    cfg = cca.PlannerConfig(); cfg.update_method = cca.UpdateMethod.INVERSE; cfg.ik_max_itr = max_itr
    iface = cca.BatchedCcAffordancePlannerInterface(cfg); iface.planner_.early_stop_ = False
    if DEVICE.type == "cuda":
        iface.generate_joint_trajectory(**kw)  # warm up
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = iface.generate_joint_trajectory(**kw)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return res.joint_trajectory, res.success, time.perf_counter() - t0


def ee_arcs(robot, joint_traj):
    joint_traj = np.asarray(joint_traj)
    B, T, n = joint_traj.shape
    flat = torch.tensor(joint_traj.reshape(B * T, n), dtype=DTYPE, device=DEVICE)
    slist = torch.tensor(np.asarray(robot.slist), dtype=DTYPE, device=DEVICE).expand(B * T, 6, n).contiguous()
    m = torch.tensor(np.asarray(robot.M), dtype=DTYPE, device=DEVICE).expand(B * T, 4, 4).contiguous()
    return bm.fkin_space(m, slist, flat).reshape(B, T, 4, 4)[..., :3, 3].detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# viser visualisation
# --------------------------------------------------------------------------- #
def valve_rim(center, axis, radius, n_pts=48):
    a = axis / np.linalg.norm(axis)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, tmp); u /= np.linalg.norm(u); v = np.cross(a, u)
    th = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    return center + radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)


def grid_offset(i, cols, spacing=0.7):
    r, c = divmod(i, cols)
    return np.array([c * spacing, r * spacing, 0.0]) - np.array([(cols - 1) * spacing / 2,
                                                                 (cols - 1) * spacing / 2, 0.0])


def visualize(robot, urdf, geom_nodes, link_meshes, full_traj, ee, centres, radii, angles, ok):
    import colorsys
    import viser

    server = viser.ViserServer(port=8080)
    print(f"\nviser server: http://localhost:{server.get_port()}")
    server.scene.add_frame("world", axes_length=0.2, axes_radius=0.004)

    ok_idx = np.where(ok)[0]
    amax = float(angles.max()) + 1e-9
    for i in ok_idx:
        off = grid_offset(i, GRID_COLS)
        hue = colorsys.hsv_to_rgb((angles[i] / amax) % 1.0, 0.85, 1.0)
        rim = valve_rim(centres[i] + off, VALVE_AXIS, radii[i])
        server.scene.add_line_segments(f"valve/{i}", points=np.stack([rim, np.roll(rim, -1, 0)], 1),
                                       colors=np.tile(hue, (rim.shape[0], 2, 1)), line_width=3.0)
        arc = ee[i] + off
        server.scene.add_line_segments(f"arc/{i}", points=np.stack([arc[:-1], arc[1:]], 1),
                                       colors=np.tile(hue, (arc.shape[0] - 1, 2, 1)), line_width=1.5)

    handles = []
    for i in ok_idx:
        off = grid_offset(i, GRID_COLS)
        row = []
        T0 = link_world_transforms(urdf, full_traj[i, 0], geom_nodes)
        for k, node in enumerate(geom_nodes):
            h = server.scene.add_mesh_trimesh(f"robot{i}/link{k}", link_meshes[node])
            h.position = T0[k][:3, 3] + off; h.wxyz = rotmat_to_wxyz(T0[k][:3, :3])
            row.append((h, off))
        handles.append((i, row))

    print(f"  {len(ok_idx)} robots turning their valves; animating. Ctrl+C to stop.")
    try:
        while True:
            for f in range(full_traj.shape[1]):
                for i, row in handles:
                    Ts = link_world_transforms(urdf, full_traj[i, f], geom_nodes)
                    for (h, off), T in zip(row, Ts):
                        h.position = T[:3, 3] + off
                        h.wxyz = rotmat_to_wxyz(T[:3, :3])
                time.sleep(0.08)
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    robot, urdf, geom_nodes, link_meshes = load_x5()
    centres, radii, angles = build_valves(robot, N_ENVS)
    print(f"Batched valve turning: {N_ENVS} x5 arms on {DEVICE}")
    full_traj, ok, dt = plan_batch(robot, centres, angles)
    print(f"  planned in {dt*1e3:.1f} ms  ({int(ok.sum())}/{N_ENVS} solved)")
    ee = ee_arcs(robot, full_traj.detach().cpu().numpy())
    visualize(robot, urdf, geom_nodes, link_meshes, full_traj.detach().cpu().numpy(),
              ee, centres, radii, angles, ok.detach().cpu().numpy())


if __name__ == "__main__":
    main()
