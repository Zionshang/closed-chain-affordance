"""Two-stage batched valve planning with the x5 arm, visualized in viser.

The valves start away from the TCP.  Every environment first plans a Cartesian
APPROACH to its valve's top rim.  The reached joint state is then fed into a
second AFFORDANCE call, which rebuilds the robot Jacobian at the grasp pose and
turns the valve.  Both stages are batched Torch calls; different environments
use different valve positions, radii, and turn angles.

Requires the viewer extras (viser + yourdfpy + trimesh)::

    pip install -e ".[viewer]"

Standalone::

    python python/demo_valve_batch.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
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
# Torch throughput defaults.  Fast mode uses a fixed iteration count (no early
# stop) and optionally compiles the IK loop; the regularized solver replaces two
# SVD applications without changing the task model or convergence criteria.
USE_FAST_MODE = True
COMPILE_FAST_MODE = False
USE_REGULARIZED_LINEAR_SOLVER = True
N_ENVS = 64
APPROACH_DENSITY = 12
TURN_DENSITY = 12
APPROACH_MAX_ITR = 15
TURN_MAX_ITR = 15

# Keep the same zero seed used by the original x5 example.  Its 6x6 robot
# Jacobian is full rank; convergence depends on the requested Cartesian path
# and discretisation, not on replacing this state with a hand-picked posture.
INITIAL_JOINTS = np.array([0, 0, 0, 0, 0, 0])


@dataclass(frozen=True)
class ValveHyperparameters:
    """Geometry and randomisation ranges for the batched valve tasks."""

    random_seed: int = 0
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    grasp_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius_range_m: tuple[float, float] = (0.05, 0.085)
    turn_angle_range_rad: tuple[float, float] = (np.pi / 3, np.pi)
    forward_offset_range_m: tuple[float, float] = (0.18, 0.22)
    lateral_offset_range_m: tuple[float, float] = (-0.01, 0.01)
    vertical_offset_range_m: tuple[float, float] = (0.18, 0.22)


@dataclass(frozen=True)
class VisualizationHyperparameters:
    port: int = 8080
    grid_columns: int = 4
    grid_spacing_m: float = 0.7
    valve_rim_points: int = 48
    animation_step_s: float = 0.08
    approach_color_rgb: tuple[float, float, float] = (0.35, 0.75, 1.0)
    failed_candidate_color_rgb: tuple[float, float, float] = (1.0, 0.15, 0.1)


VALVE_PARAMS = ValveHyperparameters()
VISUALIZATION_PARAMS = VisualizationHyperparameters()


def load_x5():
    robot = cca.build_robot_description_from_urdf(
        str(X5_URDF), str(X5_CONFIG), joint_states=INITIAL_JOINTS.copy()
    )
    urdf = yourdfpy.URDF.load(str(X5_URDF))
    geom_nodes = list(urdf.scene.graph.nodes_geometry)
    link_meshes = {}
    for n in geom_nodes:
        m = urdf.scene.geometry[n].copy()
        m.visual = trimesh.visual.ColorVisuals(
            mesh=m, vertex_colors=np.tile([150, 152, 158, 255], (len(m.vertices), 1))
        )
        link_meshes[n] = m
    return robot, urdf, geom_nodes, link_meshes


# --------------------------------------------------------------------------- #
# Valve setup + two-stage batched plan (approach then turn)
# --------------------------------------------------------------------------- #
def build_valves(robot, n, params=VALVE_PARAMS):
    """Create randomly scattered valves in front of the robot (base-frame +x)."""
    start_pose = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
    rng = np.random.default_rng(params.random_seed)
    radii = rng.uniform(*params.radius_range_m, n)
    angles = rng.uniform(*params.turn_angle_range_rad, n)
    # The x5 base frame's +x direction is "forward".  This follows the original
    # zero-seed example's reachable +x/+z approach while independently sampling
    # depth, lateral offset, and height for every environment.
    offsets = np.stack(
        [
            rng.uniform(*params.forward_offset_range_m, n),
            rng.uniform(*params.lateral_offset_range_m, n),
            rng.uniform(*params.vertical_offset_range_m, n),
        ],
        axis=-1,
    )
    grasp_poses = np.broadcast_to(start_pose, (n, 4, 4)).copy()
    grasp_poses[:, :3, 3] += offsets
    grasp_direction = np.asarray(params.grasp_direction, dtype=float)
    grasp_direction /= np.linalg.norm(grasp_direction)
    centres = grasp_poses[:, :3, 3] - radii[:, None] * grasp_direction
    return centres, radii, angles, grasp_poses


@dataclass
class TwoStageBatchResult:
    approach: cca.BatchedPlannerResult
    turn: cca.BatchedPlannerResult
    joint_trajectory: torch.Tensor
    valid_mask: torch.Tensor
    full_success: torch.Tensor
    approach_time_s: float
    turn_time_s: float
    approach_points: int


def _robot_tensors(robot, n):
    return (
        torch.tensor(np.asarray(robot.slist), dtype=DTYPE, device=DEVICE)
        .expand(n, 6, 6).contiguous(),
        torch.tensor(np.asarray(robot.M), dtype=DTYPE, device=DEVICE)
        .expand(n, 4, 4).contiguous(),
        torch.tensor(
            np.broadcast_to(robot.joint_states, (n, 6)).copy(), dtype=DTYPE, device=DEVICE
        ),
    )


def _make_planner(max_iterations):
    config = cca.PlannerConfig()
    config.accuracy = 0.1
    config.closure_err_threshold_ang = 1e-3
    config.closure_err_threshold_lin = 1e-2

    config.update_method = cca.UpdateMethod.INVERSE
    config.ik_max_itr = max_iterations
    return cca.BatchedCcAffordancePlannerInterface(
        config,
        fast_mode=USE_FAST_MODE,
        compile=COMPILE_FAST_MODE,
        fast_linear_solver=USE_REGULARIZED_LINEAR_SOLVER,
    )


def _timed_plan(iface, kwargs):
    # Warm allocator/cuSOLVER for this exact shape before reporting planner time.
    if DEVICE.type == "cuda":
        iface.generate_joint_trajectory(**kwargs)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = iface.generate_joint_trajectory(**kwargs)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - t0


def plan_batch(
    robot,
    centres,
    angles,
    grasp_poses,
    valve_params=VALVE_PARAMS,
    approach_density=APPROACH_DENSITY,
    turn_density=TURN_DENSITY,
    approach_max_itr=APPROACH_MAX_ITR,
    turn_max_itr=TURN_MAX_ITR,
):
    """Plan batched Cartesian approaches, then turns from the reached states."""
    n = len(angles)
    robot_slist, robot_home, initial_joints = _robot_tensors(robot, n)

    # Match PlanningType.CARTESIAN_GOAL: the final (arbitrary) affordance joint
    # is held at epsilon while the approach joint traverses the Cartesian screw.
    cart_axis = torch.ones(n, 3, dtype=DTYPE, device=DEVICE) / np.sqrt(3.0)
    cart_location = torch.ones(n, 3, dtype=DTYPE, device=DEVICE)
    cartesian_reference_screw = torch.cat(
        [cart_axis, torch.cross(cart_location, cart_axis, dim=-1)], dim=-1
    )
    approach_iface = _make_planner(approach_max_itr)
    approach, approach_dt = _timed_plan(approach_iface, dict(
        robot_slist=robot_slist,
        robot_m=robot_home,
        joint_states=initial_joints,
        motion_type=cca.MotionType.APPROACH,
        affordance_screw=cartesian_reference_screw,
        goal_affordance=torch.full((n,), 1e-5, dtype=DTYPE, device=DEVICE),
        trajectory_density=approach_density,
        vir_screw_order=cca.VirtualScrewOrder.NONE,
        canonical_pose=torch.tensor(grasp_poses, dtype=DTYPE, device=DEVICE),
    ))

    # Use the final APPROACH output for every environment.  For a failed
    # APPROACH this is its final IK candidate, which lets the demo diagnose and
    # visualize what a real TURN attempt from that candidate would do.
    # Passing it to the high-level interface recomputes the robot Jacobian for
    # every environment before the valve turn.
    reached_joints = approach.joint_trajectory[:, -1, :6]
    valve_axis = np.asarray(valve_params.rotation_axis, dtype=float)
    valve_axis /= np.linalg.norm(valve_axis)
    turn_screw = torch.tensor(
        np.stack(
            [np.concatenate([valve_axis, np.cross(centres[i], valve_axis)]) for i in range(n)]
        ),
        dtype=DTYPE,
        device=DEVICE,
    )
    turn_goals = torch.tensor(angles, dtype=DTYPE, device=DEVICE)
    turn_iface = _make_planner(turn_max_itr)
    turn, turn_dt = _timed_plan(turn_iface, dict(
        robot_slist=robot_slist,
        robot_m=robot_home,
        joint_states=reached_joints,
        motion_type=cca.MotionType.AFFORDANCE,
        affordance_screw=turn_screw,
        goal_affordance=turn_goals,
        trajectory_density=turn_density,
        vir_screw_order=cca.VirtualScrewOrder.XYZ,
    ))

    # The first turn point equals the final approach point; keep it only once.
    full_trajectory = torch.cat(
        [approach.joint_trajectory, turn.joint_trajectory[:, 1:]], dim=1
    )
    # One solver-validity entry per segment in ``full_trajectory``.  TURN is
    # attempted and visualized even after a failed APPROACH; ``full_success``
    # below still requires both stages to be fully valid.
    full_valid_mask = torch.cat(
        [approach.valid_mask, turn.valid_mask],
        dim=1,
    )
    return TwoStageBatchResult(
        approach=approach,
        turn=turn,
        joint_trajectory=full_trajectory,
        valid_mask=full_valid_mask,
        full_success=approach.full_success & turn.full_success,
        approach_time_s=approach_dt,
        turn_time_s=turn_dt,
        approach_points=approach.joint_trajectory.shape[1],
    )


class ValveBatchVisualizer:
    """Owns viser scene construction and animation for a planned valve batch."""

    def __init__(
        self,
        robot,
        urdf,
        geom_nodes,
        link_meshes,
        valve_params=VALVE_PARAMS,
        visualization_params=VISUALIZATION_PARAMS,
    ):
        self.robot = robot
        self.urdf = urdf
        self.geom_nodes = geom_nodes
        self.link_meshes = link_meshes
        self.valve_params = valve_params
        self.params = visualization_params

    def _link_world_transforms(self, joint6):
        cfg = np.zeros(len(self.urdf.actuated_joint_names))
        cfg[:6] = joint6
        self.urdf.update_cfg(cfg)
        graph = self.urdf.scene.graph
        return np.stack([graph.get(node)[0] for node in self.geom_nodes])

    @staticmethod
    def _rotmat_to_wxyz(rotation):
        return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]

    def _grid_offset(self, index):
        row, column = divmod(index, self.params.grid_columns)
        spacing = self.params.grid_spacing_m
        center = (self.params.grid_columns - 1) * spacing / 2
        return np.array([column * spacing - center, row * spacing - center, 0.0])

    def _valve_rim(self, center, radius):
        axis = np.asarray(self.valve_params.rotation_axis, dtype=float)
        axis /= np.linalg.norm(axis)
        reference = (
            np.array([0.0, 0.0, 1.0])
            if abs(axis[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        tangent = np.cross(axis, reference)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(axis, tangent)
        theta = np.linspace(
            0, 2 * np.pi, self.params.valve_rim_points, endpoint=False
        )
        return center + radius * (
            np.cos(theta)[:, None] * tangent
            + np.sin(theta)[:, None] * bitangent
        )

    def _end_effector_positions(self, joint_trajectory):
        joint_trajectory = np.asarray(joint_trajectory)
        batch, steps, n_joints = joint_trajectory.shape
        flat_joints = torch.tensor(
            joint_trajectory.reshape(batch * steps, n_joints),
            dtype=DTYPE,
            device=DEVICE,
        )
        slist = torch.tensor(
            np.asarray(self.robot.slist), dtype=DTYPE, device=DEVICE
        ).expand(batch * steps, 6, n_joints).contiguous()
        home = torch.tensor(
            np.asarray(self.robot.M), dtype=DTYPE, device=DEVICE
        ).expand(batch * steps, 4, 4).contiguous()
        poses = bm.fkin_space(home, slist, flat_joints)
        return poses.reshape(batch, steps, 4, 4)[..., :3, 3].cpu().numpy()

    @staticmethod
    def _add_colored_path(server, name, points, valid_mask, valid_color, invalid_color):
        """Draw every candidate segment, coloring non-converged endpoints red."""
        if points.shape[0] < 2:
            return
        segments = np.stack([points[:-1], points[1:]], axis=1)
        segment_colors = np.where(
            np.asarray(valid_mask, dtype=bool)[:, None],
            np.asarray(valid_color, dtype=float),
            np.asarray(invalid_color, dtype=float),
        )
        server.scene.add_line_segments(
            name,
            points=segments,
            colors=np.repeat(segment_colors[:, None, :], 2, axis=1),
            line_width=2.0,
        )

    def show(
        self,
        joint_trajectory,
        centres,
        radii,
        angles,
        success_mask,
        valid_mask,
        approach_points,
    ):
        import colorsys
        import viser

        trajectory = np.asarray(joint_trajectory)
        success_mask = np.asarray(success_mask, dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (trajectory.shape[0], trajectory.shape[1] - 1):
            raise ValueError(
                "valid_mask must have shape [batch, trajectory_steps - 1], got "
                f"{valid_mask.shape} for trajectory {trajectory.shape}"
            )
        ee_positions = self._end_effector_positions(trajectory)
        server = viser.ViserServer(port=self.params.port)
        print(f"\nviser server: http://localhost:{server.get_port()}")
        server.scene.add_frame("world", axes_length=0.2, axes_radius=0.004)

        all_indices = range(trajectory.shape[0])
        max_angle = float(angles.max()) + 1e-9
        invalid_color = self.params.failed_candidate_color_rgb
        for index in all_indices:
            offset = self._grid_offset(index)
            color = colorsys.hsv_to_rgb((angles[index] / max_angle) % 1.0, 0.85, 1.0)
            valve_color = color if success_mask[index] else invalid_color
            rim = self._valve_rim(centres[index] + offset, radii[index])
            server.scene.add_line_segments(
                f"valve/{index}",
                points=np.stack([rim, np.roll(rim, -1, axis=0)], axis=1),
                colors=np.tile(valve_color, (rim.shape[0], 2, 1)),
                line_width=3.0,
            )
            approach_arc = ee_positions[index, :approach_points] + offset
            turn_arc = ee_positions[index, approach_points - 1:] + offset
            approach_valid = valid_mask[index, : approach_points - 1]
            turn_valid = valid_mask[index, approach_points - 1 :]
            self._add_colored_path(
                server,
                f"approach/{index}",
                approach_arc,
                approach_valid,
                self.params.approach_color_rgb,
                invalid_color,
            )
            self._add_colored_path(
                server,
                f"turn/{index}",
                turn_arc,
                turn_valid,
                color,
                invalid_color,
            )

        robot_handles = []
        for index in all_indices:
            offset = self._grid_offset(index)
            transforms = self._link_world_transforms(trajectory[index, 0])
            handles = []
            for link_index, (node, transform) in enumerate(
                zip(self.geom_nodes, transforms)
            ):
                handle = server.scene.add_mesh_trimesh(
                    f"robot{index}/link{link_index}", self.link_meshes[node]
                )
                handle.position = transform[:3, 3] + offset
                handle.wxyz = self._rotmat_to_wxyz(transform[:3, :3])
                handles.append((handle, offset))
            robot_handles.append((index, handles))

        print(
            f"  visualizing all {trajectory.shape[0]} robots: "
            f"{int(success_mask.sum())} full trajectories, "
            f"{int((~success_mask).sum())} failed/partial trajectories in red; "
            "Ctrl+C to stop."
        )
        try:
            while True:
                for frame in range(trajectory.shape[1]):
                    for index, handles in robot_handles:
                        transforms = self._link_world_transforms(
                            trajectory[index, frame]
                        )
                        for (handle, offset), transform in zip(handles, transforms):
                            handle.position = transform[:3, 3] + offset
                            handle.wxyz = self._rotmat_to_wxyz(transform[:3, :3])
                    time.sleep(self.params.animation_step_s)
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    robot, urdf, geom_nodes, link_meshes = load_x5()
    centres, radii, angles, grasp_poses = build_valves(robot, N_ENVS, VALVE_PARAMS)
    start_pose = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
    approach_distances = np.linalg.norm(grasp_poses[:, :3, 3] - start_pose[:3, 3], axis=-1)
    print(
        f"Two-stage batched valve planning: {N_ENVS} x5 arms on {DEVICE} "
        f"(approach {approach_distances.min()*100:.1f}--{approach_distances.max()*100:.1f} cm; "
        f"dtype={str(DTYPE).removeprefix('torch.')}, "
        f"mode={'fast' if USE_FAST_MODE else 'early-stop'}, "
        f"compile={COMPILE_FAST_MODE and USE_FAST_MODE}, "
        f"linear solver={'regularized' if USE_REGULARIZED_LINEAR_SOLVER else 'SVD'})"
    )
    result = plan_batch(
        robot, centres, angles, grasp_poses, valve_params=VALVE_PARAMS
    )
    full_traj = result.joint_trajectory
    approach_ok = result.approach.full_success
    turn_solver_ok = result.turn.full_success
    turn_usable = turn_solver_ok & approach_ok
    ok = result.full_success
    approach_partial = result.approach.success & ~approach_ok
    turn_partial = result.turn.success & ~turn_solver_ok
    print(
        f"  approach: {result.approach_time_s*1e3:.1f} ms, "
        f"{int(approach_ok.sum())}/{N_ENVS} full, {int(approach_partial.sum())} partial; "
        f"mean active IK iterations={float(result.approach.active_iterations.float().mean()):.1f}"
    )
    print(
        f"  turn:     {result.turn_time_s*1e3:.1f} ms, "
        f"{int(turn_solver_ok.sum())}/{N_ENVS} candidate starts full, "
        f"{int(turn_partial.sum())} partial; "
        f"mean active IK iterations={float(result.turn.active_iterations.float().mean()):.1f}"
    )
    print(
        f"  combined: {int(ok.sum())}/{N_ENVS} full "
        f"({int(turn_usable.sum())} turns usable after a full approach), "
        f"{(result.approach_time_s + result.turn_time_s)*1e3:.1f} ms total"
    )
    visualizer = ValveBatchVisualizer(
        robot,
        urdf,
        geom_nodes,
        link_meshes,
        valve_params=VALVE_PARAMS,
        visualization_params=VISUALIZATION_PARAMS,
    )
    visualizer.show(
        full_traj.detach().cpu().numpy(),
        centres,
        radii,
        angles,
        ok.detach().cpu().numpy(),
        result.valid_mask.detach().cpu().numpy(),
        result.approach_points,
    )


if __name__ == "__main__":
    main()
