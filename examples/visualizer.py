"""Reusable Viser trajectory visualization for all examples."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation

import cca_planner as cca


@dataclass(frozen=True)
class VisualizationConfig:
    port: int = 8080
    grid_columns: int = 4
    grid_spacing_m: float = 0.7
    animation_step_s: float = 0.08
    robot_color: tuple[int, int, int] = (150, 152, 158)
    failed_color: tuple[int, int, int] = (255, 38, 26)


@dataclass(frozen=True)
class TrajectoryPhase:
    """A colored point range; ranges may overlap at a stage boundary."""

    name: str
    start_point: int
    end_point: int
    color: tuple[int, int, int]


class TaskScene(Protocol):
    """Callback that adds task geometry and optionally returns a frame updater."""

    def __call__(
        self,
        server: object,
        offsets: np.ndarray,
        success_mask: np.ndarray,
        trajectory_steps: int,
    ) -> Callable[[int], None] | None: ...


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def _rotation_and_scale(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate the rotation and per-axis scale stored in a scene transform."""
    linear = transform[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale <= np.finfo(scale.dtype).eps):
        raise ValueError("URDF visual geometry contains a zero scale")
    rotation = linear / scale
    if np.linalg.det(rotation) < 0.0:
        scale[-1] *= -1.0
        rotation[:, -1] *= -1.0
    return rotation, scale


class _CabinetUrdfScene:
    """Animate one cabinet joint using the same URDF that defines the task."""

    def __init__(
        self,
        urdf_path,
        cabinet_positions,
        cabinet_yaw,
        joint_name,
        joint_goals,
        motion_axis_segments,
        handle_positions,
        handle_axis,
        failed_color,
    ):
        self.urdf_path = urdf_path
        self.cabinet_positions = as_numpy(cabinet_positions)
        self.cabinet_yaw = float(cabinet_yaw)
        self.joint_name = joint_name
        self.joint_goals = as_numpy(joint_goals)
        self.motion_axis_segments = as_numpy(motion_axis_segments)
        self.handle_positions = as_numpy(handle_positions)
        self.handle_axis = np.asarray(handle_axis, dtype=float)
        self.failed_color = failed_color

    def __call__(self, server, offsets, success_mask, trajectory_steps):
        cabinet = yourdfpy.URDF.load(str(self.urdf_path))
        geometry_nodes = list(cabinet.scene.graph.nodes_geometry)
        meshes = {
            node: cabinet.scene.geometry[node].copy() for node in geometry_nodes
        }
        joint_index = cabinet.actuated_joint_names.index(self.joint_name)
        cabinet_rotation = Rotation.from_euler("z", self.cabinet_yaw).as_matrix()

        def transforms(joint_value, position, offset):
            config = np.zeros(len(cabinet.actuated_joint_names), dtype=float)
            config[joint_index] = joint_value
            cabinet.update_cfg(config)
            base = np.eye(4)
            base[:3, :3] = cabinet_rotation
            base[:3, 3] = position + offset
            graph = cabinet.scene.graph
            return [base @ graph.get(node)[0] for node in geometry_nodes]

        cabinet_handles = []
        normalized_handle_axis = self.handle_axis / np.linalg.norm(self.handle_axis)
        for environment, position in enumerate(self.cabinet_positions):
            environment_handles = []
            initial_transforms = transforms(0.0, position, offsets[environment])
            for link_index, (node, transform) in enumerate(
                zip(geometry_nodes, initial_transforms)
            ):
                rotation, scale = _rotation_and_scale(transform)
                handle = server.scene.add_mesh_trimesh(
                    f"tasks/cabinet/{environment}/link{link_index}",
                    meshes[node],
                    scale=tuple(scale),
                )
                handle.position = transform[:3, 3]
                handle.wxyz = _quaternion(rotation)
                environment_handles.append(handle)
            cabinet_handles.append(environment_handles)

            axis_color = (
                (20, 220, 190)
                if success_mask[environment]
                else self.failed_color
            )
            server.scene.add_line_segments(
                f"tasks/cabinet/{environment}/motion_axis",
                points=self.motion_axis_segments[environment][None, ...]
                + offsets[environment],
                colors=axis_color,
                line_width=3.0,
            )
            half_handle = 0.065 * normalized_handle_axis
            handle_position = self.handle_positions[environment]
            server.scene.add_line_segments(
                f"tasks/cabinet/{environment}/handle_axis",
                points=np.array(
                    [[handle_position - half_handle, handle_position + half_handle]]
                )
                + offsets[environment],
                colors=(255, 220, 80),
                line_width=4.0,
            )

        def update(frame):
            fraction = frame / max(trajectory_steps - 1, 1)
            for environment, handles in enumerate(cabinet_handles):
                current_transforms = transforms(
                    self.joint_goals[environment] * fraction,
                    self.cabinet_positions[environment],
                    offsets[environment],
                )
                for handle, transform in zip(handles, current_transforms):
                    rotation, _ = _rotation_and_scale(transform)
                    handle.position = transform[:3, 3]
                    handle.wxyz = _quaternion(rotation)

        return update


class _ValveScene:
    """Draw and animate a primitive valve around its configured screw axis."""

    def __init__(
        self,
        centres,
        radii,
        angles,
        rotation_axis,
        grasp_direction,
        rim_points,
        failed_color,
    ):
        self.centres = as_numpy(centres)
        self.radii = as_numpy(radii)
        self.angles = as_numpy(angles)
        self.rotation_axis = np.asarray(rotation_axis, dtype=float)
        self.grasp_direction = np.asarray(grasp_direction, dtype=float)
        self.rim_points = rim_points
        self.failed_color = failed_color

    def __call__(self, server, offsets, success_mask, trajectory_steps):
        axis = self.rotation_axis / np.linalg.norm(self.rotation_axis)
        grasp = self.grasp_direction / np.linalg.norm(self.grasp_direction)
        tangent = np.cross(axis, grasp)
        tangent /= np.linalg.norm(tangent)
        theta = np.linspace(0.0, 2.0 * np.pi, self.rim_points, endpoint=False)
        handles = []
        for index, (centre, radius) in enumerate(
            zip(self.centres, self.radii, strict=True)
        ):
            rim = radius * (
                np.cos(theta)[:, None] * grasp + np.sin(theta)[:, None] * tangent
            )
            spoke_directions = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
            spokes = np.stack(
                [
                    np.zeros((6, 3)),
                    radius
                    * (
                        np.cos(spoke_directions)[:, None] * grasp
                        + np.sin(spoke_directions)[:, None] * tangent
                    ),
                ],
                axis=1,
            )
            segments = np.concatenate(
                [np.stack([rim, np.roll(rim, -1, axis=0)], axis=1), spokes],
                axis=0,
            )
            color = (255, 160, 40) if success_mask[index] else self.failed_color
            handles.append(
                server.scene.add_line_segments(
                    f"tasks/valve/{index}",
                    points=segments,
                    colors=color,
                    line_width=3.0,
                    position=centre + offsets[index],
                )
            )

        def update(frame):
            fraction = frame / max(trajectory_steps - 1, 1)
            for handle, angle in zip(handles, self.angles, strict=True):
                half_angle = 0.5 * angle * fraction
                handle.wxyz = np.concatenate(
                    [[np.cos(half_angle)], axis * np.sin(half_angle)]
                )

        return update


def as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class ViserVisualizer:
    """Visualize parallel robot trajectories, task objects, and failure segments."""

    def __init__(
        self,
        robot: cca.RobotDescription,
        urdf_path,
        *,
        config: VisualizationConfig = VisualizationConfig(),
    ):
        self.robot = robot
        self.config = config
        self.urdf = yourdfpy.URDF.load(str(urdf_path))
        self.geometry_nodes = list(self.urdf.scene.graph.nodes_geometry)
        self.link_meshes = {}
        rgba = np.array([*config.robot_color, 255], dtype=np.uint8)
        for node in self.geometry_nodes:
            mesh = self.urdf.scene.geometry[node].copy()
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1))
            )
            self.link_meshes[node] = mesh

    def grid_offsets(self, count: int) -> np.ndarray:
        columns = min(self.config.grid_columns, count)
        rows = np.arange(count) // columns
        column_indices = np.arange(count) % columns
        center = (columns - 1) * self.config.grid_spacing_m / 2
        return np.stack(
            [
                column_indices * self.config.grid_spacing_m - center,
                rows * self.config.grid_spacing_m,
                np.zeros(count),
            ],
            axis=-1,
        )

    def end_effector_positions(self, joint_trajectory) -> np.ndarray:
        trajectory = torch.as_tensor(
            joint_trajectory,
            dtype=self.robot.joint_states.dtype,
            device=self.robot.joint_states.device,
        )
        batch_size, steps = trajectory.shape[:2]
        robot_dof = self.robot.joint_states.numel()
        flat_joints = trajectory[..., :robot_dof].reshape(-1, robot_dof)
        slist = self.robot.slist.expand(flat_joints.shape[0], -1, -1)
        home = self.robot.M.expand(flat_joints.shape[0], -1, -1)
        poses = cca.fkin_space(home, slist, flat_joints)
        return poses.reshape(batch_size, steps, 4, 4)[..., :3, 3].cpu().numpy()

    def _link_world_transforms(self, joints) -> np.ndarray:
        config = np.zeros(len(self.urdf.actuated_joint_names))
        robot_dof = self.robot.joint_states.numel()
        config[:robot_dof] = joints[:robot_dof]
        self.urdf.update_cfg(config)
        graph = self.urdf.scene.graph
        return np.stack([graph.get(node)[0] for node in self.geometry_nodes])

    @staticmethod
    def _quaternion(rotation: np.ndarray) -> np.ndarray:
        return _quaternion(rotation)

    @staticmethod
    def _rotation_and_scale(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _rotation_and_scale(transform)

    def _add_path(
        self,
        server,
        name: str,
        points: np.ndarray,
        valid_mask: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        if points.shape[0] < 2:
            return
        colors = np.where(
            valid_mask[:, None],
            np.asarray(color, dtype=np.uint8),
            np.asarray(self.config.failed_color, dtype=np.uint8),
        )
        server.scene.add_line_segments(
            name,
            points=np.stack([points[:-1], points[1:]], axis=1),
            colors=np.repeat(colors[:, None, :], 2, axis=1),
            line_width=2.0,
        )

    def show_cabinet_door(
        self,
        joint_trajectory,
        valid_mask,
        success_mask,
        cabinet_urdf_path,
        cabinet_positions,
        hinge_positions,
        handle_positions,
        open_angles,
        *,
        cabinet_yaw=np.pi,
        handle_axis=(0.0, 0.0, 1.0),
    ):
        """Visualize a cabinet door rotating about its vertical hinge."""
        steps = as_numpy(joint_trajectory).shape[1]
        hinge_positions = as_numpy(hinge_positions)
        hinge_axis = np.asarray((0.0, 0.0, 1.0))
        axis_segments = np.stack(
            [
                hinge_positions - 0.25 * hinge_axis,
                hinge_positions + 0.25 * hinge_axis,
            ],
            axis=1,
        )
        self.show(
            joint_trajectory,
            valid_mask,
            success_mask,
            [TrajectoryPhase("open_door", 0, steps, (255, 160, 40))],
            task_scene=_CabinetUrdfScene(
                cabinet_urdf_path,
                cabinet_positions,
                cabinet_yaw,
                "door_joint_1",
                open_angles,
                axis_segments,
                handle_positions,
                handle_axis,
                self.config.failed_color,
            ),
        )

    def show_cabinet_drawer_1(
        self,
        joint_trajectory,
        valid_mask,
        success_mask,
        cabinet_urdf_path,
        cabinet_positions,
        handle_positions,
        pull_distances,
        *,
        cabinet_yaw=np.pi,
        pull_axis=(-1.0, 0.0, 0.0),
        handle_axis=(0.0, -1.0, 0.0),
    ):
        """Visualize drawer 1 translating out of a three-drawer cabinet."""
        steps = as_numpy(joint_trajectory).shape[1]
        handle_positions = as_numpy(handle_positions)
        pull_distances = as_numpy(pull_distances)
        normalized_pull_axis = np.asarray(pull_axis, dtype=float)
        normalized_pull_axis /= np.linalg.norm(normalized_pull_axis)
        axis_segments = np.stack(
            [
                handle_positions,
                handle_positions + pull_distances[:, None] * normalized_pull_axis,
            ],
            axis=1,
        )
        self.show(
            joint_trajectory,
            valid_mask,
            success_mask,
            [TrajectoryPhase("open_drawer_1", 0, steps, (20, 220, 190))],
            task_scene=_CabinetUrdfScene(
                cabinet_urdf_path,
                cabinet_positions,
                cabinet_yaw,
                "drawer_joint_1",
                -pull_distances,
                axis_segments,
                handle_positions,
                handle_axis,
                self.config.failed_color,
            ),
        )

    def show_valve(
        self,
        joint_trajectory,
        valid_mask,
        success_mask,
        centres,
        radii,
        angles,
        *,
        rotation_axis=(1.0, 0.0, 0.0),
        grasp_direction=(0.0, 0.0, 1.0),
        rim_points=48,
    ):
        """Visualize a valve affordance trajectory without an approach phase."""
        steps = as_numpy(joint_trajectory).shape[1]
        self.show(
            joint_trajectory,
            valid_mask,
            success_mask,
            [TrajectoryPhase("turn_valve", 0, steps, (255, 160, 40))],
            task_scene=_ValveScene(
                centres,
                radii,
                angles,
                rotation_axis,
                grasp_direction,
                rim_points,
                self.config.failed_color,
            ),
        )

    def show(
        self,
        joint_trajectory,
        valid_mask,
        success_mask,
        phases: list[TrajectoryPhase],
        *,
        task_scene: TaskScene | None = None,
    ) -> None:
        """Start Viser and animate until interrupted with Ctrl+C."""
        import viser

        trajectory = as_numpy(joint_trajectory)
        valid_mask = as_numpy(valid_mask).astype(bool)
        success_mask = as_numpy(success_mask).astype(bool)
        if trajectory.ndim != 3:
            raise ValueError("joint_trajectory must have shape [B, T, n]")
        batch_size, steps = trajectory.shape[:2]
        if valid_mask.shape != (batch_size, steps - 1):
            raise ValueError("valid_mask must have shape [B, T - 1]")
        if success_mask.shape != (batch_size,):
            raise ValueError("success_mask must have shape [B]")
        for phase in phases:
            if not (0 <= phase.start_point < phase.end_point <= steps):
                raise ValueError(f"invalid point range for phase {phase.name!r}")

        offsets = self.grid_offsets(batch_size)
        tcp_positions = self.end_effector_positions(trajectory)
        server = viser.ViserServer(port=self.config.port)
        print(f"\nViser: http://localhost:{server.get_port()}")
        server.scene.add_frame("world", axes_length=0.2, axes_radius=0.004)

        update_task = (
            task_scene(server, offsets, success_mask, steps)
            if task_scene is not None
            else None
        )
        for environment in range(batch_size):
            for phase in phases:
                start, end = phase.start_point, phase.end_point
                self._add_path(
                    server,
                    f"paths/{environment}/{phase.name}",
                    tcp_positions[environment, start:end] + offsets[environment],
                    valid_mask[environment, start : end - 1],
                    phase.color,
                )

        robot_handles = []
        for environment in range(batch_size):
            transforms = self._link_world_transforms(trajectory[environment, 0])
            handles = []
            for link_index, (node, transform) in enumerate(
                zip(self.geometry_nodes, transforms)
            ):
                rotation, scale = self._rotation_and_scale(transform)
                handle = server.scene.add_mesh_trimesh(
                    f"robots/{environment}/link{link_index}",
                    self.link_meshes[node],
                    scale=tuple(scale),
                )
                handle.position = transform[:3, 3] + offsets[environment]
                handle.wxyz = self._quaternion(rotation)
                handles.append(handle)
            robot_handles.append(handles)

        print(
            f"  {batch_size} environments: {int(success_mask.sum())} full, "
            f"{int((~success_mask).sum())} failed/partial (red segments); Ctrl+C to stop."
        )
        try:
            while True:
                for frame in range(steps):
                    if update_task is not None:
                        update_task(frame)
                    for environment, handles in enumerate(robot_handles):
                        transforms = self._link_world_transforms(
                            trajectory[environment, frame]
                        )
                        for handle, transform in zip(handles, transforms):
                            rotation, _ = self._rotation_and_scale(transform)
                            handle.position = transform[:3, 3] + offsets[environment]
                            handle.wxyz = self._quaternion(rotation)
                    time.sleep(self.config.animation_step_s)
        except KeyboardInterrupt:
            print("\nStopped.")
