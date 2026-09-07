"""Mesh-based Viser animation for the drawer, door, and valve examples."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cabinet_demo_common import ARM_DOF, base_pose_trajectory, cca


@dataclass(frozen=True)
class VisualizationConfig:
    port: int = 8080
    animation_step_s: float = 0.08
    case_spacing_m: float = 1.0


@dataclass(frozen=True)
class TrajectoryCase:
    name: str
    result: cca.PlannerResult
    color: tuple[int, int, int]


@dataclass(frozen=True)
class CabinetTaskScene:
    urdf_path: Path
    cabinet_position: np.ndarray
    cabinet_yaw: float
    joint_name: str
    joint_goal: float
    handle_position: np.ndarray
    handle_axis: np.ndarray
    motion_path: np.ndarray


@dataclass(frozen=True)
class ValveTaskScene:
    center_position: np.ndarray
    rotation_axis: np.ndarray
    joint_goal: float
    handle_position: np.ndarray
    handle_axis: np.ndarray
    motion_path: np.ndarray
    grasp_radius: float = 0.1425
    rim_tube_radius: float = 0.0175


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to Viser's scalar-first quaternion."""
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def _color_mesh(mesh, rgba: tuple[int, int, int, int]):
    mesh.visual.face_colors = np.asarray(rgba, dtype=np.uint8)
    return mesh


def _add_procedural_valve(server, root: str, scene: ValveTaskScene, offset: np.ndarray):
    """Recreate rl_art_mj's six-spoke valve from the same primitive geometry."""
    import trimesh

    valve_root = server.scene.add_frame(
        f"{root}/valve",
        position=np.asarray(scene.center_position) + offset,
        show_axes=False,
    )
    fixed_mesh = trimesh.util.concatenate(
        [
            _color_mesh(
                trimesh.creation.cylinder(
                    radius=0.0425,
                    segment=np.asarray(((0.0, 0.0, 0.0), (0.09, 0.0, 0.0))),
                ),
                (77, 79, 84, 255),
            ),
            _color_mesh(
                trimesh.creation.cylinder(
                    radius=0.029,
                    segment=np.asarray(((-0.0375, 0.0, 0.0), (0.0475, 0.0, 0.0))),
                ),
                (77, 79, 84, 255),
            ),
        ]
    )
    server.scene.add_mesh_trimesh(f"{root}/valve/fixed", fixed_mesh)

    wheel_root = server.scene.add_frame(f"{root}/valve/wheel", show_axes=False)
    wheel_parts = [
        _color_mesh(
            trimesh.creation.cylinder(
                radius=0.045,
                segment=np.asarray(((-0.0275, 0.0, 0.0), (0.0275, 0.0, 0.0))),
            ),
            (230, 46, 31, 255),
        )
    ]
    spoke_inner_radius = 0.027
    for index in range(6):
        angle = 2.0 * np.pi * index / 6.0
        direction = np.asarray((0.0, np.cos(angle), np.sin(angle)))
        wheel_parts.append(
            _color_mesh(
                trimesh.creation.cylinder(
                    radius=0.012,
                    segment=np.stack(
                        [
                            spoke_inner_radius * direction,
                            scene.grasp_radius * direction,
                        ]
                    ),
                ),
                (191, 26, 20, 255),
            )
        )
    rim = trimesh.creation.torus(
        major_radius=scene.grasp_radius,
        minor_radius=scene.rim_tube_radius,
        major_sections=48,
        minor_sections=12,
    )
    rim.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, (0.0, 1.0, 0.0))
    )
    wheel_parts.append(_color_mesh(rim, (191, 26, 20, 255)))
    server.scene.add_mesh_trimesh(
        f"{root}/valve/wheel/geometry", trimesh.util.concatenate(wheel_parts)
    )
    return valve_root, wheel_root


class CabinetVisualizer:
    """Animate Piper-L with a cabinet or procedural valve and an SE(3) base."""

    def __init__(
        self,
        robot: cca.RobotDescription,
        robot_urdf: Path,
        *,
        initial_base_pose: np.ndarray | None = None,
        config: VisualizationConfig = VisualizationConfig(),
    ) -> None:
        self.robot = robot
        self.robot_urdf = robot_urdf
        if initial_base_pose is None:
            initial_base_pose = np.eye(4)
        self.initial_base_pose = np.asarray(initial_base_pose, dtype=float)
        if self.initial_base_pose.shape != (4, 4):
            raise ValueError("initial_base_pose must be a 4 x 4 transform")
        self.config = config

    def _world_tool_path(self, result: cca.PlannerResult, offset: np.ndarray):
        trajectory = np.asarray(result.joint_trajectory)
        base_poses = base_pose_trajectory(result)
        return np.asarray(
            [
                (
                    base_pose
                    @ np.asarray(
                        cca.fkin_space(
                            self.robot.M,
                            self.robot.slist,
                            joints[:ARM_DOF],
                        )
                    )
                )[:3, 3]
                + offset
                for joints, base_pose in zip(trajectory, base_poses)
            ]
        )

    def show(
        self,
        cases: list[TrajectoryCase],
        task_scene: CabinetTaskScene | ValveTaskScene,
        *,
        duration_s: float | None = None,
        markdown: str,
    ) -> None:
        """Start Viser and animate meshes until Ctrl+C or ``duration_s``."""
        import viser
        from viser.extras import ViserUrdf

        if not cases:
            raise ValueError("at least one trajectory case is required")

        trajectories = [np.asarray(case.result.joint_trajectory) for case in cases]
        base_poses = [base_pose_trajectory(case.result) for case in cases]
        for trajectory, poses in zip(trajectories, base_poses):
            if trajectory.ndim != 2 or poses.shape != (trajectory.shape[0], 4, 4):
                raise ValueError("planner arm/base trajectories are not point-aligned")

        server = viser.ViserServer(port=self.config.port)
        print(f"\nViser: http://localhost:{server.get_port()}")
        server.scene.add_grid(
            "/ground", width=4.0, height=3.0, cell_size=0.1, cell_thickness=0.5
        )
        server.scene.add_frame("/world", axes_length=0.25, axes_radius=0.006)
        server.gui.add_markdown(markdown)

        maximum_frames = max(trajectory.shape[0] for trajectory in trajectories)
        frame_slider = server.gui.add_slider(
            "Trajectory frame",
            min=0,
            max=maximum_frames - 1,
            step=1,
            initial_value=0,
        )
        play = server.gui.add_checkbox("Play", initial_value=True)

        case_handles = []
        for case_index, (case, trajectory, poses) in enumerate(
            zip(cases, trajectories, base_poses)
        ):
            offset = np.array(
                [
                    (case_index - (len(cases) - 1) / 2.0)
                    * self.config.case_spacing_m,
                    0.0,
                    0.0,
                ]
            )
            root = f"/cases/{case_index}"
            server.scene.add_label(
                f"{root}/label",
                case.name,
                position=offset + np.array([0.0, 0.0, 1.35]),
                anchor="bottom-center",
            )

            robot_root = server.scene.add_frame(
                f"{root}/robot", axes_length=0.14, axes_radius=0.005
            )
            robot_mesh = ViserUrdf(
                server,
                self.robot_urdf,
                root_node_name=f"{root}/robot",
                load_meshes=True,
            )
            server.scene.add_box(
                f"{root}/robot/base_platform",
                color=case.color,
                dimensions=(0.42, 0.30, 0.08),
                position=(0.0, 0.0, -0.05),
            )

            if isinstance(task_scene, CabinetTaskScene):
                cabinet_yaw_rotation = Rotation.from_euler(
                    "z", task_scene.cabinet_yaw
                ).as_matrix()
                object_root = server.scene.add_frame(
                    f"{root}/cabinet",
                    position=np.asarray(task_scene.cabinet_position) + offset,
                    wxyz=_quaternion(cabinet_yaw_rotation),
                    axes_length=0.0,
                    axes_radius=0.001,
                )
                object_mesh = ViserUrdf(
                    server,
                    task_scene.urdf_path,
                    root_node_name=f"{root}/cabinet",
                    load_meshes=True,
                )
                object_joint_index = object_mesh.get_actuated_joint_names().index(
                    task_scene.joint_name
                )
            else:
                object_root, object_mesh = _add_procedural_valve(
                    server, root, task_scene, offset
                )
                object_joint_index = None

            motion_path = np.asarray(task_scene.motion_path) + offset
            if motion_path.shape[0] > 1:
                server.scene.add_line_segments(
                    f"{root}/task_path",
                    points=np.stack([motion_path[:-1], motion_path[1:]], axis=1),
                    colors=(255, 80, 80),
                    thickness=0.01,
                )
            handle_axis = np.asarray(task_scene.handle_axis, dtype=float)
            handle_axis /= np.linalg.norm(handle_axis)
            handle_position = np.asarray(task_scene.handle_position) + offset
            half_handle = 0.065 * handle_axis
            server.scene.add_line_segments(
                f"{root}/handle_axis",
                points=np.stack(
                    [handle_position - half_handle, handle_position + half_handle]
                )[None, ...],
                colors=(255, 220, 80),
                thickness=0.018,
            )

            tool_path = self._world_tool_path(case.result, offset)
            world_base_poses = poses @ self.initial_base_pose
            base_path = world_base_poses[:, :3, 3] + offset
            if tool_path.shape[0] > 1:
                server.scene.add_line_segments(
                    f"{root}/tool_path",
                    points=np.stack([tool_path[:-1], tool_path[1:]], axis=1),
                    colors=case.color,
                    thickness=0.014,
                )
                server.scene.add_line_segments(
                    f"{root}/base_path",
                    points=np.stack([base_path[:-1], base_path[1:]], axis=1),
                    colors=(40, 220, 255),
                    thickness=0.018,
                )

            case_handles.append(
                (
                    offset,
                    robot_root,
                    robot_mesh,
                    object_root,
                    object_mesh,
                    object_joint_index,
                )
            )

        def update_scene(global_frame: int) -> None:
            for case, trajectory, poses, handles in zip(
                cases, trajectories, base_poses, case_handles
            ):
                # All plans share the same trajectory density. A partial plan
                # therefore stops at its last valid sample while the assisted
                # plan continues, making the capability boundary explicit.
                local_frame = min(global_frame, trajectory.shape[0] - 1)
                (
                    offset,
                    robot_root,
                    robot_mesh,
                    _object_root,
                    object_mesh,
                    object_joint_index,
                ) = handles
                base_pose = poses[local_frame] @ self.initial_base_pose
                robot_root.position = base_pose[:3, 3] + offset
                robot_root.wxyz = _quaternion(base_pose[:3, :3])
                robot_mesh.update_cfg(trajectory[local_frame, :ARM_DOF])

                affordance_goal = float(
                    case.result.task_description.goal.affordance
                )
                progress = 0.0
                if abs(affordance_goal) > 0.0:
                    progress = float(
                        np.clip(
                            abs(trajectory[local_frame, -1] / affordance_goal),
                            0.0,
                            1.0,
                        )
                    )
                object_angle = progress * task_scene.joint_goal
                if isinstance(task_scene, CabinetTaskScene):
                    object_configuration = np.zeros(
                        len(object_mesh.get_actuated_joint_names())
                    )
                    object_configuration[object_joint_index] = object_angle
                    object_mesh.update_cfg(object_configuration)
                else:
                    axis = np.asarray(task_scene.rotation_axis, dtype=float)
                    axis /= np.linalg.norm(axis)
                    object_mesh.wxyz = _quaternion(
                        Rotation.from_rotvec(object_angle * axis).as_matrix()
                    )

        @frame_slider.on_update
        def _(_) -> None:
            update_scene(int(frame_slider.value))

        update_scene(0)
        print("  Piper-L and articulated task geometry loaded; Ctrl+C to stop.")
        start = time.monotonic()
        frame = 0
        try:
            while duration_s is None or time.monotonic() - start < duration_s:
                if play.value:
                    frame = (frame + 1) % maximum_frames
                    frame_slider.value = frame
                    update_scene(frame)
                time.sleep(self.config.animation_step_s)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            server.stop()
