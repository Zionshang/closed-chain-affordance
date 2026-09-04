"""Mesh-based Viser animation for the cabinet drawer and door examples."""

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


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to Viser's scalar-first quaternion."""
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


class CabinetVisualizer:
    """Animate real Piper-L and cabinet meshes with an SE(3) base trajectory."""

    def __init__(
        self,
        robot: cca.RobotDescription,
        robot_urdf: Path,
        *,
        config: VisualizationConfig = VisualizationConfig(),
    ) -> None:
        self.robot = robot
        self.robot_urdf = robot_urdf
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
        task_scene: CabinetTaskScene,
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

        cabinet_yaw_rotation = Rotation.from_euler(
            "z", task_scene.cabinet_yaw
        ).as_matrix()
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

            cabinet_root = server.scene.add_frame(
                f"{root}/cabinet",
                position=np.asarray(task_scene.cabinet_position) + offset,
                wxyz=_quaternion(cabinet_yaw_rotation),
                axes_length=0.0,
                axes_radius=0.001,
            )
            cabinet_mesh = ViserUrdf(
                server,
                task_scene.urdf_path,
                root_node_name=f"{root}/cabinet",
                load_meshes=True,
            )
            cabinet_joint_names = cabinet_mesh.get_actuated_joint_names()
            cabinet_joint_index = cabinet_joint_names.index(task_scene.joint_name)

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
            base_path = poses[:, :3, 3] + offset
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
                    cabinet_root,
                    cabinet_mesh,
                    cabinet_joint_index,
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
                    _cabinet_root,
                    cabinet_mesh,
                    cabinet_joint_index,
                ) = handles
                base_pose = poses[local_frame]
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
                cabinet_configuration = np.zeros(
                    len(cabinet_mesh.get_actuated_joint_names())
                )
                cabinet_configuration[cabinet_joint_index] = (
                    progress * task_scene.joint_goal
                )
                cabinet_mesh.update_cfg(cabinet_configuration)

        @frame_slider.on_update
        def _(_) -> None:
            update_scene(int(frame_slider.value))

        update_scene(0)
        print("  Real Piper-L and cabinet meshes loaded; Ctrl+C to stop.")
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
