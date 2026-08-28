"""Open the cabinet door with Piper-L mounted on a floating base."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cabinet_demo_common import (
    ARM_DOF,
    ARM_SAFETY_HALF_RANGE,
    CABINET_URDF,
    TRAJECTORY_POINTS,
    URDF,
    assert_arm_only_partial_plan,
    assert_base_assisted_full_plan,
    cabinet_point,
    cca,
    freeze_base_mobility,
    load_piper,
    parse_args,
    planner_config,
    rotate_about_z,
    robot_at_grasp,
    solve_grasp,
    world_tool_positions,
)


CABINET_POSITION = (0.70, 0.0, 0.20)
CABINET_YAW = math.pi
DOOR_HINGE_LOCAL = (0.119816, -0.275279, 0.104384)
DOOR_HANDLE_LOCAL = (0.165816, -0.025555, 0.165822)
DOOR_HANDLE_AXIS = (0.0, 0.0, 1.0)
DOOR_HINGE_AXIS = (0.0, 0.0, 1.0)
DOOR_OPEN_ANGLE = -math.pi / 2
GRASP_ROLL = 0.0


@dataclass(frozen=True)
class DemoResult:
    robot: cca.RobotDescription
    arm_only_plan: cca.PlannerResult
    assisted_plan: cca.PlannerResult
    grasp_joints: np.ndarray
    hinge_position: np.ndarray
    handle_position: np.ndarray
    open_angle: float


def door_task(hinge_position: np.ndarray, open_angle: float) -> cca.TaskDescription:
    task = cca.TaskDescription()
    task.affordance_info.type = cca.ScrewType.ROTATION
    task.affordance_info.axis = np.asarray(DOOR_HINGE_AXIS)
    task.affordance_info.location = hinge_position
    task.goal.affordance = open_angle
    task.trajectory_density = TRAJECTORY_POINTS
    task.vir_screw_order = cca.VirtualScrewOrder.Z
    task.reserve_mobility = cca.make_floating_base_reserve_description(
        np.eye(4), translation_max_step=0.10, rotation_max_step=0.20
    )
    return task


def run_demo() -> DemoResult:
    seed_robot, robot_config = load_piper()
    hinge_position = cabinet_point(CABINET_POSITION, DOOR_HINGE_LOCAL)
    handle_position = cabinet_point(CABINET_POSITION, DOOR_HANDLE_LOCAL)
    open_angle = DOOR_OPEN_ANGLE
    grasp_joints = solve_grasp(seed_robot, handle_position, GRASP_ROLL)
    robot = robot_at_grasp(
        robot_config, grasp_joints, ARM_SAFETY_HALF_RANGE
    )
    arm_only_task = door_task(hinge_position, open_angle)
    freeze_base_mobility(arm_only_task)
    arm_only_plan = cca.PlannerInterface(
        planner_config()
    ).generate_joint_trajectory(robot, arm_only_task)
    arm_only_trajectory = assert_arm_only_partial_plan(
        arm_only_plan, grasp_joints
    )

    assisted_task = door_task(hinge_position, open_angle)
    assisted_plan = cca.PlannerInterface(
        planner_config()
    ).generate_joint_trajectory(robot, assisted_task)
    assisted_trajectory = assert_base_assisted_full_plan(
        assisted_plan, grasp_joints
    )
    expected_final = rotate_about_z(
        handle_position, hinge_position, open_angle
    )
    arm_only_tool_positions = world_tool_positions(robot, arm_only_plan)
    assisted_tool_positions = world_tool_positions(robot, assisted_plan)
    first_base_index = int(
        np.flatnonzero(np.asarray(assisted_plan.reserve_active))[0]
    )

    # Six physical joints + one free grasp-orientation coordinate + affordance.
    assert arm_only_trajectory.shape[1] == ARM_DOF + 2
    assert assisted_trajectory.shape[1] == ARM_DOF + 2
    assert arm_only_trajectory.shape[0] == first_base_index
    assert np.allclose(
        arm_only_trajectory,
        assisted_trajectory[:first_base_index],
        atol=1e-10,
    )
    assert np.linalg.norm(arm_only_tool_positions[0] - handle_position) < 2e-4
    assert np.linalg.norm(arm_only_tool_positions[-1] - expected_final) > 0.05
    assert np.linalg.norm(assisted_tool_positions[-1] - expected_final) < 5e-4

    return DemoResult(
        robot=robot,
        arm_only_plan=arm_only_plan,
        assisted_plan=assisted_plan,
        grasp_joints=grasp_joints,
        hinge_position=hinge_position,
        handle_position=handle_position,
        open_angle=open_angle,
    )


def print_diagnostics(result: DemoResult) -> None:
    arm_only = np.asarray(result.arm_only_plan.joint_trajectory)
    assisted = np.asarray(result.assisted_plan.joint_trajectory)
    base = np.asarray(result.assisted_plan.reserve_trajectory)
    expected_final = rotate_about_z(
        result.handle_position, result.hinge_position, result.open_angle
    )
    endpoint = world_tool_positions(result.robot, result.assisted_plan)[-1]
    first_base_index = int(
        np.flatnonzero(np.asarray(result.assisted_plan.reserve_active))[0]
    )
    print(f"door opening: {math.degrees(result.open_angle):.3f} deg")
    print(
        f"arm only: PARTIAL {arm_only.shape[0]}/{TRAJECTORY_POINTS}; "
        f"base assistance starts at point {first_base_index + 1}; "
        f"assisted base norm: {np.linalg.norm(base[-1]):.3e}; "
        f"max arm delta: {np.max(np.abs(assisted[:, :ARM_DOF] - result.grasp_joints)):.3e} rad; "
        f"endpoint error: {np.linalg.norm(endpoint - expected_final):.3e} m"
    )


def main() -> None:
    args = parse_args(__doc__, default_port=8080)
    result = run_demo()
    print_diagnostics(result)

    if args.visualize:
        from visualizer import (
            CabinetTaskScene,
            CabinetVisualizer,
            TrajectoryCase,
            VisualizationConfig,
        )

        fractions = np.linspace(0.0, 1.0, 40)
        motion_path = np.asarray(
            [
                rotate_about_z(
                    result.handle_position,
                    result.hinge_position,
                    result.open_angle * fraction,
                )
                for fraction in fractions
            ]
        )
        CabinetVisualizer(
            result.robot,
            URDF,
            config=VisualizationConfig(port=args.port),
        ).show(
            [
                TrajectoryCase(
                    "RM-CCA: arm first, then base assistance (90° door)",
                    result.assisted_plan,
                    (30, 210, 235),
                )
            ],
            CabinetTaskScene(
                urdf_path=CABINET_URDF,
                cabinet_position=np.asarray(CABINET_POSITION),
                cabinet_yaw=CABINET_YAW,
                joint_name="door_joint_1",
                joint_goal=result.open_angle,
                handle_position=result.handle_position,
                handle_axis=np.asarray(DOOR_HANDLE_AXIS),
                motion_path=motion_path,
            ),
            duration_s=args.duration,
            markdown=(
                "The real **Piper-L** and **cabinet** meshes are loaded from the "
                "copied assets. This is one environment with a fixed **90°** door "
                "goal and a ±0.15 rad arm operating envelope. The arm moves first; "
                "only after it reaches the boundary does the floating base supply "
                "the remaining door motion. A frozen-base PARTIAL plan is checked "
                "in the example assertions but is not rendered as a second scene."
            ),
        )


if __name__ == "__main__":
    main()
