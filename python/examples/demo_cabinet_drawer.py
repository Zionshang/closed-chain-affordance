"""Pull cabinet drawer 1 with Piper-L mounted on a floating base."""

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
    assert_configured_full_plan,
    cabinet_point,
    cca,
    freeze_base_mobility,
    load_piper,
    parse_args,
    planner_config,
    robot_at_grasp,
    solve_grasp,
    world_tool_positions,
)


CABINET_POSITION = (0.70, 0.0, 0.20)
CABINET_YAW = math.pi
DRAWER_1_HANDLE_LOCAL = (0.161706, 0.149577, 0.198016)
DRAWER_HANDLE_AXIS = (0.0, -1.0, 0.0)
DRAWER_PULL_AXIS = (-0.999816, -0.018917, -0.003302)
DRAWER_PULL_DISTANCE = 0.15
GRASP_ROLL = -math.pi / 2


@dataclass(frozen=True)
class DemoResult:
    robot: cca.RobotDescription
    arm_only_plan: cca.PlannerResult | None
    assisted_plan: cca.PlannerResult
    grasp_joints: np.ndarray
    handle_position: np.ndarray
    pull_axis: np.ndarray
    pull_distance: float
    enable_joint_limits: bool
    enable_nullspace_planning: bool


def drawer_task(
    handle_position: np.ndarray,
    pull_axis: np.ndarray,
    pull_distance: float,
) -> cca.TaskDescription:
    task = cca.TaskDescription()
    task.affordance_info.type = cca.ScrewType.TRANSLATION
    task.affordance_info.axis = pull_axis
    task.affordance_info.location = handle_position
    task.goal.affordance = pull_distance
    task.trajectory_density = TRAJECTORY_POINTS
    task.vir_screw_order = cca.VirtualScrewOrder.Y
    task.reserve_mobility = cca.make_floating_base_reserve_description(
        np.eye(4), translation_max_step=0.05, rotation_max_step=0.10
    )
    return task


def run_demo(
    *,
    enable_joint_limits: bool = True,
    enable_nullspace_planning: bool = True,
) -> DemoResult:
    seed_robot, robot_config = load_piper()
    handle_position = cabinet_point(CABINET_POSITION, DRAWER_1_HANDLE_LOCAL)
    pull_axis = np.asarray(DRAWER_PULL_AXIS, dtype=float)
    pull_axis /= np.linalg.norm(pull_axis)
    pull_distance = DRAWER_PULL_DISTANCE
    grasp_joints = solve_grasp(seed_robot, handle_position, GRASP_ROLL)
    robot = robot_at_grasp(
        robot_config, grasp_joints, ARM_SAFETY_HALF_RANGE
    )
    config = planner_config(
        enable_joint_limits=enable_joint_limits,
        enable_nullspace_planning=enable_nullspace_planning,
    )
    arm_only_plan = None
    arm_only_trajectory = None
    if enable_joint_limits and enable_nullspace_planning:
        arm_only_task = drawer_task(handle_position, pull_axis, pull_distance)
        freeze_base_mobility(arm_only_task)
        arm_only_plan = cca.PlannerInterface(config).generate_joint_trajectory(
            robot, arm_only_task
        )
        arm_only_trajectory = assert_arm_only_partial_plan(
            arm_only_plan, grasp_joints
        )

    assisted_task = drawer_task(handle_position, pull_axis, pull_distance)
    assisted_plan = cca.PlannerInterface(config).generate_joint_trajectory(
        robot, assisted_task
    )
    if enable_joint_limits and enable_nullspace_planning:
        assisted_trajectory = assert_base_assisted_full_plan(
            assisted_plan, grasp_joints
        )
    else:
        assisted_trajectory = assert_configured_full_plan(
            assisted_plan,
            grasp_joints,
            enable_joint_limits=enable_joint_limits,
            enable_nullspace_planning=enable_nullspace_planning,
        )

    expected_final = handle_position + pull_distance * pull_axis
    assisted_tool_positions = world_tool_positions(robot, assisted_plan)

    # Six physical joints + one free grasp-orientation coordinate + affordance.
    assert assisted_trajectory.shape[1] == ARM_DOF + 2
    if arm_only_plan is not None and arm_only_trajectory is not None:
        arm_only_tool_positions = world_tool_positions(robot, arm_only_plan)
        first_base_index = int(
            np.flatnonzero(np.asarray(assisted_plan.reserve_active))[0]
        )
        assert arm_only_trajectory.shape[1] == ARM_DOF + 2
        assert arm_only_trajectory.shape[0] == first_base_index
        assert np.allclose(
            arm_only_trajectory,
            assisted_trajectory[:first_base_index],
            atol=1e-10,
        )
        assert np.linalg.norm(arm_only_tool_positions[0] - handle_position) < 2e-4
        assert np.linalg.norm(arm_only_tool_positions[-1] - expected_final) > 0.02
    assert np.linalg.norm(assisted_tool_positions[-1] - expected_final) < 3e-4

    return DemoResult(
        robot=robot,
        arm_only_plan=arm_only_plan,
        assisted_plan=assisted_plan,
        grasp_joints=grasp_joints,
        handle_position=handle_position,
        pull_axis=pull_axis,
        pull_distance=pull_distance,
        enable_joint_limits=enable_joint_limits,
        enable_nullspace_planning=enable_nullspace_planning,
    )


def print_diagnostics(result: DemoResult) -> None:
    assisted = np.asarray(result.assisted_plan.joint_trajectory)
    base = np.asarray(result.assisted_plan.reserve_trajectory)
    expected_final = result.handle_position + result.pull_distance * result.pull_axis
    endpoint = world_tool_positions(result.robot, result.assisted_plan)[-1]
    print(f"drawer pull: {result.pull_distance:.6f} m")
    if result.arm_only_plan is not None:
        arm_only = np.asarray(result.arm_only_plan.joint_trajectory)
        first_base_index = int(
            np.flatnonzero(np.asarray(result.assisted_plan.reserve_active))[0]
        )
        prefix = (
            f"arm only: PARTIAL {arm_only.shape[0]}/{TRAJECTORY_POINTS}; "
            f"base assistance starts at point {first_base_index + 1}; "
        )
    else:
        prefix = (
            f"joint limits={'on' if result.enable_joint_limits else 'off'}; "
            f"null-space={'on' if result.enable_nullspace_planning else 'off'}; "
        )
    base_norm = np.linalg.norm(base[-1]) if base.size else 0.0
    print(
        prefix
        + f"base norm: {base_norm:.3e}; "
        + f"max arm delta: {np.max(np.abs(assisted[:, :ARM_DOF] - result.grasp_joints)):.3e} rad; "
        + f"endpoint error: {np.linalg.norm(endpoint - expected_final):.3e} m"
    )


def main() -> None:
    args = parse_args(__doc__, default_port=8081)
    result = run_demo(
        enable_joint_limits=args.joint_limits,
        enable_nullspace_planning=args.nullspace_planning,
    )
    print_diagnostics(result)

    from visualizer import (
        CabinetTaskScene,
        CabinetVisualizer,
        TrajectoryCase,
        VisualizationConfig,
    )

    motion_path = np.stack(
        [
            result.handle_position,
            result.handle_position + result.pull_distance * result.pull_axis,
        ]
    )
    CabinetVisualizer(
        result.robot,
        URDF,
        config=VisualizationConfig(port=args.port),
    ).show(
        [
            TrajectoryCase(
                "0.15 m drawer: configured CCA mode",
                result.assisted_plan,
                (30, 210, 235),
            )
        ],
        CabinetTaskScene(
            urdf_path=CABINET_URDF,
            cabinet_position=np.asarray(CABINET_POSITION),
            cabinet_yaw=CABINET_YAW,
            joint_name="drawer_joint_1",
            joint_goal=-result.pull_distance,
            handle_position=result.handle_position,
            handle_axis=np.asarray(DRAWER_HANDLE_AXIS),
            motion_path=motion_path,
        ),
        duration_s=args.duration,
        markdown=(
            "The real **Piper-L** and **cabinet** meshes are loaded from the "
            "copied assets. This is one environment with a fixed **0.15 m** "
            "drawer goal and two independently configurable switches. Joint limits "
            f"are **{'on' if result.enable_joint_limits else 'off'}** and null-space "
            f"planning is **{'on' if result.enable_nullspace_planning else 'off'}**. "
            "With both off, the attached floating base is ignored and this is the "
            "bit-identical legacy fixed-base CCA path."
        ),
    )


if __name__ == "__main__":
    main()
