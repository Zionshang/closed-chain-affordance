"""Pull cabinet drawer 1 with Piper-L mounted on a floating base."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cabinet_demo_common import (
    ARM_DOF,
    CABINET_URDF,
    FloatingBaseConfig,
    TRAJECTORY_POINTS,
    URDF,
    cabinet_point,
    cca,
    load_piper,
    minimum_arm_limit_clearance,
    parse_args,
    planning_mode_name,
    planner_config,
    robot_at_grasp,
    solve_grasp,
    validate_plan,
    world_tool_positions,
)


CABINET_POSITION = (0.60, 0.0, 0.20)
CABINET_YAW = math.pi
DRAWER_1_HANDLE_LOCAL = (0.161706, 0.149577, 0.198016)
DRAWER_HANDLE_AXIS = (0.0, -1.0, 0.0)
DRAWER_PULL_AXIS = (-0.999816, -0.018917, -0.003302)
DRAWER_PULL_DISTANCE = 0.15
GRASP_ROLL = -math.pi / 2
SOFT_LIMIT_RATIO = 0.8

# Configure the floating base here, not through the CLI. Dictionary keys are
# enabled relative DOFs; omitted keys ("rx" here) are fixed exactly at zero.
FLOATING_BASE = FloatingBaseConfig(
    initial_position=(0.0, -0.1, 0.0),
    initial_rpy=(0.0, 0.0, 0.0),
    dof_limits={
        "px": (-0.60, 0.60),
        "py": (-0.60, 0.60),
        "pz": (-0.50, 0.50),
        "ry": (-math.pi, math.pi),
        "rz": (-math.pi, math.pi),
    },
)


@dataclass(frozen=True)
class DemoResult:
    robot: cca.RobotDescription
    plan: cca.PlannerResult
    grasp_joints: np.ndarray
    initial_base_pose: np.ndarray
    base_config: FloatingBaseConfig
    handle_position: np.ndarray
    pull_axis: np.ndarray
    pull_distance: float
    enable_joint_limits: bool
    enable_nullspace_planning: bool
    enable_capability_aware_planning: bool
    soft_limit_ratio: float


def drawer_task(
    handle_position: np.ndarray,
    pull_axis: np.ndarray,
    pull_distance: float,
    base_config: FloatingBaseConfig = FLOATING_BASE,
) -> cca.TaskDescription:
    task = cca.TaskDescription()
    task.affordance_info.type = cca.ScrewType.TRANSLATION
    task.affordance_info.axis = pull_axis
    task.affordance_info.location = handle_position
    task.goal.affordance = pull_distance
    task.trajectory_density = TRAJECTORY_POINTS
    task.vir_screw_order = cca.VirtualScrewOrder.NONE
    task.reserve_mobility = base_config.make_description(0.05, 0.10)
    return task


def run_demo(
    *,
    enable_joint_limits: bool = True,
    enable_nullspace_planning: bool = True,
    enable_capability_aware_planning: bool = True,
    base_config: FloatingBaseConfig = FLOATING_BASE,
) -> DemoResult:
    seed_robot, robot_config = load_piper()
    handle_position = cabinet_point(CABINET_POSITION, DRAWER_1_HANDLE_LOCAL)
    pull_axis = np.asarray(DRAWER_PULL_AXIS, dtype=float)
    pull_axis /= np.linalg.norm(pull_axis)
    reserve_enabled = (
        enable_joint_limits
        or enable_nullspace_planning
        or enable_capability_aware_planning
    )
    effective_soft_limit_ratio = SOFT_LIMIT_RATIO if enable_joint_limits else 1.0
    initial_base_pose = base_config.initial_pose() if reserve_enabled else np.eye(4)
    grasp_joints = solve_grasp(
        seed_robot, handle_position, GRASP_ROLL, initial_base_pose
    )
    robot = robot_at_grasp(robot_config, grasp_joints, initial_base_pose)
    config = planner_config(
        enable_joint_limits=enable_joint_limits,
        enable_nullspace_planning=enable_nullspace_planning,
        enable_capability_aware_planning=enable_capability_aware_planning,
        soft_limit_ratio=effective_soft_limit_ratio,
    )
    task = drawer_task(
        handle_position,
        pull_axis,
        DRAWER_PULL_DISTANCE,
        base_config,
    )
    plan = cca.PlannerInterface(config).generate_joint_trajectory(robot, task)
    validate_plan(
        plan,
        robot,
        base_config,
        maximum_points=TRAJECTORY_POINTS,
        reserve_enabled=reserve_enabled,
        enforce_arm_limits=enable_joint_limits,
        soft_limit_ratio=effective_soft_limit_ratio,
    )

    expected_final = handle_position + DRAWER_PULL_DISTANCE * pull_axis
    tool_positions = world_tool_positions(robot, plan)
    assert np.linalg.norm(tool_positions[0] - handle_position) < 2e-4
    if plan.trajectory_description == cca.TrajectoryDescription.FULL:
        assert np.linalg.norm(tool_positions[-1] - expected_final) < 3e-4
    return DemoResult(
        robot=robot,
        plan=plan,
        grasp_joints=grasp_joints,
        initial_base_pose=initial_base_pose,
        base_config=base_config,
        handle_position=handle_position,
        pull_axis=pull_axis,
        pull_distance=DRAWER_PULL_DISTANCE,
        enable_joint_limits=enable_joint_limits,
        enable_nullspace_planning=enable_nullspace_planning,
        enable_capability_aware_planning=enable_capability_aware_planning,
        soft_limit_ratio=effective_soft_limit_ratio,
    )


def print_diagnostics(result: DemoResult) -> None:
    trajectory = np.asarray(result.plan.joint_trajectory)
    reserve = np.asarray(result.plan.reserve_trajectory)
    expected_final = result.handle_position + result.pull_distance * result.pull_axis
    endpoint = world_tool_positions(result.robot, result.plan)[-1]
    status = (
        "FULL"
        if result.plan.trajectory_description == cca.TrajectoryDescription.FULL
        else "PARTIAL"
    )
    translation = np.linalg.norm(reserve[-1, :3]) if reserve.size else 0.0
    rotation = np.linalg.norm(reserve[-1, 3:]) if reserve.size else 0.0
    clearance = minimum_arm_limit_clearance(
        result.robot, result.plan, result.soft_limit_ratio
    )
    print(f"drawer pull: {result.pull_distance:.6f} m; status: {status}")
    print(
        f"joint limits={'on' if result.enable_joint_limits else 'off'}; "
        f"null-space={'on' if result.enable_nullspace_planning else 'off'}; "
        f"capability-aware={'on' if result.enable_capability_aware_planning else 'off'}; "
        f"soft ratio={result.soft_limit_ratio:.2f}; "
        f"base DOFs={result.base_config.enabled_dofs}"
    )
    print(
        f"base delta: {translation:.3e} m / {rotation:.3e} rad; "
        f"minimum nominal soft-limit clearance: {clearance:.3e} rad; "
        f"max arm delta: "
        f"{np.max(np.abs(trajectory[:, :ARM_DOF] - result.grasp_joints)):.3e} rad; "
        f"endpoint error: {np.linalg.norm(endpoint - expected_final):.3e} m"
    )


def main() -> None:
    args = parse_args(__doc__, default_port=8081)
    result = run_demo(
        enable_joint_limits=args.joint_limits,
        enable_nullspace_planning=args.nullspace_planning,
        enable_capability_aware_planning=args.capability_aware_planning,
    )
    print_diagnostics(result)
    mode_name = planning_mode_name(
        result.enable_joint_limits,
        result.enable_nullspace_planning,
        result.enable_capability_aware_planning,
    )

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
        initial_base_pose=result.initial_base_pose,
        config=VisualizationConfig(port=args.port),
    ).show(
        [TrajectoryCase("0.15 m drawer", result.plan, (30, 210, 235))],
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
            f"Planning mode: **{mode_name}**; "
            f"soft-limit ratio: **{result.soft_limit_ratio:.2f}**. "
            "The Piper-L arm uses the joint limits copied from its URDF when "
            "joint-limit handling is enabled. Floating-base pose, enabled DOFs "
            f"**{result.base_config.enabled_dofs}**, and bounds are configured "
            "in this demo's `FLOATING_BASE` dictionary."
        ),
    )


if __name__ == "__main__":
    main()
