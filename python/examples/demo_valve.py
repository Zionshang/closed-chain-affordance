"""Turn the rl_art_mj six-spoke valve with floating-base Piper-L RM-CCA."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from cabinet_demo_common import (
    ARM_DOF,
    FloatingBaseConfig,
    URDF,
    cca,
    load_piper,
    minimum_arm_limit_clearance,
    parse_args,
    planner_config,
    robot_at_grasp,
    solve_grasp,
    validate_plan,
    world_tool_positions,
)


# Definitions copied from rl_art_mj/tasks/manipulation/valve/assets.py and
# valve/mdp/cca_trajectory.py.
VALVE_CENTER = np.asarray((0.45, 0.0, 0.24))
VALVE_AXIS = np.asarray((1.0, 0.0, 0.0))
VALVE_GRASP_DIRECTION = np.asarray((0.0, 0.0, 1.0))
VALVE_GRASP_RADIUS = 0.1425
VALVE_RIM_TUBE_RADIUS = 0.0175
VALVE_TARGET_ANGLE = math.pi
VALVE_TRAJECTORY_POINTS = 16
GRASP_ROLL = -math.pi / 2.0
VALVE_HANDLE_AXIS = np.asarray((0.0, 1.0, 0.0))

# Configure the floating base here, not through the CLI. Dictionary keys are
# enabled relative DOFs; omitted keys ("rx" here) are fixed exactly at zero.
FLOATING_BASE = FloatingBaseConfig(
    initial_position=(0.0, 0.0, 0.2),
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
    center_position: np.ndarray
    handle_position: np.ndarray
    target_angle: float
    enable_joint_limits: bool
    enable_nullspace_planning: bool


def rotate_about_axis(
    point: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return center + Rotation.from_rotvec(angle * axis).apply(point - center)


def valve_task(
    base_config: FloatingBaseConfig = FLOATING_BASE,
) -> cca.TaskDescription:
    task = cca.TaskDescription()
    task.affordance_info.type = cca.ScrewType.ROTATION
    task.affordance_info.axis = VALVE_AXIS
    task.affordance_info.location = VALVE_CENTER
    task.goal.affordance = VALVE_TARGET_ANGLE
    task.trajectory_density = VALVE_TRAJECTORY_POINTS
    task.vir_screw_order = cca.VirtualScrewOrder.NONE
    task.reserve_mobility = base_config.make_description(0.10, 0.20)
    return task


def run_demo(
    *,
    enable_joint_limits: bool = True,
    enable_nullspace_planning: bool = True,
    base_config: FloatingBaseConfig = FLOATING_BASE,
) -> DemoResult:
    seed_robot, robot_config = load_piper()
    handle_position = VALVE_CENTER + VALVE_GRASP_RADIUS * VALVE_GRASP_DIRECTION
    reserve_enabled = enable_joint_limits or enable_nullspace_planning
    initial_base_pose = base_config.initial_pose() if reserve_enabled else np.eye(4)
    grasp_joints = solve_grasp(
        seed_robot, handle_position, GRASP_ROLL, initial_base_pose
    )
    robot = robot_at_grasp(robot_config, grasp_joints, initial_base_pose)
    config = planner_config(
        enable_joint_limits=enable_joint_limits,
        enable_nullspace_planning=enable_nullspace_planning,
    )
    task = valve_task(base_config)
    plan = cca.PlannerInterface(config).generate_joint_trajectory(robot, task)
    validate_plan(
        plan,
        robot,
        base_config,
        maximum_points=VALVE_TRAJECTORY_POINTS,
        reserve_enabled=reserve_enabled,
        enforce_arm_limits=enable_joint_limits,
    )

    expected_final = rotate_about_axis(
        handle_position, VALVE_CENTER, VALVE_AXIS, VALVE_TARGET_ANGLE
    )
    tool_positions = world_tool_positions(robot, plan)
    assert np.linalg.norm(tool_positions[0] - handle_position) < 2e-4
    if plan.trajectory_description == cca.TrajectoryDescription.FULL:
        assert np.linalg.norm(tool_positions[-1] - expected_final) < 5e-4
    return DemoResult(
        robot=robot,
        plan=plan,
        grasp_joints=grasp_joints,
        initial_base_pose=initial_base_pose,
        base_config=base_config,
        center_position=VALVE_CENTER,
        handle_position=handle_position,
        target_angle=VALVE_TARGET_ANGLE,
        enable_joint_limits=enable_joint_limits,
        enable_nullspace_planning=enable_nullspace_planning,
    )


def print_diagnostics(result: DemoResult) -> None:
    trajectory = np.asarray(result.plan.joint_trajectory)
    reserve = np.asarray(result.plan.reserve_trajectory)
    expected_final = rotate_about_axis(
        result.handle_position,
        result.center_position,
        VALVE_AXIS,
        result.target_angle,
    )
    endpoint = world_tool_positions(result.robot, result.plan)[-1]
    status = (
        "FULL"
        if result.plan.trajectory_description == cca.TrajectoryDescription.FULL
        else "PARTIAL"
    )
    translation = np.linalg.norm(reserve[-1, :3]) if reserve.size else 0.0
    rotation = np.linalg.norm(reserve[-1, 3:]) if reserve.size else 0.0
    clearance = minimum_arm_limit_clearance(result.robot, result.plan)
    print(f"valve turn: {math.degrees(result.target_angle):.3f} deg; status: {status}")
    print(
        f"joint limits={'on' if result.enable_joint_limits else 'off'}; "
        f"null-space={'on' if result.enable_nullspace_planning else 'off'}; "
        f"base DOFs={result.base_config.enabled_dofs}"
    )
    print(
        f"base delta: {translation:.3e} m / {rotation:.3e} rad; "
        f"minimum URDF-limit clearance: {clearance:.3e} rad; "
        f"max arm delta: "
        f"{np.max(np.abs(trajectory[:, :ARM_DOF] - result.grasp_joints)):.3e} rad; "
        f"endpoint error: {np.linalg.norm(endpoint - expected_final):.3e} m"
    )


def main() -> None:
    args = parse_args(__doc__, default_port=8082)
    result = run_demo(
        enable_joint_limits=args.joint_limits,
        enable_nullspace_planning=args.nullspace_planning,
    )
    print_diagnostics(result)

    from visualizer import (
        CabinetVisualizer,
        TrajectoryCase,
        ValveTaskScene,
        VisualizationConfig,
    )

    fractions = np.linspace(0.0, 1.0, 65)
    motion_path = np.asarray(
        [
            rotate_about_axis(
                result.handle_position,
                result.center_position,
                VALVE_AXIS,
                result.target_angle * fraction,
            )
            for fraction in fractions
        ]
    )
    CabinetVisualizer(
        result.robot,
        URDF,
        initial_base_pose=result.initial_base_pose,
        config=VisualizationConfig(port=args.port),
    ).show(
        [TrajectoryCase("180° valve", result.plan, (30, 210, 235))],
        ValveTaskScene(
            center_position=result.center_position,
            rotation_axis=VALVE_AXIS,
            joint_goal=result.target_angle,
            handle_position=result.handle_position,
            handle_axis=VALVE_HANDLE_AXIS,
            motion_path=motion_path,
            grasp_radius=VALVE_GRASP_RADIUS,
            rim_tube_radius=VALVE_RIM_TUBE_RADIUS,
        ),
        duration_s=args.duration,
        markdown=(
            "This is the procedural six-spoke valve from **rl_art_mj**, with a "
            "fixed **180°** target. The Piper-L arm uses its URDF limits. "
            f"Floating-base DOFs **{result.base_config.enabled_dofs}** and bounds "
            "are configured in this demo's `FLOATING_BASE` dictionary."
        ),
    )


if __name__ == "__main__":
    main()
