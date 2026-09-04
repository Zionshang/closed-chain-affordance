"""Use Piper-L to grasp a valve rim and rotate it by half a turn.

Examples:
    python examples/demo_valve.py
    python examples/demo_valve.py --visualize
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cca_planner as cca
import torch

ROOT = Path(__file__).resolve().parents[1]
PIPER_ROOT = ROOT / "assets/robot/piper_l"
URDF = PIPER_ROOT / "urdf/piper_l_fixed_gripper.urdf"
ROBOT_CONFIG = PIPER_ROOT / "urdf/cca_config.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVIRONMENTS = 4096
VISUALIZE_COUNT = 10
TRAJECTORY_POINTS = 16
POSE_IK_ITERATIONS = 100
SEED = 42

VALVE_POSITION = (0.45, 0.0, 0.24)
VALVE_POSITION_RANDOMIZATION = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.05, 0.05),
}
VALVE_AXIS = (1.0, 0.0, 0.0)
VALVE_GRASP_DIRECTION = (0.0, 0.0, 1.0)
VALVE_GRASP_RADIUS = 0.1425
VALVE_TURN_ANGLE = math.pi
GRASP_ROLL = -math.pi / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def print_quantiles(name: str, values: torch.Tensor) -> None:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        print(f"{name}: no finite samples in this group")
        return
    quantiles = torch.quantile(
        values,
        values.new_tensor((0.5, 0.9, 0.99, 1.0)),
    )
    print(
        f"{name} q50/q90/q99/max: "
        + ", ".join(f"{value:.6f}" for value in quantiles.tolist())
    )


def main() -> None:
    args = parse_args()
    robot = cca.load_robot_from_urdf(
        URDF,
        ROBOT_CONFIG,
        joint_states=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        device=DEVICE,
    )
    planner = cca.PlannerInterface(
        cca.PlannerConfig(
            accuracy=0.02,
            ik_max_itr=150,
            secondary_goal_abs_tolerance=1e-4,
            closure_err_threshold_ang=5e-4,
            closure_err_threshold_lin=5e-4,
        ),
        early_stopping=True,
        use_regularized_normal_equations=True,
    )

    random = torch.Generator(device=DEVICE).manual_seed(SEED)
    position_samples = torch.rand(
        (NUM_ENVIRONMENTS, 3), generator=random, device=DEVICE
    )
    position_bounds = torch.tensor(
        [VALVE_POSITION_RANDOMIZATION[axis] for axis in ("x", "y", "z")],
        device=DEVICE,
    )
    lower, upper = position_bounds.unbind(dim=1)
    centres = (
        torch.tensor(VALVE_POSITION, device=DEVICE)
        + lower
        + (upper - lower) * position_samples
    )

    initial_joints = robot.joint_states.expand(NUM_ENVIRONMENTS, -1).clone()
    initial_pose = cca.fkin_space(robot.M, robot.slist, initial_joints)
    grasp_direction = torch.tensor(VALVE_GRASP_DIRECTION, device=DEVICE).expand(
        NUM_ENVIRONMENTS, -1
    )
    grasp_points = centres + VALVE_GRASP_RADIUS * grasp_direction
    grasp_roll = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(GRASP_ROLL), -math.sin(GRASP_ROLL)],
            [0.0, math.sin(GRASP_ROLL), math.cos(GRASP_ROLL)],
        ],
        device=DEVICE,
    )
    grasp_poses = initial_pose.clone()
    grasp_poses[:, :3, :3] = initial_pose[:, :3, :3] @ grasp_roll
    grasp_poses[:, :3, 3] = grasp_points

    axes = torch.tensor(VALVE_AXIS, device=DEVICE).expand(NUM_ENVIRONMENTS, -1)
    screws = cca.get_screw(cca.ScrewType.ROTATION, axes, centres)
    turn_angles = torch.full((NUM_ENVIRONMENTS,), VALVE_TURN_ANGLE, device=DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    start_time = time.perf_counter()
    grasp_joints, ik_success = planner.solve_pose_ik(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_seed=initial_joints,
        target_pose=grasp_poses,
        max_iterations=POSE_IK_ITERATIONS,
        angular_tolerance=6e-4,
        linear_tolerance=6e-4,
    )
    result = planner.generate_joint_trajectory(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_states=grasp_joints,
        affordance_screw=screws,
        goal_affordance=turn_angles,
        trajectory_density=TRAJECTORY_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.NONE,
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    planning_time = time.perf_counter() - start_time

    trajectory = result.joint_trajectory
    valid = result.valid_mask & ik_success[:, None]
    success = ik_success & result.full_success
    print(
        f"valve: {planning_time:.3f}s; "
        f"start IK {int(ik_success.sum())}/{NUM_ENVIRONMENTS}; "
        f"valve CCA {int(success.sum())}/{int(ik_success.sum())}; "
        f"full {int(success.sum())}/{NUM_ENVIRONMENTS}"
    )

    ee_poses = cca.fkin_space(robot.M, robot.slist, trajectory)
    start_error = torch.linalg.vector_norm(ee_poses[:, 0, :3, 3] - grasp_points, dim=-1)
    expected_final = centres - VALVE_GRASP_RADIUS * grasp_direction
    final_error = torch.linalg.vector_norm(
        ee_poses[:, -1, :3, 3] - expected_final, dim=-1
    )
    print_quantiles("successful start rim error [m]", start_error[success])
    print_quantiles("failed start rim error [m]", start_error[~success])
    print_quantiles("successful valve endpoint error [m]", final_error[success])
    print_quantiles("failed valve endpoint error [m]", final_error[~success])

    if args.visualize:
        from visualizer import ViserVisualizer

        score = torch.maximum(start_error, final_error)
        count = min(VISUALIZE_COUNT, NUM_ENVIRONMENTS)
        shown = torch.topk(score, count).indices
        radii = torch.full((count,), VALVE_GRASP_RADIUS, device=DEVICE)
        ViserVisualizer(robot, URDF).show_valve(
            trajectory[shown],
            valid[shown],
            success[shown],
            centres[shown],
            radii,
            turn_angles[shown],
            rotation_axis=VALVE_AXIS,
            grasp_direction=VALVE_GRASP_DIRECTION,
        )


if __name__ == "__main__":
    main()
