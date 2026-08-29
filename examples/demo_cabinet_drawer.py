"""Use Piper-L to grasp a horizontal handle and pull cabinet drawer 1 open.

Examples:
    python examples/demo_cabinet_drawer.py
    python examples/demo_cabinet_drawer.py --visualize
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

import cca_planner as cca


ROOT = Path(__file__).resolve().parents[1]
PIPER_ROOT = ROOT / "assets/robot/piper_l"
URDF = PIPER_ROOT / "urdf/piper_l_fixed_gripper.urdf"
ROBOT_CONFIG = PIPER_ROOT / "urdf/cca_config.yaml"
CABINET_URDF = ROOT / "assets/object/cabinet/urdf/cabinet.urdf"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVIRONMENTS = 4096
VISUALIZE_COUNT = 10
TRAJECTORY_POINTS = 12
POSE_IK_ITERATIONS = 100
SEED = 7

CABINET_POSITION_LOWER = (0.65, -0.10, 0.15)
CABINET_POSITION_UPPER = (0.75, 0.10, 0.25)
CABINET_YAW = math.pi  # Turn the cabinet front toward the robot.
# Values below come from drawer_joint_1 and its handle collision in cabinet.urdf.
DRAWER_1_HANDLE_LOCAL = (0.161706, 0.149577, 0.198016)
DRAWER_HANDLE_AXIS = (0.0, -1.0, 0.0)  # Local +Y after the cabinet placement yaw.
# drawer_joint_1 opens toward its negative limit, hence the extra minus sign.
DRAWER_PULL_AXIS = (-0.999816, -0.018917, -0.003302)
# drawer_joint_1 opens in the negative joint direction; this is its full travel.
DRAWER_MAX_PULL_DISTANCE = 0.22
GRASP_ROLL = -math.pi / 2


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


def cabinet_point(
    local_point: tuple[float, float, float],
    cabinet_positions: torch.Tensor,
) -> torch.Tensor:
    """Map a cabinet-local point to world after the 180-degree placement yaw."""
    yaw_pi_signs = cabinet_positions.new_tensor((-1.0, -1.0, 1.0))
    return cabinet_positions + yaw_pi_signs * cabinet_positions.new_tensor(
        local_point
    )


def main() -> None:
    args = parse_args()
    robot = cca.load_robot_from_urdf(
        URDF,
        ROBOT_CONFIG,
        joint_states=(0, 0, 0, 0, 0, 0),
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
    position_lower = torch.tensor(CABINET_POSITION_LOWER, device=DEVICE)
    position_upper = torch.tensor(CABINET_POSITION_UPPER, device=DEVICE)
    cabinet_positions = position_lower + (
        position_upper - position_lower
    ) * position_samples
    handle_positions = cabinet_point(DRAWER_1_HANDLE_LOCAL, cabinet_positions)
    pull_distances = torch.full(
        (NUM_ENVIRONMENTS,), DRAWER_MAX_PULL_DISTANCE, device=DEVICE
    )
    pull_axes = torch.tensor(DRAWER_PULL_AXIS, device=DEVICE).expand(
        NUM_ENVIRONMENTS, -1
    )
    drawer_screws = cca.get_screw(
        cca.ScrewType.TRANSLATION, pull_axes, handle_positions
    )

    initial_joints = robot.joint_states.expand(NUM_ENVIRONMENTS, -1).clone()
    initial_pose = cca.fkin_space(robot.M, robot.slist, initial_joints)
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
    grasp_poses[:, :3, 3] = handle_positions

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
    drawer = planner.generate_joint_trajectory(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_states=grasp_joints,
        affordance_screw=drawer_screws,
        goal_affordance=pull_distances,
        trajectory_density=TRAJECTORY_POINTS,
        # The handle is horizontal, so the grasp may roll only about its y axis.
        vir_screw_order=cca.VirtualScrewOrder.Y,
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    planning_time = time.perf_counter() - start_time

    trajectory = drawer.joint_trajectory
    valid = drawer.valid_mask & ik_success[:, None]
    success = ik_success & drawer.full_success
    print(
        f"cabinet drawer 1: {planning_time:.3f}s; "
        f"start IK {int(ik_success.sum())}/{NUM_ENVIRONMENTS}; "
        f"drawer CCA {int(success.sum())}/{int(ik_success.sum())}; "
        f"full {int(success.sum())}/{NUM_ENVIRONMENTS}"
    )

    ee_poses = cca.fkin_space(robot.M, robot.slist, trajectory)
    start_error = torch.linalg.vector_norm(
        ee_poses[:, 0, :3, 3] - handle_positions, dim=-1
    )
    expected_final = handle_positions + pull_distances[:, None] * pull_axes
    final_error = torch.linalg.vector_norm(
        ee_poses[:, -1, :3, 3] - expected_final, dim=-1
    )
    print_quantiles("successful start handle error [m]", start_error[success])
    print_quantiles("failed start handle error [m]", start_error[~success])
    print_quantiles(
        "successful drawer 1 endpoint error [m]", final_error[success]
    )
    print_quantiles("failed drawer 1 endpoint error [m]", final_error[~success])
    score = torch.maximum(start_error, final_error)

    if args.visualize:
        from visualizer import ViserVisualizer, VisualizationConfig

        count = min(VISUALIZE_COUNT, NUM_ENVIRONMENTS)
        shown = torch.topk(score, count).indices
        ViserVisualizer(
            robot, URDF, config=VisualizationConfig(port=8081)
        ).show_cabinet_drawer_1(
            trajectory[shown],
            valid[shown],
            success[shown],
            CABINET_URDF,
            cabinet_positions[shown],
            handle_positions[shown],
            pull_distances[shown],
            cabinet_yaw=CABINET_YAW,
            pull_axis=DRAWER_PULL_AXIS,
            handle_axis=DRAWER_HANDLE_AXIS,
        )


if __name__ == "__main__":
    main()
