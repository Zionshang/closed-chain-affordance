"""Use Piper-L to grasp a vertical cabinet-door handle and open the door.

Examples:
    python examples/demo_cabinet_door.py
    python examples/demo_cabinet_door.py --diagnose-errors
    python examples/demo_cabinet_door.py --visualize --diagnose-errors
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
SEED = 42

CABINET_POSITION = (0.70, 0.0, 0.20)
CABINET_YAW = math.pi  # Turn the cabinet front toward the robot.
# Values below come from door_joint_1 and the door handle collision in cabinet.urdf.
DOOR_HINGE_LOCAL = (0.119816, -0.275279, 0.104384)
DOOR_HANDLE_LOCAL = (0.165816, -0.025555, 0.165822)
DOOR_HANDLE_AXIS = (0.0, 0.0, 1.0)  # Vertical handle in world coordinates.
DOOR_HINGE_AXIS = (0.0, 0.0, 1.0)
DOOR_OPEN_ANGLE_RANGE = (-math.pi / 3, -math.pi / 4)
GRASP_ROLL = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--diagnose-errors", action="store_true")
    return parser.parse_args()


def print_quantiles(name: str, values: torch.Tensor) -> None:
    quantiles = torch.quantile(
        values[torch.isfinite(values)],
        values.new_tensor((0.5, 0.9, 0.99, 1.0)),
    )
    print(
        f"{name} q50/q90/q99/max: "
        + ", ".join(f"{value:.6f}" for value in quantiles.tolist())
    )


def rotate_about_z(
    points: torch.Tensor, centres: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    relative = points - centres
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    rotated = relative.clone()
    rotated[:, 0] = cosine * relative[:, 0] - sine * relative[:, 1]
    rotated[:, 1] = sine * relative[:, 0] + cosine * relative[:, 1]
    return centres + rotated


def cabinet_point(
    local_point: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Map a cabinet-local point to world after the 180-degree placement yaw."""
    return (
        CABINET_POSITION[0] - local_point[0],
        CABINET_POSITION[1] - local_point[1],
        CABINET_POSITION[2] + local_point[2],
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
    angle_samples = torch.rand(NUM_ENVIRONMENTS, generator=random, device=DEVICE)
    cabinet_positions = torch.tensor(CABINET_POSITION, device=DEVICE).expand(
        NUM_ENVIRONMENTS, -1
    ).clone()
    hinge_positions = torch.tensor(cabinet_point(DOOR_HINGE_LOCAL), device=DEVICE)
    hinge_positions = hinge_positions.expand(NUM_ENVIRONMENTS, -1).clone()
    handle_positions = torch.tensor(cabinet_point(DOOR_HANDLE_LOCAL), device=DEVICE)
    handle_positions = handle_positions.expand(NUM_ENVIRONMENTS, -1).clone()

    low_angle, high_angle = DOOR_OPEN_ANGLE_RANGE
    open_angles = low_angle + (high_angle - low_angle) * angle_samples
    hinge_axes = torch.tensor(DOOR_HINGE_AXIS, device=DEVICE).expand(
        NUM_ENVIRONMENTS, -1
    )
    door_screws = cca.get_screw(
        cca.ScrewType.ROTATION, hinge_axes, hinge_positions
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
    door = planner.generate_joint_trajectory(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_states=grasp_joints,
        affordance_screw=door_screws,
        goal_affordance=open_angles,
        trajectory_density=TRAJECTORY_POINTS,
        # The handle is vertical, so the grasp may roll only about its z axis.
        vir_screw_order=cca.VirtualScrewOrder.Z,
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    planning_time = time.perf_counter() - start_time

    trajectory = door.joint_trajectory
    valid = door.valid_mask & ik_success[:, None]
    success = ik_success & door.full_success
    print(
        f"cabinet door: {planning_time:.3f}s; "
        f"start IK {int(ik_success.sum())}/{NUM_ENVIRONMENTS}; "
        f"door CCA {int(door.full_success.sum())}/{NUM_ENVIRONMENTS}; "
        f"full {int(success.sum())}/{NUM_ENVIRONMENTS}"
    )

    score = None
    if args.diagnose_errors:
        ee_poses = cca.fkin_space(robot.M, robot.slist, trajectory)
        start_error = torch.linalg.vector_norm(
            ee_poses[:, 0, :3, 3] - handle_positions, dim=-1
        )
        expected_final = rotate_about_z(
            handle_positions, hinge_positions, open_angles
        )
        final_error = torch.linalg.vector_norm(
            ee_poses[:, -1, :3, 3] - expected_final, dim=-1
        )
        print_quantiles("start handle error [m]", start_error)
        print_quantiles("door endpoint error [m]", final_error)
        score = torch.maximum(start_error, final_error)

    if args.visualize:
        from visualizer import ViserVisualizer

        count = min(VISUALIZE_COUNT, NUM_ENVIRONMENTS)
        shown = (
            torch.arange(count, device=DEVICE)
            if score is None
            else torch.topk(score, count).indices
        )
        ViserVisualizer(robot, URDF).show_cabinet_door(
            trajectory[shown],
            valid[shown],
            success[shown],
            CABINET_URDF,
            cabinet_positions[shown],
            hinge_positions[shown],
            handle_positions[shown],
            open_angles[shown],
            cabinet_yaw=CABINET_YAW,
            handle_axis=DOOR_HANDLE_AXIS,
        )


if __name__ == "__main__":
    main()
