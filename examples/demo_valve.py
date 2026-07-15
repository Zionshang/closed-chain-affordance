"""Approach valves and turn them. Run: python examples/demo_valve.py"""

from pathlib import Path

import numpy as np
import torch

import cca_planner as cca
from visualizer import ViserVisualizer

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robot/x5/urdf/x5.urdf"
ROBOT_CONFIG = ROOT / "examples/x5_urdf_config.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NUM_ENVIRONMENTS = 64
APPROACH_POINTS = 12
TURN_POINTS = 12
APPROACH_ITERATIONS = 15
TURN_ITERATIONS = 15

# Valve task hyperparameters.
RANDOM_SEED = 0
VALVE_AXIS = (1.0, 0.0, 0.0)
GRASP_DIRECTION = (0.0, 0.0, 1.0)
RADIUS_RANGE = (0.05, 0.085)
TURN_ANGLE_RANGE = (np.pi / 3, np.pi)
POSITION_OFFSET_LOW = (0.18, -0.01, 0.18)
POSITION_OFFSET_HIGH = (0.22, 0.01, 0.22)


def make_planner(iterations):
    config = cca.PlannerConfig(
        accuracy=0.1,
        closure_err_threshold_ang=1e-3,
        closure_err_threshold_lin=1e-2,
        ik_max_itr=iterations,
        update_method=cca.UpdateMethod.INVERSE,
    )
    return cca.PlannerInterface(config, fast_mode=True, fast_linear_solver=True)


def create_tasks(robot, count):
    start_pose = cca.fkin_space(robot.M, robot.slist, robot.joint_states).cpu().numpy()
    random = np.random.default_rng(RANDOM_SEED)
    radii = random.uniform(*RADIUS_RANGE, count)
    angles = random.uniform(*TURN_ANGLE_RANGE, count)
    offsets = random.uniform(POSITION_OFFSET_LOW, POSITION_OFFSET_HIGH, (count, 3))
    grasp_poses = np.broadcast_to(start_pose, (count, 4, 4)).copy()
    grasp_poses[:, :3, 3] += offsets
    grasp_direction = np.asarray(GRASP_DIRECTION)
    centres = grasp_poses[:, :3, 3] - radii[:, None] * grasp_direction
    return centres, radii, angles, grasp_poses


def plan(robot, centres, angles, grasp_poses):
    count = len(angles)
    slist = robot.slist.expand(count, -1, -1)
    home = robot.M.expand(count, -1, -1)
    initial_joints = robot.joint_states.expand(count, -1).clone()

    reference_axis = torch.ones(count, 3, dtype=DTYPE, device=DEVICE)
    reference_axis /= torch.linalg.vector_norm(reference_axis, dim=-1, keepdim=True)
    approach_screw = cca.get_screw(
        cca.ScrewType.ROTATION, reference_axis, torch.ones_like(reference_axis)
    )
    approach = make_planner(APPROACH_ITERATIONS).generate_joint_trajectory(
        robot_slist=slist,
        robot_m=home,
        joint_states=initial_joints,
        motion_type=cca.MotionType.APPROACH,
        affordance_screw=approach_screw,
        goal_affordance=torch.full((count,), 1e-5, dtype=DTYPE, device=DEVICE),
        trajectory_density=APPROACH_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.NONE,
        canonical_pose=torch.as_tensor(grasp_poses, dtype=DTYPE, device=DEVICE),
    )

    axis = torch.as_tensor(VALVE_AXIS, dtype=DTYPE, device=DEVICE).expand(count, -1)
    valve_screw = cca.get_screw(
        cca.ScrewType.ROTATION,
        axis,
        torch.as_tensor(centres, dtype=DTYPE, device=DEVICE),
    )
    turn = make_planner(TURN_ITERATIONS).generate_joint_trajectory(
        robot_slist=slist,
        robot_m=home,
        joint_states=approach.joint_trajectory[:, -1, :6],
        motion_type=cca.MotionType.AFFORDANCE,
        affordance_screw=valve_screw,
        goal_affordance=torch.as_tensor(angles, dtype=DTYPE, device=DEVICE),
        trajectory_density=TURN_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.XYZ,
    )
    return approach, turn


def main():
    robot = cca.load_robot_from_urdf(
        URDF,
        ROBOT_CONFIG,
        joint_states=(0, 0, 0, 0, 0, 0),
        dtype=DTYPE,
        device=DEVICE,
    )
    centres, radii, angles, grasp_poses = create_tasks(robot, NUM_ENVIRONMENTS)
    approach, turn = plan(robot, centres, angles, grasp_poses)

    trajectory = torch.cat(
        [approach.joint_trajectory, turn.joint_trajectory[:, 1:]], dim=1
    )
    valid_mask = torch.cat([approach.valid_mask, turn.valid_mask], dim=1)
    success = approach.full_success & turn.full_success
    print(
        f"valve: approach={int(approach.full_success.sum())}/{NUM_ENVIRONMENTS}, "
        f"turn={int(turn.full_success.sum())}/{NUM_ENVIRONMENTS}, "
        f"full={int(success.sum())}/{NUM_ENVIRONMENTS}"
    )

    ViserVisualizer(robot, URDF).show_valves(
        trajectory,
        valid_mask,
        success,
        centres,
        radii,
        angles,
        APPROACH_POINTS,
        rotation_axis=VALVE_AXIS,
        grasp_direction=GRASP_DIRECTION,
    )


if __name__ == "__main__":
    main()

