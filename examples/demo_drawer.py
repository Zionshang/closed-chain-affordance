"""Approach drawer handles and pull them open. Run: python examples/demo_drawer.py"""

from pathlib import Path

import numpy as np
import torch

import cca_planner as cca
from visualizer import ViserVisualizer, VisualizationConfig

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robot/x5/urdf/x5.urdf"
ROBOT_CONFIG = ROOT / "examples/x5_urdf_config.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NUM_ENVIRONMENTS = 64
APPROACH_POINTS = 12
PULL_POINTS = 12
APPROACH_ITERATIONS = 15
PULL_ITERATIONS = 30

# Drawer task hyperparameters. The cabinet is in +x; the drawer opens toward -x.
RANDOM_SEED = 1
PULL_AXIS = (-1.0, 0.0, 0.0)
PULL_DISTANCE_RANGE = (0.10, 0.16)
POSITION_OFFSET_LOW = (0.20, -0.015, 0.14)
POSITION_OFFSET_HIGH = (0.24, 0.015, 0.20)


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
    distances = random.uniform(*PULL_DISTANCE_RANGE, count)
    offsets = random.uniform(POSITION_OFFSET_LOW, POSITION_OFFSET_HIGH, (count, 3))
    grasp_poses = np.broadcast_to(start_pose, (count, 4, 4)).copy()
    grasp_poses[:, :3, 3] += offsets
    return grasp_poses[:, :3, 3].copy(), distances, grasp_poses


def plan(robot, handle_positions, pull_distances, grasp_poses):
    count = len(pull_distances)
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

    pull_axis = torch.as_tensor(PULL_AXIS, dtype=DTYPE, device=DEVICE).expand(count, -1)
    drawer_screw = cca.get_screw(
        cca.ScrewType.TRANSLATION,
        pull_axis,
        torch.as_tensor(handle_positions, dtype=DTYPE, device=DEVICE),
    )
    pull = make_planner(PULL_ITERATIONS).generate_joint_trajectory(
        robot_slist=slist,
        robot_m=home,
        joint_states=approach.joint_trajectory[:, -1, :6],
        motion_type=cca.MotionType.AFFORDANCE,
        affordance_screw=drawer_screw,
        goal_affordance=torch.as_tensor(pull_distances, dtype=DTYPE, device=DEVICE),
        trajectory_density=PULL_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.XYZ,
    )
    return approach, pull


def main():
    robot = cca.load_robot_from_urdf(
        URDF,
        ROBOT_CONFIG,
        joint_states=(0, 0, 0, 0, 0, 0),
        dtype=DTYPE,
        device=DEVICE,
    )
    handles, distances, grasp_poses = create_tasks(robot, NUM_ENVIRONMENTS)
    approach, pull = plan(robot, handles, distances, grasp_poses)

    trajectory = torch.cat(
        [approach.joint_trajectory, pull.joint_trajectory[:, 1:]], dim=1
    )
    valid_mask = torch.cat([approach.valid_mask, pull.valid_mask], dim=1)
    success = approach.full_success & pull.full_success
    print(
        f"drawer: approach={int(approach.full_success.sum())}/{NUM_ENVIRONMENTS}, "
        f"pull={int(pull.full_success.sum())}/{NUM_ENVIRONMENTS}, "
        f"full={int(success.sum())}/{NUM_ENVIRONMENTS}"
    )

    viewer = ViserVisualizer(robot, URDF, config=VisualizationConfig(port=8081))
    viewer.show_drawers(
        trajectory,
        valid_mask,
        success,
        handles,
        distances,
        APPROACH_POINTS,
        pull_axis=PULL_AXIS,
    )


if __name__ == "__main__":
    main()
