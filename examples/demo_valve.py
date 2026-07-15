"""Approach valves and turn them. Run: python examples/demo_valve.py"""

from pathlib import Path

import torch

import cca_planner as cca
from visualizer import ViserVisualizer

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robot/x5/urdf/x5.urdf"
ROBOT_CONFIG = ROOT / "examples/x5_urdf_config.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NUM_ENVIRONMENTS = 64
TRAJECTORY_POINTS = 12
IK_ITERATIONS = 30

VALVE_AXIS = (1.0, 0.0, 0.0)
GRASP_DIRECTION = (0.0, 0.0, 1.0)


def main():
    robot = cca.load_robot_from_urdf(
        URDF, ROBOT_CONFIG, joint_states=(0, 0, 0, 0, 0, 0),
        dtype=DTYPE, device=DEVICE,
    )
    planner = cca.PlannerInterface(
        cca.PlannerConfig(
            ik_max_itr=IK_ITERATIONS,
            update_method=cca.UpdateMethod.INVERSE,
            closure_err_threshold_ang=1e-3,
            closure_err_threshold_lin=1e-2,
        )
    )

    # Each environment has a different valve pose, radius, and turn angle.
    random = torch.Generator(device=DEVICE).manual_seed(0)
    samples = torch.rand(
        NUM_ENVIRONMENTS, 5, generator=random, device=DEVICE, dtype=DTYPE
    )
    radii = 0.05 + 0.035 * samples[:, 0]
    angles = torch.pi / 3 + 2 * torch.pi / 3 * samples[:, 1]
    offset_low = torch.tensor((0.18, -0.01, 0.18), device=DEVICE, dtype=DTYPE)
    offset_high = torch.tensor((0.22, 0.01, 0.22), device=DEVICE, dtype=DTYPE)

    start_pose = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
    grasp_poses = start_pose.expand(NUM_ENVIRONMENTS, -1, -1).clone()
    grasp_poses[:, :3, 3] += offset_low + (offset_high - offset_low) * samples[:, 2:]
    grasp_direction = torch.tensor(GRASP_DIRECTION, device=DEVICE, dtype=DTYPE)
    centres = grasp_poses[:, :3, 3] - radii[:, None] * grasp_direction
    initial_joints = robot.joint_states.expand(NUM_ENVIRONMENTS, -1).clone()

    # Stage 1: move the TCP to each valve rim.
    reference_axis = torch.ones(NUM_ENVIRONMENTS, 3, device=DEVICE, dtype=DTYPE)
    reference_axis /= torch.linalg.vector_norm(reference_axis, dim=-1, keepdim=True)
    approach = planner.generate_joint_trajectory(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_states=initial_joints,
        motion_type=cca.MotionType.APPROACH,
        affordance_screw=cca.get_screw(
            cca.ScrewType.ROTATION, reference_axis, torch.ones_like(reference_axis)
        ),
        goal_affordance=torch.full(
            (NUM_ENVIRONMENTS,), 1e-5, device=DEVICE, dtype=DTYPE
        ),
        trajectory_density=TRAJECTORY_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.NONE,
        canonical_pose=grasp_poses,
    )

    # Stage 2: rotate around each valve centre and axis.
    valve_axes = torch.tensor(VALVE_AXIS, device=DEVICE, dtype=DTYPE).expand(
        NUM_ENVIRONMENTS, -1
    )
    turn = planner.generate_joint_trajectory(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_states=approach.joint_trajectory[:, -1, :6],
        motion_type=cca.MotionType.AFFORDANCE,
        affordance_screw=cca.get_screw(cca.ScrewType.ROTATION, valve_axes, centres),
        goal_affordance=angles,
        trajectory_density=TRAJECTORY_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.XYZ,
    )

    trajectory = torch.cat([approach.joint_trajectory, turn.joint_trajectory[:, 1:]], 1)
    valid = torch.cat([approach.valid_mask, turn.valid_mask], 1)
    success = approach.full_success & turn.full_success
    print(f"valve: {int(success.sum())}/{NUM_ENVIRONMENTS} full trajectories")
    ViserVisualizer(robot, URDF).show_valves(
        trajectory, valid, success, centres, radii, angles, TRAJECTORY_POINTS,
        rotation_axis=VALVE_AXIS, grasp_direction=GRASP_DIRECTION,
    )


if __name__ == "__main__":
    main()

