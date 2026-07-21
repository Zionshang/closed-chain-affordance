"""Solve drawer-handle IK and plan the pull. Run: python examples/demo_drawer.py"""

from pathlib import Path

import torch

import cca_planner as cca
from visualizer import ViserVisualizer, VisualizationConfig

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robot/x5/urdf/x5.urdf"
ROBOT_CONFIG = ROOT / "examples/x5_urdf_config.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NUM_ENVIRONMENTS = 64
TRAJECTORY_POINTS = 12
IK_ITERATIONS = 30
POSE_IK_ITERATIONS = 300
PULL_AXIS = (-1.0, 0.0, 0.0)  # Cabinet is in +x; pull toward the robot.


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

    # Each environment has a different handle pose and pull distance.
    random = torch.Generator(device=DEVICE).manual_seed(1)
    samples = torch.rand(
        NUM_ENVIRONMENTS, 4, generator=random, device=DEVICE, dtype=DTYPE
    )
    pull_distances = 0.10 + 0.06 * samples[:, 0]
    offset_low = torch.tensor((0.20, -0.015, 0.14), device=DEVICE, dtype=DTYPE)
    offset_high = torch.tensor((0.24, 0.015, 0.20), device=DEVICE, dtype=DTYPE)

    start_pose = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
    grasp_poses = start_pose.expand(NUM_ENVIRONMENTS, -1, -1).clone()
    grasp_poses[:, :3, 3] += offset_low + (offset_high - offset_low) * samples[:, 1:]
    handle_positions = grasp_poses[:, :3, 3]
    initial_joints = robot.joint_states.expand(NUM_ENVIRONMENTS, -1).clone()

    # Solve the handle-contact configuration directly; CCA starts at contact.
    grasp_joints, ik_success = planner.solve_pose_ik(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_seed=initial_joints,
        target_pose=grasp_poses,
        max_iterations=POSE_IK_ITERATIONS,
    )

    # Plan only the constrained drawer translation from the IK contact state.
    pull_axes = torch.tensor(PULL_AXIS, device=DEVICE, dtype=DTYPE).expand(
        NUM_ENVIRONMENTS, -1
    )
    pull = planner.generate_joint_trajectory(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_states=grasp_joints,
        motion_type=cca.MotionType.AFFORDANCE,
        affordance_screw=cca.get_screw(
            cca.ScrewType.TRANSLATION, pull_axes, handle_positions
        ),
        goal_affordance=pull_distances,
        trajectory_density=TRAJECTORY_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.XYZ,
    )

    trajectory = pull.joint_trajectory
    valid = pull.valid_mask & ik_success[:, None]
    success = ik_success & pull.full_success
    print(
        f"drawer: start IK {int(ik_success.sum())}/{NUM_ENVIRONMENTS}; "
        f"pull {int(pull.full_success.sum())}/{NUM_ENVIRONMENTS}; "
        f"full {int(success.sum())}/{NUM_ENVIRONMENTS}"
    )
    ViserVisualizer(robot, URDF, config=VisualizationConfig(port=8081)).show_drawers(
        trajectory, valid, success, handle_positions, pull_distances,
        1, pull_axis=PULL_AXIS,
    )


if __name__ == "__main__":
    main()
