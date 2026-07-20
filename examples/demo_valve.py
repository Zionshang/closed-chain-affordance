"""Plan randomized Piper-L valve trajectories.

Examples:
    python examples/demo_valve.py
    python examples/demo_valve.py --diagnose-errors
    python examples/demo_valve.py --visualize --diagnose-errors
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import cca_planner as cca
from visualizer import ViserVisualizer


ROOT = Path(__file__).resolve().parents[1]
PIPER_ROOT = ROOT / "assets/robot/piper_l"
URDF = PIPER_ROOT / "urdf/piper_l_fixed_gripper.urdf"
ROBOT_CONFIG = PIPER_ROOT / "urdf/cca_config.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 规划计算设备
DTYPE = torch.float32  # 规划张量精度
NUM_ENVIRONMENTS = 4096  # 并行规划环境数
VISUALIZE_COUNT = 32  # Viser 中显示的环境数
TRAJECTORY_POINTS = 12  # 每个阶段的轨迹点数
IK_ITERATIONS = 50  # 每个轨迹点的最大 IK 迭代次数
SEED = 42  # 随机种子

VALVE_POSITION = (0.55, 0.0, 0.24)  # 阀门中心的基准位置
VALVE_POSITION_RANDOMIZATION = {  # 各坐标轴的位置随机偏移范围
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (0.0, 0.0),
}
VALVE_OUTER_DIAMETERS = (0.28, 0.32, 0.36)  # 可选阀门外径
VALVE_RIM_DIAMETER = 0.035  # 外圈管材直径
VALVE_AXIS = (1.0, 0.0, 0.0)  # 阀门旋转轴方向
VALVE_UP = (0.0, 0.0, 1.0)  # 初始抓取点的径向方向
TURN_ANGLE = torch.pi  # 阀门目标转角
GRASP_ROLL = torch.pi / 2  # 接近阶段的末端滚转角

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize", action="store_true", help="Open the Viser viewer.")
    parser.add_argument(
        "--diagnose-errors",
        action="store_true",
        help="Print endpoint-error and waypoint-jump quantiles.",
    )
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


def main() -> None:
    args = parse_args()
    robot = cca.load_robot_from_urdf(
        URDF,
        ROBOT_CONFIG,
        joint_states=(0, 0, 0, 0, 0, 0),
        dtype=DTYPE,
        device=DEVICE,
    )
    planner = cca.PlannerInterface(
        cca.PlannerConfig(
            ik_max_itr=IK_ITERATIONS,
            update_method=cca.UpdateMethod.INVERSE,
            closure_err_threshold_ang=1e-3,
            closure_err_threshold_lin=1e-2,
        ),
        fast_mode=True,
        compile=False,
        fast_linear_solver=True,
    )

    random = torch.Generator(device=DEVICE).manual_seed(SEED)
    position_sample = torch.rand(
        NUM_ENVIRONMENTS, 3, generator=random, device=DEVICE, dtype=DTYPE
    )
    position = torch.tensor(VALVE_POSITION, device=DEVICE, dtype=DTYPE)
    position_bounds = torch.tensor(
        [VALVE_POSITION_RANDOMIZATION[axis] for axis in ("x", "y", "z")],
        device=DEVICE,
        dtype=DTYPE,
    )
    position_low, position_high = position_bounds.unbind(dim=1)
    centres = position + position_low + (position_high - position_low) * position_sample

    diameter_values = torch.tensor(
        VALVE_OUTER_DIAMETERS, device=DEVICE, dtype=DTYPE
    )
    diameter_indices = torch.randint(
        len(VALVE_OUTER_DIAMETERS),
        (NUM_ENVIRONMENTS,),
        generator=random,
        device=DEVICE,
    )
    outer_diameters = diameter_values[diameter_indices]
    grasp_radii = 0.5 * (outer_diameters - VALVE_RIM_DIAMETER)

    initial_joints = robot.joint_states.expand(NUM_ENVIRONMENTS, -1).clone()
    initial_pose = cca.fkin_space(robot.M, robot.slist, initial_joints)
    up = torch.tensor(VALVE_UP, device=DEVICE, dtype=DTYPE).expand(
        NUM_ENVIRONMENTS, -1
    )
    grasp_points = centres + grasp_radii.unsqueeze(-1) * up
    approach_delta = grasp_points - initial_pose[:, :3, 3]
    approach_distance = torch.linalg.vector_norm(approach_delta, dim=-1)
    approach_direction = approach_delta / approach_distance.clamp_min(1e-6).unsqueeze(-1)
    approach_orientation = torch.zeros(
        NUM_ENVIRONMENTS, 3, device=DEVICE, dtype=DTYPE
    )
    approach_orientation[:, 0] = GRASP_ROLL

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    start_time = time.perf_counter()

    approach = planner.generate_joint_trajectory(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_states=initial_joints,
        motion_type=cca.MotionType.AFFORDANCE,
        affordance_screw=cca.get_screw(
            cca.ScrewType.TRANSLATION,
            approach_direction,
            torch.zeros_like(approach_direction),
        ),
        goal_affordance=approach_distance,
        trajectory_density=TRAJECTORY_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.XYZ,
        goal_ee_orientation=approach_orientation,
    )

    valve_axes = torch.tensor(VALVE_AXIS, device=DEVICE, dtype=DTYPE).expand(
        NUM_ENVIRONMENTS, -1
    )
    turn_angles = torch.full(
        (NUM_ENVIRONMENTS,), TURN_ANGLE, device=DEVICE, dtype=DTYPE
    )
    turn = planner.generate_joint_trajectory(
        robot_slist=robot.slist,
        robot_m=robot.M,
        joint_states=approach.joint_trajectory[:, -1],
        motion_type=cca.MotionType.AFFORDANCE,
        affordance_screw=cca.get_screw(
            cca.ScrewType.ROTATION, valve_axes, centres
        ),
        goal_affordance=turn_angles,
        trajectory_density=TRAJECTORY_POINTS,
        vir_screw_order=cca.VirtualScrewOrder.NONE,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    planning_time = time.perf_counter() - start_time

    trajectory = torch.cat(
        (approach.joint_trajectory, turn.joint_trajectory[:, 1:]), dim=1
    )
    valid = torch.cat((approach.valid_mask, turn.valid_mask), dim=1)
    success = approach.full_success & turn.full_success
    print(
        f"planning: {planning_time:.3f}s; "
        f"approach {int(approach.full_success.sum())}/{NUM_ENVIRONMENTS}; "
        f"turn {int(turn.full_success.sum())}/{NUM_ENVIRONMENTS}; "
        f"full {int(success.sum())}/{NUM_ENVIRONMENTS}"
    )

    score = None
    if args.diagnose_errors:
        ee_poses = cca.fkin_space(robot.M, robot.slist, trajectory)
        approach_error = torch.linalg.vector_norm(
            ee_poses[:, TRAJECTORY_POINTS - 1, :3, 3] - grasp_points, dim=-1
        )
        final_grasp_points = centres - grasp_radii.unsqueeze(-1) * up
        turn_error = torch.linalg.vector_norm(
            ee_poses[:, -1, :3, 3] - final_grasp_points, dim=-1
        )
        waypoint_jump = torch.linalg.vector_norm(
            torch.diff(ee_poses[:, :, :3, 3], dim=1), dim=-1
        ).amax(dim=-1)
        print_quantiles("approach endpoint error [m]", approach_error)
        print_quantiles("turn endpoint error [m]", turn_error)
        print_quantiles("maximum TCP waypoint jump [m]", waypoint_jump)
        score = torch.maximum(approach_error, turn_error)

    if args.visualize:
        if score is None:
            shown = torch.arange(VISUALIZE_COUNT, device=DEVICE)
        else:
            shown = torch.topk(score, VISUALIZE_COUNT).indices
        print("visualized environment ids:", shown.tolist())
        ViserVisualizer(robot, URDF).show_valves(
            trajectory[shown],
            valid[shown],
            success[shown],
            centres[shown],
            0.5 * outer_diameters[shown],
            turn_angles[shown],
            TRAJECTORY_POINTS,
            rotation_axis=VALVE_AXIS,
            grasp_direction=VALVE_UP,
        )


if __name__ == "__main__":
    main()
