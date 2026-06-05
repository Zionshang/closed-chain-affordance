from __future__ import annotations
from pathlib import Path

import numpy as np

import closed_chain_affordance as cca
from meshcat_viewer import MeshcatViewer


VALVE_RADIUS_M = 0.08
VALVE_OPEN_ANGLE_RAD = np.pi


def build_x5_robot_description() -> cca.RobotDescription:
    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = repo_root / "assets" / "robot" / "x5" / "urdf" / "x5.urdf"
    config_path = repo_root / "python" / "x5_urdf_config.yaml"

    return cca.build_robot_description_from_urdf(
        str(urdf_path),
        str(config_path),
        joint_states=np.zeros(6, dtype=float),
    )


def build_approach_task(current_pose: np.ndarray) -> cca.TaskDescription:
    task = cca.TaskDescription(cca.PlanningType.CARTESIAN_GOAL)
    target_pose = current_pose.copy()
    target_pose[0, 3] += 0.2
    target_pose[2, 3] += 0.2
    task.goal.canonical_pose = target_pose
    task.trajectory_density = 20
    return task


def build_affordance_task(tcp_pose_on_valve: np.ndarray) -> cca.TaskDescription:
    task = cca.TaskDescription()
    affordance = cca.ScrewInfo()
    affordance.type = cca.ScrewType.ROTATION
    affordance.axis = cca.axis_to_vec(cca.Axis.X_MINUS)
    valve_center = tcp_pose_on_valve[:3, 3].copy()
    valve_center[2] -= VALVE_RADIUS_M
    affordance.location = valve_center
    task.affordance_info = affordance
    task.goal.affordance = VALVE_OPEN_ANGLE_RAD
    task.trajectory_density = 20
    return task


def plan_x5_trajectory() -> tuple[cca.RobotDescription, cca.PlannerResult]:
    robot = build_x5_robot_description()
    current_pose = cca.fkin_space(robot.M, robot.slist, robot.joint_states)

    print("robot screw list shape:", robot.slist.shape, flush=True)
    print("home/tool pose:", flush=True)
    print(robot.M, flush=True)
    print("current pose from FK:", flush=True)
    print(current_pose, flush=True)

    planner_config = cca.PlannerConfig()
    planner_config.update_method = cca.UpdateMethod.INVERSE
    planner = cca.CcAffordancePlannerInterface(planner_config)

    approach_task = build_approach_task(current_pose)
    approach_result = planner.generate_joint_trajectory(robot, approach_task)

    print(f"approach success: {approach_result.success}", flush=True)
    print(f"approach trajectory description: {approach_result.trajectory_description}", flush=True)
    print(f"approach update trail: {approach_result.update_trail}", flush=True)
    print(f"approach points: {len(approach_result.joint_trajectory)}", flush=True)

    if not approach_result.success:
        raise RuntimeError("Approach planner did not find a solution.")

    robot_at_valve = cca.RobotDescription()
    robot_at_valve.slist = robot.slist
    robot_at_valve.M = robot.M
    robot_at_valve.joint_states = np.asarray(approach_result.joint_trajectory[-1][:6], dtype=float)
    robot_at_valve.gripper_state = robot.gripper_state
    valve_pose = cca.fkin_space(robot_at_valve.M, robot_at_valve.slist, robot_at_valve.joint_states)

    print("pose after approach:", flush=True)
    print(valve_pose, flush=True)

    affordance_task = build_affordance_task(valve_pose)
    print("valve center:", affordance_task.affordance_info.location, flush=True)
    print("valve radius:", VALVE_RADIUS_M, flush=True)
    affordance_result = planner.generate_joint_trajectory(robot_at_valve, affordance_task)

    print(f"affordance success: {affordance_result.success}", flush=True)
    print(f"affordance trajectory description: {affordance_result.trajectory_description}", flush=True)
    print(f"affordance update trail: {affordance_result.update_trail}", flush=True)
    print(f"affordance points: {len(affordance_result.joint_trajectory)}", flush=True)

    if not affordance_result.success:
        raise RuntimeError("Affordance planner did not find a solution.")

    result = cca.PlannerResult()
    result.success = True
    result.trajectory_description = affordance_result.trajectory_description
    result.joint_trajectory = approach_result.joint_trajectory + affordance_result.joint_trajectory
    result.planning_time = approach_result.planning_time + affordance_result.planning_time
    result.update_method = planner_config.update_method
    result.update_trail = f"{approach_result.update_trail} -> {affordance_result.update_trail}"
    result.includes_gripper_trajectory = (
        approach_result.includes_gripper_trajectory or affordance_result.includes_gripper_trajectory
    )
    result.task_description = affordance_result.task_description

    print(f"combined points: {len(result.joint_trajectory)}", flush=True)
    print("first non-zero trajectory point:", flush=True)
    print(result.joint_trajectory[1], flush=True)
    print("final trajectory point:", flush=True)
    print(result.joint_trajectory[-1], flush=True)

    return robot, result


def main(show_viewer: bool = False) -> None:
    _, result = plan_x5_trajectory()

    if not show_viewer:
        return

    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = repo_root / "assets" / "robot" / "x5" / "urdf" / "x5.urdf"
    x5_package_root = repo_root / "assets" / "robot" / "x5"
    viewer = MeshcatViewer(
        urdf_path=urdf_path,
        package_map={
            "x5_description": x5_package_root,
            "x5": x5_package_root,
        },
        root_path="x5",
    )
    viewer.open()
    viewer.animate_trajectory(
        trajectory=[np.asarray(point[:6], dtype=float) for point in result.joint_trajectory],
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        fixed_joint_values={"joint7": 0.0, "joint8": 0.0},
        trajectory_frame_name="ee_link",
    )


if __name__ == "__main__":
    main(show_viewer=True)
