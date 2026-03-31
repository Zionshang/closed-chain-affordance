import numpy as np

import closed_chain_affordance_py as cca


def build_ur5_robot_description() -> cca.RobotDescription:
    mconv = 1000.0
    w2 = 82.0 / mconv
    l1 = 425.0 / mconv
    l2 = 392.0 / mconv
    h1 = 89.0 / mconv
    h2 = 95.0 / mconv
    w3 = 135.85 / mconv
    w4 = 119.7 / mconv
    w6 = 93.0 / mconv

    q = np.array(
        [
            [0.0, 0.0, l1, l1 + l2, l1 + l2, l1 + l2],
            [0.0, w3, w3 - w4, w3 - w4, w3 - w4 + w6, w3 - w4 + w6],
            [h1, h1, h1, h1, h1, h1 - h2],
        ],
        dtype=float,
    )

    w = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        ],
        dtype=float,
    )

    robot_slist = np.zeros((6, 6), dtype=float)
    for i in range(6):
        axis = w[:, i]
        location = q[:, i]
        robot_slist[:, i] = np.concatenate([axis, -np.cross(axis, location)])

    home_pose = np.eye(4, dtype=float)
    home_pose[:3, 3] = [l1 + l2, w3 - w4 + w6 + w2, h1 - h2]

    robot = cca.RobotDescription()
    robot.slist = robot_slist
    robot.M = home_pose
    robot.joint_states = np.zeros(6, dtype=float)
    robot.gripper_state = 0.0
    return robot


def build_task_description() -> cca.TaskDescription:
    mconv = 1000.0
    w2 = 82.0 / mconv
    l1 = 425.0 / mconv
    l2 = 392.0 / mconv
    h1 = 89.0 / mconv
    h2 = 95.0 / mconv
    w3 = 135.85 / mconv
    w4 = 119.7 / mconv
    w6 = 93.0 / mconv
    aff_offset = -100.0 / mconv

    affordance = cca.ScrewInfo()
    affordance.type = cca.ScrewType.ROTATION
    affordance.axis = cca.axis_to_vec(cca.Axis.X_MINUS)
    affordance.location = np.array(
        [l1 + l2, w3 - w4 + w6 + w2 + aff_offset, h1 - h2],
        dtype=float,
    )

    task = cca.TaskDescription()
    task.affordance_info = affordance
    task.trajectory_density = 5
    task.gripper_goal_type = cca.GripperGoalType.CONTINUOUS
    task.goal.affordance = 0.4
    task.goal.gripper = 0.4
    return task


def main() -> None:
    robot = build_ur5_robot_description()
    task = build_task_description()

    planner_config = cca.PlannerConfig()
    planner_config.accuracy = 0.01
    planner_config.update_method = cca.UpdateMethod.INVERSE

    planner = cca.CcAffordancePlannerInterface(planner_config)
    result = planner.generate_joint_trajectory(robot, task)

    print(f"planning success: {result.success}")
    print(f"trajectory description: {result.trajectory_description}")
    print(f"update trail: {result.update_trail}")
    print(f"number of trajectory points: {len(result.joint_trajectory)}")

    if not result.success:
        raise RuntimeError("Planner did not find a solution.")

    print("first non-zero trajectory point:")
    print(result.joint_trajectory[1])
    print("final trajectory point:")
    print(result.joint_trajectory[-1])


if __name__ == "__main__":
    main()
