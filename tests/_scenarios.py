"""Scenario registry shared by the worker and the equivalence test.

Each scenario is a function ``scenario(cca) -> (robot, task, cfg)`` built from
the same canonical numeric inputs, so the C++ binding and the pure-Python
package can be driven identically. Results are serialized to plain JSON in a
subprocess (see ``_worker.py``) so the two BLAS-backed native libraries never
share a process.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- #
# Canonical inputs
# --------------------------------------------------------------------------- #
def ur5_slist_and_home():
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
    slist = np.zeros((6, 6))
    for i in range(6):
        slist[:, i] = np.concatenate([w[:, i], -np.cross(w[:, i], q[:, i])])

    home = np.eye(4)
    home[:3, 3] = [l1 + l2, w3 - w4 + w6 + w2, h1 - h2]
    return slist, home


def valve_location():
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
    return np.array([l1 + l2, w3 - w4 + w6 + w2 + aff_offset, h1 - h2], dtype=float)


# --------------------------------------------------------------------------- #
# Parallel object builders (build the same values for either module)
# --------------------------------------------------------------------------- #
def build_robot(cca, slist, home, joint_states=None, gripper_state=float("nan")):
    robot = cca.RobotDescription()
    robot.slist = np.asarray(slist, dtype=float)
    robot.M = np.asarray(home, dtype=float)
    robot.joint_states = (
        np.zeros(slist.shape[1]) if joint_states is None else np.asarray(joint_states, dtype=float)
    )
    robot.gripper_state = gripper_state
    return robot


def build_config(cca, accuracy=0.01, update_method=None, ik_max_itr=None):
    cfg = cca.PlannerConfig()
    cfg.accuracy = accuracy
    if update_method is not None:
        cfg.update_method = update_method
    if ik_max_itr is not None:
        cfg.ik_max_itr = ik_max_itr
    return cfg


def build_screw_info(cca, screw_type, axis=None, location=None, screw=None, pitch=float("nan")):
    aff = cca.ScrewInfo()
    aff.type = screw_type
    if axis is not None:
        aff.axis = np.asarray(axis, dtype=float)
    if location is not None:
        aff.location = np.asarray(location, dtype=float)
    if screw is not None:
        aff.screw = np.asarray(screw, dtype=float)
    if not math.isnan(pitch):
        aff.pitch = pitch
    return aff


def build_task(cca, affordance_info=None, affordance_goal=None, trajectory_density=None, motion_type=None,
               vir_screw_order=None, ee_orientation=None, gripper_goal=float("nan"),
               gripper_goal_type=None, canonical_pose=None, planning_type=None):
    task = cca.TaskDescription(planning_type) if planning_type is not None else cca.TaskDescription()
    if affordance_info is not None:
        task.affordance_info = affordance_info
    if affordance_goal is not None:
        task.goal.affordance = affordance_goal
    if trajectory_density is not None:
        task.trajectory_density = trajectory_density
    if motion_type is not None:
        task.motion_type = motion_type
    if vir_screw_order is not None:
        task.vir_screw_order = vir_screw_order
    if ee_orientation is not None:
        task.goal.ee_orientation = np.asarray(ee_orientation, dtype=float)
    if not math.isnan(gripper_goal):
        task.goal.gripper = gripper_goal
    if gripper_goal_type is not None:
        task.gripper_goal_type = gripper_goal_type
    if canonical_pose is not None:
        task.goal.canonical_pose = np.asarray(canonical_pose, dtype=float)
    return task


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def _rotation_affordance(cca, *, update_method, density=5, gripper_goal=float("nan"),
                         gripper_goal_type=None, ee_orientation=None, gripper_state=float("nan")):
    slist, home = ur5_slist_and_home()
    robot = build_robot(cca, slist, home, gripper_state=gripper_state)
    aff = build_screw_info(
        cca, cca.ScrewType.ROTATION, axis=cca.axis_to_vec(cca.Axis.X_MINUS), location=valve_location()
    )
    task = build_task(
        cca, aff, affordance_goal=0.4, trajectory_density=density, motion_type=cca.MotionType.AFFORDANCE,
        gripper_goal=gripper_goal, gripper_goal_type=gripper_goal_type, ee_orientation=ee_orientation,
    )
    cfg = build_config(cca, update_method=update_method)
    return robot, task, cfg


def scenario_rotation_inverse(cca):
    return _rotation_affordance(cca, update_method=cca.UpdateMethod.INVERSE)


def scenario_rotation_transpose(cca):
    return _rotation_affordance(cca, update_method=cca.UpdateMethod.TRANSPOSE)


def scenario_rotation_with_gripper(cca):
    return _rotation_affordance(
        cca, update_method=cca.UpdateMethod.INVERSE, gripper_goal=0.4,
        gripper_goal_type=cca.GripperGoalType.CONTINUOUS, gripper_state=0.0,
    )


def scenario_rotation_with_ee_orientation(cca):
    return _rotation_affordance(
        cca, update_method=cca.UpdateMethod.INVERSE, density=8, ee_orientation=np.array([0.1, 0.0, 0.1]),
    )


def scenario_translation(cca):
    slist, home = ur5_slist_and_home()
    robot = build_robot(cca, slist, home)
    aff = build_screw_info(cca, cca.ScrewType.TRANSLATION, axis=np.array([1.0, 0.0, 0.0]), location=np.zeros(3))
    task = build_task(cca, aff, affordance_goal=0.2, trajectory_density=6, motion_type=cca.MotionType.AFFORDANCE)
    cfg = build_config(cca, update_method=cca.UpdateMethod.INVERSE)
    return robot, task, cfg


def scenario_screw(cca):
    slist, home = ur5_slist_and_home()
    robot = build_robot(cca, slist, home)
    aff = build_screw_info(cca, cca.ScrewType.SCREW, axis=np.array([0.0, 0.0, 1.0]), location=np.zeros(3), pitch=0.05)
    task = build_task(cca, aff, affordance_goal=0.5, trajectory_density=6, motion_type=cca.MotionType.AFFORDANCE)
    cfg = build_config(cca, update_method=cca.UpdateMethod.INVERSE)
    return robot, task, cfg


def build_x5_robot(cca):
    """Build the x5 arm from its URDF (converges better than the UR5 here)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = str(repo_root / "assets" / "robot" / "x5" / "urdf" / "x5.urdf")
    config_path = str(repo_root / "python" / "x5_urdf_config.yaml")
    return cca.build_robot_description_from_urdf(urdf_path, config_path, joint_states=np.zeros(6))


def scenario_x5_cartesian_approach(cca):
    """APPROACH motion to a Cartesian target with the x5 arm (converges to a
    PARTIAL trajectory, exercising the approach-motion code path)."""
    robot = build_x5_robot(cca)
    current_pose = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
    canonical = current_pose.copy()
    canonical[0, 3] += 0.2
    canonical[2, 3] += 0.2
    # The CARTESIAN_GOAL constructor fills in affordance_info and goal.affordance;
    # only the canonical pose and density are overridden here.
    task = build_task(
        cca, trajectory_density=20,
        planning_type=cca.PlanningType.CARTESIAN_GOAL, canonical_pose=canonical,
    )
    cfg = build_config(cca, update_method=cca.UpdateMethod.INVERSE)
    return robot, task, cfg


def scenario_cartesian_goal(cca):
    """UR5 CARTESIAN_GOAL (infeasible for this robot). Both implementations must
    agree on the failure (UNSET), confirming the approach-motion path matches
    even when no solution exists."""
    slist, home = ur5_slist_and_home()
    robot = build_robot(cca, slist, home)
    canonical = home.copy()
    canonical[0, 3] += 0.1
    canonical[2, 3] += 0.1
    task = build_task(
        cca, trajectory_density=15,
        planning_type=cca.PlanningType.CARTESIAN_GOAL, canonical_pose=canonical,
    )
    cfg = build_config(cca, update_method=cca.UpdateMethod.INVERSE)
    return robot, task, cfg


def scenario_best(cca):
    return _rotation_affordance(cca, update_method=cca.UpdateMethod.BEST)


SCENARIOS = {
    "rotation_inverse": scenario_rotation_inverse,
    "rotation_transpose": scenario_rotation_transpose,
    "rotation_with_gripper": scenario_rotation_with_gripper,
    "rotation_with_ee_orientation": scenario_rotation_with_ee_orientation,
    "translation": scenario_translation,
    "screw": scenario_screw,
    "x5_cartesian_approach": scenario_x5_cartesian_approach,
    "cartesian_goal": scenario_cartesian_goal,
    "best": scenario_best,
}

# Scenarios where the winning update method (and thus the exact joints) may be
# scheduling-dependent; only success / description / length are compared.
NONDETERMINISTIC = {"best"}
