"""Regression tests for the single-task pure-Python C++ reimplementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import closed_chain_affordance as cca  # noqa: E402
from _scenarios import scenario_cartesian_goal, scenario_rotation_inverse  # noqa: E402
from closed_chain_affordance.affordance_util import (  # noqa: E402
    compute_gripper_joint_trajectory,
    get_vir_screw_axes,
)
from closed_chain_affordance.robot_builder import _origin_to_matrix  # noqa: E402


CPP_PUBLIC_API = {
    "Axis",
    "PoseSpecificationMethod",
    "GripperGoalType",
    "ScrewType",
    "VirtualScrewOrder",
    "EeOrientationConstraint",
    "PlanningType",
    "MotionType",
    "TrajectoryDescription",
    "UpdateMethod",
    "VecInfo",
    "PoseFrom",
    "ScrewInfoFrom",
    "ScrewInfo",
    "RobotDescription",
    "Goal",
    "TaskDescription",
    "PlannerConfig",
    "PlannerResult",
    "CcAffordancePlannerInterface",
    "axis_to_vec",
    "get_screw",
    "fkin_space",
    "plan",
    "build_robot_description_from_yaml",
    "build_robot_description_from_urdf",
}


def test_top_level_api_matches_cpp_binding():
    assert set(cca.__all__) == CPP_PUBLIC_API
    assert not hasattr(cca, "BatchedCcAffordancePlannerInterface")
    assert not hasattr(cca, "plan_batch")


@pytest.mark.parametrize(
    "enum_type",
    [
        cca.Axis,
        cca.PoseSpecificationMethod,
        cca.GripperGoalType,
        cca.ScrewType,
        cca.VirtualScrewOrder,
        cca.EeOrientationConstraint,
        cca.PlanningType,
        cca.MotionType,
        cca.TrajectoryDescription,
        cca.UpdateMethod,
    ],
)
def test_enum_values_follow_cpp_zero_based_order(enum_type):
    assert [member.value for member in enum_type] == list(range(len(enum_type)))


def test_binding_struct_constructors_are_no_arg():
    with pytest.raises(TypeError):
        cca.PlannerConfig(accuracy=0.01)
    with pytest.raises(TypeError):
        cca.ScrewInfo(type=cca.ScrewType.ROTATION)
    with pytest.raises(TypeError):
        cca.TaskDescription("APPROACH")
    with pytest.raises(TypeError):
        cca.TaskDescription(None)
    with pytest.raises(TypeError):
        cca.CcAffordancePlannerInterface(None)


def test_interface_copies_config_and_result_task():
    robot, task, config = scenario_rotation_inverse(cca)
    planner = cca.CcAffordancePlannerInterface(config)

    config.update_method = cca.UpdateMethod.TRANSPOSE
    assert planner.planner_config_.update_method == cca.UpdateMethod.INVERSE

    result = planner.generate_joint_trajectory(robot, task)
    assert result.task_description is not task
    planned_goal = result.task_description.goal.affordance
    task.goal.affordance = planned_goal + 1.0
    assert result.task_description.goal.affordance == planned_goal


def test_failed_plan_with_continuous_gripper_does_not_divide_by_zero():
    robot, task, config = scenario_cartesian_goal(cca)
    robot.gripper_state = 0.0
    task.goal.gripper = 0.5
    task.gripper_goal_type = cca.GripperGoalType.CONTINUOUS

    result = cca.CcAffordancePlannerInterface(config).generate_joint_trajectory(robot, task)

    assert not result.success
    assert result.includes_gripper_trajectory
    assert result.joint_trajectory == []
    assert compute_gripper_joint_trajectory(cca.GripperGoalType.CONTINUOUS, 0.0, 0.5, 1) == [0.5]


def test_urdf_rpy_uses_extrinsic_xyz_convention():
    rpy = np.array([0.2, -0.3, 0.4])
    actual = _origin_to_matrix(np.zeros(3), rpy)[:3, :3]
    expected = Rotation.from_euler("z", rpy[2]).as_matrix()
    expected = expected @ Rotation.from_euler("y", rpy[1]).as_matrix()
    expected = expected @ Rotation.from_euler("x", rpy[0]).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (cca.VirtualScrewOrder.XY, np.eye(3)[:, [0, 1]]),
        (cca.VirtualScrewOrder.YZ, np.eye(3)[:, [1, 2]]),
        (cca.VirtualScrewOrder.ZX, np.eye(3)[:, [2, 0]]),
    ],
)
def test_two_axis_virtual_screw_orders(order, expected):
    np.testing.assert_array_equal(get_vir_screw_axes(order), expected)


def test_canonical_pose_can_be_resolved_from_fk_without_redundant_goal_pose():
    robot, _, config = scenario_rotation_inverse(cca)
    task = cca.TaskDescription(cca.PlanningType.CARTESIAN_GOAL)
    task.canonical_pose_from.method = cca.PoseSpecificationMethod.FROM_FK
    task.canonical_pose_from.post_transform = np.eye(4)
    task.canonical_pose_from.post_transform[0, 3] = 0.05
    task.trajectory_density = 2

    result = cca.CcAffordancePlannerInterface(config).generate_joint_trajectory(robot, task)

    expected = cca.fkin_space(robot.M, robot.slist, robot.joint_states) @ task.canonical_pose_from.post_transform
    np.testing.assert_allclose(result.task_description.goal.canonical_pose, expected, atol=1e-12)
