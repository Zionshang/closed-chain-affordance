"""Regression tests for batched-only semantics and throughput modes.

These tests use ``unittest`` so they can run in the lightweight ``cca``
environment without installing pytest::

    conda run -n cca python tests/test_batched_features.py
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import closed_chain_affordance as cca  # noqa: E402
import closed_chain_affordance.batched_math as bm  # noqa: E402
import closed_chain_affordance.batched_planner as bp  # noqa: E402
from _scenarios import SCENARIOS, build_x5_robot  # noqa: E402
from test_batched_vs_numpy import scenario_to_batched_kwargs  # noqa: E402


class BatchedFeatureTests(unittest.TestCase):
    def test_float32_small_angle_log_keeps_sub_acos_resolution(self):
        # In float32, trace(R) rounds to exactly 3 for angles at this scale.
        # An acos(trace) implementation therefore returns zero even though the
        # skew part still contains an accurate first-order rotation signal.
        rotation_vector = torch.tensor([3e-5, -4e-5, 2e-5], dtype=torch.float32)
        rotation = bm.matrix_exp3(bm.vec_to_so3(rotation_vector))
        self.assertEqual(float(torch.trace(rotation)), 3.0)
        recovered = bm.so3_log_map(rotation)
        torch.testing.assert_close(recovered, rotation_vector, atol=2e-7, rtol=2e-4)

    def test_float32_cartesian_approach_matches_numpy_full_success(self):
        robot = build_x5_robot(cca)
        robot.joint_states = np.zeros(6)
        start = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
        target = start.copy()
        target[:3, 3] += np.array([0.20, 0.0, 0.20])

        affordance = cca.ScrewInfo()
        affordance.type = cca.ScrewType.ROTATION
        affordance.axis = np.ones(3) / np.sqrt(3.0)
        affordance.location = np.ones(3)
        task = cca.TaskDescription()
        task.motion_type = cca.MotionType.APPROACH
        task.vir_screw_order = cca.VirtualScrewOrder.NONE
        task.affordance_info = affordance
        task.goal.affordance = 1e-5
        task.goal.canonical_pose = target
        task.trajectory_density = 10
        cfg = cca.PlannerConfig()
        cfg.update_method = cca.UpdateMethod.INVERSE
        cfg.ik_max_itr = 200

        reference = cca.CcAffordancePlannerInterface(cfg).generate_joint_trajectory(robot, task)
        self.assertEqual(reference.trajectory_description, cca.TrajectoryDescription.FULL)

        result = cca.BatchedCcAffordancePlannerInterface(cfg).generate_joint_trajectory(
            robot_slist=torch.tensor(robot.slist, dtype=torch.float32),
            robot_m=torch.tensor(robot.M, dtype=torch.float32),
            joint_states=torch.tensor(robot.joint_states, dtype=torch.float32),
            motion_type=task.motion_type,
            affordance_screw=torch.tensor(cca.get_screw(affordance), dtype=torch.float32),
            goal_affordance=torch.tensor(1e-5, dtype=torch.float32),
            trajectory_density=task.trajectory_density,
            vir_screw_order=task.vir_screw_order,
            canonical_pose=torch.tensor(target, dtype=torch.float32),
        )
        self.assertTrue(bool(result.full_success[0]))

    def test_virtual_screw_orders_match_cpp_matrices(self):
        expected = {
            cca.VirtualScrewOrder.XYZ: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            cca.VirtualScrewOrder.YZX: np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            cca.VirtualScrewOrder.ZXY: np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            cca.VirtualScrewOrder.XY: np.eye(3)[:, [0, 1]],
            # Exact matrices from the original C++ comma initialisers.
            cca.VirtualScrewOrder.YZ: np.array([[0, 1], [1, 0], [0, 0]]),
            cca.VirtualScrewOrder.ZX: np.array([[0, 0], [1, 0], [1, 0]]),
        }
        q = torch.zeros(2, 3, dtype=torch.float64)
        for order, axes in expected.items():
            np.testing.assert_array_equal(cca.get_vir_screw_axes(order), axes)
            screws = bp.build_virtual_slist(order, q)
            np.testing.assert_array_equal(screws[0, :3].numpy(), axes)

    def test_unbatched_inputs_are_rank_broadcast_even_when_size_equals_batch(self):
        robot, task, cfg = SCENARIOS["rotation_inverse"](cca)
        cfg.ik_max_itr = 20
        batch = 6  # deliberately equals the leading dimension of an unbatched slist
        screw = torch.tensor(cca.get_screw(task.affordance_info), dtype=torch.float64)
        result = cca.BatchedCcAffordancePlannerInterface(cfg).generate_joint_trajectory(
            robot_slist=torch.tensor(robot.slist, dtype=torch.float64),
            robot_m=torch.tensor(robot.M, dtype=torch.float64),
            joint_states=torch.tensor(np.broadcast_to(robot.joint_states, (batch, 6)).copy()),
            motion_type=task.motion_type,
            affordance_screw=screw,
            goal_affordance=torch.full((batch,), task.goal.affordance, dtype=torch.float64),
            trajectory_density=3,
            vir_screw_order=task.vir_screw_order,
        )
        self.assertEqual(result.joint_trajectory.shape[:2], (batch, 3))

    def test_mixed_gripper_nan_holds_start_without_nan_output(self):
        robot, task, cfg = SCENARIOS["rotation_inverse"](cca)
        cfg.ik_max_itr = 30
        kwargs = scenario_to_batched_kwargs(robot, task, cfg, 2, torch.float64)
        kwargs["trajectory_density"] = 3
        kwargs["gripper_state"] = torch.tensor([0.1, 0.2], dtype=torch.float64)
        kwargs["goal_gripper"] = torch.tensor([0.4, float("nan")], dtype=torch.float64)
        result = cca.BatchedCcAffordancePlannerInterface(cfg).generate_joint_trajectory(**kwargs)
        self.assertTrue(result.includes_gripper)
        self.assertEqual(result.gripper_active_mask.tolist(), [True, False])
        self.assertFalse(bool(torch.isnan(result.joint_trajectory).any()))
        torch.testing.assert_close(
            result.joint_trajectory[1, :, robot.joint_states.size],
            torch.full((3,), 0.2, dtype=torch.float64),
        )

    def test_best_selects_expected_result_per_environment(self):
        robot, task, cfg = SCENARIOS["rotation_inverse"](cca)
        cfg.ik_max_itr = 40
        kwargs = scenario_to_batched_kwargs(robot, task, cfg, 3, torch.float64)
        kwargs["trajectory_density"] = 4
        core = cca.BatchedCcAffordancePlanner(cfg)

        cc, _ = bp.compose_cc_model_slist(
            kwargs["robot_slist"], kwargs["robot_m"], kwargs["joint_states"],
            kwargs["affordance_screw"], None, kwargs["vir_screw_order"],
        )
        goals = torch.tensor([[0.2], [0.4], [0.8]], dtype=torch.float64)
        inverse = core.generate_motion_joint_trajectory(
            cc, goals, 1, 4, update_method=cca.UpdateMethod.INVERSE
        )
        transpose = core.generate_motion_joint_trajectory(
            cc, goals, 1, 4, update_method=cca.UpdateMethod.TRANSPOSE
        )
        best = core.generate_motion_joint_trajectory(
            cc, goals, 1, 4, update_method=cca.UpdateMethod.BEST
        )
        expected = core._select_best(inverse, transpose)
        torch.testing.assert_close(best.joint_trajectory, expected.joint_trajectory)
        self.assertTrue(torch.equal(best.valid_mask, expected.valid_mask))
        self.assertTrue(torch.equal(best.description_codes, expected.description_codes))

    def test_heterogeneous_batch_matches_individual_numpy_plans(self):
        robot, task, cfg = SCENARIOS["rotation_inverse"](cca)
        goals = [0.2, 0.35, 0.5]
        kwargs = scenario_to_batched_kwargs(robot, task, cfg, len(goals), torch.float64)
        kwargs["goal_affordance"] = torch.tensor(goals, dtype=torch.float64)
        result = cca.BatchedCcAffordancePlannerInterface(cfg).generate_joint_trajectory(**kwargs)

        for env, goal in enumerate(goals):
            scalar_task = copy.deepcopy(task)
            scalar_task.goal.affordance = goal
            reference = cca.CcAffordancePlannerInterface(cfg).generate_joint_trajectory(
                robot, scalar_task
            )
            valid_points = result.joint_trajectory[env, 1:][result.valid_mask[env]]
            torch.testing.assert_close(
                valid_points,
                torch.tensor(
                    np.asarray(reference.joint_trajectory[1:])[:, : robot.joint_states.size],
                    dtype=torch.float64,
                ),
                atol=1e-6,
                rtol=1e-5,
            )

    def test_failed_environment_is_masked_without_corrupting_successful_one(self):
        robot, task, cfg = SCENARIOS["rotation_inverse"](cca)
        cfg.ik_max_itr = 30
        kwargs = scenario_to_batched_kwargs(robot, task, cfg, 2, torch.float64)
        kwargs["goal_affordance"] = torch.tensor([0.4, 100.0], dtype=torch.float64)
        result = cca.BatchedCcAffordancePlannerInterface(cfg).generate_joint_trajectory(**kwargs)

        self.assertEqual(result.full_success.tolist(), [True, False])
        self.assertEqual(result.valid_mask.sum(dim=-1).tolist(), [4, 0])
        # Invalid points expose the final IK candidates instead of being replaced
        # by the start/last-valid state.  The initial point itself is still exact.
        torch.testing.assert_close(result.joint_trajectory[1, 0], kwargs["joint_states"][1])
        failed_candidates = result.joint_trajectory[1, 1:]
        self.assertTrue(bool(torch.isfinite(failed_candidates).all()))
        self.assertTrue(bool((failed_candidates != kwargs["joint_states"][1]).any()))
        # One unresolved environment determines whole-batch latency.
        self.assertTrue(bool((result.executed_iterations == cfg.ik_max_itr).all()))

    def test_fast_solver_reports_work_and_preserves_closure(self):
        robot, task, cfg = SCENARIOS["rotation_inverse"](cca)
        cfg.ik_max_itr = 40
        kwargs = scenario_to_batched_kwargs(robot, task, cfg, 8, torch.float32)
        kwargs["goal_affordance"] = torch.linspace(0.2, 0.6, 8)
        iface = cca.BatchedCcAffordancePlannerInterface(cfg).enable_fast_linear_solver()
        result = iface.generate_joint_trajectory(**kwargs)
        self.assertTrue(bool(result.full_success.all()))
        self.assertTrue(bool((result.executed_iterations <= cfg.ik_max_itr).all()))
        self.assertTrue(torch.equal(result.full_success, result.valid_mask.all(dim=-1)))

        cc, _ = bp.compose_cc_model_slist(
            kwargs["robot_slist"], kwargs["robot_m"], kwargs["joint_states"],
            kwargs["affordance_screw"], None, kwargs["vir_screw_order"],
        )
        points = result.differential_trajectory[result.valid_mask]
        slists = cc[:, None].expand(-1, result.valid_mask.shape[1], -1, -1)[result.valid_mask]
        tse = bm.fkin_space(torch.eye(4, dtype=points.dtype), slists, points)
        rho = (
            bm.adjoint(tse)
            @ bm.se3_to_vec(bm.matrix_log6(bm.trans_inv(tse))).unsqueeze(-1)
        ).squeeze(-1)
        self.assertLess(float(rho[:, 3:].norm(dim=-1).max()), 5e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_float32_smoke(self):
        robot, task, cfg = SCENARIOS["rotation_inverse"](cca)
        kwargs = scenario_to_batched_kwargs(robot, task, cfg, 16, torch.float32)
        kwargs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        result = (
            cca.BatchedCcAffordancePlannerInterface(cfg)
            .enable_fast_linear_solver()
            .generate_joint_trajectory(**kwargs)
        )
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(result.joint_trajectory).all()))
        self.assertTrue(bool(result.success.all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
