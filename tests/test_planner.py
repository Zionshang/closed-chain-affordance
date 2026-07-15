from __future__ import annotations

import unittest
from pathlib import Path

import torch

import cca_planner as cca
import cca_planner.math as cm


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robot/x5/urdf/x5.urdf"
CONFIG = ROOT / "examples/x5_urdf_config.yaml"
REFERENCE_MODE = dict(fast_mode=False, compile=False, fast_linear_solver=False)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = cca.load_robot_from_urdf(URDF, CONFIG, dtype=torch.float64)

    def _turn_inputs(self, goals: torch.Tensor):
        batch = goals.numel()
        q = self.robot.joint_states.expand(batch, -1).clone()
        tcp = cca.fkin_space(self.robot.M, self.robot.slist, self.robot.joint_states)[:3, 3]
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype).expand(batch, -1)
        # Geometry is a deterministic function of the goal so that an item has
        # exactly the same task when planned alone or as part of a batch.
        radii = 0.05 + 0.03 * goals
        centres = tcp - radii[:, None] * torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype)
        return dict(
            robot_slist=self.robot.slist,
            robot_m=self.robot.M,
            joint_states=q,
            motion_type=cca.MotionType.AFFORDANCE,
            affordance_screw=cca.get_screw(cca.ScrewType.ROTATION, axis, centres),
            goal_affordance=goals,
            trajectory_density=5,
            vir_screw_order=cca.VirtualScrewOrder.XYZ,
        )

    def test_robot_loader_returns_torch_model(self):
        self.assertEqual(self.robot.slist.shape, (6, 6))
        self.assertEqual(self.robot.M.shape, (4, 4))
        self.assertEqual(self.robot.joint_states.shape, (6,))
        self.assertTrue(torch.is_tensor(self.robot.slist))
        torch.testing.assert_close(
            cca.fkin_space(self.robot.M, self.robot.slist, self.robot.joint_states),
            self.robot.M,
        )

    def test_virtual_screw_orders(self):
        expected = {
            cca.VirtualScrewOrder.XYZ: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            cca.VirtualScrewOrder.YZX: ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
            cca.VirtualScrewOrder.ZXY: ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
            cca.VirtualScrewOrder.XY: ((1, 0, 0), (0, 1, 0)),
            cca.VirtualScrewOrder.YZ: ((0, 1, 0), (1, 0, 0)),
            cca.VirtualScrewOrder.ZX: ((0, 1, 1), (0, 0, 0)),
        }
        location = torch.zeros(2, 3, dtype=torch.float64)
        for order, columns in expected.items():
            screws = cca.build_virtual_slist(order, location)
            wanted = torch.tensor(columns, dtype=torch.float64).T
            torch.testing.assert_close(screws[0, :3], wanted)

    def test_heterogeneous_batch_matches_individual_calls(self):
        goals = torch.tensor([0.2, 0.35, 0.5], dtype=torch.float64)
        config = cca.PlannerConfig(ik_max_itr=40, update_method=cca.UpdateMethod.INVERSE)
        batch = cca.PlannerInterface(config, **REFERENCE_MODE).generate_joint_trajectory(
            **self._turn_inputs(goals)
        )
        self.assertTrue(bool(batch.full_success.all()))
        for index, goal in enumerate(goals):
            single = cca.PlannerInterface(config, **REFERENCE_MODE).generate_joint_trajectory(
                **self._turn_inputs(goal.reshape(1))
            )
            torch.testing.assert_close(batch.joint_trajectory[index], single.joint_trajectory[0])

    def test_failed_points_return_final_candidates(self):
        goals = torch.tensor([0.4, 100.0], dtype=torch.float64)
        config = cca.PlannerConfig(ik_max_itr=30, update_method=cca.UpdateMethod.INVERSE)
        inputs = self._turn_inputs(goals)
        result = cca.PlannerInterface(config, **REFERENCE_MODE).generate_joint_trajectory(**inputs)
        self.assertEqual(result.full_success.tolist(), [True, False])
        self.assertEqual(result.valid_mask.sum(dim=-1).tolist(), [4, 0])
        failed = result.joint_trajectory[1, 1:]
        self.assertTrue(bool(torch.isfinite(failed).all()))
        self.assertTrue(bool((failed != inputs["joint_states"][1]).any()))

    def test_default_mode_is_fixed_iteration_regularized_solver(self):
        config = cca.PlannerConfig(ik_max_itr=8, update_method=cca.UpdateMethod.INVERSE)
        interface = cca.PlannerInterface(config)
        self.assertFalse(interface.planner_.early_stop_)
        self.assertTrue(interface.planner_.fast_solve_)
        self.assertFalse(interface.planner_._compile_requested_)
        result = interface.generate_joint_trajectory(
            **self._turn_inputs(torch.tensor([0.2, 0.3], dtype=torch.float64))
        )
        self.assertTrue(bool((result.executed_iterations == config.ik_max_itr).all()))

    def test_mixed_gripper_goal(self):
        inputs = self._turn_inputs(torch.tensor([0.2, 0.3], dtype=torch.float64))
        inputs["gripper_state"] = torch.tensor([0.1, 0.2], dtype=torch.float64)
        inputs["goal_gripper"] = torch.tensor([0.4, float("nan")], dtype=torch.float64)
        config = cca.PlannerConfig(ik_max_itr=40, update_method=cca.UpdateMethod.INVERSE)
        result = cca.PlannerInterface(config, **REFERENCE_MODE).generate_joint_trajectory(**inputs)
        self.assertTrue(result.includes_gripper)
        self.assertEqual(result.gripper_active_mask.tolist(), [True, False])
        torch.testing.assert_close(
            result.joint_trajectory[1, :, 6], torch.full((5,), 0.2, dtype=torch.float64)
        )

    def test_translation_affordance_for_drawer(self):
        batch = 2
        q = self.robot.joint_states.expand(batch, -1).clone()
        tcp = cca.fkin_space(self.robot.M, self.robot.slist, q)[:, :3, 3]
        pull_axes = torch.tensor([-1.0, 0.0, 0.0], dtype=q.dtype).expand(batch, -1)
        drawer_screws = cca.get_screw(cca.ScrewType.TRANSLATION, pull_axes, tcp)
        config = cca.PlannerConfig(
            ik_max_itr=80, update_method=cca.UpdateMethod.INVERSE
        )
        result = cca.PlannerInterface(config, **REFERENCE_MODE).generate_joint_trajectory(
            robot_slist=self.robot.slist,
            robot_m=self.robot.M,
            joint_states=q,
            motion_type=cca.MotionType.AFFORDANCE,
            affordance_screw=drawer_screws,
            goal_affordance=torch.tensor([0.08, 0.12], dtype=q.dtype),
            trajectory_density=5,
            vir_screw_order=cca.VirtualScrewOrder.XYZ,
        )
        self.assertTrue(bool(result.full_success.all()))
        self.assertEqual(result.joint_trajectory.shape, (batch, 5, 6))

    def test_converged_points_close_the_chain(self):
        goals = torch.tensor([0.2, 0.4], dtype=torch.float64)
        inputs = self._turn_inputs(goals)
        config = cca.PlannerConfig(ik_max_itr=40, update_method=cca.UpdateMethod.INVERSE)
        result = cca.PlannerInterface(config, **REFERENCE_MODE).generate_joint_trajectory(**inputs)
        cc_slist, _ = cca.compose_cc_model_slist(
            inputs["robot_slist"].expand(goals.numel(), -1, -1),
            inputs["robot_m"].expand(goals.numel(), -1, -1),
            inputs["joint_states"],
            inputs["affordance_screw"],
            vir_screw_order=inputs["vir_screw_order"],
        )
        points = result.differential_trajectory[result.valid_mask]
        slists = cc_slist[:, None].expand(-1, 4, -1, -1)[result.valid_mask]
        transforms = cm.fkin_space(torch.eye(4, dtype=torch.float64), slists, points)
        rho = cm.se3_to_vec(cm.matrix_log6(transforms))
        self.assertLess(float(rho[..., :3].norm(dim=-1).max()), 2e-4)
        self.assertLess(float(rho[..., 3:].norm(dim=-1).max()), 2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
