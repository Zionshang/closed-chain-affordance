"""Benchmark fixed-iteration CCA against whole-batch early stopping.

Run with output enabled::

    pytest -s tests/test_early_stopping.py

``CCA_BENCHMARK_ENVIRONMENTS`` and ``CCA_BENCHMARK_REPEATS`` can temporarily
reduce the workload; their defaults are 4096 environments and three timed runs.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import torch

import cca_planner as cca


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robot/x5/urdf/x5.urdf"
ROBOT_CONFIG = ROOT / "examples/x5_urdf_config.yaml"
NUM_ENVIRONMENTS = int(os.getenv("CCA_BENCHMARK_ENVIRONMENTS", "4096"))
REPEATS = int(os.getenv("CCA_BENCHMARK_REPEATS", "3"))
DEVICE = torch.device(
    os.getenv("CCA_BENCHMARK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
)


def _synchronize() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _build_inputs() -> dict[str, torch.Tensor | int | cca.VirtualScrewOrder]:
    robot = cca.load_robot_from_urdf(URDF, ROBOT_CONFIG, device=DEVICE)
    joints = robot.joint_states.expand(NUM_ENVIRONMENTS, -1).clone()
    tcp = cca.fkin_space(robot.M, robot.slist, joints)[:, :3, 3]
    axes = torch.tensor((1.0, 0.0, 0.0), device=DEVICE).expand(
        NUM_ENVIRONMENTS, -1
    )
    goals = torch.linspace(0.15, 0.35, NUM_ENVIRONMENTS, device=DEVICE)
    radii = torch.linspace(0.05, 0.08, NUM_ENVIRONMENTS, device=DEVICE)
    centres = tcp - radii[:, None] * torch.tensor(
        (0.0, 0.0, 1.0), device=DEVICE
    )
    return {
        "robot_slist": robot.slist,
        "robot_m": robot.M,
        "joint_states": joints,
        "affordance_screw": cca.get_screw(cca.ScrewType.ROTATION, axes, centres),
        "goal_affordance": goals,
        "trajectory_density": 5,
        "vir_screw_order": cca.VirtualScrewOrder.XYZ,
    }


def _measure(
    planner: cca.PlannerInterface,
    inputs: dict[str, torch.Tensor | int | cca.VirtualScrewOrder],
) -> tuple[cca.PlannerResult, float]:
    planner.generate_joint_trajectory(**inputs)
    durations = []
    result = None
    for _ in range(REPEATS):
        _synchronize()
        start = time.perf_counter()
        result = planner.generate_joint_trajectory(**inputs)
        _synchronize()
        durations.append(time.perf_counter() - start)
    assert result is not None
    return result, statistics.median(durations)


def test_early_stopping_timing() -> None:
    inputs = _build_inputs()
    config = cca.PlannerConfig(ik_max_itr=40)
    fixed_result, fixed_time = _measure(
        cca.PlannerInterface(config, early_stopping=False), inputs
    )
    early_result, early_time = _measure(
        cca.PlannerInterface(config, early_stopping=True), inputs
    )

    torch.testing.assert_close(
        early_result.joint_trajectory, fixed_result.joint_trajectory
    )
    assert torch.equal(early_result.full_success, fixed_result.full_success)
    assert bool(
        (early_result.executed_iterations <= fixed_result.executed_iterations).all()
    )

    success_rate = float(early_result.full_success.float().mean())
    fixed_iterations = float(fixed_result.executed_iterations.float().mean())
    early_iterations = float(early_result.executed_iterations.float().mean())
    print("\nEarly-stopping benchmark")
    print(f"device={DEVICE}; environments={NUM_ENVIRONMENTS}; repeats={REPEATS}")
    print(
        f"fixed iterations: {fixed_time:.6f}s; "
        f"mean executed iterations={fixed_iterations:.2f}"
    )
    print(
        f"early stopping:   {early_time:.6f}s; "
        f"mean executed iterations={early_iterations:.2f}"
    )
    print(f"fixed/early time ratio: {fixed_time / early_time:.3f}x")
    print(f"full success rate: {success_rate:.2%}")


if __name__ == "__main__":
    test_early_stopping_timing()
