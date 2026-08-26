"""Benchmark regularized normal equations against the SVD reference path.

Run with output enabled::

    pytest -s tests/test_regularized_normal_equations.py

The default workload contains 4096 environments. Environment variables
``CCA_BENCHMARK_ENVIRONMENTS`` and ``CCA_BENCHMARK_REPEATS`` may override the
batch size and number of timed runs.
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


def _build_problem() -> tuple[
    dict[str, torch.Tensor | int | cca.VirtualScrewOrder], torch.Tensor
]:
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
    affordance_screws = cca.get_screw(cca.ScrewType.ROTATION, axes, centres)
    inputs = {
        "robot_slist": robot.slist,
        "robot_m": robot.M,
        "joint_states": joints,
        "affordance_screw": affordance_screws,
        "goal_affordance": goals,
        "trajectory_density": 5,
        "vir_screw_order": cca.VirtualScrewOrder.XYZ,
    }
    cc_slist = cca.compose_cc_model_slist(
        robot.slist.expand(NUM_ENVIRONMENTS, -1, -1),
        robot.M.expand(NUM_ENVIRONMENTS, -1, -1),
        joints,
        affordance_screws,
        cca.VirtualScrewOrder.XYZ,
    )
    return inputs, cc_slist


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


def _accuracy_metrics(
    result: cca.PlannerResult,
    cc_slist: torch.Tensor,
    goals: torch.Tensor,
) -> dict[str, float]:
    final_state = result.differential_trajectory[:, -1]
    affordance_error = (final_state[:, -1] + goals).abs()
    closure = cca.fkin_space(
        torch.eye(4, device=DEVICE), cc_slist, final_state
    )
    closure_twist = cca.se3_to_vec(cca.matrix_log6(closure))
    angular_error = closure_twist[:, :3].norm(dim=-1)
    linear_error = closure_twist[:, 3:].norm(dim=-1)
    return {
        "success_rate": float(result.full_success.float().mean()),
        "affordance_mean": float(affordance_error.mean()),
        "affordance_max": float(affordance_error.max()),
        "closure_angular_mean": float(angular_error.mean()),
        "closure_angular_max": float(angular_error.max()),
        "closure_linear_mean": float(linear_error.mean()),
        "closure_linear_max": float(linear_error.max()),
    }


def _print_metrics(name: str, duration: float, metrics: dict[str, float]) -> None:
    print(
        f"{name}: {duration:.6f}s; success={metrics['success_rate']:.2%}; "
        f"affordance error mean/max="
        f"{metrics['affordance_mean']:.3e}/{metrics['affordance_max']:.3e}; "
        f"closure angular mean/max="
        f"{metrics['closure_angular_mean']:.3e}/{metrics['closure_angular_max']:.3e}; "
        f"closure linear mean/max="
        f"{metrics['closure_linear_mean']:.3e}/{metrics['closure_linear_max']:.3e}"
    )


def test_regularized_normal_equations_timing_and_accuracy() -> None:
    inputs, cc_slist = _build_problem()
    config = cca.PlannerConfig(ik_max_itr=40)
    regularized_result, regularized_time = _measure(
        cca.PlannerInterface(
            config,
            early_stopping=False,
            use_regularized_normal_equations=True,
        ),
        inputs,
    )
    svd_result, svd_time = _measure(
        cca.PlannerInterface(
            config,
            early_stopping=False,
            use_regularized_normal_equations=False,
        ),
        inputs,
    )
    goals = inputs["goal_affordance"]
    assert isinstance(goals, torch.Tensor)
    regularized_metrics = _accuracy_metrics(regularized_result, cc_slist, goals)
    svd_metrics = _accuracy_metrics(svd_result, cc_slist, goals)

    assert regularized_time > 0.0 and svd_time > 0.0
    assert bool(torch.isfinite(regularized_result.joint_trajectory).all())
    assert bool(torch.isfinite(svd_result.joint_trajectory).all())

    print("\nRegularized-normal-equations benchmark")
    print(f"device={DEVICE}; environments={NUM_ENVIRONMENTS}; repeats={REPEATS}")
    _print_metrics("regularized normal equations", regularized_time, regularized_metrics)
    _print_metrics("SVD reference", svd_time, svd_metrics)
    print(f"SVD/regularized time ratio: {svd_time / regularized_time:.3f}x")


if __name__ == "__main__":
    test_regularized_normal_equations_timing_and_accuracy()
