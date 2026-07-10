"""Equivalence tests: batched PyTorch planner vs. the single-shot NumPy reference.

Two levels of comparison:

1. **Core planner** — the closed-chain model is composed once with the NumPy
   helpers, then fed to BOTH the NumPy single-shot solver and the batched
   solver. Their differential trajectories are compared (isolates IK + stepping).

2. **Full interface** — the batched interface is driven from the same scenario
   inputs and its absolute robot trajectory / success / description are compared
   against the single-shot interface (covers composition + conversion too).

Scenarios are split by numerical character:

* **STRICT** — well-conditioned problems. Matched point-by-point to a tight
  tolerance (the IK tracks the reference to ~1e-7 over the whole trajectory).
* **UNSET** — infeasible problems. Both implementations must agree on the
  failure (success=False, UNSET, empty trajectory); no joint comparison.
* **SENSITIVE** — near-singular problems (e.g. ``x5_cartesian_approach``). The
  closed-chain IK skirks a singularity, so the *exact* set of converged steps is
  not reproducible across linear-algebra backends (a ~1e-9 per-iteration
  difference compounds and flips which steps converge — the same reason the
  repo's own test marks BEST mode nondeterministic). For these we check that
  both implementations agree on success / description *category* and that every
  converged point the torch planner returns actually satisfies the closed-chain
  closure constraint (an implementation-independent correctness check).

Run standalone::

    python tests/test_batched_vs_numpy.py

or with pytest::

    pytest tests/test_batched_vs_numpy.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import closed_chain_affordance as cca  # noqa: E402
import closed_chain_affordance.batched_math as bm  # noqa: E402
from _scenarios import (  # noqa: E402
    build_x5_robot,
    SCENARIOS,
    ur5_slist_and_home,
    build_robot,
    build_config,
)

JOINT_ATOL = 1e-6
JOINT_RTOL = 1e-5
B = 5  # batch size used to exercise batching
DTYPE = torch.float64  # parity in float64; float32 GPU smoke tested separately


# --------------------------------------------------------------------------- #
# Extra well-conditioned scenarios (APPROACH motion with real affordance goals;
# the UR5 / x5 cartesian-approach scenarios are near-singular, see SENSITIVE).
# --------------------------------------------------------------------------- #
def _x5_approach(offset, affordance, density):
    robot = build_x5_robot(cca)
    cur = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
    canonical = cur.copy()
    canonical[:3, 3] += np.asarray(offset, dtype=float)
    aff = cca.ScrewInfo()
    aff.type = cca.ScrewType.ROTATION
    aff.axis = cca.axis_to_vec(cca.Axis.Z)
    aff.location = cur[:3, 3]
    task = cca.TaskDescription()
    task.motion_type = cca.MotionType.APPROACH
    task.vir_screw_order = cca.VirtualScrewOrder.NONE
    task.goal.canonical_pose = canonical
    task.trajectory_density = density
    task.goal.affordance = affordance
    task.affordance_info = aff
    return robot, task, build_config(cca, update_method=cca.UpdateMethod.INVERSE)


def scenario_x5_approach_full(cca_mod):
    return _x5_approach([0.1, 0.0, 0.1], 0.6, 8)


def scenario_x5_approach_full_2(cca_mod):
    return _x5_approach([0.15, 0.05, 0.1], 0.8, 10)


EXTRA_SCENARIOS = {
    "x5_approach_full": scenario_x5_approach_full,
    "x5_approach_full_2": scenario_x5_approach_full_2,
}

ALL_SCENARIOS = {**SCENARIOS, **EXTRA_SCENARIOS}

STRICT = {
    "rotation_inverse", "rotation_transpose", "rotation_with_gripper",
    "translation", "screw", "best",
    "x5_approach_full", "x5_approach_full_2",
}
UNSET = {"rotation_with_ee_orientation", "cartesian_goal"}
SENSITIVE = {"x5_cartesian_approach"}


# --------------------------------------------------------------------------- #
# Scenario -> batched tensor inputs
# --------------------------------------------------------------------------- #
def scenario_to_batched_kwargs(robot, task, cfg, B, dtype):
    aff_screw = np.asarray(cca.get_screw(task.affordance_info), dtype=float)
    joint_states = np.asarray(robot.joint_states, dtype=float)

    def rep(x):
        t = torch.tensor(np.asarray(x, dtype=float), dtype=dtype)
        return t.unsqueeze(0).expand(B, *t.shape).contiguous()

    kwargs = dict(
        robot_slist=torch.tensor(np.asarray(robot.slist, dtype=float), dtype=dtype)
        .unsqueeze(0).expand(B, *robot.slist.shape).contiguous(),
        robot_m=torch.tensor(np.asarray(robot.M, dtype=float), dtype=dtype)
        .unsqueeze(0).expand(B, 4, 4).contiguous(),
        joint_states=rep(joint_states),
        motion_type=task.motion_type,
        affordance_screw=rep(aff_screw),
        goal_affordance=rep(np.asarray(task.goal.affordance, dtype=float)),
        trajectory_density=int(task.trajectory_density),
        vir_screw_order=task.vir_screw_order,
        gripper_goal_type=task.gripper_goal_type,
    )
    ee = np.asarray(task.goal.ee_orientation, dtype=float)
    if ee.size > 0:
        kwargs["goal_ee_orientation"] = rep(ee)
    if task.motion_type == cca.MotionType.APPROACH:
        cp = np.asarray(task.goal.canonical_pose, dtype=float)
        kwargs["canonical_pose"] = (
            torch.tensor(cp, dtype=dtype).unsqueeze(0).expand(B, 4, 4).contiguous()
        )
    if not np.isnan(task.goal.gripper):
        kwargs["goal_gripper"] = rep(np.asarray(task.goal.gripper, dtype=float))
        if not np.isnan(robot.gripper_state):
            kwargs["gripper_state"] = rep(np.asarray(robot.gripper_state, dtype=float))
    return kwargs


def _scenario_inputs(name):
    robot, task, cfg = ALL_SCENARIOS[name](cca)
    if cfg.update_method == cca.UpdateMethod.BEST:
        cfg.update_method = cca.UpdateMethod.INVERSE  # BEST is scheduling-dependent
    return robot, task, cfg


# --------------------------------------------------------------------------- #
# NumPy core-level reference
# --------------------------------------------------------------------------- #
def _numpy_core_result(robot, task, cfg):
    aff = cca.ScrewInfo(
        type=task.affordance_info.type,
        axis=np.asarray(task.affordance_info.axis, dtype=float),
        location=np.asarray(task.affordance_info.location, dtype=float),
        screw=np.asarray(task.affordance_info.screw, dtype=float),
        pitch=task.affordance_info.pitch,
    )
    has_approach = task.motion_type == cca.MotionType.APPROACH
    canonical = task.goal.canonical_pose if has_approach else None
    cc_model = cca.compose_cc_model_slist(robot, aff, canonical, task.vir_screw_order)

    if has_approach:
        cc_slist = cc_model.slist
        approach_limit = np.asarray(cc_model.approach_limit, dtype=float)
        secondary = np.concatenate([
            np.asarray(task.goal.ee_orientation, dtype=float).reshape(-1),
            np.array([approach_limit]), np.array([task.goal.affordance]),
        ])
    else:
        cc_slist = np.asarray(cc_model, dtype=float)
        secondary = np.concatenate([
            np.asarray(task.goal.ee_orientation, dtype=float).reshape(-1),
            np.array([task.goal.affordance]),
        ])
    tau = secondary.size

    method = cfg.update_method if cfg.update_method != cca.UpdateMethod.BEST else cca.UpdateMethod.INVERSE
    cls = {cca.UpdateMethod.INVERSE: cca.CcAffordancePlannerInverse,
           cca.UpdateMethod.TRANSPOSE: cca.CcAffordancePlannerTranspose}[method]
    planner = cls(cfg)
    gen = (planner.generate_approach_motion_joint_trajectory if has_approach
           else planner.generate_affordance_motion_joint_trajectory)
    res = gen(cc_slist, secondary, tau, int(task.trajectory_density))
    return res, cc_slist, secondary, tau, has_approach, method


# --------------------------------------------------------------------------- #
# Core planner comparison (STRICT + UNSET; SENSITIVE handled at interface level)
# --------------------------------------------------------------------------- #
def compare_core(name):
    robot, task, cfg = _scenario_inputs(name)
    res_np, cc_slist, secondary, tau, has_approach, method = _numpy_core_result(robot, task, cfg)

    cc_t = torch.tensor(np.broadcast_to(cc_slist, (B, *cc_slist.shape)).copy(), dtype=DTYPE)
    sec_t = torch.tensor(np.broadcast_to(secondary, (B, *secondary.shape)).copy(), dtype=DTYPE)
    planner = cca.BatchedCcAffordancePlanner(cfg)
    out = planner.generate_motion_joint_trajectory(
        cc_t, sec_t, tau, int(task.trajectory_density),
        has_approach=has_approach, update_method=method,
    )

    if name in UNSET:
        assert not res_np.success and len(res_np.joint_trajectory) == 0, f"[{name}/core] expected UNSET"
        assert bool((out.description_codes == 0).all()), f"[{name}/core] torch expected UNSET"
        return 0.0

    np_traj = np.asarray(res_np.joint_trajectory, dtype=float) if res_np.joint_trajectory else np.zeros((0, cc_slist.shape[0]))
    valid = out.valid_mask.numpy()
    maxdiff = 0.0
    for env in range(B):
        torch_pts = out.joint_trajectory[env].numpy()[valid[env]]
        assert np_traj.shape == torch_pts.shape, (
            f"[{name}/core env{env}] shape np {np_traj.shape} vs torch {torch_pts.shape}"
        )
        d = float(np.max(np.abs(np_traj - torch_pts))) if np_traj.size else 0.0
        maxdiff = max(maxdiff, d)
        assert np.allclose(np_traj, torch_pts, atol=JOINT_ATOL, rtol=JOINT_RTOL), (
            f"[{name}/core env{env}] diff trajectory differs (max abs diff={d:.3e})"
        )
    expected = {cca.TrajectoryDescription.UNSET: 0, cca.TrajectoryDescription.PARTIAL: 1,
                cca.TrajectoryDescription.FULL: 2}[res_np.trajectory_description]
    assert bool((out.description_codes == expected).all()), (
        f"[{name}/core] description np={res_np.trajectory_description} torch={out.description_codes.tolist()}"
    )
    return maxdiff


# --------------------------------------------------------------------------- #
# Interface-level comparison
# --------------------------------------------------------------------------- #
def _closure_residual(cc_slist_t, thetalist_t):
    eye4 = torch.eye(4, dtype=cc_slist_t.dtype, device=cc_slist_t.device)
    tse = bm.fkin_space(eye4, cc_slist_t, thetalist_t)
    rho = (bm.adjoint(tse) @ bm.se3_to_vec(bm.matrix_log6(bm.trans_inv(tse))).unsqueeze(-1)).squeeze(-1)
    return rho.norm(dim=-1)  # [...]


def compare_interface(name):
    robot, task, cfg = _scenario_inputs(name)
    ref = cca.CcAffordancePlannerInterface(cfg).generate_joint_trajectory(robot, task)
    n_robot = int(np.asarray(robot.joint_states).size)
    has_gripper = ref.includes_gripper_trajectory
    n_out = n_robot + (1 if has_gripper else 0)
    np_traj = (np.asarray(ref.joint_trajectory, dtype=float)[:, :n_out]
               if ref.joint_trajectory else np.zeros((0, n_out)))

    kwargs = scenario_to_batched_kwargs(robot, task, cfg, B, DTYPE)
    result = cca.BatchedCcAffordancePlannerInterface(cfg).generate_joint_trajectory(**kwargs)
    assert result.includes_gripper == has_gripper, f"[{name}/iface] gripper flag mismatch"

    valid = result.valid_mask.numpy()
    for env in range(B):
        assert bool(result.success[env]) == ref.success, (
            f"[{name}/iface env{env}] success np={ref.success} torch={bool(result.success[env])}"
        )
        assert result.description[env] == ref.trajectory_description, (
            f"[{name}/iface env{env}] description np={ref.trajectory_description} torch={result.description[env]}"
        )

        traj = result.joint_trajectory[env].numpy()
        if name in UNSET:
            assert int(valid[env].sum()) == 0, f"[{name}/iface env{env}] UNSET should have no valid steps"
            continue
        if name in SENSITIVE:
            continue  # point-parity checked separately (see compare_interface_sensitive)

        torch_valid_pts = traj[1:][valid[env]][:, :n_out]
        assert np_traj.shape[0] == 1 + torch_valid_pts.shape[0], (
            f"[{name}/iface env{env}] length np={np_traj.shape[0]} torch={1 + torch_valid_pts.shape[0]}"
        )
        assert np.allclose(np_traj[0], traj[0, :n_out], atol=JOINT_ATOL, rtol=JOINT_RTOL), (
            f"[{name}/iface env{env}] start state differs"
        )
        d = float(np.max(np.abs(np_traj[1:] - torch_valid_pts))) if torch_valid_pts.size else 0.0
        assert np.allclose(np_traj[1:], torch_valid_pts, atol=JOINT_ATOL, rtol=JOINT_RTOL), (
            f"[{name}/iface env{env}] abs trajectory differs (max abs diff={d:.3e})"
        )
    return 0.0


def compare_interface_sensitive(name):
    """For SENSITIVE scenarios: agree on success/description category, and every
    converged differential point satisfies the closed-chain closure constraint."""
    robot, task, cfg = _scenario_inputs(name)
    ref = cca.CcAffordancePlannerInterface(cfg).generate_joint_trajectory(robot, task)
    kwargs = scenario_to_batched_kwargs(robot, task, cfg, B, DTYPE)
    result = cca.BatchedCcAffordancePlannerInterface(cfg).generate_joint_trajectory(**kwargs)

    for env in range(B):
        assert bool(result.success[env]) == ref.success, f"[{name} env{env}] success mismatch"
        # same description CATEGORY (FULL vs not-FULL). PARTIAL/UNSET exact step
        # counts are not comparable across backends for near-singular problems.
        ref_full = ref.trajectory_description == cca.TrajectoryDescription.FULL
        assert (result.description[env] == cca.TrajectoryDescription.FULL) == ref_full, (
            f"[{name} env{env}] FULL-ness mismatch"
        )

    # Independent correctness check: converged IK points close the chain.
    diff = result.differential_trajectory  # [B, steps, n_cc]
    valid = result.valid_mask  # [B, steps]
    eps_lin = cfg.closure_err_threshold_lin
    cc_slist_t = _interface_cc_slist(robot, task)
    worst = 0.0
    for env in range(B):
        pts = diff[env][valid[env]]  # [k, n_cc]
        if pts.shape[0] == 0:
            continue
        rho = _closure_residual(cc_slist_t[env], pts)  # per-env slist; pts is [k, n_cc]
        worst = max(worst, float(rho.max()))
        # closure residual magnitude (twist norm) must be at the linear threshold scale
        assert float(rho.max()) < max(1e-3, 10 * eps_lin), (
            f"[{name} env{env}] converged point violates closure: rho={float(rho.max()):.3e}"
        )
    return worst


def _interface_cc_slist(robot, task):
    aff = cca.ScrewInfo(
        type=task.affordance_info.type, axis=np.asarray(task.affordance_info.axis, float),
        location=np.asarray(task.affordance_info.location, float),
        screw=np.asarray(task.affordance_info.screw, float), pitch=task.affordance_info.pitch)
    canonical = task.goal.canonical_pose if task.motion_type == cca.MotionType.APPROACH else None
    cc = cca.compose_cc_model_slist(robot, aff, canonical, task.vir_screw_order)
    cc_slist = cc.slist if hasattr(cc, "slist") else np.asarray(cc)
    return torch.tensor(np.broadcast_to(np.asarray(cc_slist, float), (B, *cc_slist.shape)).copy(), dtype=DTYPE)


# --------------------------------------------------------------------------- #
# Math primitive spot-check
# --------------------------------------------------------------------------- #
def test_batched_math_vs_numpy():
    import closed_chain_affordance.math_utils as nm
    rng = np.random.default_rng(0)
    for _ in range(20):
        V = rng.standard_normal(6) * 0.5
        T = nm.matrix_exp6(nm.vec_to_se3(V))
        t_T = torch.tensor(T, dtype=DTYPE)
        np.testing.assert_allclose(
            bm.matrix_exp6(bm.vec_to_se3(torch.tensor(V, dtype=DTYPE))).numpy(), T, atol=1e-9)
        np.testing.assert_allclose(
            bm.se3_to_vec(bm.matrix_log6(t_T)).numpy(), nm.se3_to_vec(nm.matrix_log6(T)), atol=1e-7)
        np.testing.assert_allclose(bm.adjoint(t_T).numpy(), nm.adjoint(T), atol=1e-9)
        np.testing.assert_allclose(bm.trans_inv(t_T).numpy(), nm.trans_inv(T), atol=1e-9)


def _run_all():
    passed = failed = 0

    def run(label, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS  {label}")
            return True
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {label}: {e}")
            return False

    print("== batched math ==")
    run("math primitives", test_batched_math_vs_numpy)

    print("== core planner (STRICT) ==")
    for name in STRICT:
        d = run(f"core  {name}", lambda _n=name: (compare_core(_n), None)[1])
        if d:
            pass  # pass

    print("== core planner (UNSET) ==")
    for name in UNSET:
        run(f"core  {name} (unset)", lambda _n=name: compare_core(_n))

    print("== full interface (STRICT) ==")
    for name in STRICT:
        run(f"iface {name}", lambda _n=name: compare_interface(_n))

    print("== full interface (UNSET) ==")
    for name in UNSET:
        run(f"iface {name} (unset)", lambda _n=name: compare_interface(_n))

    print("== full interface (SENSITIVE) ==")
    for name in SENSITIVE:
        run(f"iface {name} (sensitive)", lambda _n=name: compare_interface_sensitive(_n))

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
