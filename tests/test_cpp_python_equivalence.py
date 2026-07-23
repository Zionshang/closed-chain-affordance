"""Macro equivalence tests: compiled C++ binding vs. pure-Python implementation.

Each planning scenario is executed in TWO separate subprocesses (one importing
only the C++ binding, one importing only the pure-Python package). Splitting
them across processes avoids the BLAS/LAPACK conflict that occurs when two
native linear-algebra backends (Eigen via the binding, OpenBLAS via NumPy) share
a single interpreter, which otherwise makes NumPy's SVD intermittently fail to
converge. The two JSON results are then compared.

Run standalone:

    python tests/test_cpp_python_equivalence.py

or with pytest (if installed):

    pytest tests/test_cpp_python_equivalence.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import CPP_SO, load_both  # noqa: E402
from _scenarios import NONDETERMINISTIC, SCENARIOS  # noqa: E402

pytestmark = pytest.mark.skipif(
    not CPP_SO.exists(),
    reason="C++ reference binding is not staged under tests/_cpp_ref",
)

TESTS_DIR = Path(__file__).resolve().parent
WORKER = TESTS_DIR / "_worker.py"

# Tolerance for joint trajectories. The IK loop relies on pseudoinverses, and
# the two implementations use different linear-algebra backends: the C++ binding
# uses Eigen's complete-orthogonal-decomposition pseudoinverse, while the
# pure-Python path uses NumPy's SVD pseudoinverse. The Moore-Penrose
# pseudoinverse is mathematically unique, so on well-conditioned matrices the
# two agree to ~1e-13; near singularities (which the closed-chain IK routinely
# skirts) the algorithms round off differently, and over a trajectory of ~20
# Newton-Raphson steps the differences accumulate to ~1e-7. A real divergence
# (wrong sign, wrong formula, off-by-one) shows up at O(0.1) or larger, so this
# tolerance catches bugs while tolerating cross-backend round-off.
JOINT_ATOL = 1e-6
JOINT_RTOL = 1e-5
STRICT_ATOL = 1e-9

CPP_PUBLIC_API = {
    "Axis", "PoseSpecificationMethod", "GripperGoalType", "ScrewType", "VirtualScrewOrder",
    "EeOrientationConstraint", "PlanningType", "MotionType", "TrajectoryDescription", "UpdateMethod",
    "VecInfo", "PoseFrom", "ScrewInfoFrom", "ScrewInfo", "RobotDescription", "Goal",
    "TaskDescription", "PlannerConfig", "PlannerResult", "CcAffordancePlannerInterface",
    "axis_to_vec", "get_screw", "fkin_space", "plan", "build_robot_description_from_yaml",
    "build_robot_description_from_urdf",
}


def _require_cpp_binding():
    if not CPP_SO.exists():
        raise FileNotFoundError(
            f"C++ binding not found at {CPP_SO}. Build it with "
            "`cmake -S . -B build -DPython3_EXECUTABLE=$(which python) && cmake --build build -j4`, "
            "then move python/closed_chain_affordance.so into tests/_cpp_ref/."
        )


def _run_worker(impl: str, scenario: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(WORKER), impl, scenario],
        capture_output=True,
        text=True,
        cwd=str(TESTS_DIR.parent),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker({impl}, {scenario}) exited {proc.returncode}\nstderr:\n{proc.stderr}"
        )
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return json.loads(last_line)


def _compare(cpp: dict, py: dict, scenario: str, *, nondeterministic: bool = False):
    if "error" in cpp:
        raise AssertionError(f"[{scenario}] C++ worker error: {cpp['error']}")
    if "error" in py:
        raise AssertionError(f"[{scenario}] Python worker error: {py['error']}")

    assert cpp["success"] == py["success"], f"[{scenario}] success mismatch"
    assert cpp["description"] == py["description"], (
        f"[{scenario}] description mismatch: {cpp['description']} vs {py['description']}"
    )
    assert cpp["includes_gripper"] == py["includes_gripper"], (
        f"[{scenario}] includes_gripper mismatch"
    )

    assert len(cpp["trajectory"]) == len(py["trajectory"]), (
        f"[{scenario}] trajectory length mismatch: cpp={len(cpp['trajectory'])} py={len(py['trajectory'])}"
    )

    if nondeterministic:
        # BEST mode: the winning method depends on scheduling; only structural
        # properties are guaranteed to match.
        return

    if not cpp["trajectory"]:
        return
    cpp_arr = np.array(cpp["trajectory"], dtype=float)
    py_arr = np.array(py["trajectory"], dtype=float)
    assert cpp_arr.shape == py_arr.shape, f"[{scenario}] point shape mismatch"
    max_diff = float(np.max(np.abs(cpp_arr - py_arr))) if cpp_arr.size else 0.0
    assert np.allclose(cpp_arr, py_arr, atol=JOINT_ATOL, rtol=JOINT_RTOL), (
        f"[{scenario}] joint trajectory differs (max abs diff={max_diff:.3e})"
    )


# --------------------------------------------------------------------------- #
# Free-function checks (lightweight; safe to run in-process).
# --------------------------------------------------------------------------- #
def test_public_api_contract():
    cca_cpp, cca_py = load_both()
    assert set(cca_py.__all__) == CPP_PUBLIC_API
    assert CPP_PUBLIC_API.issubset(set(dir(cca_cpp)))

    enum_names = [
        "Axis", "PoseSpecificationMethod", "GripperGoalType", "ScrewType", "VirtualScrewOrder",
        "EeOrientationConstraint", "PlanningType", "MotionType", "TrajectoryDescription", "UpdateMethod",
    ]
    for enum_name in enum_names:
        cpp_enum = getattr(cca_cpp, enum_name)
        py_enum = getattr(cca_py, enum_name)
        for expected_value, member_name in enumerate(py_enum.__members__):
            assert getattr(py_enum, member_name).value == expected_value
            assert getattr(cpp_enum, member_name).value == expected_value


def test_axis_to_vec():
    cca_cpp, cca_py = load_both()
    for axis_name in ["X", "Y", "Z", "X_MINUS", "Y_MINUS", "Z_MINUS", "ORIGIN"]:
        v_cpp = np.asarray(cca_cpp.axis_to_vec(getattr(cca_cpp.Axis, axis_name)), dtype=float)
        v_py = np.asarray(cca_py.axis_to_vec(getattr(cca_py.Axis, axis_name)), dtype=float)
        np.testing.assert_allclose(v_py, v_cpp, atol=STRICT_ATOL)


def test_get_screw():
    import math

    cca_cpp, cca_py = load_both()
    cases = [
        ("ROTATION", np.array([0.0, 0.0, 1.0]), np.array([0.1, -0.2, 0.3]), float("nan")),
        ("TRANSLATION", np.array([1.0, 0.0, 0.0]), np.zeros(3), float("nan")),
        ("SCREW", np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.5),
    ]
    for stype_name, axis, location, pitch in cases:
        si_cpp = cca_cpp.ScrewInfo()
        si_cpp.type = getattr(cca_cpp.ScrewType, stype_name)
        si_cpp.axis = axis
        si_cpp.location = location
        if not math.isnan(pitch):
            si_cpp.pitch = pitch
        si_py = cca_py.ScrewInfo()
        si_py.type = getattr(cca_py.ScrewType, stype_name)
        si_py.axis = axis
        si_py.location = location
        if not math.isnan(pitch):
            si_py.pitch = pitch
        np.testing.assert_allclose(
            np.asarray(cca_py.get_screw(si_py), dtype=float),
            np.asarray(cca_cpp.get_screw(si_cpp), dtype=float),
            atol=STRICT_ATOL,
        )


def test_fkin_space():
    cca_cpp, cca_py = load_both()
    from _scenarios import ur5_slist_and_home

    slist, home = ur5_slist_and_home()
    thetalist = np.array([0.3, -0.7, 0.4, 0.2, -0.5, 0.6])
    np.testing.assert_allclose(
        np.asarray(cca_py.fkin_space(home, slist, thetalist), dtype=float),
        np.asarray(cca_cpp.fkin_space(home, slist, thetalist), dtype=float),
        atol=STRICT_ATOL,
    )


def test_urdf_robot_builder():
    cca_cpp, cca_py = load_both()
    repo_root = TESTS_DIR.parent
    urdf_path = str(repo_root / "assets" / "robot" / "x5" / "urdf" / "x5.urdf")
    config_path = str(repo_root / "python" / "x5_urdf_config.yaml")

    rd_cpp = cca_cpp.build_robot_description_from_urdf(urdf_path, config_path, joint_states=np.zeros(6))
    rd_py = cca_py.build_robot_description_from_urdf(urdf_path, config_path, joint_states=np.zeros(6))
    np.testing.assert_allclose(np.asarray(rd_py.slist), np.asarray(rd_cpp.slist), atol=STRICT_ATOL,
                               err_msg="URDF slist mismatch")
    np.testing.assert_allclose(np.asarray(rd_py.M), np.asarray(rd_cpp.M), atol=STRICT_ATOL,
                               err_msg="URDF M mismatch")


# --------------------------------------------------------------------------- #
# Planning scenarios (run in isolated subprocesses).
# --------------------------------------------------------------------------- #
def _make_scenario_test(name, nondeterministic=False):
    def _test():  # noqa: ANN202
        _require_cpp_binding()
        cpp = _run_worker("cpp", name)
        py = _run_worker("py", name)
        _compare(cpp, py, name, nondeterministic=nondeterministic)

    _test.__name__ = f"test_scenario_{name}"
    return _test


# Dynamically create one test function per scenario.
for _name in SCENARIOS:
    globals()[f"test_scenario_{_name}"] = _make_scenario_test(_name, nondeterministic=(_name in NONDETERMINISTIC))


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #
def _collect_tests():
    tests = [
        ("public_api_contract", test_public_api_contract),
        ("axis_to_vec", test_axis_to_vec),
        ("get_screw", test_get_screw),
        ("fkin_space", test_fkin_space),
        ("urdf_robot_builder", test_urdf_robot_builder),
    ]
    for name in SCENARIOS:
        tests.append((f"scenario:{name}", globals()[f"test_scenario_{name}"]))
    return tests


def _run_all() -> bool:
    _require_cpp_binding()
    passed = failed = 0
    for label, test in _collect_tests():
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {label}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  PASS  {label}")
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
