"""Subprocess worker for the C++ vs. Python equivalence tests.

Usage:
    python tests/_worker.py <cpp|py> <scenario_name>

Loads exactly one implementation (so the two BLAS-backed native libraries never
coexist in a single process), runs the named scenario, and prints the result as
a single JSON object on stdout. The parent test compares the JSON from the two
runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_cpp, load_py  # noqa: E402
from _scenarios import SCENARIOS  # noqa: E402


def run(impl: str, scenario_name: str) -> dict:
    cca = load_cpp() if impl == "cpp" else load_py()
    robot, task, cfg = SCENARIOS[scenario_name](cca)
    planner = cca.CcAffordancePlannerInterface(cfg)
    result = planner.generate_joint_trajectory(robot, task)

    trajectory = []
    for point in result.joint_trajectory:
        trajectory.append([float(x) for x in np_as_list(point)])

    return {
        "success": bool(result.success),
        "description": result.trajectory_description.name,
        "includes_gripper": bool(result.includes_gripper_trajectory),
        "trajectory": trajectory,
    }


def np_as_list(point):
    # Avoid importing numpy at module import time in the signature; the modules
    # above already imported it transitively.
    import numpy as np

    return np.asarray(point, dtype=float).ravel()


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in ("cpp", "py") or argv[2] not in SCENARIOS:
        print(
            f"Usage: {argv[0]} <cpp|py> <scenario> ; scenarios: {sorted(SCENARIOS)}",
            file=sys.stderr,
        )
        return 2
    try:
        payload = run(argv[1], argv[2])
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
