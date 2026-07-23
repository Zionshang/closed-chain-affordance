# Closed-Chain Affordance Planner

A pure-Python implementation of the Closed-Chain Affordance (CCA) planning framework. It uses NumPy and SciPy for kinematics and closed-chain inverse kinematics, PyYAML for robot configuration, and the Python standard library for URDF parsing.

CCA describes manipulation tasks using a screw axis, a location and a goal. Typical applications include turning a valve, pulling a drawer, following a Cartesian approach path and controlling end-effector orientation.

## Installation

```bash
pip install .
```

For editable development:

```bash
pip install -e .
```

MeshCat visualization is optional:

```bash
pip install -e ".[viewer]"
```

## Quick start

```python
from pathlib import Path

import numpy as np
import closed_chain_affordance as cca

root = Path.cwd()
robot = cca.build_robot_description_from_urdf(
    str(root / "assets/robot/x5/urdf/x5.urdf"),
    str(root / "examples/x5_urdf_config.yaml"),
    joint_states=np.zeros(6),
)

task = cca.TaskDescription()
task.affordance_info.type = cca.ScrewType.ROTATION
task.affordance_info.axis = cca.axis_to_vec(cca.Axis.X_MINUS)

tcp_pose = cca.fkin_space(robot.M, robot.slist, robot.joint_states)
task.affordance_info.location = tcp_pose[:3, 3] + np.array([0.0, 0.0, -0.08])
task.goal.affordance = np.pi
task.trajectory_density = 20

config = cca.PlannerConfig()
config.update_method = cca.UpdateMethod.INVERSE
result = cca.plan(robot, task, config)

print(result.success)
print(result.trajectory_description)
```

## Example

The x5 example plans a Cartesian approach followed by a valve rotation:

```bash
python examples/demo_x5_urdf.py
```

To animate the result in MeshCat:

```bash
python examples/demo_x5_urdf.py --viewer
```

The first point in `result.joint_trajectory` is the starting state. Later points also contain internal virtual closed-chain joints; select the first `n` values when sending commands to an `n`-joint robot.

## Project layout

```text
src/closed_chain_affordance/  Python package
examples/                     Example and viewer
assets/                       x5 URDF and meshes
pyproject.toml                Build and dependency configuration
```

## Reference

Panthi, Janak, Farshid Alambeigi, and Mitch Pryor. “A Closed-Chain Approach to Generating Affordance Joint Trajectories for Robotic Manipulators.” IEEE Transactions on Robotics, 2025.
