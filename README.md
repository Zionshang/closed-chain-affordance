# cca-planner

`cca-planner` is a Torch-native, batched closed-chain affordance trajectory
planner. One call plans different goals for `B` parallel environments and
returns dense tensors suitable for GPU simulation and reinforcement learning.

## Install

```bash
pip install -e .
```

Install the optional cabinet-demo viewer dependencies with:

```bash
pip install -e ".[viewer]"
```

The import name is `cca_planner`:

```python
import cca_planner as cca
```

## Minimal batched cabinet-door task

```python
from pathlib import Path
import math

import torch
import cca_planner as cca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

robot = cca.load_robot_from_urdf(
    Path("assets/robot/piper_l/urdf/piper_l_fixed_gripper.urdf"),
    Path("assets/robot/piper_l/urdf/cca_config.yaml"),
    device=device,
)
q0 = robot.joint_states.expand(batch_size, -1).clone()

# The cabinet URDF is placed at (0.70, 0, 0.20) with a pi yaw so its front
# faces the robot. These local points come from door_joint_1 and its handle.
cabinet_positions = torch.tensor((0.70, 0.0, 0.20), device=device).expand(batch_size, -1)
yaw_pi_signs = torch.tensor((-1.0, -1.0, 1.0), device=device)
hinge_local = torch.tensor((0.119816, -0.275279, 0.104384), device=device)
handle_local = torch.tensor((0.165816, -0.025555, 0.165822), device=device)
hinge_positions = cabinet_positions + yaw_pi_signs * hinge_local
handle_positions = cabinet_positions + yaw_pi_signs * handle_local
hinge_axes = torch.tensor((0.0, 0.0, 1.0), device=device).expand(batch_size, -1)
door_screws = cca.get_screw(cca.ScrewType.ROTATION, hinge_axes, hinge_positions)
open_angles = torch.linspace(-math.pi / 3, -math.pi / 4, batch_size, device=device)

planner = cca.PlannerInterface(
    cca.PlannerConfig(ik_max_itr=150),
    early_stopping=True,
    use_regularized_normal_equations=True,
)

# CCA starts at contact, so solve the cabinet-handle grasp pose first.
initial_pose = cca.fkin_space(robot.M, robot.slist, q0)
grasp_roll = torch.tensor(
    [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], device=device
)
grasp_pose = initial_pose.clone()
grasp_pose[:, :3, :3] = initial_pose[:, :3, :3] @ grasp_roll
grasp_pose[:, :3, 3] = handle_positions
grasp_joints, ik_success = planner.solve_pose_ik(
    robot_slist=robot.slist,
    robot_m=robot.M,
    joint_seed=q0,
    target_pose=grasp_pose,
)

result = planner.generate_joint_trajectory(
    robot_slist=robot.slist,
    robot_m=robot.M,
    joint_states=grasp_joints,
    affordance_screw=door_screws,
    goal_affordance=open_angles,
    trajectory_density=12,
    # The handle is vertical: allow grasp roll only around the handle's z axis.
    vir_screw_order=cca.VirtualScrewOrder.Z,
)

trajectory = result.joint_trajectory  # [B, 12, n]
full_success = ik_success & result.full_success
valid_steps = result.valid_mask       # [B, 11]
```

`PlannerConfig` controls convergence: `accuracy`,
`secondary_goal_min_magnitude`, `secondary_goal_abs_tolerance`,
`closure_err_threshold_ang`, `closure_err_threshold_lin`, and `ik_max_itr`.
Exact zero secondary goals remain zero; the independent
absolute tolerance prevents small goals from demanding unrealistic relative
precision. CCA always uses the inverse update and automatically switches to a
damped least-squares update near singularities. The interface exposes only two
execution choices: `early_stopping` and `use_regularized_normal_equations`.
Planner tensors must use `torch.float32`.

`PlannerInterface.solve_pose_ik()` provides batched damped-least-squares
endpoint IK and returns `(joint_states, converged)`. It can resolve a Cartesian
canonical pose independently of trajectory-planning success.

## Examples

- [`examples/demo_cabinet_door.py`](examples/demo_cabinet_door.py): Piper grasps
  the cabinet's vertical door handle, remains free to roll about that handle,
  and opens the door about its hinge.
- [`examples/demo_cabinet_drawer.py`](examples/demo_cabinet_drawer.py): Piper
  grasps drawer 1's horizontal handle, remains free to roll about the handle,
  and pulls the drawer along its rail.

Each script contains its complete minimal planning flow. Only the reusable
[`ViserVisualizer`](examples/visualizer.py) is shared. Run them with:

```bash
python examples/demo_cabinet_door.py
python examples/demo_cabinet_drawer.py
```

Install the development extra with `pip install -e ".[dev]"`, then run the two
4096-environment timing comparisons with:

```bash
pytest -s
```

Set `CCA_BENCHMARK_ENVIRONMENTS` or `CCA_BENCHMARK_REPEATS` to use a smaller
batch or fewer timed repetitions during quick local checks.
