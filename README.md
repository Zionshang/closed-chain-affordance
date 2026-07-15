# cca-planner

`cca-planner` is a Torch-native, batched closed-chain affordance trajectory
planner. One call plans different goals for `B` parallel environments and
returns dense tensors suitable for GPU simulation and reinforcement learning.

## Install

```bash
pip install -e .
```

Install the optional valve-demo viewer dependencies with:

```bash
pip install -e ".[viewer]"
```

The import name is `cca_planner`:

```python
import cca_planner as cca
```

## Minimal batched valve turn

```python
from pathlib import Path

import torch
import cca_planner as cca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
batch_size = 64

# Load the space screw axes (slist), home TCP pose (M), and initial arm joints.
robot = cca.load_robot_from_urdf(
    Path("assets/robot/x5/urdf/x5.urdf"),
    Path("examples/x5_urdf_config.yaml"),
    dtype=dtype,
    device=device,
)
slist = robot.slist                         # [6, n], robot joint screw axes
home_pose = robot.M                         # [4, 4], TCP pose at zero joints
q0 = robot.joint_states.expand(batch_size, -1).clone()  # [B, n]

# Current TCP positions are used to place one slightly different valve per env.
tcp = cca.fkin_space(home_pose, slist, q0)[:, :3, 3]     # [B, 3]
axis = torch.tensor([1., 0., 0.], device=device, dtype=dtype).expand(batch_size, -1)
radii = torch.linspace(0.05, 0.085, batch_size, device=device, dtype=dtype)
centres = tcp - radii[:, None] * torch.tensor([0., 0., 1.], device=device, dtype=dtype)
valve_screws = cca.get_screw(cca.ScrewType.ROTATION, axis, centres)  # [B, 6]
turn_angles = torch.linspace(0.6, 1.2, batch_size, device=device, dtype=dtype)

planner = cca.PlannerInterface(
    cca.PlannerConfig(ik_max_itr=50),
    fast_mode=True,              # fixed-iteration masked Torch loop
    compile=False,               # opt in only after benchmarking the target workload
    fast_linear_solver=True,     # regularized normal-equation solver
)
result = planner.generate_joint_trajectory(
    robot_slist=slist,            # shared [6, n] or batched [B, 6, n]
    robot_m=home_pose,            # shared [4, 4] or batched [B, 4, 4]
    joint_states=q0,              # initial arm state, [B, n]
    motion_type=cca.MotionType.AFFORDANCE,
    affordance_screw=valve_screws,# valve rotation screw for every environment
    goal_affordance=turn_angles,  # different requested turn angle per environment
    trajectory_density=12,        # output points including the initial state
    vir_screw_order=cca.VirtualScrewOrder.XYZ,
)

trajectory = result.joint_trajectory  # [B, 12, n]
full_success = result.full_success    # [B]
valid_steps = result.valid_mask       # [B, 11]
```

`PlannerConfig` controls convergence: `accuracy`,
`closure_err_threshold_ang`, `closure_err_threshold_lin`, `ik_max_itr`, and
`update_method`. By default the package uses fixed iterations
(`fast_mode=True`, no early stop), the fast linear solver, and no
`torch.compile`. Call `enable_chunked_early_stop()` if host-side convergence
checks are desirable.

## Examples

- [`examples/demo_valve.py`](examples/demo_valve.py): approach randomized valve
  rims, then turn each valve around its rotation screw.
- [`examples/demo_drawer.py`](examples/demo_drawer.py): approach randomized
  drawer handles, then pull each drawer along its translation screw.

Each script contains its complete minimal planning flow. Only the reusable
[`ViserVisualizer`](examples/visualizer.py) is shared. Run them with:

```bash
python examples/demo_valve.py
python examples/demo_drawer.py
```

Install the development extra with `pip install -e ".[dev]"`, then run
`pytest`.
