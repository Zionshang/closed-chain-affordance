# cca-planner

`cca-planner` 是一个完全基于 Torch 的批量闭链可供性轨迹规划包。一次调用可以输入
`B` 组不同的机器人状态、目标和物体位置，并直接返回适合 GPU 仿真及强化学习使用的
批量轨迹 Tensor。

## 安装

在仓库根目录执行：

```bash
pip install -e .
```

如需运行带 Viser 可视化的 cabinet Demo：

```bash
pip install -e ".[viewer]"
```

安装后的导入名是：

```python
import cca_planner as cca
```

## 最小示例：批量打开 cabinet door

```python
from pathlib import Path
import math

import torch
import cca_planner as cca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 64

# 从 URDF 构造规划器所需的机器人运动学描述。
robot = cca.load_robot_from_urdf(
    Path("assets/robot/piper_l/urdf/piper_l_fixed_gripper.urdf"),
    Path("assets/robot/piper_l/urdf/cca_config.yaml"),
    device=device,
)
q0 = robot.joint_states.expand(B, -1).clone()

# Cabinet URDF 放在 (0.70, 0, 0.20)，并绕 z 轴旋转 pi 使正面朝向机器人。
# 下面两个局部坐标直接来自 door_joint_1 与 door 把手。
cabinet_positions = torch.tensor((0.70, 0.0, 0.20), device=device).expand(B, -1)
yaw_pi_signs = torch.tensor((-1.0, -1.0, 1.0), device=device)
hinge_local = torch.tensor((0.119816, -0.275279, 0.104384), device=device)
handle_local = torch.tensor((0.165816, -0.025555, 0.165822), device=device)
hinge_positions = cabinet_positions + yaw_pi_signs * hinge_local
handle_positions = cabinet_positions + yaw_pi_signs * handle_local
hinge_axes = torch.tensor((0.0, 0.0, 1.0), device=device).expand(B, -1)
door_screws = cca.get_screw(cca.ScrewType.ROTATION, hinge_axes, hinge_positions)
open_angles = torch.linspace(-math.pi / 3, -math.pi / 4, B, device=device)

config = cca.PlannerConfig(ik_max_itr=150)
planner = cca.PlannerInterface(
    config,
    early_stopping=True,
    use_regularized_normal_equations=True,
)

# CCA 从接触状态开始，因此先用 Pose IK 求解把手抓取位姿。
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
    # Door 把手竖直，因此抓取只允许绕把手的 z 轴转动。
    vir_screw_order=cca.VirtualScrewOrder.Z,
)

trajectory = result.joint_trajectory  # [B, 12, n]，绝对机器人关节轨迹
full_success = ik_success & result.full_success
valid_steps = result.valid_mask       # [B, 11]，逐目标点的收敛状态
```

## 收敛与 Torch 执行参数

`PlannerConfig` 是任务无关的统一收敛配置：

- `accuracy=0.1`：离散目标的相对容差。
- `secondary_goal_min_magnitude=1e-5`：仅对非零次级目标施加的最小幅值。
- `secondary_goal_abs_tolerance=1e-5`：独立于目标幅值的绝对容差下限；精确零目标仍保持为零。
- `closure_err_threshold_ang=1e-4`：闭链旋转残差阈值。
- `closure_err_threshold_lin=1e-5`：闭链平移残差阈值，单位为米。
- `ik_max_itr=200`：每个轨迹点的最大 IK 迭代次数。

CCA 始终使用 inverse 更新，并在接近奇异时根据条件数自动切换到阻尼最小二乘。
接口只保留两个 Torch 执行选项：

- `early_stopping=False`：默认固定迭代次数，并用布尔 mask 冻结已收敛环境；设为 `True`
  后在整个批次收敛时提前结束，但 GPU 每轮都需要同步到主机检查。
- `use_regularized_normal_equations=True`：默认用正则化法方程代替两处 SVD 伪逆乘积。

规划器输入统一要求使用 `torch.float32`。失败时返回张量仍保留每个目标点的最后候选解，
是否真正收敛必须以 `valid_mask` / `full_success` 为准。

`PlannerInterface.solve_pose_ik()` 提供批量阻尼最小二乘末端位姿 IK，返回
`(joint_states, converged)`，可独立于轨迹规划结果求解 canonical pose 对应的关节状态。

## 示例

- [`examples/demo_cabinet_door.py`](examples/demo_cabinet_door.py)：Piper 抓住
  cabinet 的竖直 door 把手，允许绕把手轴转动，再绕铰链打开门。
- [`examples/demo_cabinet_drawer.py`](examples/demo_cabinet_drawer.py)：Piper 抓住
  drawer 1 的水平把手，允许绕把手轴转动，再沿滑轨拉开抽屉。
- [`examples/demo_valve.py`](examples/demo_valve.py)：Piper 抓取随机位置的阀门
  轮缘顶部，并沿旋转螺旋完成半圈转动。

每个脚本都独立包含完整的最小规划流程；只有机器人网格、物体几何、轨迹着色和动画
共用 [`ViserVisualizer`](examples/visualizer.py)。

```bash
python examples/demo_cabinet_door.py
python examples/demo_cabinet_drawer.py
python examples/demo_valve.py
```

先执行 `pip install -e ".[dev]"` 安装测试依赖，再运行三个默认 4096 环境的性能对比：

```bash
pytest -s
```

快速本地检查时，可以通过 `CCA_BENCHMARK_ENVIRONMENTS` 和
`CCA_BENCHMARK_REPEATS` 减少环境数与计时重复次数。
