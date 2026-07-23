# 闭链可供性规划器

这是 Closed-Chain Affordance（CCA）规划框架的纯 Python 实现。运动学和闭链逆运动学使用 NumPy 与 SciPy，机器人配置使用 PyYAML，URDF 由 Python 标准库解析。

CCA 使用螺旋轴、轴位置和运动目标描述操作任务，可用于旋转阀门、拉动抽屉、笛卡尔接近以及末端姿态控制。

## 安装

```bash
pip install .
```

开发时可使用可编辑安装：

```bash
pip install -e .
```

MeshCat 可视化是可选功能：

```bash
pip install -e ".[viewer]"
```

## 基本用法

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

## 示例

x5 示例会先规划笛卡尔 APPROACH，再规划阀门旋转：

```bash
python examples/demo_x5_urdf.py
```

显示 MeshCat 动画：

```bash
python examples/demo_x5_urdf.py --viewer
```

`result.joint_trajectory` 的第一个点是机器人起始状态。后续点还包含闭链模型内部的虚拟关节；向具有 `n` 个关节的机器人发送命令时，应截取每个轨迹点的前 `n` 个值。

## 目录结构

```text
src/closed_chain_affordance/  Python 包
examples/                     示例和可视化工具
assets/                       x5 URDF 与网格资源
pyproject.toml                构建和依赖配置
```

## 论文

Panthi, Janak, Farshid Alambeigi, and Mitch Pryor. “A Closed-Chain Approach to Generating Affordance Joint Trajectories for Robotic Manipulators.” IEEE Transactions on Robotics, 2025.
