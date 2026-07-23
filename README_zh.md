# 闭链可供性（CCA）纯 Python 规划器

本仓库提供原始 C++ Closed-Chain Affordance 规划器的纯 Python 复现。默认安装只使用 NumPy、SciPy 和 PyYAML，不需要编译 C++，也不依赖 Torch。

原始 C++ 实现仍保留在 `affordance_util/`、`cc_affordance_planner/` 和 `bindings/` 中，作为算法基准和等价测试参考。

## 安装

```bash
pip install -e .
```

如果需要 MeshCat 可视化：

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
    str(root / "python/x5_urdf_config.yaml"),
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
print(result.success, result.trajectory_description)
```

`result.joint_trajectory` 的第一个点是机器人起始状态。其后的点包含机器人关节以及闭链模型内部的虚拟关节状态；控制机器人时应按机器人实际关节数量截取前部元素。

## APPROACH 与 AFFORDANCE

- `PlanningType.CARTESIAN_GOAL`：生成到笛卡尔目标位姿的 APPROACH 轨迹。
- `MotionType.AFFORDANCE`：TCP 已接触目标后，沿旋转、平移或一般螺旋运动。
- `PlanningType.EE_ORIENTATION_ONLY`：保持 TCP 位置并调整末端姿态。

典型任务应先规划 APPROACH，再以到达状态构造新的 `RobotDescription` 并规划 AFFORDANCE。完整示例见 `python/demo_x5_urdf.py`。

```bash
# 只规划
python python/demo_x5_urdf.py

# 规划并显示 MeshCat 动画
python python/demo_x5_urdf.py --viewer
```

## 与 C++ 对照测试

构建 C++ reference 需要 Eigen、yaml-cpp、urdfdom 和 pybind11：

```bash
sudo apt install libyaml-cpp-dev liburdfdom-dev libeigen3-dev
cmake -S . -B build -DPython3_EXECUTABLE=$(which python)
cmake --build build -j4
mkdir -p tests/_cpp_ref
mv python/closed_chain_affordance.so tests/_cpp_ref/
pytest tests/test_cpp_python_equivalence.py -q
```

两套实现使用不同的线性代数后端：C++ 使用 Eigen 的 Complete Orthogonal Decomposition，Python 使用 NumPy SVD。因此等价测试采用浮点容差，而不是要求逐位相同。
