# Torch 批量 CCA 规划器

本项目以原始 C++ CCA 实现为基准，提供 NumPy 单任务实现和 Torch 批量实现。Torch 版本使用同一套闭链任务描述和 IK 算法，只增加批量 Tensor 输入、GPU 执行和批量诊断信息。

```bash
pip install -e .
```

## 最小示例

```python
import numpy as np
import torch
import closed_chain_affordance as cca

# 一次并行规划 128 个阀门任务；B 是 batch size，不是机器人关节数。
B = 128

# 所有输入 Tensor 必须位于同一设备并使用同一种 dtype。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从 URDF 和配置文件读取机器人运动学模型。
# robot 包含：
#   robot.slist        各机器人关节在零位时的空间螺旋轴，形状 [6, n_robot]
#   robot.M            所有关节为 0 时，TCP 相对机器人基座的位姿，形状 [4, 4]
#   robot.joint_states 当前机器人关节角，形状 [n_robot]
robot = cca.build_robot_description_from_urdf(
    "assets/robot/x5/urdf/x5.urdf",
    "python/x5_urdf_config.yaml",
    joint_states=np.zeros(6),
)

# q[b] 是第 b 个环境的机器人起始关节角。本例让 128 台机器人都从零位开始。
q = torch.tensor(robot.joint_states, device=device, dtype=torch.float32)
q = q.expand(B, -1).clone()                                         # [B, n_robot]

# slist 的每一列是一个 6 维关节 screw [角速度部分; 线速度部分]。
# 它描述“每个机器人关节绕哪根轴运动”，不是轨迹，也不是当前姿态下的 Jacobian。
# 这里所有环境使用同一种机器人，所以直接共享一份 [6, n_robot] 模型即可。
slist = torch.tensor(robot.slist, device=device, dtype=q.dtype)       # [6, n_robot]

# home 是机器人零关节角时的 TCP 齐次变换矩阵：前三列是方向，最后一列是位置。
# 它同样由所有环境共享；规划器会结合 home、slist 和 q 计算当前 TCP 位姿。
home = torch.tensor(robot.M, device=device, dtype=q.dtype)           # [4, 4]

# axis[b] 是第 b 个阀门的旋转轴方向，必须是单位向量。
# [1, 0, 0] 表示阀门绕机器人基座坐标系的 +x 方向旋转。
axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=q.dtype)
axis = axis.expand(B, -1)                                           # [B, 3]

# TCP（Tool Center Point）是机械臂末端实际执行任务的参考点。
# fkin_space 用前向运动学计算当前 q 下的 TCP 位姿；[:3, 3] 取出其 xyz 位置。
tcp = torch.tensor(
    cca.fkin_space(robot.M, robot.slist, robot.joint_states)[:3, 3],
    device=device,
    dtype=q.dtype,
)                                                                      # [3]

# radii[b] 是第 b 个阀门的物理半径，单位为米；本例从 5 cm 到 8.5 cm。
radii = torch.linspace(0.05, 0.085, B, device=device, dtype=q.dtype) # [B]

# grasp_direction 表示从阀门圆心指向抓取点的方向。
# 本例把当前 TCP 当作阀门顶部的抓取点，因此满足：
#   TCP = valve_center + radius * grasp_direction
grasp_direction = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=q.dtype)

# centres[b] 是第 b 个阀门的圆心，也是阀门旋转轴经过的空间点。
# 由上面的几何关系反推出：valve_center = TCP - radius * grasp_direction。
centres = tcp - radii[:, None] * grasp_direction                     # [B, 3]

# 纯旋转任务的 6 维 screw 为 [w; c × w]：
#   w = axis          阀门转轴方向
#   c = centres       转轴经过的点
# screws[b] 因而完整描述“第 b 个阀门绕空间中的哪根轴旋转”。
screws = torch.cat([axis, torch.cross(centres, axis, dim=-1)], dim=-1)

# goals[b] 是第 b 个阀门希望转过的角度，单位为弧度；这里为 60° 到 180°。
goals = torch.linspace(np.pi / 3, np.pi, B, device=device, dtype=q.dtype)

# 高层 Torch 批量接口：内部会构造闭链模型并并行求解 B 条关节轨迹。
# 默认启用 fast mode 和正则化线性求解，并关闭 early stop；torch.compile 单独选择。
planner = cca.BatchedCcAffordancePlannerInterface()
result = planner.generate_joint_trajectory(
    robot_slist=slist,                         # 机器人关节 screw 模型
    robot_m=home,                              # 机器人零位时的 TCP 位姿
    joint_states=q,                            # 每个环境的起始关节角
    motion_type=cca.MotionType.AFFORDANCE,     # 已接触阀门，规划沿阀门转轴运动
    affordance_screw=screws,                   # 每个阀门的旋转 screw
    goal_affordance=goals,                     # 每个阀门的目标转角
    trajectory_density=12,                     # 每条输出轨迹包含 12 个关节状态
    vir_screw_order=cca.VirtualScrewOrder.XYZ, # 允许三个虚拟末端姿态自由度
)

# trajectory[b, t] 是环境 b 在轨迹点 t 的绝对机器人关节角。
trajectory = result.joint_trajectory                  # [B, 12, n_robot]

# full_success[b] 表示环境 b 的所有 11 个 IK 步骤是否全部收敛。
full_success = result.full_success                     # [B]

# valid_mask[b, t] 可进一步查看每一个离散 IK 步骤是否收敛。
valid_mask = result.valid_mask                         # [B, 11]
```

这个例子从 `AFFORDANCE` 阶段开始，因此假设 TCP 已经位于阀门边缘并与阀门建立了接触。阀门离机械臂较远时，应先规划 `APPROACH`；两阶段示例见 `python/demo_valve_batch.py`。

几个最容易混淆的名字：

| 名字 | 物理含义 | 是否随批次变化 |
|---|---|---|
| `slist` | 机器人自身各关节的 6 维空间螺旋轴，决定机器人运动学结构。 | 本例共享，也可传 `[B,6,n_robot]`。 |
| `home` / `robot.M` | 机器人零位时 TCP 相对基座的完整位姿。 | 本例共享，也可传 `[B,4,4]`。 |
| `tcp` | 当前关节角下末端工具中心点的位置，不是阀门圆心。 | 本例所有环境相同。 |
| `axis` | 阀门的旋转轴方向，例如 `[1,0,0]` 表示沿基座 `+x`。 | 可以为每个阀门分别指定。 |
| `radii` | 阀门半径；用于由抓取点反推阀门圆心，不直接传给规划器。 | 每个阀门不同。 |
| `centres` | 阀门圆心，同时是旋转轴经过的一点。 | 每个阀门不同。 |
| `screws` | 由 `axis` 和 `centres` 合成的完整阀门旋转约束。 | 每个阀门不同。 |
| `goals` | 每个阀门最终需要旋转的角度。 | 每个阀门不同。 |

## Torch 特有公开接口

| 接口 | 功能 |
|---|---|
| `BatchedCcAffordancePlannerInterface` | 推荐的高层批量接口；接收机器人状态、任务和目标 Tensor，返回绝对关节轨迹。 |
| `BatchedCcAffordancePlanner` | 低层批量 IK；输入已经组合好的闭链 screw matrix 和 secondary goals。 |
| `plan_batch(...)` | 使用默认 `PlannerConfig` 创建高层接口并规划一次。 |
| `compose_cc_model_slist_batched(...)` | 批量构造闭链模型；高级用法。 |
| `BatchedPlannerResult` | 高层批量规划结果。 |
| `BatchedMotionResult` | 低层闭链差分轨迹和求解诊断。 |

原有 `MotionType`、`VirtualScrewOrder`、`UpdateMethod` 和 `PlannerConfig` 与 C++/NumPy 共用，不是 Torch 专属语义。

## Torch 特有输入规则

以下参数可以逐环境不同：

| 参数 | 形状 | 含义 |
|---|---|---|
| `joint_states` | `[B, n]` | 每个环境的当前关节角。 |
| `affordance_screw` | `[B, 6]` | 每个环境的任务 screw。 |
| `goal_affordance` | `[B]` | 每个环境的任务目标。 |
| `canonical_pose` | `[B, 4, 4]` | `APPROACH` 的目标位姿。 |
| `goal_ee_orientation` | `[B, k]` | 可选虚拟 EE 关节目标。 |
| `gripper_state` / `goal_gripper` | `[B]` | 可选夹爪状态和目标；`goal_gripper=NaN` 表示该环境保持初值。 |

`robot_slist` 和 `robot_m` 可传共享的 `[6,n]`、`[4,4]`，也可传批量形式。所有 Tensor 必须使用相同 device 和 dtype。

一个批次内必须共享 `motion_type`、`trajectory_density`、`vir_screw_order`、`gripper_goal_type` 和 `update_method`。

## Torch 特有结果字段

| 字段 | 功能 |
|---|---|
| `joint_trajectory [B,T,n]` | 定长绝对关节轨迹；失败点保留该次 IK 的最后候选解，必须结合 `valid_mask` 使用。 |
| `valid_mask [B,T-1]` | 每个离散 IK 点是否收敛。 |
| `success [B]` | 至少一个轨迹点成功。 |
| `full_success [B]` | 所有轨迹点都成功。 |
| `differential_trajectory` | 闭链内部差分轨迹，用于诊断。 |
| `active_iterations [B,T-1]` | 每个环境、每个轨迹点实际参与的 IK 迭代数。 |
| `executed_iterations [T-1]` | 每个轨迹点整个批次实际执行的迭代数。 |
| `gripper_active_mask [B]` | 哪些环境启用了夹爪目标。 |

## Torch 特有执行开关

默认配置面向批量吞吐：启用固定迭代 `fast_mode`、启用正则化线性求解，并关闭 `early_stop`。已经收敛的环境仍会由 mask 冻结，但整个批次固定执行 `ik_max_itr` 次。`torch.compile` 默认关闭，因为首次编译成本可能远大于小矩阵规划本身；需要重复复用同一形状时可显式开启。

| 方法 | 功能 |
|---|---|
| `enable_fast_linear_solver(enabled=True)` | 开关正则化法方程；默认开启。它替代两处 SVD，通常更快，但属于近似数值后端。 |
| `enable_fast_mode(compile=True, fast_solve=True)` | 固定执行 `ik_max_itr` 并关闭 early stop；默认已处于 fast mode，但构造器默认 `compile=False`。显式调用本方法会请求懒编译。 |
| `enable_chunked_early_stop(check_interval=4, fast_solve=True)` | 在第一次规划前切回 early stop；每若干次迭代检查一次批量收敛。 |

这些开关只改变 Torch 的数值执行方式，不改变 APPROACH、AFFORDANCE 或闭链任务定义。

需要与 C++/NumPy 做精确数值对照时，在构造时显式使用 reference 配置：

```python
planner = cca.BatchedCcAffordancePlannerInterface(
    fast_mode=False,            # 恢复逐迭代 early stop
    compile=False,              # 不使用 torch.compile
    fast_linear_solver=False,   # 恢复 SVD 伪逆
)
```
