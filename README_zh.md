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

B = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
robot = cca.build_robot_description_from_urdf(
    "assets/robot/x5/urdf/x5.urdf",
    "python/x5_urdf_config.yaml",
    joint_states=np.zeros(6),
)

q = torch.tensor(robot.joint_states, device=device, dtype=torch.float32).expand(B, -1).clone()
slist = torch.tensor(robot.slist, device=device, dtype=q.dtype)       # [6, n]，批量共享
home = torch.tensor(robot.M, device=device, dtype=q.dtype)           # [4, 4]，批量共享

axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=q.dtype).expand(B, -1)
tcp = torch.tensor(
    cca.fkin_space(robot.M, robot.slist, robot.joint_states)[:3, 3],
    device=device,
    dtype=q.dtype,
)
radii = torch.linspace(0.05, 0.085, B, device=device, dtype=q.dtype)
centres = tcp - radii[:, None] * torch.tensor(
    [0.0, 0.0, 1.0], device=device, dtype=q.dtype
)
screws = torch.cat([axis, torch.cross(centres, axis, dim=-1)], dim=-1)
goals = torch.linspace(np.pi / 3, np.pi, B, device=device, dtype=q.dtype)

planner = cca.BatchedCcAffordancePlannerInterface()
result = planner.generate_joint_trajectory(
    robot_slist=slist,
    robot_m=home,
    joint_states=q,
    motion_type=cca.MotionType.AFFORDANCE,
    affordance_screw=screws,
    goal_affordance=goals,
    trajectory_density=12,
    vir_screw_order=cca.VirtualScrewOrder.XYZ,
)

trajectory = result.joint_trajectory   # [B, T, n_robot]
valid = result.full_success            # [B]
```

两阶段 `APPROACH -> AFFORDANCE` 示例见 `python/demo_valve_batch.py`。

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

默认模式使用 SVD 伪逆并逐迭代检查收敛，最接近 NumPy/C++ 参考行为。

| 方法 | 功能 |
|---|---|
| `enable_fast_linear_solver()` | 用正则化法方程替代两处 SVD；通常更快，但属于近似数值后端。 |
| `enable_chunked_early_stop(check_interval=4, fast_solve=False)` | 每若干次迭代检查一次批量收敛，减少 GPU/CPU 同步。 |
| `enable_fast_mode(compile=True, fast_solve=False)` | 固定执行 `ik_max_itr`，可配合 `torch.compile`；适合重复使用同一形状的批量任务。 |

这些开关只改变 Torch 的数值执行方式，不改变 APPROACH、AFFORDANCE 或闭链任务定义。
