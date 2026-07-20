# cca-planner

`cca-planner` 是一个完全基于 Torch 的批量闭链可供性轨迹规划包。一次调用可以输入
`B` 组不同的机器人状态、目标和物体位置，并直接返回适合 GPU 仿真及强化学习使用的
批量轨迹 Tensor。

## 安装

在仓库根目录执行：

```bash
pip install -e .
```

如需运行带 Viser 可视化的阀门 Demo：

```bash
pip install -e ".[viewer]"
```

安装后的导入名是：

```python
import cca_planner as cca
```

## 最小示例：批量旋转阀门

```python
from pathlib import Path

import torch
import cca_planner as cca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
B = 64  # 并行环境数；以下每个环境都可以有不同的阀门和目标

# 从 URDF 构造规划器所需的机器人运动学描述。
robot = cca.load_robot_from_urdf(
    Path("assets/robot/x5/urdf/x5.urdf"),
    Path("examples/x5_urdf_config.yaml"),
    dtype=dtype,
    device=device,
)

# slist：空间螺旋轴矩阵，形状 [6, n]。每一列描述一个机器人关节的
# 旋转/平移轴及其空间位置，规划器用它计算正运动学和雅可比矩阵。
slist = robot.slist

# M（这里命名为 home_pose）：所有关节为零时，TCP（工具中心点/末端）
# 相对机器人基座的 4x4 齐次变换矩阵。
home_pose = robot.M

# q0：每个环境的初始关节角，形状 [B, n]。这里先复制相同零位；在 Isaac Lab
# 中可直接传入每个环境当前的机器人关节位置。
q0 = robot.joint_states.expand(B, -1).clone()

# tcp：由 slist、M 和 q0 算出的当前末端位置，形状 [B, 3]。
tcp = cca.fkin_space(home_pose, slist, q0)[:, :3, 3]

# axis：每个阀门的转轴方向，形状 [B, 3]；这里都绕基座坐标系 +x 轴旋转。
axis = torch.tensor([1., 0., 0.], device=device, dtype=dtype).expand(B, -1)

# radii：每个阀门的半径，形状 [B]。使用不同半径演示异构批输入。
radii = torch.linspace(0.05, 0.085, B, device=device, dtype=dtype)

# centres：阀门中心，形状 [B, 3]。假定 TCP 当前位于阀门上沿，因此阀门中心
# 等于 TCP 沿 -z 方向移动一个半径。若阀门离末端较远，应先规划 APPROACH。
grasp_direction = torch.tensor([0., 0., 1.], device=device, dtype=dtype)
centres = tcp - radii[:, None] * grasp_direction

# valve_screws：把“绕 axis、经过 centres 旋转”编码为 6 维可供性螺旋轴。
# 这就是闭链任务中物体一侧的运动约束。
valve_screws = cca.get_screw(cca.ScrewType.ROTATION, axis, centres)

# turn_angles：每个环境期望转动的弧度，形状 [B]，可以全部不同。
turn_angles = torch.linspace(0.6, 1.2, B, device=device, dtype=dtype)

# 收敛参数集中在 PlannerConfig；这里只覆盖最大 IK 迭代次数。
config = cca.PlannerConfig(ik_max_itr=50)
planner = cca.PlannerInterface(
    config,
    fast_mode=True,           # 固定迭代次数、用 mask 冻结已收敛环境；默认开启
    compile=False,            # 是否 torch.compile；默认关闭，需按实际批规模测试收益
    fast_linear_solver=True,  # 用正则化法方程代替两处 SVD；默认开启
)

result = planner.generate_joint_trajectory(
    robot_slist=slist,             # [6, n] 可由所有环境共享，也可传 [B, 6, n]
    robot_m=home_pose,             # 零位 TCP 位姿：[4, 4] 或 [B, 4, 4]
    joint_states=q0,               # 每个环境的初始关节角：[B, n]
    motion_type=cca.MotionType.AFFORDANCE,
    affordance_screw=valve_screws, # 每个阀门的 6 维运动螺旋轴：[B, 6]
    goal_affordance=turn_angles,   # 每个阀门的目标转角：[B]
    trajectory_density=12,         # 输出点数，包含初始状态，因此内部求解 11 个目标点
    vir_screw_order=cca.VirtualScrewOrder.XYZ, # 允许末端姿态通过 XYZ 虚拟关节调整
)

trajectory = result.joint_trajectory  # [B, 12, n]，绝对机器人关节轨迹
full_success = result.full_success    # [B]，该环境所有轨迹点是否都收敛
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
- `update_method=UpdateMethod.BEST`：比较逆法和转置法并逐环境选较优结果。

下面三个构造参数是 Torch 批执行特有的实现策略，不改变 APPROACH 与
AFFORDANCE 的统一任务描述：

- `fast_mode=True`：默认开启固定次数迭代，并用布尔 mask 冻结已收敛环境；因此默认
  **不 early stop**，可避免 GPU 每轮同步到 CPU。
- `fast_linear_solver=True`：默认使用正则化法方程，减少小矩阵 SVD 的高额调度开销。
- `compile=False`：默认不调用 `torch.compile`。在 Isaac Lab 中，只有批大小、自由度和
  轨迹密度较稳定且规划器长期复用时，编译成本才更可能被摊薄。

若确实希望提前终止，可在首次规划前调用
`planner.enable_chunked_early_stop(check_interval=4)`。失败时返回张量中仍保留每个目标点
的最后候选解，是否真正收敛必须以 `valid_mask` / `full_success` 为准。

`PlannerInterface.solve_pose_ik()` 提供批量阻尼最小二乘末端位姿 IK，返回
`(joint_states, converged)`，可独立于轨迹规划结果求解 canonical pose 对应的关节状态。

## 示例

- [`examples/demo_valve.py`](examples/demo_valve.py)：先接近随机化的阀门抓取点，
  再沿旋转螺旋轴扭转阀门。
- [`examples/demo_drawer.py`](examples/demo_drawer.py)：先接近随机化的抽屉把手，
  再沿平移螺旋轴将抽屉拉开。

每个脚本都独立包含完整的最小规划流程；只有机器人网格、物体几何、轨迹着色和动画
共用 [`ViserVisualizer`](examples/visualizer.py)。

```bash
python examples/demo_valve.py
python examples/demo_drawer.py
```

先执行 `pip install -e ".[dev]"` 安装测试依赖，再运行 `pytest`。
