# Closed-Chain Affordance 方法讲解

这份文档不是使用手册，而是结合论文 **A Closed-Chain Approach to Generating Affordance Joint Trajectories for Robotic Manipulators** 和本仓库代码，把 Closed-Chain Affordance (CCA) 方法的原理讲清楚。

读完后你应该能回答这些问题：

- 为什么“开阀门、拉抽屉、推门、拧螺丝”可以统一成 screw affordance？
- 为什么论文要把任务和机器人建成一个闭链机构？
- `affordance.axis`、`affordance.location`、`goal.affordance` 分别是什么意思？
- 代码里的 `compose_cc_model_slist`、`generate_joint_trajectory`、`call_cc_ik_solver` 在论文算法里对应哪一步？
- 为什么 X5 demo 里阀门圆心不能直接等于 TCP 位置？

论文参考：

- Janak Panthi, Farshid Alambeigi, and Mitch Pryor, "A Closed-Chain Approach to Generating Affordance Joint Trajectories for Robotic Manipulators," IEEE Transactions on Robotics, 2025.
- IEEE: <https://ieeexplore.ieee.org/abstract/document/11049010>
- Demo video: <https://www.youtube.com/watch?v=Ukv93hbNrOM>

## 1. 这个方法到底想解决什么

很多操作任务不是“末端到达某个点”这么简单，而是“末端沿着某个约束运动”：

- 阀门：绕固定轴旋转。
- 抽屉：沿固定方向平移。
- 门：绕铰链轴旋转。
- 螺丝：绕轴旋转，同时沿轴前进。

传统做法常常会把这些任务离散成很多末端 waypoint，然后让机器人逐点跟踪。但这样有几个问题：

- waypoint 是人为采样出来的，任务本身的几何约束没有进入机构模型。
- 如果运行中发现初始抓取位姿、物体轴、姿态约束不对，重新规划不够直接。
- 很难自然表达“末端姿态要保持”、“末端姿态可以自由”、“只控制某几个姿态分量”。
- 逐点速度控制通常只看局部，不容易提前知道完整关节轨迹是否会碰到奇异或不舒服的构型。

CCA 的目标是：给定机器人当前状态和任务 affordance，直接生成一整段关节轨迹。

论文的问题定义可以概括成：

```text
输入：
  机器人描述、初始关节角、任务 affordance screw、任务目标和可选末端姿态目标

输出：
  Q = {q0, q1, ..., qs}
  一段机器人关节空间轨迹，使 EE 沿 affordance path 运动到目标
```

注意，论文和这个仓库处理的是“已经准备开始执行 affordance 的轨迹生成”。它不是全局避障 planner，也不负责在复杂环境中找 grasp。

## 2. Affordance 是什么

这里的 affordance 可以理解成“物体允许机器人怎么动”。

阀门允许“绕阀门轴转”；抽屉允许“沿导轨方向拉”；螺丝允许“绕轴转且沿轴前进”。这些都可以用 screw 表示。

一个 screw 通常由三样东西描述：

```text
axis      轴方向
location  轴经过空间中的哪个点
pitch     每转 1 rad 同时沿轴平移多少；纯旋转 pitch = 0
```

在代码里对应 `affordance_util::ScrewInfo` / Python 的 `cca.ScrewInfo`：

```python
affordance = cca.ScrewInfo()
affordance.type = cca.ScrewType.ROTATION
affordance.axis = cca.axis_to_vec(cca.Axis.X_MINUS)
affordance.location = valve_center
```

### 2.1 纯旋转

对阀门、门铰链这类任务：

```text
S = [ w
      q x w ]
```

其中：

- `w` 是旋转轴单位向量。
- `q` 是轴上的一点。
- `q x w` 是空间 screw 的线速度部分。

代码里是：

```cpp
screw.head(3) = si.axis;
screw.tail(3) = si.location.cross(si.axis);
```

也就是 [affordance_util.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/affordance_util/src/affordance_util/affordance_util.cpp:776) 里的 `get_screw(const ScrewInfo&)`。

### 2.2 纯平移

对抽屉：

```text
S = [ 0
      v ]
```

其中 `v` 是平移方向。代码里 `ScrewType::TRANSLATION` 会把角速度部分设成 0，线速度部分设成 `axis`。

### 2.3 螺旋运动

对螺丝：

```text
S = [ w
      q x w + h w ]
```

`h` 是 pitch。代码里 `ScrewType::SCREW` 对应：

```cpp
screw.tail(3) = si.location.cross(si.axis) + si.pitch * si.axis;
```

## 3. 为什么要把任务建成闭链

这是论文最关键的思想。

先用阀门举例。机械臂抓住阀门边缘上的某一点以后，这个点不能随便动，它必须沿阀门圆周运动。也就是说，机器人末端和阀门之间产生了一个约束。

论文的做法不是说“我给末端生成一串圆弧 waypoint”。它说：

1. 机器人本体是一条开链。
2. 从机器人末端到阀门轴之间想象一段 affordance link。
3. 阀门轴本身是一个 affordance joint。
4. 从 affordance joint 再想象一段 ground link 回到机器人 base。
5. 机器人 + affordance chain 合起来形成一个闭链机构。

直观地说：

```text
机器人 base -> 机器人关节 -> EE/TCP -> 虚拟 EE joint -> affordance link
-> affordance joint -> ground link -> 回到机器人 base
```

闭链的意义是：任务约束不再是外部附加的 waypoint，而是机构本身的闭合约束。

论文用 Kirchhoff-Davies circulation law 写这个闭链约束：

```text
N(q) * theta_dot = 0
```

这里：

- `N` 是 closed-chain network matrix，每列是一根 screw。
- `theta_dot` 是所有真实/虚拟/affordance joint 的速度。
- 等式为 0 表示整个闭链没有断开。

代码里这个 `N` 不是固定矩阵，而是在当前 joint state 附近通过 screw Jacobian 算出来：

```cpp
Eigen::MatrixXd jac = affordance_util::JacobianSpace(slist, thetalist);
Eigen::MatrixXd Np = jac.leftCols(nof_pjoints_);
Eigen::MatrixXd Ns = jac.rightCols(nof_sjoints_);
```

位置在 [cc_affordance_planner.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner.cpp:617)。

## 4. 虚拟 EE joint 是干什么的

论文里一个重要细节是：末端姿态可以被控制，也可以被释放。

比如开阀门至少有三种策略：

- 手掌姿态保持不变，只让接触点绕阀门走。
- 手掌跟着阀门一起转。
- 手掌姿态部分自由，只保证接触点走在 affordance path 上。

为了表达这些情况，论文在 EE 和 affordance link 之间加入一个虚拟 spherical joint。三维空间里它可以有 3 个虚拟旋转自由度。这样一来，闭链中可控制的东西包括：

```text
affordance 运动量
EE orientation 的若干分量
```

对 6 自由度机器人，若加入 3 个虚拟 EE 旋转 joint 和 1 个 affordance joint，则总 joint 数是：

```text
6 个机器人 joint + 3 个虚拟 EE joint + 1 个 affordance joint = 10
```

SE(3) screw 长度是 6，所以闭链 mobility：

```text
m = j - l = 10 - 6 = 4
```

这 4 个可控量就是：

```text
3 个 EE orientation 分量 + 1 个 affordance 分量
```

代码里虚拟 EE screw 在 `compose_cc_model_slist` 中追加：

```cpp
const Eigen::MatrixXd &w_vir = affordance_util::get_vir_screw_axes(vir_screw_order);
vir_slist.col(i) = get_screw(w_vir.col(i), q_vir);
slist << robot_jacobian, vir_slist, aff.screw;
```

位置在 [affordance_util.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/affordance_util/src/affordance_util/affordance_util.cpp:169)。

这里有一个实现细节：论文讨论了可以在不同 frame 里定义虚拟 joint。当前仓库实现中，虚拟 EE screws 是在 affordance 起点处、按 space frame 对齐来构造的。代码注释也写了这一点：这样控制 gripper orientation 更直观。

如果 `vir_screw_order == NONE`，代码就不加入虚拟 EE joints，只加入 affordance screw。这表示不控制 EE orientation。

## 5. Primary 和 Secondary 怎么理解

论文把闭链里的 joint 分成两类：

- primary joints：求解器要算的未知量。
- secondary joints：你指定目标的任务量。

这不是物理上“主动/被动”的唯一划分，而是建模时的选择。

在最常见的 affordance motion 中：

```text
primary   = 机器人 joints + 可能的虚拟 EE joints
secondary = affordance joint，外加需要控制的 EE orientation 分量
```

你给 secondary 一个目标：

```text
阀门转 pi
或者 阀门转 pi，同时 EE roll 保持/改变到某个值
```

求解器去找 primary 的变化：

```text
机器人关节怎么动，虚拟 EE joint 怎么变化
```

代码里 `task_offset_tau` / `nof_secondary_joints` 就是在表达 secondary joints 的数量：

```cpp
nof_pjoints_ = slist.cols() - task_offset_tau;
nof_sjoints_ = task_offset_tau;
```

位置在 [cc_affordance_planner.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner.cpp:318)。

`nof_secondary_joints` 在接口层由任务目标决定：

```cpp
size_t nof_secondary_joints = 1; // at least the affordance
nof_secondary_joints += task_description.goal.ee_orientation.size();
```

位置在 [cc_affordance_planner_interface.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner_interface.cpp:70)。

## 6. 从闭链约束到 IK

闭链约束是：

```text
N theta_dot = 0
```

把 joint 分成 primary 和 secondary：

```text
[ Ns  Np ] [ theta_s_dot ] = 0
          [ theta_p_dot ]
```

于是：

```text
Ns theta_s_dot = -Np theta_p_dot
```

如果你知道 primary 怎么动，可以算 secondary：

```text
theta_s_dot = -Ns^dagger Np theta_p_dot
```

反过来，如果你希望 secondary 达到目标，就要求 primary：

```text
theta_p_dot = -Np^dagger Ns theta_s_dot
```

这就是论文说的一个漂亮点：闭链 forward velocity kinematics 和 inverse velocity kinematics 形式很像。

但规划需要的是位置轨迹，不只是速度。所以论文把问题写成求根：

```text
f(theta_p) = theta_sd - theta_s
```

意思是：给定想要的 secondary 目标 `theta_sd`，找一组 primary `theta_p`，让当前 secondary `theta_s` 跟目标一致。

代码里每个小步都会调用：

```cpp
call_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd)
```

其中：

- `theta_pg` 是 primary 初值。
- `theta_sg` 是 secondary 初值。
- `theta_sd` 是当前小步的 secondary 目标。

## 7. 为什么还要 closure error

如果只做 Newton-Raphson 数值迭代，闭链可能会“裂开”。直观地说，机器人和虚拟 affordance chain 的末端不再正好闭合。

论文把这个裂开的误差写成一个 twist：

```text
rho
```

算法 3 做的事情是：

1. 用 FK 算闭链末端 `Tse`。
2. 用 SE(3) log 算出闭合误差 twist `rho`。
3. 用闭链 Jacobian 的伪逆把这个误差分配回 primary 和 secondary joint，修正 joint 值。

代码对应：

```cpp
Eigen::Matrix4d Tse = FKinSpace(I, slist, thetalist);
rho = Adjoint(Tse) * se3ToVec(MatrixLog6(TransInv(Tse)));

Nc << Np, Ns;
delta_theta = pinv_Nc * rho;
theta_p += delta_theta.head(nof_pjoints_);
theta_s += delta_theta.tail(nof_sjoints_);
```

位置在 [cc_affordance_planner.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner.cpp:661)。

这一步是 CCA 和普通“沿路径 IK”很不一样的地方：它显式地维护闭链闭合。

## 8. 整个轨迹生成算法

论文算法 1 的思想很简单：

```text
不要一次求完整 pi rad 的阀门转动。
把目标拆成很多小步。
每一步都解一次闭链 IK。
把上一步结果作为下一步初值。
```

代码里 affordance motion 的主循环是：

```cpp
deltatheta_a = theta_adf / (trajectory_density - 1);

while (loop_counter_k < trajectory_density - 1) {
    theta_sd(nof_sjoints_ - 1) -= deltatheta_a;
    ik_result = call_cc_ik_solver(...);
    if (ik_result) {
        joint_trajectory.push_back(ik_result.value());
        theta_sg = traj_point.tail(nof_sjoints_);
        theta_pg = traj_point.head(nof_pjoints_);
    }
}
```

位置在 [cc_affordance_planner.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner.cpp:396)。

这个“上一点作为下一点初值”的机制很重要。它让连续轨迹更平滑，也减少每一步搜索的难度。

最后接口层会把“闭链增量轨迹”转换成机器人实际关节轨迹：

```cpp
cc_start_joint_states.head(nof_robot_joints) = start_joint_states;
return point + cc_start_joint_states;
```

位置在 [cc_affordance_planner_interface.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner_interface.cpp:275)。

这就是论文公式：

```text
Q = q0 + {0, theta_p1(1:n), ..., theta_ps(1:n)}
```

## 9. 代码里的三种规划模式

### 9.1 AFFORDANCE

这是最标准的论文场景：

```text
当前 EE 已经在任务起点。
现在沿 affordance path 运动。
```

代码路径：

```text
CcAffordancePlannerInterface::generate_joint_trajectory
  -> compose_cc_model_slist(robot, aff, vir_screw_order)
  -> generate_affordance_motion_joint_trajectory
  -> call_cc_ik_solver
```

### 9.2 APPROACH

Approach 是本仓库额外封装出来的一类任务：从当前 pose 移到某个 canonical pose，同时也可以放进 affordance 约束。

代码里 approach 会额外构造一个 approach screw：

```cpp
approach_twist = Adjoint(start_pose) * se3ToVec(Log(Inv(start_pose) * approach_end_pose));
approach_screw = approach_twist / norm(approach_twist);
cc_model.approach_limit = norm(approach_twist);
```

位置在 [affordance_util.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/affordance_util/src/affordance_util/affordance_util.cpp:73)。

然后 closed-chain screw list 变成：

```text
robot Jacobian + optional virtual EE screws + approach screw + affordance screw
```

### 9.3 CARTESIAN_GOAL

`CARTESIAN_GOAL` 是 `APPROACH` 的特殊情况。它把目标位姿当成 canonical pose，并给 affordance 一个很小的 dummy 目标。

代码里 `TaskDescription(PlanningType::CARTESIAN_GOAL)` 会设置：

```cpp
motion_type = APPROACH;
vir_screw_order = NONE;
goal.affordance = eps;
```

位置在 [cc_affordance_planner.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner.cpp:20)。

这也是 X5 demo 里 approach 阶段的模式。

## 10. INVERSE、TRANSPOSE、BEST 是什么

闭链 IK 里更新 primary joint 的核心是：

```text
theta_p <- theta_p + something * (theta_sd - theta_s)
```

仓库有两种更新方式：

- `INVERSE`：用伪逆；如果条件数太大，则切换到 damped least squares。
- `TRANSPOSE`：用转置近似。

代码：

```cpp
delta_theta_p = pinv_N * (theta_sd - theta_s);
```

或：

```cpp
delta_theta_p = N.transpose() * (theta_sd - theta_s);
```

位置分别在 [cc_affordance_planner.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner.cpp:57) 和 [cc_affordance_planner.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner.cpp:46)。

`BEST` 会并行跑 inverse 和 transpose，谁先给出完整解就用谁；如果都不完整，就选更好的 partial trajectory。接口层逻辑在 [cc_affordance_planner_interface.cpp](/home/zishang/cpp-workspace/closed-chain-affordance/cc_affordance_planner/src/cc_affordance_planner/cc_affordance_planner_interface.cpp:153)。

## 11. X5 开阀门 demo 怎么理解

`python/demo_x5_urdf.py` 做两段：

```text
1. approach：把 TCP 移到阀门边缘上的接触点。
2. affordance：让 TCP 绕阀门圆心转。
```

关键不是“末端原地旋转”，而是“末端点绕 affordance axis 运动”。

demo 中 affordance axis 是：

```python
affordance.axis = cca.axis_to_vec(cca.Axis.X_MINUS)
```

也就是说阀门轴沿 `-X`。那么阀门圆面在 `Y-Z` 平面内。TCP 如果在阀门圆周上，圆心应该和 TCP 在 `Y-Z` 平面内相差一个半径。

代码里：

```python
VALVE_RADIUS_M = 0.08
valve_center = tcp_pose_on_valve[:3, 3].copy()
valve_center[2] -= VALVE_RADIUS_M
affordance.location = valve_center
```

这里的含义是：

```text
TCP 当前点 = 阀门边缘点
valve_center = 阀门旋转轴经过的点
axis = 阀门轴方向
goal.affordance = 阀门转动角度
```

如果你把 `affordance.location` 设成 TCP 当前点，那么 affordance 轴穿过 TCP，结果就会变成 TCP 绕自己所在的轴运动，看起来像“纯旋转”或“原地转”，而不是开阀门的圆周运动。

这也是为什么：

```text
有 axis 还不够，还必须有正确的 location。
```

axis 决定“绕哪个方向转”；location 决定“绕空间中哪条轴转”。

## 12. URDF 里的 end_effector.frame_name 和这个方法的关系

当前仓库版本里：

```yaml
end_effector:
  - frame_name: ee_link
```

这个 `frame_name` 就是规划使用的 EE/TCP frame。换句话说，`RobotDescription.M` 是这个 frame 在 home configuration 下的位姿，FK 出来的 `current_pose` 也是这个 frame 的位姿。

如果你的真实 TCP 不在末端 link 原点，应该在 URDF 里用 fixed joint 建一个虚拟 TCP link，然后让 `frame_name` 指向那个 link。

这跟 CCA 原理有关：affordance 约束施加在 EE/TCP 上。如果 TCP frame 定错，闭链约束会被施加到错误的点上，轨迹看起来就会不对。

## 13. 这个仓库和论文的实现差异/取舍

需要注意几处实现细节：

1. 论文是方法框架，仓库是一个具体 C++ 实现。它把很多论文算法行直接落实成函数，例如 `generate_affordance_motion_joint_trajectory`、`call_cc_ik_solver`、`adjust_for_closure_error`。

2. 代码中 closed-chain `slist` 的前几列不是机器人 home screw list 原样，而是当前状态下的 robot space Jacobian：

   ```cpp
   robot_jacobian = JacobianSpace(robot_description.slist, robot_description.joint_states);
   ```

   这对应闭链建模时从当前 grasp/start pose 出发做 differential trajectory。

3. 虚拟 EE screws 当前按 space frame 方向构造，而不是更复杂地随 gripper frame 动态重建。代码注释说明这是为了让 orientation 控制更直观。

4. Python demo 里的 X5 阀门不是论文原实验平台，而是这个仓库为了演示 URDF 构建和 MeshCat 可视化加入的例子。

5. 本仓库当前版本已经把 `tool` 标签和 `gripper_joint_name` 从 YAML 配置语义中去掉；`end_effector.frame_name` 被视为 TCP/tool frame。

## 14. 用一句话总结

CCA 的核心不是“让末端跟一条路径”，而是：

```text
把任务 affordance 作为虚拟 joint 接到机器人末端，
把机器人和任务合成一个闭链机构，
然后通过闭链 IK 生成让任务 joint 达到目标的机器人关节轨迹。
```

所以理解这个方法时，最重要的是不要把 `affordance` 看成普通目标位姿。它是闭链机构里的一个 joint；`goal.affordance` 是这个 joint 要走的量；机器人关节轨迹是为了让这个虚拟任务 joint 运动起来而被求出来的。

