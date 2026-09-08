# Closed-Chain Affordance (CCA) Planning Framework

Can you think of a robot manipulation task in terms of an axis, a location, and a goal? For example:
- Turning a valve 90 degrees about its central axis
- Pulling a drawer 20 cm straight outwards to open it
- Fastening a screw with a few turns, considering its axis and pitch

Many common manipulation tasks can be approached this way. The **Closed-Chain Affordance (CCA) Framework** enables you to plan robot joint trajectories for such tasks using these intuitive inputs. Additionally, it offers the flexibility to:
- **Control or free the end-effector (EE) orientation** along the task path
- **Adjust the EE orientation while keeping its position fixed**, useful for tasks like reconfiguration, aligning objects, etc.

This repository contains two C++ packages, `affordance_util` and `cc_affordance_planner`, which together form a standalone library framework for CCA. It utilizes the closed-chain affordance model described in the following IEEE Transactions on Robotics (T-RO) paper. A demonstration video showcasing simulation and real-world tasks is available [here](https://www.youtube.com/watch?v=Ukv93hbNrOM).

## Paper Reference
- Panthi, Janak, Farshid Alambeigi, and Mitch Pryor. "A Closed-Chain Approach to Generating Affordance Joint Trajectories for Robotic Manipulators." IEEE Transactions on Robotics (2025). [Link](https://ieeexplore.ieee.org/abstract/document/11049010)

## Notable Dependencies
1. `C++20`
2. `eigen3`
3. `urdfdom`

Install with `sudo apt install liburdfdom-dev libeigen3-dev`

## Installation Instructions
Follow these steps to install the `affordance_util` and `cc_affordance_planner` libraries:

1. Create a temporary directory and clone this repository in there:

   ```bash
   mkdir ~/temp_cca_ws && cd ~/temp_cca_ws
   ```
   ```
   git clone git@github.com:UTNuclearRoboticsPublic/closed-chain-affordance.git
   ```

2. Build and install the `affordance_util` package.

   ```bash
   cd ~/temp_cca_ws/closed-chain-affordance/affordance_util/ && mkdir build && cd build && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release && cmake --build . && sudo cmake --install .
   ```

3. Build and install the `cc_affordance_planner` package.

   ```bash
   cd ~/temp_cca_ws/closed-chain-affordance/cc_affordance_planner/ && mkdir build && cd build && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release && cmake --build . && sudo cmake --install .
   ```

4. Remove the temporary directory along with this repo clone since we don't need them anymore:

   ```bash
   cd && rm -rf ~/temp_cca_ws
   ```

## ROS2 Implementation

The planner is designed for direct use in your C++ project (see the Usage section below). However, for easier implementation on physical robots, a ROS2 interface (essentially a wrapper around this C++ library) is available. Additionally, an optional user-friendly RViz plugin is provided for intuitive, code-free planning and execution.

👉 [ROS2 Implementation Instructions](https://github.com/UTNuclearRoboticsPublic/closed_chain_affordance_ros.git)

## Usage Information
This section provides detailed instructions on configuring your project's CMakeLists.txt file (see Housekeeping) and writing code (see Code) to utilize the planner.

### Housekeeping
You can include these libraries in your cpp project by including the following in your `CMakeLists.txt`.
```bash
find_package(affordance_util REQUIRED)
find_package(cc_affordance_planner REQUIRED)
```

Link against your targets as:
```bash
target_link_libraries(<target_name> PUBLIC affordance_util::affordance_util PUBLIC cc_affordance_planner::cc_affordance_planner)
```

### Code
This section describes the required and optional code setup for this planner.
#### Required Setup
Using the planner is straightforward and requires just instantiating the planner interface object and calling a method on it by passing robot and task descriptions. Follow these 5 steps:
1. Include these headers:
```cpp
#include <affordance_util/affordance_util.hpp>
#include <cc_affordance_planner/cc_affordance_planner.hpp>
#include <cc_affordance_planner/cc_affordance_planner_interface.hpp>
```
2. Instantiate the planner interface object:
```cpp
cc_affordance_planner::CcAffordancePlannerInterface ccAffordancePlannerInterface;
```
3. Provide the robot description. You can generate this automatically from a URDF or YAML file using the `affordance_util::robot_builder` functions. Sample YAML and URDF files are available in `affordance_util/src/test`. If you'd prefer to manually specify the description, you can do so as follows:

```cpp
affordance_util::RobotDescription robot_description;
robot_description.slist = ; // Eigen::MatrixXd with robot joint screws as its columns in order
robot_description.M = ; // Eigen::Matrix4d representing the homogenous transformation matrix for the EE (palm) at home position
robot_description.joint_states = ; // Eigen::VectorXd representing the joint states of the robot in order at the start config of the affordance
robot_description.joint_lower_limits = ; // Optional absolute lower limits; empty means unbounded
robot_description.joint_upper_limits = ; // Optional absolute upper limits; empty means unbounded
```

4. Furnish task description. Here is an example for rotation motion. More examples in Task Examples section.
```cpp
cc_affordance_planner::TaskDescription task_description;

// Affordance
affordance_util::ScrewInfo aff;
aff.type = affordance_util::ScrewType::ROTATION; // Possible values are ROTATION, TRANSLATION, SCREW
aff.axis = Eigen::Vector3d(1, 0, 0); // Eigen::Vector3d representing the affordance screw axis, [1,0,0] for example
aff.location = Eigen::Vector3d(0, 0, 0); // Eigen::Vector3d representing the location of the affordance screw axis, [0,0,0] for example

task_description.affordance_info = aff;

// Goals
task_description.goal.affordance = 0.4; // Goal for the affordance, 0.4 for instance
```
Here is an example for planning EE orientation adjustment about reference frame `x-axis` while keeping its position fixed.

```cpp
cc_affordance_planner::TaskDescription task_description(cc_affordance_planner::PlanningType::EE_ORIENTATION_ONLY);
req.task_description.affordance_info.axis = Eigen::Vector3d(1, 0, 0); // Axis
req.task_description.goal.affordance = M_PI / 2.0; // Goal
```

5. Generating the joint trajectory to accomplish the specified task:
```cpp
try
{
plannerResult = ccAffordancePlannerInterface.generate_joint_trajectory(robot_description, task_description);
}
catch (const std::invalid_argument &e)
{
std::cerr << "Planner returned exception: " << e.what() << std::endl;
}
```

Reading the result:
```cpp
if (plannerResult.success)
{
std::vector<Eigen::VectorXd> solution = plannerResult.joint_trajectory;// Contains the entire closed-chain trajectory. For robot trajectory that you can send to a j-joint robot, extract the first j joint positions in each point in the trajectory.
// Additional planning result info
switch (plannerResult.trajectory_description)
        {

        case cc_affordance_planner::TrajectoryDescription::FULL:
            std::cout<<"Trajectory description: FULL.\n";
            break;
        case cc_affordance_planner::TrajectoryDescription::PARTIAL:
            std::cout<<"Trajectory description: PARTIAL. Execute with caution.\n";
            break;
        default:
            std::cout<<"Trajectory description: UNSET.\n";
            break;
        }
std::cout << "The entire planning took " << plannerResult.planning_time.count() << " microseconds\n";
}
else
{
std::cerr << "Planner did not find a solution." << std::endl;
}
```

#### Optional Features
Below are optional and advanced features of this framework that one may choose to use as needed.

##### Gripper Trajectory Consideration
The framework can compute a joint trajectory that considers the gripper. To use this feature simply, provide the current state of the gripper in robot description and specify the desired goal state in task description. Optionally, provide gripper goal type.
```cpp
robot_description.gripper_state = 0.0; // Joint value of the gripper
task_description.goal.gripper = 0.4;
task_description.gripper_goal_type = affordance_util::GripperGoalType::CONSTANT; // Possible values are CONSTANT and CONTINUOUS. CONSTANT is default and means the gripper joint will have the desired value, for instance, 0.4 for all points in the trajectory. CONTINUOUS means it will take trajectory_density number of points to go from the current gripper state to the goal gripper state.
```

##### Optional and Advanced Settings
It is possible to instantiate the planner interface object with some desired settings. Here is an example that reflects default values that can be modified.
```cpp
cc_affordance_planner::PlannerConfig plannerConfig;
plannerConfig.accuracy = 10.0/100; // accuracy of the planner, 10% for example

// Then instantiate the planner interface object
cc_affordance_planner::CcAffordancePlannerInterface ccAffordancePlannerInterface(plannerConfig);
```
Optional advanced planner config parameters with default values:
```cpp
plannerConfig.update_method = cc_affordance_planner::UpdateMethod::INVERSE; // method used to solve closed-chain IK. Choices are INVERSE, TRANSPOSE, and BEST. BEST runs INVERSE and TRANSPOSE in parallel and returns the best result.
plannerConfig.closure_err_threshold_ang = 1e-4; // Threshold for the closed-chain closure angular error
plannerConfig.closure_err_threshold_lin = 1e-5; // Threshold for the closed-chain closure linear error
plannerConfig.ik_max_itr = 200; // Limit for the number of iterations for the closed-chain IK solver
```

##### Reserve-Mobility CCA (RM-CCA)

The planner offers two reserve-mobility allocation methods. The existing strict null-space method remains available.
It first completes the CCA correction `Jc * dx = e`, then keeps the floating base stationary in the task null space.
With `dx0 = Jc^+ e`, `Nc = I - Jc^+ Jc`, and `B` selecting reserve coordinates, it uses

```text
dx = dx0 - Nc (B Nc)^+ B dx0.
```

This gives exact base stationarity whenever the arm has a feasible solution. The new Capability-Aware RM-CCA mode
instead solves one state-dependent metric inverse:

```text
G(q) = diag(Ga(q), Gb)
dx   = G^-1 Jc^T (Jc G^-1 Jc^T)^+ e
```

`Gb = diag(lambda_t, lambda_t, lambda_t, lambda_r, lambda_r, lambda_r)` makes base translation and rotation expensive
on their own physical scales. Each finite arm joint has

```text
r_i = (q_i - q_mid_i) / ((q_max_i - q_min_i) / 2)
w_i = w_a0 + lambda_l r_i^2 / (1 - r_i^2 + epsilon)^2.
```

The implementation evaluates the weighted inverse through Jacobian whitening, avoiding normal-equation inversion.
When the arm is healthy, its lower metric cost makes it dominate. Near a limit, `Ga(q)` rises continuously; near an
arm singularity, the arm block of `Jc` loses task mobility. Either effect makes the expensive base participate
smoothly without an activation gate, QP, or MPC. Finite closure recovery uses the same block metric
`diag(Ga(q), Gb, Gs)`, so it cannot silently revert to an unweighted base-heavy correction.

Both methods retain the direction-aware active set. A correction is clipped at the first configured feasibility
boundary, frozen only for that active-set solve, and re-solved over the remaining DoFs. Every new Newton iteration
reactivates the joint, allowing immediate inward motion. Pseudoinverses use the independent cutoff
`max(svd_absolute_tolerance, svd_relative_tolerance * sigma_max)`. The default absolute value `1e-8` preserves the
former RM-CCA numerical floor while making it independently tunable.

```cpp
// Prefer this helper after robot_builder(): it copies URDF/YAML limits into RobotDescription.
auto robot_description = affordance_util::make_robot_description(
    robot_config, current_arm_joint_states);

cc_affordance_planner::TaskDescription task_description;
// ...set affordance and goal as in the examples above...

task_description.reserve_mobility =
    affordance_util::make_floating_base_reserve_description(
        Eigen::Matrix4d::Identity(),
        0.02, // maximum translation increment per Newton iteration [m]
        0.05  // maximum rotation increment per Newton iteration [rad]
    );

cc_affordance_planner::PlannerConfig config;
config.enable_joint_limits = true;
config.enable_nullspace_planning = true;
config.enable_capability_aware_planning = true; // Select the metric method; takes precedence over strict null-space.
config.svd_relative_tolerance = 1e-8;
config.svd_absolute_tolerance = 1e-8;
config.residual_mobility_tolerance = 1e-10;
config.soft_limit_ratio = 0.8; // Central 80% of each finite URDF interval is the effective planning interval.
config.arm_mobility_weight = 1.0;
config.joint_limit_barrier_gain = 1.0;
config.joint_limit_barrier_epsilon = 1e-3;
config.base_translation_weight = 400.0;
config.base_rotation_weight = 25.0;
config.closure_secondary_weight = 1.0;
config.joint_limit_margin = 1e-6;

cc_affordance_planner::CcAffordancePlannerInterface planner(config);
auto result = planner.generate_joint_trajectory(robot_description, task_description);
```

The three switches have the following precedence:

| Switch configuration | Behavior |
|---|---|
| `enable_capability_aware_planning=true` | Capability-aware metric allocation; this takes precedence over `enable_nullspace_planning`. |
| capability-aware off, `enable_nullspace_planning=true` | Existing strict CCA-first/base-stationarity method. |
| only `enable_joint_limits=true` | Bound-aware whole-body ordinary pseudoinverse. |
| all three switches `false` | Exact legacy fixed-base CCA; the attached reserve chain is ignored. |

All three switches default to `true`. Because capability-aware allocation has precedence, it is the effective default
algorithm. Disable only that switch to return to the existing strict null-space method.

`enable_joint_limits` independently controls both the arm proximity barrier and arm active-set enforcement. Disabling
it provides a clean limit-handling ablation while retaining Jacobian-driven singularity allocation in capability mode.
Reserve-coordinate limits always remain active whenever an RM method is in use. `soft_limit_ratio` defaults to `1.0`;
values below one contract each finite URDF interval about its center for both the barrier and active set. If the
supplied initial state is already outside that soft interval but still inside the URDF interval, the initial state
remains admissible and only inward motion is encouraged/allowed. The original URDF bounds are never enlarged or
overwritten.

Whenever at least one extension is enabled together with reserve mobility, the planner uses its pseudoinverse path and
does not start the transpose thread, even if `update_method` is `BEST`. `result.joint_trajectory` retains the
arm/secondary layout used by the fixed-base interface, while `result.reserve_trajectory` and
`result.reserve_pose_trajectory` contain the separate base trajectory. In exact legacy mode these reserve outputs are
empty. `reserve_active`, `residual_mobility_norm` (the maximum reserve correction norm used while solving that point),
and `active_arm_dof_count` are aligned with the RM trajectory points.

##### Python bindings and cabinet Viser demos

The optional `cca_cpp` pybind11 module exposes robot construction from URDF/YAML, the robot/task/config structures,
the planner interface, floating-base reserve construction, RM diagnostics, and FK/Jacobian helpers. The Python
superbuild compiles both C++ libraries from this checkout, so it does not accidentally bind a stale system install.

The dependencies have been installed in the `cca_fb` Conda environment on this workspace. To reproduce the setup:

```bash
source /home/zishang/miniconda3/etc/profile.d/conda.sh
conda activate cca_fb
python -m pip install -r python/requirements-viewer.txt
```

Build the extension (this machine obtains `urdfdom` from ROS Jazzy):

```bash
cmake -S python -B python/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/ros/jazzy \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build python/build --target cca_cpp --parallel
```

Three Piper-L manipulation demos are provided. Each command runs its pose-IK, RM-CCA, joint-limit, and endpoint
assertions, then starts Viser by default:

```bash
python python/examples/demo_cabinet_drawer.py
python python/examples/demo_cabinet_door.py
python python/examples/demo_valve.py
```

All algorithm selections are available directly in every example through paired boolean flags. Numerical metric and
floating-base values remain code-side:

```bash
# Default: Capability-Aware RM-CCA
python python/examples/demo_cabinet_drawer.py

# Existing strict null-space RM-CCA
python python/examples/demo_cabinet_drawer.py --no-capability-aware-planning

# Capability metric without arm limit barrier/active set
python python/examples/demo_cabinet_drawer.py --no-joint-limits

# Bound-aware ordinary whole-body pseudoinverse
python python/examples/demo_cabinet_drawer.py \
  --no-capability-aware-planning --no-nullspace-planning

# Original fixed-base closed-chain-affordance behavior: all three switches off
python python/examples/demo_cabinet_drawer.py \
  --no-joint-limits --no-nullspace-planning --no-capability-aware-planning
```

Door uses `localhost:8080`, drawer uses `localhost:8081`, and valve uses `localhost:8082`. Use `--port` to override the
port or `--duration` to stop the viewer automatically after a specified number of seconds.

Floating-base numerical configuration is deliberately code-side only. Each demo has one `FLOATING_BASE` value near
the task constants:

```python
FLOATING_BASE = FloatingBaseConfig(
    initial_position=(0.0, 0.0, 0.0),
    initial_rpy=(0.0, 0.0, 0.0),
    dof_limits={
        "px": (-0.60, 0.60),
        "py": (-0.60, 0.60),
        "pz": (-0.50, 0.50),
        "ry": (-math.pi, math.pi),
        "rz": (-math.pi, math.pi),
    },
)
```

The dictionary is the enable mask and the limit configuration at the same time. Its permitted keys are `px`, `py`,
`pz`, `rx`, `ry`, and `rz`; the value is the `(lower, upper)` displacement from the configured initial pose. A missing
key is disabled by assigning that reserve coordinate the exact bound `[0, 0]`. Thus the example above enables only
`px py pz ry rz` and fixes `rx`. Every enabled interval must contain zero. Initial orientation uses fixed-axis RPY in
radians and is composed as `Rz @ Ry @ Rx`. The Viser robot root and base path include this initial pose.

All six Piper-L lower and upper limits are copied unchanged from `piper_l_fixed_gripper.urdf`; the robot description is
never assigned a fabricated workspace. The demos set `SOFT_LIMIT_RATIO = 0.8`, so limit-aware planning uses the central
80% as its conservative effective interval while still validating the original URDF hard bounds. Set it to `1.0` to
use the complete URDF interval. In strict null-space mode, an arm-feasible motion keeps the base stationary; in
capability-aware mode, base motion rises continuously as joint-limit proximity or differential arm capability gets
worse. Changing the initial pose therefore changes the allocation naturally and is not required to preserve a fixed
activation pattern. Use `dof_limits={}` to disable all base DoFs when checking a fixed-base version of the same task.

The Python-side numerical metric defaults live in `CapabilityMetricConfig` / `CAPABILITY_METRIC` in
`python/examples/cabinet_demo_common.py`. They are deliberately not CLI arguments:

```python
CAPABILITY_METRIC = CapabilityMetricConfig(
    arm_weight=1.0,
    joint_limit_barrier_gain=1.0,
    joint_limit_barrier_epsilon=1e-3,
    base_translation_weight=400.0,
    base_rotation_weight=25.0,
    closure_secondary_weight=1.0,
    svd_relative_tolerance=1e-8,
    svd_absolute_tolerance=1e-8,
)
```

The cabinet tasks preserve their cabinet placement, handle/hinge definitions, and free virtual handle axes. Their
deterministic goals are a `0.15 m` drawer pull and a `-pi/2` (90-degree) door opening. The real Piper-L and articulated
cabinet DAE meshes are loaded directly from `python/assets`.

The valve demo follows `~/py-workspace/rl_art_mj`: a six-spoke wheel centered at `(0.45, 0, 0.24)`, rotating about its
local X axis, with a top-rim grasp radius of `0.1425 m`, `VirtualScrewOrder.NONE`, 16 trajectory points, and a fixed
`pi` (180-degree) goal. Viser reconstructs the reference valve's hub, six spokes, rim, shaft, and mount from the same
primitive dimensions.

Optional task description parameters:
```cpp
task_description.goal.ee_orientation = Eigen::Vector3d(0.1, 0.0, 0.1); // EE orientation to maintain along the affordance path, rpy = [0.1,0.0,0.1] for instance. Can specify one or more aspections of the orientation in the order specified by VirScrewOrder below.
task_description.trajectory_density = 10; // Number of points in the solved joint trajectory, 10 for example
task_description.vir_screw_order = affordance_util::VirtualScrewOrder::XYZ; // Order of the axes in the closed-chain model virtual gripper joint. Possible values include X, Y, Z, XY, YZ, ZX, XYZ, YZX, ZXY, and NONE.

// For affordance, one can directly specify the 6x1 screw vector for affordance instead of axis and location
aff.type = affordance_util::ScrewType::TRANSLATION;
aff.screw = Eigen::Vector6d(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
```
#### Task Examples
##### Affordance - Rotation
```cpp
cc_affordance_planner::TaskDescription task_description;

// Affordance
aff.type = affordance_util::ScrewType::ROTATION;
aff.axis = Eigen::Vector3d(0, 0, 1);
aff.location = Eigen::Vector3d(0.0, 0.0, 0.0);
aff_goal = (Eigen::VectorXd(1) << (1.0 / 2.0) * M_PI).finished();
task_description.affordance_info = aff;

// Goal
task_description.goal.affordance = (1.0 / 2.0) * M_PI;
```

##### Affordance - Translation
```cpp
cc_affordance_planner::TaskDescription task_description;

// Affordance
aff.type = affordance_util::ScrewType::TRANSLATION;
aff.axis = Eigen::Vector3d(1.0, 0.0, 0.0);
aff.location = Eigen::Vector3d(0.0, 0.0, 0.0);
task_description.affordance_info = aff;

// Goal
task_description.goal.affordance = 0.5;
```

##### Affordance - Screw
```cpp
cc_affordance_planner::TaskDescription task_description;

// Affordance
aff.type = affordance_util::ScrewType::SCREW;
aff.axis = Eigen::Vector3d(0, 0, 1);
aff.location = Eigen::Vector3d(0.0, 0.0, 0.0);
aff.pitch = 0.5;
task_description.affordance_info = aff;

// Goal
task_description.goal.affordance = (1.0 / 2.0) * M_PI;
```

##### Affordance - Rotation with EE Orientation Control
```cpp
cc_affordance_planner::TaskDescription task_description;

// Affordance
aff.type = affordance_util::ScrewType::ROTATION;
aff.axis = Eigen::Vector3d(0, 0, 1);
aff.location = Eigen::Vector3d(0.0, 0.0, 0.0);
aff_goal = (Eigen::VectorXd(1) << (1.0 / 2.0) * M_PI).finished();
task_description.affordance_info = aff;

// Goal
task_description.goal.affordance = (1.0 / 2.0) * M_PI;
task_description.goal.ee_orientation = Eigen::Vector3d(0.1, 0.0, 0.1); # Gripper-frame orientation per vir_screw_order. Default as roll-pitch-yaw.
```

### Author
Janak Panthi aka Crasun Jans
