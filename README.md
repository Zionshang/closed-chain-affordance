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

RM-CCA uses one strict task hierarchy throughout every Newton iteration:

1. complete the original CCA correction `Jc * dx = e`;
2. keep a floating base (or another reserve chain) stationary inside the primary-task null space.

With `dx0 = Jc^+ e`, `Nc = I - Jc^+ Jc`, and `B` selecting reserve coordinates, the correction is

```text
dx = dx0 - Nc (B Nc)^+ B dx0.
```

Therefore, whenever an arm-only solution exists, the base correction is zero exactly (up to floating-point error).
When joint bounds or truncated-SVD rank loss remove feasible arm directions, the same expression is recomputed over
the remaining feasible variables and retains only the base motion that can no longer be eliminated. Bound handling is
direction-aware: a joint is frozen only for the current active-set solve and can reactivate inward on the next Newton
iteration. Closure correction uses the identical hierarchy and preserves the CCA secondary state already advanced by
the main Newton step. There is no arm/base phase switch, QP, weighting cost, or optimizer.

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
config.svd_relative_tolerance = 1e-8;
config.residual_mobility_tolerance = 1e-10;
config.joint_limit_margin = 1e-6;

cc_affordance_planner::CcAffordancePlannerInterface planner(config);
auto result = planner.generate_joint_trajectory(robot_description, task_description);
```

When reserve mobility is enabled, the planner always uses its pseudoinverse path and does not start the transpose
thread, even if `update_method` is `BEST`. `result.joint_trajectory` retains the arm/secondary layout used by the
fixed-base interface, while `result.reserve_trajectory` and `result.reserve_pose_trajectory` contain the separate base
trajectory. `reserve_active`, `residual_mobility_norm` (the maximum reserve correction norm used while solving that
point), and `active_arm_dof_count` are aligned with those trajectory points.

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

Only the two migrated cabinet tasks are kept as executable Python examples. Run their pose-IK, RM-CCA, joint-limit,
and endpoint assertions without a viewer:

```bash
python python/examples/demo_cabinet_drawer.py
python python/examples/demo_cabinet_door.py
```

Start either real-mesh Viser animation:

```bash
python python/examples/demo_cabinet_drawer.py --visualize  # localhost:8081
python python/examples/demo_cabinet_door.py --visualize    # localhost:8080
```

Both tasks preserve the Piper-L, cabinet placement, handle/hinge definitions, and free virtual handle axis from the
other project. The goals are deterministic: drawer pull is fixed at `0.15 m` and door opening is fixed at `-pi/2`
(90 degrees). For a reproducible capability-exhaustion demonstration, the real URDF limits are intersected with a
±0.15-rad operating envelope around the grasp. Each example internally verifies that a numerically frozen base returns
`PARTIAL`, and that the assisted plan has a bit-identical arm-only prefix before the base moves and returns `FULL`.
Viser renders only that single assisted environment: drawer base assistance starts at point 6 after the arm-only plan
stops at 5/12; door assistance starts at point 3 after the arm-only plan stops at 2/12. The real Piper-L and articulated
cabinet DAE meshes are loaded directly from `python/assets`.

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
