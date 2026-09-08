#include <affordance_util/affordance_util.hpp>
#include <cc_affordance_planner/cc_affordance_planner_interface.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace cc_affordance_planner
{

CcAffordancePlannerInterface::CcAffordancePlannerInterface()
    : ccAffordancePlannerInverse_(), ccAffordancePlannerTranspose_()
{
}

CcAffordancePlannerInterface::CcAffordancePlannerInterface(const PlannerConfig &planner_config)
    : planner_config_(planner_config),
      ccAffordancePlannerInverse_(planner_config),
      ccAffordancePlannerTranspose_(planner_config)

{
    // Validate planner config

    if ((planner_config.accuracy <= 0) || (planner_config.accuracy > 1.0))
    {
        throw std::invalid_argument("Planner config: 'accuracy' must be in the range (0,1]");
    }

    if (planner_config.closure_err_threshold_ang < 1e-10)
    {

        throw std::invalid_argument(
            "Planner config: 'closure_err_threshold_ang' cannot be unrealistically small, i.e. less than 1e-10.");
    }

    if (planner_config.closure_err_threshold_lin < 1e-10)
    {

        throw std::invalid_argument(
            "Planner config: 'closure_err_threshold_lin' cannot be unrealistically small, i.e. less than 1e-10.");
    }

    if ((planner_config.ik_max_itr < 2) || (planner_config.ik_max_itr > 100000))
    {

        throw std::invalid_argument("Planner config: 'ik_max_itr' must be in the range [2, 100000]");
    }
    if (!(planner_config.svd_relative_tolerance > 0.0) ||
        !std::isfinite(planner_config.svd_relative_tolerance))
    {
        throw std::invalid_argument("Planner config: 'svd_relative_tolerance' must be finite and positive.");
    }
    if (planner_config.svd_absolute_tolerance < 0.0 ||
        !std::isfinite(planner_config.svd_absolute_tolerance))
    {
        throw std::invalid_argument("Planner config: 'svd_absolute_tolerance' must be finite and non-negative.");
    }
    if (planner_config.residual_mobility_tolerance < 0.0 ||
        !std::isfinite(planner_config.residual_mobility_tolerance))
    {
        throw std::invalid_argument("Planner config: 'residual_mobility_tolerance' must be finite and non-negative.");
    }
    if (!(planner_config.soft_limit_ratio > 0.0) || planner_config.soft_limit_ratio > 1.0 ||
        !std::isfinite(planner_config.soft_limit_ratio))
    {
        throw std::invalid_argument("Planner config: 'soft_limit_ratio' must be in the range (0,1].");
    }
    const std::array positive_metric_parameters = {
        planner_config.arm_mobility_weight, planner_config.joint_limit_barrier_epsilon,
        planner_config.base_translation_weight, planner_config.base_rotation_weight,
        planner_config.closure_secondary_weight};
    if (std::any_of(positive_metric_parameters.begin(), positive_metric_parameters.end(),
                    [](const double value) { return !(value > 0.0) || !std::isfinite(value); }) ||
        planner_config.joint_limit_barrier_gain < 0.0 ||
        !std::isfinite(planner_config.joint_limit_barrier_gain))
    {
        throw std::invalid_argument(
            "Planner config: mobility weights/epsilon must be finite and positive; barrier gain must be non-negative.");
    }
    if (planner_config.joint_limit_margin < 0.0 || !std::isfinite(planner_config.joint_limit_margin) ||
        planner_config.joint_limit_tolerance < 0.0 || !std::isfinite(planner_config.joint_limit_tolerance))
    {
        throw std::invalid_argument("Planner config: joint-limit margin and tolerance must be finite and non-negative.");
    }
}

PlannerResult CcAffordancePlannerInterface::generate_joint_trajectory(
    const affordance_util::RobotDescription &robot_description, const TaskDescription &task_description)
{

    // Check the input for any potential errors
    this->validate_input_(robot_description, task_description);

    // Output of the function
    PlannerResult plannerResult;

    affordance_util::ReserveMobilityDescription reserve_mobility = task_description.reserve_mobility;
    // Disabling every RM extension is an explicit compatibility mode. Ignore the attached reserve model entirely and
    // use the same model composition, solver selection, and trajectory conversion as the original fixed-base CCA.
    const bool rm_planning_enabled =
        reserve_mobility.enabled &&
        (planner_config_.enable_joint_limits || planner_config_.enable_nullspace_planning ||
         planner_config_.enable_capability_aware_planning);
    Eigen::Index feasible_arm_dof_for_diagnostics = robot_description.joint_states.size();
    if (rm_planning_enabled)
    {
        const Eigen::Index reserve_dof = reserve_mobility.slist.cols();
        const Eigen::Index virtual_dof = task_description.vir_screw_order == affordance_util::VirtualScrewOrder::NONE
                                             ? 0
                                             : affordance_util::get_vir_screw_axes(task_description.vir_screw_order).cols();
        const Eigen::Index controlled_virtual_dof = task_description.goal.ee_orientation.size();
        if (controlled_virtual_dof > virtual_dof)
        {
            throw std::invalid_argument("EE-orientation goals exceed the selected virtual-screw DOF.");
        }
        // Uncontrolled virtual EE joints are free primary grasp motions. They belong on the arm side of the
        // reserve partition with a zero initial state and unbounded limits; otherwise RM-CCA would leave primary
        // columns outside both the feasible-arm and reserve sets.
        const Eigen::Index free_virtual_dof = virtual_dof - controlled_virtual_dof;
        const Eigen::Index feasible_arm_dof = robot_description.joint_states.size() + free_virtual_dof;
        feasible_arm_dof_for_diagnostics = feasible_arm_dof;
        if (reserve_mobility.lower_limits.size() == 0)
        {
            reserve_mobility.lower_limits =
                Eigen::VectorXd::Constant(reserve_dof, -std::numeric_limits<double>::infinity());
            reserve_mobility.upper_limits =
                Eigen::VectorXd::Constant(reserve_dof, std::numeric_limits<double>::infinity());
        }
        if (reserve_mobility.max_step.size() == 0)
        {
            reserve_mobility.max_step =
                Eigen::VectorXd::Constant(reserve_dof, std::numeric_limits<double>::infinity());
        }

        Eigen::VectorXd arm_lower = robot_description.joint_lower_limits;
        Eigen::VectorXd arm_upper = robot_description.joint_upper_limits;
        if (arm_lower.size() == 0)
        {
            arm_lower = Eigen::VectorXd::Constant(robot_description.joint_states.size(),
                                                   -std::numeric_limits<double>::infinity());
            arm_upper = Eigen::VectorXd::Constant(robot_description.joint_states.size(),
                                                   std::numeric_limits<double>::infinity());
        }

        std::vector<size_t> reserve_indices(static_cast<size_t>(reserve_dof));
        std::iota(reserve_indices.begin(), reserve_indices.end(), 0);
        Eigen::VectorXd feasible_arm_start = Eigen::VectorXd::Zero(feasible_arm_dof);
        feasible_arm_start.head(robot_description.joint_states.size()) = robot_description.joint_states;
        Eigen::VectorXd feasible_arm_lower =
            Eigen::VectorXd::Constant(feasible_arm_dof, -std::numeric_limits<double>::infinity());
        Eigen::VectorXd feasible_arm_upper =
            Eigen::VectorXd::Constant(feasible_arm_dof, std::numeric_limits<double>::infinity());
        feasible_arm_lower.head(robot_description.joint_states.size()) = arm_lower;
        feasible_arm_upper.head(robot_description.joint_states.size()) = arm_upper;

        std::vector<size_t> arm_indices(static_cast<size_t>(feasible_arm_dof));
        std::iota(arm_indices.begin(), arm_indices.end(), static_cast<size_t>(reserve_dof));
        ccAffordancePlannerInverse_.configure_reserve_mobility(
            reserve_mobility, feasible_arm_start, feasible_arm_lower, feasible_arm_upper, reserve_indices, arm_indices);
    }
    else
    {
        ccAffordancePlannerInverse_.disable_reserve_mobility();
    }

    // Extract task description
    // Affordance info -- handle if asked to get from FK
    affordance_util::ScrewInfo aff = task_description.affordance_info;
    if (task_description.affordance_info_from.method==affordance_util::PoseSpecificationMethod::FROM_FK){
	const affordance_util::VecInfo vec_info = affordance_util::get_affordance_info_from_fk(task_description.affordance_info_from, robot_description);
        aff.location = vec_info.location;
        // If available, use axis info from FK. If it is not set, it means it was provided in affordance_info and we keep that.
        if (!vec_info.axis.hasNaN()){
	    aff.axis = vec_info.axis;
	} 
    }

    const affordance_util::VirtualScrewOrder vir_screw_order = task_description.vir_screw_order;

    // Compute the nof secondary joints
    size_t nof_secondary_joints = 1; // secondary joints contain at least the affordance
    nof_secondary_joints += task_description.goal.ee_orientation.size(); // add relevant virtual ee joints
    // TODO: Remove the need to pass nof_secondary_joints to the planner and get it directly from the size of the
    // theta_sdf vector.

    // Extract additional task description
    Eigen::Matrix4d canonical_pose = task_description.goal.canonical_pose;
    if (task_description.motion_type == MotionType::APPROACH)
    {
        // Canonical pose -- handle if asked to get from FK
        if (task_description.canonical_pose_from.method==affordance_util::PoseSpecificationMethod::FROM_FK){
            canonical_pose = affordance_util::get_pose_from_fk(task_description.canonical_pose_from, robot_description);
        }
        else {
            canonical_pose = task_description.goal.canonical_pose;
        }

        // Compose the closed-chain model screws and determine the limit for the approach screw
        const affordance_util::CcModel cc_model = rm_planning_enabled
            ? affordance_util::compose_rm_cc_model_slist(robot_description, aff, canonical_pose, reserve_mobility,
                                                         task_description.approach_gamma, vir_screw_order)
            : affordance_util::compose_cc_model_slist(robot_description, aff, canonical_pose,
                                                      task_description.approach_gamma, vir_screw_order);

        // Extract and construct the secondary joint goals
        nof_secondary_joints += 1; // add approach joint
        const Eigen::VectorXd secondary_joint_goals =
            (Eigen::VectorXd(nof_secondary_joints) << task_description.goal.ee_orientation, cc_model.approach_limit,
             task_description.goal.affordance)
                .finished();

        // Solve the joint trajectory for the given task. This function calls the Cc Affordance Planner inside.
        plannerResult = this->generate_specified_motion_joint_trajectory_(
            &CcAffordancePlanner::generate_approach_motion_joint_trajectory,
            &CcAffordancePlanner::generate_approach_motion_joint_trajectory, cc_model.slist, secondary_joint_goals,
            nof_secondary_joints, task_description.trajectory_density, rm_planning_enabled);
    }

    else // task_description.motion_type == MotionType::AFFORDANCE
    {
        // Compose the closed-chain model screws
        const Eigen::MatrixXd cc_slist = rm_planning_enabled
            ? affordance_util::compose_rm_cc_model_slist(robot_description, aff, reserve_mobility, vir_screw_order)
            : affordance_util::compose_cc_model_slist(robot_description, aff, vir_screw_order);

        // Extract and construct the secondary joint goals
        const Eigen::VectorXd secondary_joint_goals =
            (Eigen::VectorXd(nof_secondary_joints) << task_description.goal.ee_orientation,
             task_description.goal.affordance)
                .finished();

        // Solve the joint trajectory for the given task. This function calls the Cc Affordance Planner inside.
        plannerResult = this->generate_specified_motion_joint_trajectory_(
            &CcAffordancePlanner::generate_affordance_motion_joint_trajectory,
            &CcAffordancePlanner::generate_affordance_motion_joint_trajectory, cc_slist, secondary_joint_goals,
            nof_secondary_joints, task_description.trajectory_density, rm_planning_enabled);
    }

    std::vector<double> gripper_joint_trajectory;

    // Compute gripper trajectory if gripper goal is provided
    if (!std::isnan(task_description.goal.gripper))
    {
        const int trajectory_size = static_cast<int>(plannerResult.joint_trajectory.size()) + 1;
        gripper_joint_trajectory = affordance_util::compute_gripper_joint_trajectory(
            task_description.gripper_goal_type, robot_description.gripper_state, task_description.goal.gripper,
            trajectory_size);

        // Indicate result includes gripper trajectory
        plannerResult.includes_gripper_trajectory = true;
    }

    // Convert the differential closed-chain joint trajectory to robot joint trajectory
    if (rm_planning_enabled && !plannerResult.joint_trajectory.empty())
    {
        plannerResult.reserve_trajectory.insert(plannerResult.reserve_trajectory.begin(),
                                                reserve_mobility.initial_state);
        plannerResult.reserve_pose_trajectory.insert(
            plannerResult.reserve_pose_trajectory.begin(),
            affordance_util::FKinSpace(Eigen::Matrix4d::Identity(), reserve_mobility.slist,
                                       reserve_mobility.initial_state));
        plannerResult.residual_mobility_norm.insert(plannerResult.residual_mobility_norm.begin(), 0.0);
        plannerResult.reserve_active.insert(plannerResult.reserve_active.begin(), false);
        plannerResult.active_arm_dof_count.insert(plannerResult.active_arm_dof_count.begin(),
                                                  static_cast<size_t>(feasible_arm_dof_for_diagnostics));
    }
    this->convert_cc_traj_to_robot_traj_(
        plannerResult.joint_trajectory, robot_description.joint_states,
        gripper_joint_trajectory); // Will insert gripper_joint_trajectory if task description included a gripper goal

    // Record the task description used for planning
    plannerResult.task_description = task_description;
    plannerResult.task_description.reserve_mobility = reserve_mobility;
    plannerResult.task_description.affordance_info = aff; // Update affordance info if obtained from FK
    plannerResult.task_description.goal.canonical_pose = canonical_pose; // Update canonical pose if obtained from FK

    // Return the result
    return plannerResult;
}

PlannerResult CcAffordancePlannerInterface::generate_specified_motion_joint_trajectory_(
    const Gsmt &generate_specified_motion_joint_trajectory,
    const Gsmt_st &generate_specified_motion_joint_trajectory_st, const Eigen::MatrixXd &slist,
    const Eigen::VectorXd &secondary_joint_goals, const size_t &nof_secondary_joints, const int &trajectory_density,
    const bool reserve_mobility_enabled)

{
    PlannerResult transposeResult; // Result from the transpose planner
    PlannerResult inverseResult;   // Result from the inverse planner
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> transpose_result_obtained{false};
    std::atomic<bool> inverse_result_obtained{false};

    // Construct inverse and transpose planner objects
    CcAffordancePlanner *ccAffordancePlannerInversePtr(&ccAffordancePlannerInverse_);
    CcAffordancePlanner *ccAffordancePlannerTransposePtr(&ccAffordancePlannerTranspose_);

    if (reserve_mobility_enabled)
    {
        inverseResult = (ccAffordancePlannerInversePtr->*generate_specified_motion_joint_trajectory)(
            slist, secondary_joint_goals, nof_secondary_joints, trajectory_density);
        inverseResult.update_method = UpdateMethod::INVERSE;
        if (planner_config_.enable_capability_aware_planning)
        {
            inverseResult.update_trail += "capability-aware rm-cca inverse";
        }
        else if (planner_config_.enable_joint_limits && planner_config_.enable_nullspace_planning)
        {
            inverseResult.update_trail += "rm-cca inverse";
        }
        else if (planner_config_.enable_joint_limits)
        {
            inverseResult.update_trail += "bound-aware whole-body inverse";
        }
        else
        {
            inverseResult.update_trail += "null-space rm-cca inverse (joint limits disabled)";
        }
        return inverseResult;
    }

    // If a specific update method is requested, run the planner using that
    if (planner_config_.update_method == UpdateMethod::INVERSE)
    {
        inverseResult = (ccAffordancePlannerInversePtr->*generate_specified_motion_joint_trajectory)(
            slist, secondary_joint_goals, nof_secondary_joints, trajectory_density);
        inverseResult.update_method = UpdateMethod::INVERSE;
        inverseResult.update_trail += "inverse";
        return inverseResult;
    }

    if (planner_config_.update_method == UpdateMethod::TRANSPOSE)
    {
        transposeResult = (ccAffordancePlannerTransposePtr->*generate_specified_motion_joint_trajectory)(
            slist, secondary_joint_goals, nof_secondary_joints, trajectory_density);
        transposeResult.update_method = UpdateMethod::TRANSPOSE;
        transposeResult.update_trail += "transpose";
        return transposeResult;
    }

    // If a specific update method is not requested, run the inverse and transpose planners concurrently in separate
    // threads
    std::jthread inverse_thread([&](std::stop_token st) {
        inverseResult = (ccAffordancePlannerInversePtr->*generate_specified_motion_joint_trajectory_st)(
            slist, secondary_joint_goals, nof_secondary_joints, trajectory_density, st);
        {
            std::unique_lock<std::mutex> lock(mtx);
            inverseResult.update_method = UpdateMethod::INVERSE;
            inverseResult.update_trail += "inverse";
            inverse_result_obtained = true;
        }
        cv.notify_all();
    });

    std::jthread transpose_thread([&](std::stop_token st) {
        transposeResult = (ccAffordancePlannerTransposePtr->*generate_specified_motion_joint_trajectory_st)(
            slist, secondary_joint_goals, nof_secondary_joints, trajectory_density, st);
        {
            std::unique_lock<std::mutex> lock(mtx);
            transposeResult.update_method = UpdateMethod::TRANSPOSE;
            transposeResult.update_trail += "transpose";
            transpose_result_obtained = true;
        }
        cv.notify_all();
    });

    // Wake the main thread up when a result is found
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&inverse_result_obtained, &transpose_result_obtained]() {
            return (inverse_result_obtained.load() || transpose_result_obtained.load());
        });
    }

    // Analyze the result
    if (inverse_result_obtained.load()) // Inverse planner returned first
    {
        if (inverseResult.trajectory_description == TrajectoryDescription::FULL)
        {

            transpose_thread.request_stop();
            return inverseResult;
        }
        else
        // If inverse planner returned partial or no trajectory then, wait for the transpose planner
        {
            transpose_thread.join();
        }
    }
    else // Transpose planner must have returned first
    {
        if (transposeResult.trajectory_description == TrajectoryDescription::FULL)
        {

            inverse_thread.request_stop();
            return transposeResult;
        }
        else
        // If transpose planner returned partial or no trajectory then, wait for the inverse planner
        {
            inverse_thread.join();
        }
    }

    // At this point, both planners have run. Analyze which trajectory is fuller.
    if ((inverseResult.trajectory_description == TrajectoryDescription::PARTIAL) &&
        (transposeResult.trajectory_description == TrajectoryDescription::UNSET))
    {
        inverseResult.update_trail += " --> transpose unset --> inverse partial";
        return inverseResult;
    }
    else if ((inverseResult.trajectory_description == TrajectoryDescription::UNSET) &&
             (transposeResult.trajectory_description == TrajectoryDescription::PARTIAL))
    {
        transposeResult.update_trail += " --> inverse unset --> transpose partial";
        return transposeResult;
    }
    else // both must be partial so, return whichever has a longer trajectory
    {
        inverseResult.update_trail += " --> transpose and inverse partial --> inverse longer traj";
        transposeResult.update_trail += " --> transpose and inverse partial --> transpose longer traj";
        return (inverseResult.joint_trajectory.size() > transposeResult.joint_trajectory.size()) ? inverseResult
                                                                                                 : transposeResult;
    }
}

void CcAffordancePlannerInterface::convert_cc_traj_to_robot_traj_(std::vector<Eigen::VectorXd> &cc_trajectory,
                                                                  const Eigen::VectorXd &start_joint_states,
                                                                  const std::vector<double> &gripper_joint_trajectory)
{
    // Return if the trajectory is empty, no need to attempt conversion
    if (cc_trajectory.empty())
    {
        return;
    }

    const size_t nof_robot_joints = start_joint_states.size();
    const size_t total_joints = cc_trajectory[0].size() + (gripper_joint_trajectory.empty() ? 0 : 1);

    Eigen::VectorXd cc_start_joint_states = Eigen::VectorXd::Zero(total_joints);
    cc_start_joint_states.head(nof_robot_joints) = start_joint_states;

    // Lambda expression to handle both cases of transform with or without gripper trajectory
    auto transform_point = [&](const Eigen::VectorXd &point,
                               std::optional<double> gripper_value = std::nullopt) -> Eigen::VectorXd {
        if (gripper_value)
        {
            Eigen::VectorXd new_point(total_joints);
            new_point.head(nof_robot_joints) = point.head(nof_robot_joints);
            new_point[nof_robot_joints] = 0.0;
            new_point.tail(point.size() - nof_robot_joints) = point.tail(point.size() - nof_robot_joints);
            cc_start_joint_states[nof_robot_joints] = *gripper_value;
            return new_point + cc_start_joint_states;
        }
        else
        {
            return point + cc_start_joint_states;
        }
    };

    // If gripper trajectory is provided, we insert it in the final trajectory
    if (!gripper_joint_trajectory.empty())
    {
        std::transform(cc_trajectory.begin(), cc_trajectory.end(), gripper_joint_trajectory.begin() + 1,
                       cc_trajectory.begin(), transform_point);

        // We start with the second point in the gripper trajectory for std::transform cuz we'll insert the first
        // element below

        cc_start_joint_states[nof_robot_joints] = gripper_joint_trajectory[0];
    }

    // Insert the start state at the beginning of the trajectory
    cc_trajectory.insert(cc_trajectory.begin(), cc_start_joint_states);

    // If gripper trajectory is not provided, we transform the trajectory without gripper consideration
    if (gripper_joint_trajectory.empty())
    {
        std::transform(cc_trajectory.begin() + 1, cc_trajectory.end(), cc_trajectory.begin() + 1,
                       [&](Eigen::VectorXd &point) { return transform_point(point); });
    }
}

void CcAffordancePlannerInterface::validate_input_(const affordance_util::RobotDescription &robot_description,
                                                   const TaskDescription &task_description)
{
    /* Validate robot description */
    if (robot_description.slist.size() == 0)
    {
        throw std::invalid_argument("Robot description: 'slist' cannot be empty.");
    }

    if (robot_description.M.hasNaN())
    {
        throw std::invalid_argument("Robot description: 'M' (palm HTM) must be specified.");
    }

    // Lambda to check if a matrix is a valid homogeneous transformation matrix
    const double tolerance = 1e-4;
    auto is_valid_htm = [tolerance](const Eigen::Matrix4d& T) -> bool {
        const Eigen::Matrix3d R = T.block<3,3>(0,0);
        
        // Check if R is a proper rotation matrix
        bool valid_rotation = 
            R.isUnitary(tolerance) &&  // R^T * R = I
            (std::abs(R.determinant() - 1.0) < tolerance);  // det(R) = +1
        
        // Check if bottom row is [0, 0, 0, 1]
        bool valid_bottom = 
            T.row(3).head(3).isZero(tolerance) &&
            (std::abs(T(3,3) - 1.0) < tolerance);
        
        return valid_rotation && valid_bottom;
    };

    if (!is_valid_htm(robot_description.M))
    {
        throw std::invalid_argument("Robot description: 'M' is not a valid transformation matrix.");
    }

    if (robot_description.joint_states.size() == 0)
    {
        throw std::invalid_argument("Robot description: 'joint_states' cannot be empty.");
    }

    if (robot_description.slist.cols() != robot_description.joint_states.size())
    {
        throw std::invalid_argument(
            "Robot description: 'joint_states' size must match the number of columns in 'slist'.");
    }

    const bool has_lower_limits = robot_description.joint_lower_limits.size() != 0;
    const bool has_upper_limits = robot_description.joint_upper_limits.size() != 0;
    if (has_lower_limits != has_upper_limits)
    {
        throw std::invalid_argument("Robot description: joint lower and upper limits must both be supplied or empty.");
    }
    if (has_lower_limits &&
        (robot_description.joint_lower_limits.size() != robot_description.joint_states.size() ||
         robot_description.joint_upper_limits.size() != robot_description.joint_states.size()))
    {
        throw std::invalid_argument("Robot description: joint-limit sizes must match the robot DOF.");
    }
    for (Eigen::Index i = 0; i < robot_description.joint_lower_limits.size(); ++i)
    {
        const double lower = robot_description.joint_lower_limits(i);
        const double upper = robot_description.joint_upper_limits(i);
        if (std::isnan(lower) || std::isnan(upper) || !(lower < upper))
        {
            throw std::invalid_argument("Robot description: every joint lower limit must be less than its upper limit.");
        }
        if (robot_description.joint_states(i) < lower || robot_description.joint_states(i) > upper)
        {
            throw std::invalid_argument("Robot description: a starting joint state lies outside its joint limits.");
        }
        const double effective_width =
            std::isfinite(lower) && std::isfinite(upper)
                ? planner_config_.soft_limit_ratio * (upper - lower)
                : std::numeric_limits<double>::infinity();
        if (task_description.reserve_mobility.enabled && planner_config_.enable_joint_limits &&
            !(2.0 * planner_config_.joint_limit_margin < effective_width))
        {
            throw std::invalid_argument("Planner soft joint limit and margin leave no safe interval for an arm joint.");
        }
    }

    if ((!std::isnan(task_description.goal.gripper)) && (std::isnan(robot_description.gripper_state)))
    {
        throw std::invalid_argument("Robot description: 'gripper_state' must be supplied and cannot be NaN when "
                                    "goal.gripper is specified in task description.");
    }

    const affordance_util::ReserveMobilityDescription &reserve = task_description.reserve_mobility;
    const bool rm_planning_enabled =
        reserve.enabled &&
        (planner_config_.enable_joint_limits || planner_config_.enable_nullspace_planning ||
         planner_config_.enable_capability_aware_planning);
    if (rm_planning_enabled)
    {
        const Eigen::Index reserve_dof = reserve.slist.cols();
        if (reserve.slist.rows() != 6 || reserve_dof == 0 || !reserve.slist.allFinite())
        {
            throw std::invalid_argument("Reserve mobility: 'slist' must be a finite 6 x n matrix with n > 0.");
        }
        if (reserve.initial_state.size() != reserve_dof || !reserve.initial_state.allFinite())
        {
            throw std::invalid_argument("Reserve mobility: 'initial_state' must be finite and match reserve DOF.");
        }
        const bool reserve_has_lower = reserve.lower_limits.size() != 0;
        const bool reserve_has_upper = reserve.upper_limits.size() != 0;
        if (reserve_has_lower != reserve_has_upper ||
            (reserve_has_lower &&
             (reserve.lower_limits.size() != reserve_dof || reserve.upper_limits.size() != reserve_dof)))
        {
            throw std::invalid_argument("Reserve mobility: lower/upper limits must both be empty or match reserve DOF.");
        }
        for (Eigen::Index i = 0; i < reserve.lower_limits.size(); ++i)
        {
            if (std::isnan(reserve.lower_limits(i)) || std::isnan(reserve.upper_limits(i)) ||
                reserve.lower_limits(i) > reserve.upper_limits(i) ||
                reserve.initial_state(i) < reserve.lower_limits(i) || reserve.initial_state(i) > reserve.upper_limits(i))
            {
                throw std::invalid_argument("Reserve mobility: invalid limits or initial state outside limits.");
            }
        }
        if (reserve.max_step.size() != 0 && reserve.max_step.size() != reserve_dof)
        {
            throw std::invalid_argument("Reserve mobility: 'max_step' must be empty or match reserve DOF.");
        }
        for (Eigen::Index i = 0; i < reserve.max_step.size(); ++i)
        {
            if (!(reserve.max_step(i) > 0.0))
            {
                throw std::invalid_argument("Reserve mobility: every max-step value must be positive.");
            }
        }
    }
    /* Validate task description */

    if (task_description.affordance_info.type == affordance_util::ScrewType::UNSET)
    {
        throw std::invalid_argument("Task description: 'affordance_info.type' must be specified.");
    }

    if (task_description.affordance_info_from.method != affordance_util::PoseSpecificationMethod::FROM_FK &&
        ((task_description.affordance_info.axis.hasNaN() ||
         task_description.affordance_info.location.hasNaN()) && task_description.affordance_info.screw.hasNaN()))
    {
        throw std::invalid_argument("Task description: Either 'affordance_info.axis' and 'affordance_info.location', "
                                    "or 'affordance_info.screw' must be specified.");
    }

    if (task_description.affordance_info_from.method == affordance_util::PoseSpecificationMethod::FROM_FK &&
        (task_description.affordance_info.axis.hasNaN() && task_description.affordance_info_from.axis_in_final_pose.hasNaN()))
    {
        throw std::invalid_argument(
            "Task description: For 'affordance_info_from.method = FROM_FK', either 'affordance_info.axis' or 'affordance_info_from.axis_in_final_pose' must be specified.");
    }

    if (task_description.affordance_info.type == affordance_util::ScrewType::SCREW &&
        task_description.affordance_info.screw.hasNaN() && std::isnan(task_description.affordance_info.pitch))
    {
        throw std::invalid_argument("Task description: For 'SCREW' type affordance_info, if screw is not filled out, "
                                    "'pitch' must be specified.");
    }

    if (!task_description.affordance_info.axis.hasNaN() &&
        std::abs(task_description.affordance_info.axis.norm() - 1) > tolerance)
    {
        throw std::invalid_argument("Task description: 'affordance_info.axis' must be a unit vector");
    }

    if (std::isnan(task_description.goal.affordance))
    {

        throw std::invalid_argument("Task description: 'goal.affordance' must be specified and cannot be NaN.");
    }

    if ((task_description.motion_type == MotionType::APPROACH) && (!is_valid_htm(task_description.goal.canonical_pose)))
    {
        throw std::invalid_argument("Task description: 'canonical_pose' is not a valid transformation matrix. "
            "Valid canonical pose is needed for approach motion.");
    }

    if (task_description.trajectory_density < 2)
    {

        throw std::invalid_argument("Task description: 'trajectory_density' must be >= 2.");
    }

    const Eigen::Index virtual_dof = task_description.vir_screw_order == affordance_util::VirtualScrewOrder::NONE
                                         ? 0
                                         : affordance_util::get_vir_screw_axes(task_description.vir_screw_order).cols();
    if (task_description.goal.ee_orientation.size() > virtual_dof)
    {
        throw std::invalid_argument("Task description: EE-orientation goals exceed the selected virtual-screw DOF.");
    }
}

} // namespace cc_affordance_planner
