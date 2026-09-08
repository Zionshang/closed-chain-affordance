#include <affordance_util/affordance_util.hpp>
#include <cc_affordance_planner/cc_affordance_planner.hpp>
#include <algorithm>
#include <limits>
#include <unordered_set>

namespace cc_affordance_planner
{

TaskDescription::TaskDescription(const PlanningType &planningType)
{

    if (planningType == PlanningType::EE_ORIENTATION_ONLY)
    {
        // Model Setting -- EE_ORIENTATION_ONLY is simply a special case of the AFFORDANCE motion
        motion_type = cc_affordance_planner::MotionType::AFFORDANCE;
        vir_screw_order = affordance_util::VirtualScrewOrder::NONE;

        // Affordance Info
        affordance_info.type = affordance_util::ScrewType::ROTATION;
        affordance_info_from.method = affordance_util::PoseSpecificationMethod::FROM_FK;
    }
    else if (planningType == PlanningType::CARTESIAN_GOAL)
    {
        // Model Setting -- CARTESIAN_GOAL is simply a special case of the APPROACH motion
        motion_type = cc_affordance_planner::MotionType::APPROACH;
        vir_screw_order = affordance_util::VirtualScrewOrder::NONE;

        // Affordance Info
        affordance_info.type = affordance_util::ScrewType::ROTATION;

        // The cartesian goal is simply the affordance reference pose i.e. a pose at which the affordance is zero for
        // the APPROACH motion. Doesn't matter what affordance we chose since at its zero, the affordance has not caused
        // any transformation yet.
        affordance_info.axis = Eigen::Vector3d::Ones().normalized(); // A random axis
        affordance_info.location = Eigen::Vector3d::Ones(); // A random location
        constexpr double eps = 1e-5; // An epsilon
        goal.affordance = eps;
    }
    else if (planningType == PlanningType::APPROACH)
    {
        // Model Setting -- Set to common use-case default values
        motion_type = cc_affordance_planner::MotionType::APPROACH;
        vir_screw_order = affordance_util::VirtualScrewOrder::NONE; // Constrain the EE orientation to what's dictated
                                                                    // by the approach path
    }
}

void CcAffordancePlannerTranspose::update_theta_p(Eigen::VectorXd &theta_p, const Eigen::VectorXd &theta_sd,
                                                  const Eigen::VectorXd &theta_s, const Eigen::MatrixXd &N)
{

    //**Alg2:L11: Update theta_p using Eqn. 24 but approximate the inverse with transpose
    const Eigen::VectorXd delta_theta_p = N.transpose() * (theta_sd - theta_s);

    // Update thate_p using Newton-Raphson
    theta_p += delta_theta_p;
}

void CcAffordancePlannerInverse::update_theta_p(Eigen::VectorXd &theta_p, const Eigen::VectorXd &theta_sd,
                                                const Eigen::VectorXd &theta_s, const Eigen::MatrixXd &N)
{

    const Eigen::MatrixXd pinv_N = N.completeOrthogonalDecomposition().pseudoInverse(); // pseudo-inverse of N
    const double cond_N = N.norm() * pinv_N.norm();
    Eigen::VectorXd delta_theta_p(nof_pjoints_);

    // If N is near-singular use Damped Least Squares
    if (cond_N > cond_N_threshold_)
    {
        dls_flag_ = true;
        const Eigen::MatrixXd NNt = N * N.transpose();
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(NNt.rows(), NNt.cols());

        // Compute delta_theta_p using the Damped-Least-Squares method
        delta_theta_p = N.transpose() *
                        ((NNt + std::pow(lambda_, 2) * I).completeOrthogonalDecomposition().pseudoInverse()) *
                        (theta_sd - theta_s);
    }
    else // Use the regular inverse method
    {
        //**Alg2:L11: Update theta_p using Eqn. 24
        delta_theta_p = pinv_N * (theta_sd - theta_s);
    }

    // Update thate_p using Newton-Raphson
    theta_p += delta_theta_p;
}
CcAffordancePlanner::CcAffordancePlanner(const PlannerConfig &plannerConfig)
    : accuracy_(plannerConfig.accuracy),
      eps_rw_(plannerConfig.closure_err_threshold_ang),
      eps_rv_(plannerConfig.closure_err_threshold_lin),
      max_itr_l_(plannerConfig.ik_max_itr),
      enable_joint_limits_(plannerConfig.enable_joint_limits),
      enable_nullspace_planning_(plannerConfig.enable_nullspace_planning),
      enable_capability_aware_planning_(plannerConfig.enable_capability_aware_planning),
      svd_relative_tolerance_(plannerConfig.svd_relative_tolerance),
      svd_absolute_tolerance_(plannerConfig.svd_absolute_tolerance),
      residual_mobility_tolerance_(plannerConfig.residual_mobility_tolerance),
      soft_limit_ratio_(plannerConfig.soft_limit_ratio),
      arm_mobility_weight_(plannerConfig.arm_mobility_weight),
      joint_limit_barrier_gain_(plannerConfig.joint_limit_barrier_gain),
      joint_limit_barrier_epsilon_(plannerConfig.joint_limit_barrier_epsilon),
      base_translation_weight_(plannerConfig.base_translation_weight),
      base_rotation_weight_(plannerConfig.base_rotation_weight),
      closure_secondary_weight_(plannerConfig.closure_secondary_weight),
      joint_limit_margin_(plannerConfig.joint_limit_margin),
      joint_limit_tolerance_(plannerConfig.joint_limit_tolerance)
{
}

void CcAffordancePlanner::configure_reserve_mobility(
    const affordance_util::ReserveMobilityDescription &reserve_mobility,
    const Eigen::VectorXd &arm_start_absolute, const Eigen::VectorXd &arm_lower_limits,
    const Eigen::VectorXd &arm_upper_limits, const std::vector<size_t> &reserve_primary_indices,
    const std::vector<size_t> &arm_primary_indices)
{
    const Eigen::Index reserve_dof = reserve_mobility.slist.cols();
    if (!reserve_mobility.enabled || reserve_mobility.slist.rows() != 6 || reserve_dof == 0 ||
        reserve_mobility.initial_state.size() != reserve_dof)
    {
        throw std::invalid_argument("Invalid enabled reserve-mobility description.");
    }
    if (reserve_primary_indices.size() != static_cast<size_t>(reserve_dof) ||
        arm_start_absolute.size() != static_cast<Eigen::Index>(arm_primary_indices.size()) ||
        arm_lower_limits.size() != arm_start_absolute.size() || arm_upper_limits.size() != arm_start_absolute.size())
    {
        throw std::invalid_argument("RM-CCA partition, state, and arm-limit dimensions are inconsistent.");
    }
    const bool reserve_has_lower = reserve_mobility.lower_limits.size() != 0;
    const bool reserve_has_upper = reserve_mobility.upper_limits.size() != 0;
    if (reserve_has_lower != reserve_has_upper ||
        (reserve_has_lower && (reserve_mobility.lower_limits.size() != reserve_dof ||
                               reserve_mobility.upper_limits.size() != reserve_dof)) ||
        (reserve_mobility.max_step.size() != 0 && reserve_mobility.max_step.size() != reserve_dof))
    {
        throw std::invalid_argument("Reserve limit and max-step vectors must be empty or match reserve DOF.");
    }

    std::unordered_set<size_t> primary_partition;
    for (const size_t index : reserve_primary_indices)
    {
        if (!primary_partition.insert(index).second)
        {
            throw std::invalid_argument("Reserve primary indices contain duplicates.");
        }
    }
    for (const size_t index : arm_primary_indices)
    {
        if (!primary_partition.insert(index).second)
        {
            throw std::invalid_argument("Arm and reserve primary indices overlap or contain duplicates.");
        }
    }
    if (primary_partition.size() != reserve_primary_indices.size() + arm_primary_indices.size())
    {
        throw std::invalid_argument("Invalid RM-CCA primary partition.");
    }
    for (Eigen::Index i = 0; i < arm_start_absolute.size(); ++i)
    {
        if (std::isnan(arm_lower_limits(i)) || std::isnan(arm_upper_limits(i)) ||
            !(arm_lower_limits(i) < arm_upper_limits(i)))
        {
            throw std::invalid_argument("Every arm lower limit must be strictly less than its upper limit.");
        }
    }
    for (Eigen::Index i = 0; i < reserve_mobility.lower_limits.size(); ++i)
    {
        if (std::isnan(reserve_mobility.lower_limits(i)) || std::isnan(reserve_mobility.upper_limits(i)) ||
            reserve_mobility.lower_limits(i) > reserve_mobility.upper_limits(i) ||
            reserve_mobility.initial_state(i) < reserve_mobility.lower_limits(i) ||
            reserve_mobility.initial_state(i) > reserve_mobility.upper_limits(i))
        {
            throw std::invalid_argument("Invalid reserve limits or initial reserve state outside limits.");
        }
    }
    for (Eigen::Index i = 0; i < reserve_mobility.max_step.size(); ++i)
    {
        if (!(reserve_mobility.max_step(i) > 0.0))
        {
            throw std::invalid_argument("Every reserve max-step value must be positive.");
        }
    }

    reserve_mobility_enabled_ = true;
    reserve_mobility_ = reserve_mobility;
    arm_start_absolute_ = arm_start_absolute;
    arm_lower_limits_ = arm_lower_limits;
    arm_upper_limits_ = arm_upper_limits;
    reserve_primary_indices_ = reserve_primary_indices;
    arm_primary_indices_ = arm_primary_indices;
}

void CcAffordancePlanner::disable_reserve_mobility()
{
    reserve_mobility_enabled_ = false;
    reserve_primary_indices_.clear();
    arm_primary_indices_.clear();
}

Eigen::VectorXd CcAffordancePlanner::make_primary_start_guess() const
{
    Eigen::VectorXd theta_pg = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nof_pjoints_));
    if (reserve_mobility_enabled_)
    {
        scatter_update(theta_pg, reserve_primary_indices_, reserve_mobility_.initial_state);
    }
    return theta_pg;
}

Eigen::VectorXd CcAffordancePlanner::make_primary_mobility_metric(const Eigen::VectorXd &theta_p) const
{
    if (theta_p.size() != static_cast<Eigen::Index>(nof_pjoints_))
    {
        throw std::invalid_argument("Capability metric state does not match the primary-coordinate count.");
    }

    Eigen::VectorXd metric = Eigen::VectorXd::Constant(theta_p.size(), arm_mobility_weight_);
    for (size_t i = 0; i < reserve_primary_indices_.size(); ++i)
    {
        metric(static_cast<Eigen::Index>(reserve_primary_indices_[i])) =
            i < 3 ? base_translation_weight_ : base_rotation_weight_;
    }
    if (!enable_joint_limits_)
    {
        return metric;
    }
    for (size_t i = 0; i < arm_primary_indices_.size(); ++i)
    {
        const Eigen::Index metric_index = static_cast<Eigen::Index>(arm_primary_indices_[i]);
        const double lower = arm_lower_limits_(static_cast<Eigen::Index>(i));
        const double upper = arm_upper_limits_(static_cast<Eigen::Index>(i));
        if (!std::isfinite(lower) || !std::isfinite(upper))
        {
            continue;
        }
        const double center = lower + 0.5 * (upper - lower);
        const double soft_half_range = 0.5 * soft_limit_ratio_ * (upper - lower);
        const double absolute_state = arm_start_absolute_(static_cast<Eigen::Index>(i)) + theta_p(metric_index);
        const double normalized_distance = std::min(1.0, std::abs(absolute_state - center) / soft_half_range);
        const double squared_distance = normalized_distance * normalized_distance;
        const double denominator = 1.0 - squared_distance + joint_limit_barrier_epsilon_;
        metric(metric_index) +=
            joint_limit_barrier_gain_ * squared_distance / (denominator * denominator);
    }
    return metric;
}

void CcAffordancePlanner::append_trajectory_point(PlannerResult &result,
                                                  const Eigen::VectorXd &closed_chain_point) const
{
    if (!reserve_mobility_enabled_)
    {
        result.joint_trajectory.push_back(closed_chain_point);
        return;
    }

    const Eigen::VectorXd theta_p = closed_chain_point.head(static_cast<Eigen::Index>(nof_pjoints_));
    const Eigen::VectorXd arm_state = select_columns(theta_p.transpose(), arm_primary_indices_).transpose();
    const Eigen::VectorXd reserve_state = select_columns(theta_p.transpose(), reserve_primary_indices_).transpose();
    Eigen::VectorXd arm_and_secondary(arm_state.size() + static_cast<Eigen::Index>(nof_sjoints_));
    arm_and_secondary << arm_state, closed_chain_point.tail(static_cast<Eigen::Index>(nof_sjoints_));
    result.joint_trajectory.push_back(arm_and_secondary);
    result.reserve_trajectory.push_back(reserve_state);
    result.reserve_pose_trajectory.push_back(
        affordance_util::FKinSpace(Eigen::Matrix4d::Identity(), reserve_mobility_.slist, reserve_state));
    result.residual_mobility_norm.push_back(last_rm_ik_diagnostics_.residual_norm);
    result.active_arm_dof_count.push_back(last_rm_ik_diagnostics_.active_arm_dof_count);

    const bool was_active = !result.reserve_active.empty() && result.reserve_active.back();
    result.reserve_active.push_back(last_rm_ik_diagnostics_.reserve_active);
    if (last_rm_ik_diagnostics_.reserve_active && !was_active)
    {
        ++result.reserve_activation_count;
    }
    result.reserve_mobility_used = result.reserve_mobility_used || last_rm_ik_diagnostics_.reserve_active;
}

PlannerResult CcAffordancePlanner::generate_approach_motion_joint_trajectory(const Eigen::MatrixXd &slist,
                                                                             const Eigen::VectorXd &theta_sdf,
                                                                             const size_t &task_offset_tau,
                                                                             const int &stepper_max_itr_m,
                                                                             std::stop_token st)
{

    auto start_time = std::chrono::high_resolution_clock::now(); // Monitor clock to track planning time

    PlannerResult plannerResult; // Result of the planner

    // Clamp theta_sdf to minimum goal magnitude to avoid potential numerical issues
    const Eigen::VectorXd theta_sdf_clamped = affordance_util::clamp_to_magnitude_minimum(theta_sdf, goal_min_);

    // Extract affordance and approach goals
    const double theta_adf = theta_sdf_clamped.tail(1)(0); // affordance screw goal
    const double theta_pdf = theta_sdf_clamped.tail(2)(0); // approach screw goal

    //**Alg1:L1: Define affordance step, deltatheta_a
    const double deltatheta_a = theta_adf / (stepper_max_itr_m - 1);
    const double deltatheta_p = theta_pdf / (stepper_max_itr_m - 1);

    //** Alg1:L2: Determine relevant matrix and vector sizes based on task_offset_tau
    nof_pjoints_ = slist.cols() - task_offset_tau;
    nof_sjoints_ = task_offset_tau;

    /// Compute the elementwise error tolerance for seconday joint goals
    theta_s_tol_ = theta_sdf_clamped;
    theta_s_tol_.tail(1)(0) = deltatheta_a;
    theta_s_tol_.tail(2)(0) = deltatheta_p;
    theta_s_tol_ = (accuracy_ * theta_s_tol_).cwiseAbs();

    //**Alg1:L3 and Alg1:L2: Set start guesses and step goal
    Eigen::VectorXd theta_sg = Eigen::VectorXd::Zero(nof_sjoints_);
    Eigen::VectorXd theta_pg = this->make_primary_start_guess();
    Eigen::VectorXd theta_sd = theta_sdf_clamped; // We set the affordance goal in the loop in reference to the start state
    theta_sd.tail(2).setConstant(0); // start approach and affordance goals at 0 but gripper orientation as specified

    //**Alg1:L4: Compute no. of iterations, stepper_max_itr_m to final goal: Passed in as planner config

    //**Alg1:L5: Initialize loop counter, loop_counter_k; success counter, success_counter_s
    int loop_counter_k = 0;
    int success_counter_s = 0;

    while (loop_counter_k < (stepper_max_itr_m - 1) && !st.stop_requested()) //**Alg1:L6
    {
        loop_counter_k = loop_counter_k + 1; //**Alg1:L7:

        //**Alg1:L8: Update aff and approach step goal:
        // Set the affordance step goal as aff_step away from the current pose. Affordance is the last element of
        // theta_sd
        theta_sd(nof_sjoints_ - 1) = theta_sd(nof_sjoints_ - 1) - deltatheta_a;
        theta_sd(nof_sjoints_ - 2) = theta_sd(nof_sjoints_ - 2) - deltatheta_p;

        //**Alg1:L13: Call Algorithm 2 with args, theta_sd, theta_pg, theta_sg, slist
        std::optional<Eigen::VectorXd> ik_result = this->call_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd, st);

        if (ik_result.has_value()) //**Alg1:L14
        {
            //**Alg1:L15: Record solution, theta_p, theta_sg
            const Eigen::VectorXd &traj_point = ik_result.value();
            this->append_trajectory_point(plannerResult, traj_point);

            //**Alg1:L16 Update guesses, theta_pg, theta_sg
            theta_sg = traj_point.tail(nof_sjoints_);
            theta_pg = traj_point.head(nof_pjoints_);

            success_counter_s = success_counter_s + 1; //**Alg1:L17
        }                                              //**Alg1:L18

    } //**Alg1:L19

    // Capture the planning time
    auto end_time = std::chrono::high_resolution_clock::now();
    plannerResult.planning_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Set planning result
    if (!plannerResult.joint_trajectory.empty())
    {
        plannerResult.success = true;

        if (loop_counter_k == success_counter_s)
        {
            plannerResult.trajectory_description = TrajectoryDescription::FULL;
        }
        else
        {
            plannerResult.trajectory_description = TrajectoryDescription::PARTIAL;
        }
    }
    else
    {
        plannerResult.success = false;
        plannerResult.trajectory_description = TrajectoryDescription::UNSET;
    }

    // Indicate if DLS was used
    plannerResult.update_trail = dls_flag_ ? "dls and " : plannerResult.update_trail;

    return plannerResult;
}

PlannerResult CcAffordancePlanner::generate_approach_motion_joint_trajectory(const Eigen::MatrixXd &slist,
                                                                             const Eigen::VectorXd &theta_sdf,
                                                                             const size_t &task_offset_tau,
                                                                             const int &stepper_max_itr_m)
{

    auto start_time = std::chrono::high_resolution_clock::now(); // Monitor clock to track planning time

    PlannerResult plannerResult; // Result of the planner

    // Clamp theta_sdf to minimum goal magnitude to avoid potential numerical issues
    const Eigen::VectorXd theta_sdf_clamped = affordance_util::clamp_to_magnitude_minimum(theta_sdf, goal_min_);

    // Extract affordance and approach goals
    const double theta_adf = theta_sdf_clamped.tail(1)(0); // affordance screw goal
    const double theta_pdf = theta_sdf_clamped.tail(2)(0); // approach screw goal

    //**Alg1:L1: Define affordance step, deltatheta_a
    const double deltatheta_a = theta_adf / (stepper_max_itr_m - 1);
    const double deltatheta_p = theta_pdf / (stepper_max_itr_m - 1);

    //** Alg1:L2: Determine relevant matrix and vector sizes based on task_offset_tau
    nof_pjoints_ = slist.cols() - task_offset_tau;
    nof_sjoints_ = task_offset_tau;

    /// Compute the elementwise error tolerance for seconday joint goals
    theta_s_tol_ = theta_sdf_clamped;
    theta_s_tol_.tail(1)(0) = deltatheta_a;
    theta_s_tol_.tail(2)(0) = deltatheta_p;
    theta_s_tol_ = (accuracy_ * theta_s_tol_).cwiseAbs();

    //**Alg1:L3 and Alg1:L2: Set start guesses and step goal
    Eigen::VectorXd theta_sg = Eigen::VectorXd::Zero(nof_sjoints_);
    Eigen::VectorXd theta_pg = this->make_primary_start_guess();
    Eigen::VectorXd theta_sd = theta_sdf_clamped; // We set the affordance goal in the loop in reference to the start state
    theta_sd.tail(2).setConstant(0); // start approach and affordance goals at 0 but gripper orientation as specified

    //**Alg1:L4: Compute no. of iterations, stepper_max_itr_m to final goal: Passed in as planner config

    //**Alg1:L5: Initialize loop counter, loop_counter_k; success counter, success_counter_s
    int loop_counter_k = 0;
    int success_counter_s = 0;

    while (loop_counter_k < (stepper_max_itr_m - 1)) //**Alg1:L6
    {

        loop_counter_k = loop_counter_k + 1; //**Alg1:L7:

        //**Alg1:L8: Update aff and approach step goal:
        // Set the affordance step goal as aff_step away from the current pose. Affordance is the last element of
        // theta_sd
        theta_sd(nof_sjoints_ - 1) = theta_sd(nof_sjoints_ - 1) - deltatheta_a;
        theta_sd(nof_sjoints_ - 2) = theta_sd(nof_sjoints_ - 2) - deltatheta_p;

        //**Alg1:L13: Call Algorithm 2 with args, theta_sd, theta_pg, theta_sg, slist
        std::optional<Eigen::VectorXd> ik_result = this->call_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd);

        if (ik_result.has_value()) //**Alg1:L14
        {
            //**Alg1:L15: Record solution, theta_p, theta_sg
            const Eigen::VectorXd &traj_point = ik_result.value();
            this->append_trajectory_point(plannerResult, traj_point);

            //**Alg1:L16 Update guesses, theta_pg, theta_sg
            theta_sg = traj_point.tail(nof_sjoints_);
            theta_pg = traj_point.head(nof_pjoints_);

            success_counter_s = success_counter_s + 1; //**Alg1:L17
        }                                              //**Alg1:L18

    } //**Alg1:L19

    // Capture the planning time
    auto end_time = std::chrono::high_resolution_clock::now();
    plannerResult.planning_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Set planning result
    if (!plannerResult.joint_trajectory.empty())
    {
        plannerResult.success = true;

        if (loop_counter_k == success_counter_s)
        {
            plannerResult.trajectory_description = TrajectoryDescription::FULL;
        }
        else
        {
            plannerResult.trajectory_description = TrajectoryDescription::PARTIAL;
        }
    }
    else
    {
        plannerResult.success = false;
        plannerResult.trajectory_description = TrajectoryDescription::UNSET;
    }

    // Indicate if DLS was used
    plannerResult.update_trail = dls_flag_ ? "dls and " : plannerResult.update_trail;

    return plannerResult;
}

PlannerResult CcAffordancePlanner::generate_affordance_motion_joint_trajectory(const Eigen::MatrixXd &slist,
                                                                               const Eigen::VectorXd &theta_sdf,
                                                                               const size_t &task_offset_tau,
                                                                               const int &stepper_max_itr_m,
                                                                               std::stop_token st)
{

    auto start_time = std::chrono::high_resolution_clock::now(); // Monitor clock to track planning time

    PlannerResult plannerResult; // Result of the planner

    // Clamp theta_sdf to minimum goal magnitude to avoid potential numerical issues
    const Eigen::VectorXd theta_sdf_clamped = affordance_util::clamp_to_magnitude_minimum(theta_sdf, goal_min_);

    // Extract affordance goal
    const double theta_adf = theta_sdf_clamped.tail(1)(0);

    //**Alg1:L1: Define affordance step, deltatheta_a
    const double deltatheta_a = theta_adf / (stepper_max_itr_m - 1);

    //** Alg1:L2: Determine relevant matrix and vector sizes based on task_offset_tau
    nof_pjoints_ = slist.cols() - task_offset_tau;
    nof_sjoints_ = task_offset_tau;

    /// Compute the elementwise error tolerance for seconday joint goals
    theta_s_tol_ = theta_sdf_clamped;
    theta_s_tol_.tail(1)(0) = deltatheta_a;
    theta_s_tol_ = (accuracy_ * theta_s_tol_).cwiseAbs();

    //**Alg1:L3 and Alg1:L2: Set start guesses and step goal
    Eigen::VectorXd theta_sg = Eigen::VectorXd::Zero(nof_sjoints_);
    Eigen::VectorXd theta_pg = this->make_primary_start_guess();
    Eigen::VectorXd theta_sd = theta_sdf_clamped; // We set the affordance goal in the loop in reference to the start state
    theta_sd.tail(1).setConstant(0);      // start affordance at 0 but gripper orientation as specified

    //**Alg1:L4: Compute no. of iterations, stepper_max_itr_m to final goal: Passed in as planner config

    //**Alg1:L5: Initialize loop counter, loop_counter_k; success counter, success_counter_s
    int loop_counter_k = 0;
    int success_counter_s = 0;

    while (loop_counter_k < (stepper_max_itr_m - 1) && !st.stop_requested()) //**Alg1:L6
    {
        loop_counter_k = loop_counter_k + 1; //**Alg1:L7:

        //**Alg1:L8: Update aff step goal:
        // Set the affordance step goal as aff_step away from the current pose. Affordance is the last element of
        // theta_sd
        theta_sd(nof_sjoints_ - 1) = theta_sd(nof_sjoints_ - 1) - deltatheta_a;

        //**Alg1:L13: Call Algorithm 2 with args, theta_sd, theta_pg, theta_sg, slist
        std::optional<Eigen::VectorXd> ik_result = this->call_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd, st);

        if (ik_result.has_value()) //**Alg1:L14
        {
            //**Alg1:L15: Record solution, theta_p, theta_sg
            const Eigen::VectorXd &traj_point = ik_result.value();
            this->append_trajectory_point(plannerResult, traj_point);

            //**Alg1:L16 Update guesses, theta_pg, theta_sg
            theta_sg = traj_point.tail(nof_sjoints_);
            theta_pg = traj_point.head(nof_pjoints_);

            success_counter_s = success_counter_s + 1; //**Alg1:L17
        }                                              //**Alg1:L18

    } //**Alg1:L19

    // Capture the planning time
    auto end_time = std::chrono::high_resolution_clock::now();
    plannerResult.planning_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Set planning result
    if (!plannerResult.joint_trajectory.empty())
    {
        plannerResult.success = true;

        if (loop_counter_k == success_counter_s)
        {
            plannerResult.trajectory_description = TrajectoryDescription::FULL;
        }
        else
        {
            plannerResult.trajectory_description = TrajectoryDescription::PARTIAL;
        }
    }
    else
    {
        plannerResult.success = false;
        plannerResult.trajectory_description = TrajectoryDescription::UNSET;
    }

    // Indicate if DLS was used
    plannerResult.update_trail = dls_flag_ ? "dls and " : plannerResult.update_trail;

    return plannerResult;
}

PlannerResult CcAffordancePlanner::generate_affordance_motion_joint_trajectory(const Eigen::MatrixXd &slist,
                                                                               const Eigen::VectorXd &theta_sdf,
                                                                               const size_t &task_offset_tau,
                                                                               const int &stepper_max_itr_m)
{

    auto start_time = std::chrono::high_resolution_clock::now(); // Monitor clock to track planning time

    PlannerResult plannerResult; // Result of the planner

    // Clamp theta_sdf to minimum goal magnitude to avoid potential numerical issues
    const Eigen::VectorXd theta_sdf_clamped = affordance_util::clamp_to_magnitude_minimum(theta_sdf, goal_min_);

    // Extract affordance goal
    const double theta_adf = theta_sdf_clamped.tail(1)(0);

    //**Alg1:L1: Define affordance step, deltatheta_a
    const double deltatheta_a = theta_adf / (stepper_max_itr_m - 1);

    //** Alg1:L2: Determine relevant matrix and vector sizes based on task_offset_tau
    nof_pjoints_ = slist.cols() - task_offset_tau;
    nof_sjoints_ = task_offset_tau;

    /// Compute the elementwise error tolerance for seconday joint goals
    theta_s_tol_ = theta_sdf_clamped;
    theta_s_tol_.tail(1)(0) = deltatheta_a;
    theta_s_tol_ = (accuracy_ * theta_s_tol_).cwiseAbs();

    //**Alg1:L3 and Alg1:L2: Set start guesses and step goal
    Eigen::VectorXd theta_sg = Eigen::VectorXd::Zero(nof_sjoints_);
    Eigen::VectorXd theta_pg = this->make_primary_start_guess();
    Eigen::VectorXd theta_sd = theta_sdf_clamped; // We set the affordance goal in the loop in reference to the start state
    theta_sd.tail(1).setConstant(0);      // start affordance at 0 but gripper orientation as specified

    //**Alg1:L4: Compute no. of iterations, stepper_max_itr_m to final goal: Passed in as planner config

    //**Alg1:L5: Initialize loop counter, loop_counter_k; success counter, success_counter_s
    int loop_counter_k = 0;
    int success_counter_s = 0;

    while (loop_counter_k < (stepper_max_itr_m - 1)) //**Alg1:L6
    {

        loop_counter_k = loop_counter_k + 1; //**Alg1:L7:

        //**Alg1:L8: Update aff step goal:
        // Set the affordance step goal as aff_step away from the current pose. Affordance is the last element of
        // theta_sd
        theta_sd(nof_sjoints_ - 1) = theta_sd(nof_sjoints_ - 1) - deltatheta_a;

        //**Alg1:L13: Call Algorithm 2 with args, theta_sd, theta_pg, theta_sg, slist
        std::optional<Eigen::VectorXd> ik_result = this->call_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd);

        if (ik_result.has_value()) //**Alg1:L14
        {
            //**Alg1:L15: Record solution, theta_p, theta_sg
            const Eigen::VectorXd &traj_point = ik_result.value();
            this->append_trajectory_point(plannerResult, traj_point);

            //**Alg1:L16 Update guesses, theta_pg, theta_sg
            theta_sg = traj_point.tail(nof_sjoints_);
            theta_pg = traj_point.head(nof_pjoints_);

            success_counter_s = success_counter_s + 1; //**Alg1:L17
        }                                              //**Alg1:L18

    } //**Alg1:L19

    // Capture the planning time
    auto end_time = std::chrono::high_resolution_clock::now();
    plannerResult.planning_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Set planning result
    if (!plannerResult.joint_trajectory.empty())
    {
        plannerResult.success = true;

        if (loop_counter_k == success_counter_s)
        {
            plannerResult.trajectory_description = TrajectoryDescription::FULL;
        }
        else
        {
            plannerResult.trajectory_description = TrajectoryDescription::PARTIAL;
        }
    }
    else
    {
        plannerResult.success = false;
        plannerResult.trajectory_description = TrajectoryDescription::UNSET;
    }

    // Indicate if DLS was used
    plannerResult.update_trail = dls_flag_ ? "dls and " : plannerResult.update_trail;

    return plannerResult;
}

std::optional<Eigen::VectorXd> CcAffordancePlanner::call_cc_ik_solver(const Eigen::MatrixXd &slist,
                                                                      const Eigen::VectorXd &theta_pg,
                                                                      const Eigen::VectorXd &theta_sg,
                                                                      const Eigen::VectorXd &theta_sd,
                                                                      std::stop_token st)
{
    if (reserve_mobility_enabled_)
    {
        return this->call_rm_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd, st);
    }

    /* Eigen resizings */
    Eigen::VectorXd thetalist; // helper variable holding theta_p, theta_s
    thetalist.conservativeResize(slist.cols());

    //**Alg2:L1: Set max. no. of iterations, max_itr_l_, and error thresholds, p_task_err_threshold_eps_s,
    // eps_r_: Defined as class public variables

    //** Alg2:L2: Set dt as small time increment
    const double dt = 1e-2; // time step to compute joint velocities

    // Alg2:L3: Capture guesses
    Eigen::VectorXd theta_p = theta_pg;
    Eigen::VectorXd theta_s = theta_sg;

    Eigen::VectorXd oldtheta_p =
        Eigen::VectorXd::Zero(nof_pjoints_); // capture theta_p to compute joint velocities below for Alg2:L8

    //**Alg2:L4: Initialize loop counter, loop_counter_i
    int loop_counter_i = 0;

    //**Alg2:L5: Start closure error at 0
    Eigen::VectorXd rho = Eigen::VectorXd::Zero(twist_length_); // twist length is 6

    // Check error
    Eigen::VectorXd theta_s_err = (theta_sd - theta_s).cwiseAbs(); // secondary joint goal error

    bool err = ((theta_s_err.array() > theta_s_tol_.array()).any() || rho.head(3).norm() > eps_rw_ ||
                rho.tail(3).norm() > eps_rv_);

    while (err && loop_counter_i < max_itr_l_ && !st.stop_requested()) //**Alg2:L6
    {
        loop_counter_i = loop_counter_i + 1; //**Alg2:L7

        //**Alg2:L8: Compute Np, Ns as screw-based Jacobians
        thetalist << theta_p, theta_s;
        Eigen::MatrixXd jac = affordance_util::JacobianSpace(slist, thetalist);
        Eigen::MatrixXd Np = jac.leftCols(nof_pjoints_);
        Eigen::MatrixXd Ns = jac.rightCols(nof_sjoints_);

        //**Alg2:L9: Compute theta_pdot using Eqn. 23
        Eigen::MatrixXd theta_pdot = (theta_p - oldtheta_p) / dt;

        //**Alg2:L10: Compute N using Eqn. 22
        Eigen::MatrixXd pinv_Ns; // pseudo-inverse of Ns
        pinv_Ns = Ns.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXd pinv_theta_pdot; // pseudo-inverse of theta_pdot
        pinv_theta_pdot = theta_pdot.completeOrthogonalDecomposition().pseudoInverse();

        Eigen::MatrixXd N = -pinv_Ns * (Np + rho * pinv_theta_pdot);

        oldtheta_p = theta_p; // capture theta_p to compute joint velocities below for Alg2:L8

        // Update theta_p using Newton-Raphson
        this->update_theta_p(theta_p, theta_sd, theta_s, N); // returned by reference

        //**Alg2:L12: Call Algorithm 3 with args, theta_s, theta_p, slist, Np, Ns
        this->adjust_for_closure_error(slist, Np, Ns, theta_p, theta_s,
                                       rho); // theta_s and theta_p returned by reference

        // Check error
        theta_s_err = (theta_sd - theta_s).cwiseAbs(); // secondary joint goal error
        err = ((theta_s_err.array() > theta_s_tol_.array()).any() || rho.head(3).norm() > eps_rw_ ||
               rho.tail(3).norm() > eps_rv_);

    } //**Alg2:L13

    if (!err) //**Alg2:L14
    {
        thetalist << theta_p, theta_s; // return thetalist corrected by closure_error_optimizer
        return thetalist;              //** Alg2:L15
    }
    else
    {
        return std::nullopt; //** Alg2:L15 // Represents no value (similar to nullptr for pointers)
    }
}

std::optional<Eigen::VectorXd> CcAffordancePlanner::call_cc_ik_solver(const Eigen::MatrixXd &slist,
                                                                      const Eigen::VectorXd &theta_pg,
                                                                      const Eigen::VectorXd &theta_sg,
                                                                      const Eigen::VectorXd &theta_sd)
{
    if (reserve_mobility_enabled_)
    {
        return this->call_rm_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd, std::stop_token());
    }

    /* Eigen resizings */
    Eigen::VectorXd thetalist; // helper variable holding theta_p, theta_s
    thetalist.conservativeResize(slist.cols());

    //**Alg2:L1: Set max. no. of iterations, p_max_itr_l, and error thresholds, p_task_err_threshold_eps_s,
    // p_closure_err_threshold_eps_r: Defined as class public variables

    //** Alg2:L2: Set dt as small time increment
    const double dt = 1e-2; // time step to compute joint velocities

    // Alg2:L3: Capture guesses
    Eigen::VectorXd theta_p = theta_pg;
    Eigen::VectorXd theta_s = theta_sg;

    Eigen::VectorXd oldtheta_p =
        Eigen::VectorXd::Zero(nof_pjoints_); // capture theta_p to compute joint velocities below for Alg2:L8

    //**Alg2:L4: Initialize loop counter, loop_counter_i
    int loop_counter_i = 0;

    //**Alg2:L5: Start closure error at 0
    Eigen::VectorXd rho = Eigen::VectorXd::Zero(twist_length_); // twist length is 6

    // Check error
    Eigen::VectorXd theta_s_err = (theta_sd - theta_s).cwiseAbs(); // secondary joint goal error

    bool err = ((theta_s_err.array() > theta_s_tol_.array()).any() || rho.head(3).norm() > eps_rw_ ||
                rho.tail(3).norm() > eps_rv_);

    while (err && loop_counter_i < max_itr_l_) //**Alg2:L6
    {
        loop_counter_i = loop_counter_i + 1; //**Alg2:L7

        //**Alg2:L8: Compute Np, Ns as screw-based Jacobians
        thetalist << theta_p, theta_s;
        Eigen::MatrixXd jac = affordance_util::JacobianSpace(slist, thetalist);
        Eigen::MatrixXd Np = jac.leftCols(nof_pjoints_);
        Eigen::MatrixXd Ns = jac.rightCols(nof_sjoints_);

        //**Alg2:L9: Compute theta_pdot using Eqn. 23
        Eigen::MatrixXd theta_pdot = (theta_p - oldtheta_p) / dt;

        //**Alg2:L10: Compute N using Eqn. 22
        Eigen::MatrixXd pinv_Ns; // pseudo-inverse of Ns
        pinv_Ns = Ns.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXd pinv_theta_pdot; // pseudo-inverse of theta_pdot
        pinv_theta_pdot = theta_pdot.completeOrthogonalDecomposition().pseudoInverse();

        Eigen::MatrixXd N = -pinv_Ns * (Np + rho * pinv_theta_pdot);

        oldtheta_p = theta_p; // capture theta_p to compute joint velocities below for Alg2:L8

        // Update theta_p using Newton-Raphson
        this->update_theta_p(theta_p, theta_sd, theta_s, N); // returned by reference

        //**Alg2:L12: Call Algorithm 3 with args, theta_s, theta_p, slist, Np, Ns
        this->adjust_for_closure_error(slist, Np, Ns, theta_p, theta_s,
                                       rho); // theta_s and theta_p returned by reference

        // Check error
        theta_s_err = (theta_sd - theta_s).cwiseAbs(); // secondary joint goal error
        err = ((theta_s_err.array() > theta_s_tol_.array()).any() || rho.head(3).norm() > eps_rw_ ||
               rho.tail(3).norm() > eps_rv_);

    } //**Alg2:L13

    if (!err) //**Alg2:L14
    {
        thetalist << theta_p, theta_s; // return thetalist corrected by closure_error_optimizer
        return thetalist;              //** Alg2:L15
    }
    else
    {
        return std::nullopt; //** Alg2:L15 // Represents no value (similar to nullptr for pointers)
    }
}

std::optional<Eigen::VectorXd> CcAffordancePlanner::call_rm_cc_ik_solver(
    const Eigen::MatrixXd &slist, const Eigen::VectorXd &theta_pg, const Eigen::VectorXd &theta_sg,
    const Eigen::VectorXd &theta_sd, std::stop_token st)
{
    if (theta_pg.size() != static_cast<Eigen::Index>(nof_pjoints_) ||
        theta_sg.size() != static_cast<Eigen::Index>(nof_sjoints_) || theta_sd.size() != theta_sg.size() ||
        slist.cols() != theta_pg.size() + theta_sg.size())
    {
        throw std::invalid_argument("RM-CCA solver input dimensions are inconsistent.");
    }

    std::unordered_set<size_t> partition;
    for (const size_t index : reserve_primary_indices_)
    {
        if (index >= nof_pjoints_ || !partition.insert(index).second)
        {
            throw std::invalid_argument("Invalid reserve primary index partition.");
        }
    }
    for (const size_t index : arm_primary_indices_)
    {
        if (index >= nof_pjoints_ || !partition.insert(index).second)
        {
            throw std::invalid_argument("Invalid arm primary index partition.");
        }
    }
    if (partition.size() != nof_pjoints_)
    {
        throw std::invalid_argument("Arm and reserve indices must cover every RM-CCA primary joint.");
    }

    Eigen::VectorXd theta_p = theta_pg;
    Eigen::VectorXd theta_s = theta_sg;
    Eigen::VectorXd oldtheta_p = theta_p;
    Eigen::VectorXd rho = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(twist_length_));

    Eigen::VectorXd primary_lower =
        Eigen::VectorXd::Constant(static_cast<Eigen::Index>(nof_pjoints_), -std::numeric_limits<double>::infinity());
    Eigen::VectorXd primary_upper =
        Eigen::VectorXd::Constant(static_cast<Eigen::Index>(nof_pjoints_), std::numeric_limits<double>::infinity());
    // Reserve-coordinate limits also encode the user's enabled-DOF mask, so they remain active independently of the
    // arm joint-limit ablation switch.
    for (size_t i = 0; i < reserve_primary_indices_.size(); ++i)
    {
        const Eigen::Index primary_index = static_cast<Eigen::Index>(reserve_primary_indices_[i]);
        primary_lower(primary_index) = reserve_mobility_.lower_limits(static_cast<Eigen::Index>(i));
        primary_upper(primary_index) = reserve_mobility_.upper_limits(static_cast<Eigen::Index>(i));
    }
    if (enable_joint_limits_)
    {
        for (size_t i = 0; i < arm_primary_indices_.size(); ++i)
        {
            const Eigen::Index primary_index = static_cast<Eigen::Index>(arm_primary_indices_[i]);
            double lower = arm_lower_limits_(static_cast<Eigen::Index>(i));
            double upper = arm_upper_limits_(static_cast<Eigen::Index>(i));
            if (std::isfinite(lower) && std::isfinite(upper))
            {
                const double center = lower + 0.5 * (upper - lower);
                const double soft_half_range = 0.5 * soft_limit_ratio_ * (upper - lower);
                lower = center - soft_half_range;
                upper = center + soft_half_range;
            }
            // The safe interval normally excludes a small margin. If a supplied start state is already inside that
            // margin, keep the start itself admissible so the joint can move back toward the interior.
            primary_lower(primary_index) =
                std::min(0.0, lower + joint_limit_margin_ - arm_start_absolute_(static_cast<Eigen::Index>(i)));
            primary_upper(primary_index) =
                std::max(0.0, upper - joint_limit_margin_ - arm_start_absolute_(static_cast<Eigen::Index>(i)));
        }
    }

    const std::vector<size_t> no_reserve_stationarity;
    const std::vector<size_t> &stationary_reserve_indices =
        enable_nullspace_planning_ ? reserve_primary_indices_ : no_reserve_stationarity;

    auto reserve_values = [&](const Eigen::VectorXd &primary) {
        Eigen::VectorXd values(static_cast<Eigen::Index>(reserve_primary_indices_.size()));
        for (size_t i = 0; i < reserve_primary_indices_.size(); ++i)
        {
            values(static_cast<Eigen::Index>(i)) =
                primary(static_cast<Eigen::Index>(reserve_primary_indices_[i]));
        }
        return values;
    };

    auto limit_reserve_increment = [&](Eigen::VectorXd &delta, const Eigen::VectorXd &already_used) {
        double scale = 1.0;
        for (size_t i = 0; i < reserve_primary_indices_.size(); ++i)
        {
            const double maximum = reserve_mobility_.max_step(static_cast<Eigen::Index>(i));
            const double increment =
                delta(static_cast<Eigen::Index>(reserve_primary_indices_[i]));
            const double remaining = std::max(0.0, maximum - already_used(static_cast<Eigen::Index>(i)));
            if (std::isfinite(maximum) && std::abs(increment) > remaining && std::abs(increment) > 0.0)
            {
                scale = std::min(scale, remaining / std::abs(increment));
            }
        }
        delta *= std::clamp(scale, 0.0, 1.0);
    };

    last_rm_ik_diagnostics_ = RmIkDiagnostics{};
    last_rm_ik_diagnostics_.active_arm_dof_count = arm_primary_indices_.size();
    Eigen::VectorXd theta_s_error = (theta_sd - theta_s).cwiseAbs();
    bool error = ((theta_s_error.array() > theta_s_tol_.array()).any() || rho.head(3).norm() > eps_rw_ ||
                  rho.tail(3).norm() > eps_rv_);

    constexpr double derivative_dt = 1e-2;
    int iteration = 0;
    while (error && iteration < max_itr_l_ && !st.stop_requested())
    {
        ++iteration;
        Eigen::VectorXd thetalist(slist.cols());
        thetalist << theta_p, theta_s;
        const Eigen::MatrixXd jacobian = affordance_util::JacobianSpace(slist, thetalist);
        const Eigen::MatrixXd Np = jacobian.leftCols(static_cast<Eigen::Index>(nof_pjoints_));
        const Eigen::MatrixXd Ns = jacobian.rightCols(static_cast<Eigen::Index>(nof_sjoints_));
        const Eigen::VectorXd theta_pdot = (theta_p - oldtheta_p) / derivative_dt;

        // Closed-chain constraint mapping: Jc = d(theta_s) / d(theta_p).
        const Eigen::MatrixXd Jc =
            compute_closed_chain_mapping(Np, Ns, rho, theta_pdot, svd_relative_tolerance_,
                                         svd_absolute_tolerance_);
        oldtheta_p = theta_p;

        const Eigen::VectorXd task_error = theta_sd - theta_s;
        const Eigen::VectorXd reserve_before = reserve_values(theta_p);

        // The capability-aware mode uses one heterogeneous metric. The original mode retains strict base
        // stationarity in the feasible CCA null space. Both use the same direction-aware configured-bound active set.
        FeasibleNullSpaceStep task_step;
        if (enable_capability_aware_planning_)
        {
            const Eigen::VectorXd primary_metric = make_primary_mobility_metric(theta_p);
            task_step = compute_feasible_metric_step(
                Jc, task_error, theta_p, primary_lower, primary_upper, primary_metric, arm_primary_indices_,
                reserve_primary_indices_, svd_relative_tolerance_, joint_limit_tolerance_, svd_absolute_tolerance_);
        }
        else
        {
            task_step = compute_feasible_nullspace_step(
                Jc, task_error, theta_p, primary_lower, primary_upper, arm_primary_indices_,
                stationary_reserve_indices, svd_relative_tolerance_, joint_limit_tolerance_,
                svd_absolute_tolerance_);
        }
        Eigen::VectorXd primary_delta = task_step.delta;
        limit_reserve_increment(primary_delta,
                                Eigen::VectorXd::Zero(static_cast<Eigen::Index>(reserve_primary_indices_.size())));
        // A common globalization gain preserves both strict task priority and exact base stationarity while avoiding
        // alternating overshoot between the Newton and finite-closure corrections near an active bound.
        constexpr double hierarchy_newton_gain = 0.5;
        primary_delta *= hierarchy_newton_gain;
        theta_p += primary_delta;
        // Jc maps a primary correction to its closed-chain secondary correction. Advance both sides of that
        // differential together; the following finite closure solve then removes only the linearization error.
        theta_s += hierarchy_newton_gain * task_error;

        // Closure recovery uses the selected allocation rule as well. Capability-aware mode solves directly over
        // [primary, secondary] with block metric diag(Ga, Gb, Gs); strict mode preserves the advanced secondary state.
        Eigen::VectorXd closure_state(slist.cols());
        closure_state << theta_p, theta_s;
        Eigen::VectorXd closure_lower =
            Eigen::VectorXd::Constant(slist.cols(), -std::numeric_limits<double>::infinity());
        Eigen::VectorXd closure_upper =
            Eigen::VectorXd::Constant(slist.cols(), std::numeric_limits<double>::infinity());
        closure_lower.head(static_cast<Eigen::Index>(nof_pjoints_)) = primary_lower;
        closure_upper.head(static_cast<Eigen::Index>(nof_pjoints_)) = primary_upper;
        const Eigen::MatrixXd raw_closure_jacobian = affordance_util::JacobianSpace(slist, closure_state);
        const Eigen::VectorXd closure_error = compute_closure_error(slist, theta_p, theta_s);
        FeasibleNullSpaceStep closure_step;
        if (enable_capability_aware_planning_)
        {
            Eigen::VectorXd closure_metric =
                Eigen::VectorXd::Constant(closure_state.size(), closure_secondary_weight_);
            closure_metric.head(static_cast<Eigen::Index>(nof_pjoints_)) =
                make_primary_mobility_metric(theta_p);
            closure_step = compute_feasible_metric_step(
                raw_closure_jacobian, closure_error, closure_state, closure_lower, closure_upper, closure_metric,
                arm_primary_indices_, reserve_primary_indices_, svd_relative_tolerance_, joint_limit_tolerance_,
                svd_absolute_tolerance_);
        }
        else
        {
            // Strict RM-CCA must not undo the CCA secondary state that the primary step just advanced.
            Eigen::MatrixXd closure_jacobian(closure_error.size() + theta_s.size(), closure_state.size());
            closure_jacobian.topRows(closure_error.size()) = raw_closure_jacobian;
            closure_jacobian.bottomRows(theta_s.size()).setZero();
            closure_jacobian.bottomRightCorner(theta_s.size(), theta_s.size()).setIdentity();
            Eigen::VectorXd closure_task_error(closure_error.size() + theta_s.size());
            closure_task_error << closure_error, Eigen::VectorXd::Zero(theta_s.size());
            closure_step = compute_feasible_nullspace_step(
                closure_jacobian, closure_task_error, closure_state, closure_lower, closure_upper,
                arm_primary_indices_, stationary_reserve_indices, svd_relative_tolerance_, joint_limit_tolerance_,
                svd_absolute_tolerance_);
        }
        Eigen::VectorXd closure_delta = closure_step.delta;
        Eigen::VectorXd task_reserve_use(static_cast<Eigen::Index>(reserve_primary_indices_.size()));
        for (size_t i = 0; i < reserve_primary_indices_.size(); ++i)
        {
            task_reserve_use(static_cast<Eigen::Index>(i)) =
                std::abs(primary_delta(static_cast<Eigen::Index>(reserve_primary_indices_[i])));
        }
        limit_reserve_increment(closure_delta, task_reserve_use);
        closure_delta *= hierarchy_newton_gain;
        theta_p += closure_delta.head(static_cast<Eigen::Index>(nof_pjoints_));
        theta_s += closure_delta.tail(static_cast<Eigen::Index>(nof_sjoints_));
        rho = compute_closure_error(slist, theta_p, theta_s);

        const Eigen::VectorXd reserve_change = reserve_values(theta_p) - reserve_before;
        last_rm_ik_diagnostics_.residual_norm =
            std::max(last_rm_ik_diagnostics_.residual_norm, reserve_change.norm());
        last_rm_ik_diagnostics_.reserve_active =
            last_rm_ik_diagnostics_.reserve_active || reserve_change.norm() > residual_mobility_tolerance_;
        last_rm_ik_diagnostics_.active_arm_dof_count =
            std::min({last_rm_ik_diagnostics_.active_arm_dof_count, task_step.active_arm_dof_count,
                      closure_step.active_arm_dof_count});

        theta_s_error = (theta_sd - theta_s).cwiseAbs();
        error = ((theta_s_error.array() > theta_s_tol_.array()).any() || rho.head(3).norm() > eps_rw_ ||
                 rho.tail(3).norm() > eps_rv_);
    }

    if (error)
    {
        return std::nullopt;
    }
    Eigen::VectorXd result(slist.cols());
    result << theta_p, theta_s;
    return result;
}

void CcAffordancePlanner::adjust_for_closure_error(
    const Eigen::MatrixXd &slist, const Eigen::MatrixXd &Np, const Eigen::MatrixXd &Ns, Eigen::VectorXd &theta_p,
    Eigen::VectorXd &theta_s, Eigen::VectorXd &rho) //**Alg3:L5 // theta_s and theta_p returned by reference
{

    /* Eigen resizings */
    Eigen::VectorXd thetalist;
    thetalist.conservativeResize(slist.cols()); // helper variable holding theta_p, theta_s

    //**Alg3:L1: Compute forward kinematics to chain's end link, Tse
    const Eigen::Matrix4d des_endlink_htm_ = Eigen::Matrix4d::Identity(); // Desired HTM for the end link
    thetalist << theta_p, theta_s;
    Eigen::Matrix4d Tse =
        affordance_util::FKinSpace(des_endlink_htm_, slist, thetalist); // HTM of actual end of ground link

    //**Alg3:L2: Compute closure error
    rho =
        /* Eigen::Matrix<double, twist_length_, 1> rho = */
        affordance_util::Adjoint(Tse) *
        affordance_util::se3ToVec(affordance_util::MatrixLog6(affordance_util::TransInv(Tse)));

    //**Alg3:L3: Adjust joint angles for closure error
    Eigen::MatrixXd Nc(Np.rows(), Np.cols() + Ns.cols());
    Nc << Np, Ns; // Combine Np and Ns horizontally
    Eigen::MatrixXd pinv_Nc;
    pinv_Nc = Nc.completeOrthogonalDecomposition().pseudoInverse(); // pseudo-inverse of N
    const Eigen::VectorXd delta_theta = pinv_Nc * rho;              // correction differential

    // Correct the joint angles
    theta_p = theta_p + delta_theta.head(nof_pjoints_);
    theta_s = theta_s + delta_theta.tail(nof_sjoints_);

    // Compute final error
    thetalist << theta_p, theta_s;
    Tse = affordance_util::FKinSpace(des_endlink_htm_, slist, thetalist);
    rho = affordance_util::Adjoint(Tse) *
          affordance_util::se3ToVec(affordance_util::MatrixLog6(affordance_util::TransInv(Tse)));
}

} // namespace cc_affordance_planner
