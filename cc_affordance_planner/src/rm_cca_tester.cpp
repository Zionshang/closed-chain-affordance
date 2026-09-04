#include <affordance_util/affordance_util.hpp>
#include <cc_affordance_planner/cc_affordance_planner_interface.hpp>
#include <cc_affordance_planner/rm_cca.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
void require(const bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

std::string read_file(const std::string &path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Cannot open test data: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void test_hierarchy_keeps_base_stationary()
{
    Eigen::MatrixXd jacobian(1, 2);
    jacobian << 1.0, 1.0;
    const Eigen::VectorXd error = Eigen::VectorXd::Ones(1);
    const auto step =
        cc_affordance_planner::compute_base_stationary_step(jacobian, error, {1}, 1e-12);

    require(step.task_residual.norm() < 1e-12,
            "Hierarchy test: the primary CCA correction was changed.");
    require(step.base_delta.norm() < 1e-12,
            "Hierarchy test: base moved despite an arm-only solution.");
    require(std::abs(step.delta(0) - 1.0) < 1e-12,
            "Hierarchy test: null-space redistribution did not transfer motion to the arm.");
}

void test_projected_numerical_zero_is_truncated()
{
    const Eigen::Matrix2d numerical_zero =
        10.0 * std::numeric_limits<double>::epsilon() * Eigen::Matrix2d::Identity();
    const Eigen::MatrixXd pseudoinverse =
        cc_affordance_planner::pseudo_inverse_svd(numerical_zero, 1e-8);
    require(pseudoinverse.norm() == 0.0,
            "SVD-floor test: projected floating-point noise became a false mobility direction.");
}

void test_bound_exhaustion_keeps_unavoidable_base_motion()
{
    Eigen::MatrixXd jacobian(1, 2);
    jacobian << 1.0, 1.0;
    const Eigen::VectorXd error = Eigen::VectorXd::Ones(1);
    Eigen::Vector2d state(0.9, 0.0);
    Eigen::Vector2d lower(-1.0, -10.0);
    Eigen::Vector2d upper(1.0, 10.0);
    const auto step = cc_affordance_planner::compute_feasible_nullspace_step(
        jacobian, error, state, lower, upper, {0}, {1}, 1e-12, 1e-12);

    require(step.task_residual.norm() < 1e-12,
            "Bound test: arm plus base did not preserve the primary task.");
    require(std::abs(step.delta(0) - 0.1) < 1e-12,
            "Bound test: arm correction was not clipped at its first boundary.");
    require(std::abs(step.delta(1) - 0.9) < 1e-12,
            "Bound test: unavoidable motion was not retained by the base.");
    require(step.frozen_variable_indices == std::vector<size_t>{0} && step.active_arm_dof_count == 0,
            "Bound test: the boundary active set is incorrect.");
}

void test_boundary_joint_reactivates_inward()
{
    Eigen::MatrixXd jacobian(1, 2);
    jacobian << 1.0, 1.0;
    const Eigen::VectorXd error = Eigen::VectorXd::Constant(1, -0.2);
    Eigen::Vector2d state(1.0, 0.0);
    Eigen::Vector2d lower(-1.0, -10.0);
    Eigen::Vector2d upper(1.0, 10.0);
    const auto step = cc_affordance_planner::compute_feasible_nullspace_step(
        jacobian, error, state, lower, upper, {0}, {1}, 1e-12, 1e-12);

    require(step.task_residual.norm() < 1e-12,
            "Reactivation test: the inward task correction was not completed.");
    require(std::abs(step.delta(0) + 0.2) < 1e-12 && step.base_delta.norm() < 1e-12,
            "Reactivation test: an upper-bound arm joint did not reactivate inward.");
    require(step.frozen_variable_indices.empty() && step.active_arm_dof_count == 1,
            "Reactivation test: an inward-moving joint was incorrectly frozen.");
}

void test_rank_loss_keeps_only_unavoidable_base_motion()
{
    Eigen::Matrix2d jacobian;
    jacobian << 1.0, 0.0,
                0.0, 1.0;
    const Eigen::Vector2d error(1.0, 1.0);
    const auto step =
        cc_affordance_planner::compute_base_stationary_step(jacobian, error, {1}, 1e-12);

    require(step.task_residual.norm() < 1e-12,
            "Rank-loss test: the whole-body correction did not complete the task.");
    require(std::abs(step.delta(0) - 1.0) < 1e-12 && std::abs(step.delta(1) - 1.0) < 1e-12,
            "Rank-loss test: the differential direction unavailable to the arm was not assigned to the base.");
}

void test_closure_uses_same_hierarchy()
{
    Eigen::MatrixXd closure_jacobian(1, 3);
    closure_jacobian << 1.0, 1.0, -1.0;
    const Eigen::VectorXd closure_error = Eigen::VectorXd::Constant(1, 0.3);
    const auto step = cc_affordance_planner::compute_feasible_nullspace_step(
        closure_jacobian, closure_error, Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Constant(-10.0), Eigen::Vector3d::Constant(10.0),
        {0}, {1}, 1e-12, 1e-12);

    require(step.task_residual.norm() < 1e-12,
            "Closure hierarchy test: the closure correction was not preserved.");
    require(step.base_delta.norm() < 1e-12,
            "Closure hierarchy test: base moved despite an arm/secondary closure solution.");
}

void test_finite_rotation_reserve_closure()
{
    // A numerically locked dummy arm holds a point away from a Z hinge. The
    // floating base must supply an 0.8-rad finite rotation and its associated
    // translation. This specifically exercises nonlinear closure correction
    // after reserve activation, rather than only an infinitesimal update.
    affordance_util::RobotDescription robot;
    robot.slist = Eigen::MatrixXd::Zero(6, 1);
    robot.M = Eigen::Matrix4d::Identity();
    robot.M.block<3, 1>(0, 3) = Eigen::Vector3d(0.5, 0.2, 0.1);
    robot.joint_states = Eigen::VectorXd::Zero(1);
    robot.joint_lower_limits = Eigen::VectorXd::Constant(1, -2e-5);
    robot.joint_upper_limits = Eigen::VectorXd::Constant(1, 2e-5);

    cc_affordance_planner::TaskDescription task;
    task.affordance_info.type = affordance_util::ScrewType::ROTATION;
    task.affordance_info.axis = Eigen::Vector3d::UnitZ();
    task.affordance_info.location = Eigen::Vector3d(0.1, 0.1, 0.0);
    task.goal.affordance = -0.8;
    task.goal.ee_orientation = Eigen::VectorXd::Zero(1);
    task.trajectory_density = 12;
    task.vir_screw_order = affordance_util::VirtualScrewOrder::Z;
    task.reserve_mobility =
        affordance_util::make_floating_base_reserve_description(Eigen::Matrix4d::Identity(), 0.1, 0.2);

    cc_affordance_planner::PlannerConfig config;
    config.accuracy = 0.001;
    config.ik_max_itr = 5000;
    config.closure_err_threshold_ang = 1e-4;
    config.closure_err_threshold_lin = 1e-4;
    config.residual_mobility_tolerance = 1e-10;
    config.joint_limit_margin = 1e-6;
    const auto result =
        cc_affordance_planner::CcAffordancePlannerInterface(config).generate_joint_trajectory(robot, task);

    require(result.success && result.trajectory_description == cc_affordance_planner::TrajectoryDescription::FULL,
            "Finite-rotation case: reserve did not complete the nonlinear hinge trajectory.");
    require(result.reserve_mobility_used && result.reserve_trajectory.size() == 12,
            "Finite-rotation case: reserve activation or trajectory alignment is missing.");
    require(std::abs(result.reserve_trajectory.back()(5) + 0.8) < 2e-4,
            "Finite-rotation case: floating-base yaw did not track the hinge goal.");

    const Eigen::Vector3d start = robot.M.block<3, 1>(0, 3);
    const Eigen::Vector3d hinge = task.affordance_info.location;
    const Eigen::Vector3d expected = hinge + Eigen::AngleAxisd(task.goal.affordance, Eigen::Vector3d::UnitZ()) *
                                                   (start - hinge);
    const Eigen::Vector3d actual =
        (result.reserve_pose_trajectory.back() * robot.M).block<3, 1>(0, 3);
    require((actual - expected).norm() < 3e-4,
            "Finite-rotation case: floating base did not keep the point on the hinge arc.");
}

affordance_util::RobotDescription make_kinova_description()
{
    const std::string data_directory = CCA_TEST_DATA_DIR;
    const auto builder_info = affordance_util::extract_info_for_urdf_robot_builder(
        data_directory + "/cca_kinova_gen3_7dof_urdf.yaml");
    const auto robot_config = affordance_util::robot_builder(
        read_file(data_directory + "/kinova_gen3_7dof.urdf"), builder_info);
    require(robot_config.joint_lower_limits.size() == 7 && robot_config.joint_upper_limits.size() == 7,
            "URDF limit extraction did not follow the seven-joint Kinova chain.");
    require(std::isinf(robot_config.joint_lower_limits(0)) &&
                std::abs(robot_config.joint_lower_limits(1) + 2.41) < 1e-12,
            "URDF continuous/bounded joint limits were not extracted correctly.");

    const auto legacy_yaml = affordance_util::robot_builder(
        data_directory + "/cca_kinova_gen3_7dof_description.yaml");
    require(legacy_yaml.joint_lower_limits.size() == 7 && std::isinf(legacy_yaml.joint_lower_limits(0)) &&
                std::abs(legacy_yaml.joint_lower_limits(1) + 2.41) < 1e-12,
            "YAML limits were not extracted or omitted legacy limits were not treated as unbounded.");
    return affordance_util::make_robot_description(robot_config, Eigen::VectorXd::Zero(7));
}

cc_affordance_planner::TaskDescription make_rotation_task(const affordance_util::RobotDescription &robot,
                                                          const double goal, const int density)
{
    cc_affordance_planner::TaskDescription task;
    task.affordance_info.type = affordance_util::ScrewType::ROTATION;
    task.affordance_info.screw = robot.slist.rightCols(1);
    task.goal.affordance = goal;
    task.trajectory_density = density;
    task.vir_screw_order = affordance_util::VirtualScrewOrder::NONE;
    task.reserve_mobility =
        affordance_util::make_floating_base_reserve_description(Eigen::Matrix4d::Identity(), 0.02, 0.05);
    return task;
}

void test_disabled_extensions_are_exactly_legacy_cca()
{
    const auto robot = make_kinova_description();
    const auto rm_task = make_rotation_task(robot, 0.02, 3);
    auto legacy_task = rm_task;
    legacy_task.reserve_mobility.enabled = false;

    cc_affordance_planner::PlannerConfig config;
    config.update_method = cc_affordance_planner::UpdateMethod::INVERSE;
    config.enable_joint_limits = false;
    config.enable_nullspace_planning = false;

    const auto legacy =
        cc_affordance_planner::CcAffordancePlannerInterface(config).generate_joint_trajectory(robot, legacy_task);
    const auto compatibility =
        cc_affordance_planner::CcAffordancePlannerInterface(config).generate_joint_trajectory(robot, rm_task);

    require(legacy.success == compatibility.success &&
                legacy.trajectory_description == compatibility.trajectory_description &&
                legacy.update_method == compatibility.update_method && legacy.update_trail == compatibility.update_trail,
            "Compatibility-switch test: planner status differs from fixed-base legacy CCA.");
    require(legacy.joint_trajectory.size() == compatibility.joint_trajectory.size(),
            "Compatibility-switch test: trajectory length differs from fixed-base legacy CCA.");
    for (size_t i = 0; i < legacy.joint_trajectory.size(); ++i)
    {
        require((legacy.joint_trajectory[i].array() == compatibility.joint_trajectory[i].array()).all(),
                "Compatibility-switch test: trajectory is not bit-identical to fixed-base legacy CCA.");
    }
    require(compatibility.reserve_trajectory.empty() && compatibility.reserve_pose_trajectory.empty() &&
                compatibility.reserve_active.empty() && !compatibility.reserve_mobility_used,
            "Compatibility-switch test: disabled extensions still emitted reserve-mobility output.");
}

void test_extensions_can_be_switched_independently()
{
    const auto robot = make_kinova_description();
    const auto task = make_rotation_task(robot, 0.02, 3);

    cc_affordance_planner::PlannerConfig no_nullspace_config;
    no_nullspace_config.update_method = cc_affordance_planner::UpdateMethod::INVERSE;
    no_nullspace_config.enable_joint_limits = true;
    no_nullspace_config.enable_nullspace_planning = false;
    const auto no_nullspace = cc_affordance_planner::CcAffordancePlannerInterface(no_nullspace_config)
                                  .generate_joint_trajectory(robot, task);
    require(no_nullspace.success && no_nullspace.reserve_mobility_used,
            "Independent-switch test: disabling null-space planning did not produce a whole-body solution.");

    auto tightly_limited_robot = robot;
    tightly_limited_robot.joint_lower_limits = Eigen::VectorXd::Constant(7, -2e-5);
    tightly_limited_robot.joint_upper_limits = Eigen::VectorXd::Constant(7, 2e-5);
    cc_affordance_planner::PlannerConfig no_limits_config;
    no_limits_config.update_method = cc_affordance_planner::UpdateMethod::INVERSE;
    no_limits_config.enable_joint_limits = false;
    no_limits_config.enable_nullspace_planning = true;
    const auto no_limits = cc_affordance_planner::CcAffordancePlannerInterface(no_limits_config)
                               .generate_joint_trajectory(tightly_limited_robot, task);
    require(no_limits.success &&
                no_limits.trajectory_description == cc_affordance_planner::TrajectoryDescription::FULL,
            "Independent-switch test: null-space planning without bounds did not complete.");
    require(!no_limits.reserve_mobility_used &&
                no_limits.joint_trajectory.back().head(7).cwiseAbs().maxCoeff() > 2e-5,
            "Independent-switch test: disabled joint limits were still enforced or the base was not stationary.");
}

void test_kinova_integration()
{
    cc_affordance_planner::PlannerConfig config;
    config.update_method = cc_affordance_planner::UpdateMethod::BEST; // RM-CCA must force inverse-only execution.
    config.accuracy = 0.01;
    config.ik_max_itr = 500;
    config.closure_err_threshold_lin = 3e-5;
    config.residual_mobility_tolerance = 1e-10;
    config.joint_limit_margin = 1e-6;
    cc_affordance_planner::CcAffordancePlannerInterface planner(config);

    const auto reachable_robot = make_kinova_description();
    auto reachable_task = make_rotation_task(reachable_robot, 0.02, 3);
    const auto reachable = planner.generate_joint_trajectory(reachable_robot, reachable_task);
    std::cout << "KINOVA_CASE_A_SUCCESS=" << reachable.success << " POINTS=" << reachable.joint_trajectory.size()
              << " DESCRIPTION=" << static_cast<int>(reachable.trajectory_description) << '\n';
    require(reachable.success && reachable.trajectory_description == cc_affordance_planner::TrajectoryDescription::FULL,
            "Kinova Case A: arm-reachable trajectory did not complete.");
    double maximum_base_motion = 0.0;
    for (const auto &state : reachable.reserve_trajectory)
    {
        maximum_base_motion = std::max(maximum_base_motion, state.norm());
    }
    require(!reachable.reserve_mobility_used && maximum_base_motion < 1e-10,
            "Kinova Case A: floating base moved for an arm-reachable task.");

    auto limited_robot = reachable_robot;
    limited_robot.joint_lower_limits = Eigen::VectorXd::Constant(7, -2e-5);
    limited_robot.joint_upper_limits = Eigen::VectorXd::Constant(7, 2e-5);
    auto unreachable_task = make_rotation_task(limited_robot, 0.04, 5);
    const auto unreachable = planner.generate_joint_trajectory(limited_robot, unreachable_task);
    require(unreachable.success &&
                unreachable.trajectory_description == cc_affordance_planner::TrajectoryDescription::FULL,
            "Kinova Case B: reserve mobility did not continue beyond arm limits.");
    require(unreachable.reserve_mobility_used, "Kinova Case B: reserve was never activated.");

    size_t first_activation = unreachable.reserve_active.size();
    for (size_t i = 0; i < unreachable.reserve_active.size(); ++i)
    {
        if (unreachable.reserve_active[i])
        {
            first_activation = i;
            break;
        }
        require(unreachable.reserve_trajectory[i].norm() < 1e-10,
                "Kinova Case B: reserve diagnostics did not align with the retained base motion.");
    }
    require(first_activation < unreachable.reserve_active.size(),
            "Kinova Case B: no reserve activation sample was recorded.");

    std::cout << "integration_case,index,residual_norm,active_arm_dof,reserve_active,base_norm\n";
    for (size_t i = 0; i < unreachable.reserve_trajectory.size(); ++i)
    {
        std::cout << "kinova_unreachable," << i << ',' << unreachable.residual_mobility_norm[i] << ','
                  << unreachable.active_arm_dof_count[i] << ',' << unreachable.reserve_active[i] << ','
                  << unreachable.reserve_trajectory[i].norm() << '\n';
    }
    std::cout << "KINOVA_ARM_REACHABLE_MAX_BASE_MOTION=" << maximum_base_motion << '\n';
    std::cout << "KINOVA_ARM_UNREACHABLE_FIRST_ACTIVATION_INDEX=" << first_activation << '\n';
}
} // namespace

int main()
{
    try
    {
        test_hierarchy_keeps_base_stationary();
        test_projected_numerical_zero_is_truncated();
        test_bound_exhaustion_keeps_unavoidable_base_motion();
        test_boundary_joint_reactivates_inward();
        test_rank_loss_keeps_only_unavoidable_base_motion();
        test_closure_uses_same_hierarchy();
        test_finite_rotation_reserve_closure();
        test_disabled_extensions_are_exactly_legacy_cca();
        test_extensions_can_be_switched_independently();
        test_kinova_integration();
        std::cout << "All RM-CCA deterministic and integration tests passed.\n";
        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "RM-CCA test failure: " << exception.what() << '\n';
        return 1;
    }
}
