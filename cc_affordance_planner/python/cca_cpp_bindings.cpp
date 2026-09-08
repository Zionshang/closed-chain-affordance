#include <affordance_util/affordance_util.hpp>
#include <cc_affordance_planner/cc_affordance_planner.hpp>
#include <cc_affordance_planner/cc_affordance_planner_interface.hpp>

#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace py = pybind11;
namespace au = affordance_util;
namespace cca = cc_affordance_planner;

PYBIND11_MODULE(cca_cpp, module)
{
    module.doc() = "Python bindings for the C++ Closed-Chain Affordance planner and RM-CCA";

    py::enum_<au::ScrewType>(module, "ScrewType")
        .value("ROTATION", au::ScrewType::ROTATION)
        .value("TRANSLATION", au::ScrewType::TRANSLATION)
        .value("SCREW", au::ScrewType::SCREW)
        .value("UNSET", au::ScrewType::UNSET);
    py::enum_<au::VirtualScrewOrder>(module, "VirtualScrewOrder")
        .value("XYZ", au::VirtualScrewOrder::XYZ)
        .value("YZX", au::VirtualScrewOrder::YZX)
        .value("ZXY", au::VirtualScrewOrder::ZXY)
        .value("XY", au::VirtualScrewOrder::XY)
        .value("YZ", au::VirtualScrewOrder::YZ)
        .value("ZX", au::VirtualScrewOrder::ZX)
        .value("X", au::VirtualScrewOrder::X)
        .value("Y", au::VirtualScrewOrder::Y)
        .value("Z", au::VirtualScrewOrder::Z)
        .value("NONE", au::VirtualScrewOrder::NONE);
    py::enum_<au::GripperGoalType>(module, "GripperGoalType")
        .value("CONSTANT", au::GripperGoalType::CONSTANT)
        .value("CONTINUOUS", au::GripperGoalType::CONTINUOUS);
    py::enum_<cca::EeOrientationConstraint>(module, "EeOrientationConstraint")
        .value("PRESERVE", cca::EeOrientationConstraint::PRESERVE)
        .value("DEFAULT", cca::EeOrientationConstraint::DEFAULT);
    py::enum_<cca::PlanningType>(module, "PlanningType")
        .value("APPROACH", cca::PlanningType::APPROACH)
        .value("AFFORDANCE", cca::PlanningType::AFFORDANCE)
        .value("EE_ORIENTATION_ONLY", cca::PlanningType::EE_ORIENTATION_ONLY)
        .value("CARTESIAN_GOAL", cca::PlanningType::CARTESIAN_GOAL);
    py::enum_<cca::MotionType>(module, "MotionType")
        .value("APPROACH", cca::MotionType::APPROACH)
        .value("AFFORDANCE", cca::MotionType::AFFORDANCE);
    py::enum_<cca::TrajectoryDescription>(module, "TrajectoryDescription")
        .value("FULL", cca::TrajectoryDescription::FULL)
        .value("PARTIAL", cca::TrajectoryDescription::PARTIAL)
        .value("UNSET", cca::TrajectoryDescription::UNSET);
    py::enum_<cca::UpdateMethod>(module, "UpdateMethod")
        .value("INVERSE", cca::UpdateMethod::INVERSE)
        .value("TRANSPOSE", cca::UpdateMethod::TRANSPOSE)
        .value("BEST", cca::UpdateMethod::BEST);

    py::class_<au::ScrewInfo>(module, "ScrewInfo")
        .def(py::init<>())
        .def_readwrite("type", &au::ScrewInfo::type)
        .def_readwrite("axis", &au::ScrewInfo::axis)
        .def_readwrite("location", &au::ScrewInfo::location)
        .def_readwrite("screw", &au::ScrewInfo::screw)
        .def_readwrite("pitch", &au::ScrewInfo::pitch);

    py::class_<au::RobotDescription>(module, "RobotDescription")
        .def(py::init<>())
        .def_readwrite("slist", &au::RobotDescription::slist)
        .def_readwrite("M", &au::RobotDescription::M)
        .def_readwrite("joint_states", &au::RobotDescription::joint_states)
        .def_readwrite("joint_lower_limits", &au::RobotDescription::joint_lower_limits)
        .def_readwrite("joint_upper_limits", &au::RobotDescription::joint_upper_limits)
        .def_readwrite("gripper_state", &au::RobotDescription::gripper_state);

    py::class_<au::ReserveMobilityDescription>(module, "ReserveMobilityDescription")
        .def(py::init<>())
        .def_readwrite("enabled", &au::ReserveMobilityDescription::enabled)
        .def_readwrite("slist", &au::ReserveMobilityDescription::slist)
        .def_readwrite("initial_state", &au::ReserveMobilityDescription::initial_state)
        .def_readwrite("lower_limits", &au::ReserveMobilityDescription::lower_limits)
        .def_readwrite("upper_limits", &au::ReserveMobilityDescription::upper_limits)
        .def_readwrite("max_step", &au::ReserveMobilityDescription::max_step);

    py::class_<au::RobotConfig::JointNames>(module, "RobotJointNames")
        .def(py::init<>())
        .def_readwrite("robot", &au::RobotConfig::JointNames::robot)
        .def_readwrite("gripper", &au::RobotConfig::JointNames::gripper);
    py::class_<au::RobotConfig::FrameNames>(module, "RobotFrameNames")
        .def(py::init<>())
        .def_readwrite("ref", &au::RobotConfig::FrameNames::ref)
        .def_readwrite("ee", &au::RobotConfig::FrameNames::ee)
        .def_readwrite("tool", &au::RobotConfig::FrameNames::tool);
    py::class_<au::RobotConfig::KinematicChain>(module, "RobotKinematicChain")
        .def(py::init<>())
        .def_readwrite("base_joint_name", &au::RobotConfig::KinematicChain::base_joint_name)
        .def_readwrite("end_joint_name", &au::RobotConfig::KinematicChain::end_joint_name);
    py::class_<au::RobotConfig>(module, "RobotConfig")
        .def(py::init<>())
        .def_readwrite("slist", &au::RobotConfig::Slist)
        .def_readwrite("M", &au::RobotConfig::M)
        .def_readwrite("joint_lower_limits", &au::RobotConfig::joint_lower_limits)
        .def_readwrite("joint_upper_limits", &au::RobotConfig::joint_upper_limits)
        .def_readwrite("joint_names", &au::RobotConfig::joint_names)
        .def_readwrite("frame_names", &au::RobotConfig::frame_names)
        .def_readwrite("ee_to_tool_offset", &au::RobotConfig::ee_to_tool_offset)
        .def_readwrite("kinematic_chain", &au::RobotConfig::kinematic_chain);

    py::class_<cca::Goal>(module, "Goal")
        .def(py::init<>())
        .def_readwrite("affordance", &cca::Goal::affordance)
        .def_readwrite("ee_orientation", &cca::Goal::ee_orientation)
        .def_readwrite("canonical_pose", &cca::Goal::canonical_pose)
        .def_readwrite("gripper", &cca::Goal::gripper);
    py::class_<cca::TaskDescription>(module, "TaskDescription")
        .def(py::init<>())
        .def(py::init<const cca::PlanningType &>(), py::arg("planning_type"))
        .def_readwrite("affordance_info", &cca::TaskDescription::affordance_info)
        .def_readwrite("goal", &cca::TaskDescription::goal)
        .def_readwrite("trajectory_density", &cca::TaskDescription::trajectory_density)
        .def_readwrite("approach_gamma", &cca::TaskDescription::approach_gamma)
        .def_readwrite("motion_type", &cca::TaskDescription::motion_type)
        .def_readwrite("vir_screw_order", &cca::TaskDescription::vir_screw_order)
        .def_readwrite("gripper_goal_type", &cca::TaskDescription::gripper_goal_type)
        .def_readwrite("ee_orientation_constraint", &cca::TaskDescription::ee_orientation_constraint)
        .def_readwrite("reserve_mobility", &cca::TaskDescription::reserve_mobility);

    py::class_<cca::PlannerConfig>(module, "PlannerConfig")
        .def(py::init<>())
        .def_readwrite("accuracy", &cca::PlannerConfig::accuracy)
        .def_readwrite("closure_err_threshold_ang", &cca::PlannerConfig::closure_err_threshold_ang)
        .def_readwrite("closure_err_threshold_lin", &cca::PlannerConfig::closure_err_threshold_lin)
        .def_readwrite("ik_max_itr", &cca::PlannerConfig::ik_max_itr)
        .def_readwrite("update_method", &cca::PlannerConfig::update_method)
        .def_readwrite("enable_joint_limits", &cca::PlannerConfig::enable_joint_limits)
        .def_readwrite("enable_nullspace_planning", &cca::PlannerConfig::enable_nullspace_planning)
        .def_readwrite("enable_capability_aware_planning", &cca::PlannerConfig::enable_capability_aware_planning)
        .def_readwrite("svd_relative_tolerance", &cca::PlannerConfig::svd_relative_tolerance)
        .def_readwrite("svd_absolute_tolerance", &cca::PlannerConfig::svd_absolute_tolerance)
        .def_readwrite("residual_mobility_tolerance", &cca::PlannerConfig::residual_mobility_tolerance)
        .def_readwrite("soft_limit_ratio", &cca::PlannerConfig::soft_limit_ratio)
        .def_readwrite("arm_mobility_weight", &cca::PlannerConfig::arm_mobility_weight)
        .def_readwrite("joint_limit_barrier_gain", &cca::PlannerConfig::joint_limit_barrier_gain)
        .def_readwrite("joint_limit_barrier_epsilon", &cca::PlannerConfig::joint_limit_barrier_epsilon)
        .def_readwrite("base_translation_weight", &cca::PlannerConfig::base_translation_weight)
        .def_readwrite("base_rotation_weight", &cca::PlannerConfig::base_rotation_weight)
        .def_readwrite("closure_secondary_weight", &cca::PlannerConfig::closure_secondary_weight)
        .def_readwrite("joint_limit_margin", &cca::PlannerConfig::joint_limit_margin)
        .def_readwrite("joint_limit_tolerance", &cca::PlannerConfig::joint_limit_tolerance);

    py::class_<cca::PlannerResult>(module, "PlannerResult")
        .def_readonly("success", &cca::PlannerResult::success)
        .def_readonly("trajectory_description", &cca::PlannerResult::trajectory_description)
        .def_readonly("joint_trajectory", &cca::PlannerResult::joint_trajectory)
        .def_property_readonly("planning_time_us", [](const cca::PlannerResult &result) {
            return result.planning_time.count();
        })
        .def_readonly("update_method", &cca::PlannerResult::update_method)
        .def_readonly("update_trail", &cca::PlannerResult::update_trail)
        .def_readonly("includes_gripper_trajectory", &cca::PlannerResult::includes_gripper_trajectory)
        .def_readonly("task_description", &cca::PlannerResult::task_description)
        .def_readonly("reserve_mobility_used", &cca::PlannerResult::reserve_mobility_used)
        .def_readonly("reserve_activation_count", &cca::PlannerResult::reserve_activation_count)
        .def_readonly("reserve_trajectory", &cca::PlannerResult::reserve_trajectory)
        .def_readonly("reserve_pose_trajectory", &cca::PlannerResult::reserve_pose_trajectory)
        .def_readonly("residual_mobility_norm", &cca::PlannerResult::residual_mobility_norm)
        .def_readonly("reserve_active", &cca::PlannerResult::reserve_active)
        .def_readonly("active_arm_dof_count", &cca::PlannerResult::active_arm_dof_count);

    py::class_<cca::CcAffordancePlannerInterface>(module, "PlannerInterface")
        .def(py::init<>())
        .def(py::init<const cca::PlannerConfig &>(), py::arg("config"))
        .def("generate_joint_trajectory", &cca::CcAffordancePlannerInterface::generate_joint_trajectory,
             py::arg("robot"), py::arg("task"), py::call_guard<py::gil_scoped_release>());

    module.def("robot_builder_from_yaml",
               py::overload_cast<const std::string &>(&au::robot_builder), py::arg("config_path"));
    module.def("extract_info_for_urdf_robot_builder", &au::extract_info_for_urdf_robot_builder,
               py::arg("config_path"));
    module.def("robot_builder_from_urdf_string",
               py::overload_cast<const std::string &, const au::RobotConfig &>(&au::robot_builder),
               py::arg("urdf_string"), py::arg("builder_info"));
    module.def("make_robot_description", &au::make_robot_description, py::arg("robot_config"),
               py::arg("joint_states"), py::arg("gripper_state") = std::numeric_limits<double>::quiet_NaN());
    module.def("make_floating_base_reserve_description", &au::make_floating_base_reserve_description,
               py::arg("initial_base_pose") = Eigen::Matrix4d::Identity(),
               py::arg("translation_max_step") = std::numeric_limits<double>::infinity(),
               py::arg("rotation_max_step") = std::numeric_limits<double>::infinity());
    module.def("fkin_space", &au::FKinSpace, py::arg("M"), py::arg("slist"), py::arg("joint_states"));
    module.def("jacobian_space", &au::JacobianSpace, py::arg("slist"), py::arg("joint_states"));
    module.def("adjoint", &au::Adjoint, py::arg("transform"));
    module.def("trans_inv", &au::TransInv, py::arg("transform"));
    module.def("matrix_log6", &au::MatrixLog6, py::arg("transform"));
    module.def("se3_to_vec", &au::se3ToVec, py::arg("se3"));
    module.def(
        "solve_pose_ik",
        [](const au::RobotDescription &robot, const Eigen::Matrix4d &target_pose, int max_iterations,
           double angular_tolerance, double linear_tolerance, double damping, double max_joint_step) {
            if (max_iterations < 1 || angular_tolerance <= 0.0 || linear_tolerance <= 0.0 || damping <= 0.0 ||
                max_joint_step <= 0.0)
            {
                throw std::invalid_argument("Pose-IK iteration, tolerance, damping, and step parameters must be positive.");
            }
            if (robot.slist.rows() != 6 || robot.slist.cols() != robot.joint_states.size())
            {
                throw std::invalid_argument("Pose IK requires a 6 x n screw list matching the joint seed.");
            }

            Eigen::VectorXd joints = robot.joint_states;
            const bool use_limits = robot.joint_lower_limits.size() != 0;
            if (use_limits && (robot.joint_lower_limits.size() != joints.size() ||
                               robot.joint_upper_limits.size() != joints.size()))
            {
                throw std::invalid_argument("Pose-IK joint limits must both match the robot DOF.");
            }

            auto error_twist = [&](const Eigen::VectorXd &state) {
                const Eigen::Matrix4d current_pose = au::FKinSpace(robot.M, robot.slist, state);
                const Eigen::VectorXd body_error =
                    au::se3ToVec(au::MatrixLog6(au::TransInv(current_pose) * target_pose));
                const Eigen::VectorXd space_error = au::Adjoint(current_pose) * body_error;
                return space_error;
            };

            bool converged = false;
            for (int iteration = 0; iteration < max_iterations; ++iteration)
            {
                const Eigen::VectorXd error = error_twist(joints);
                if (error.head(3).norm() <= angular_tolerance && error.tail(3).norm() <= linear_tolerance)
                {
                    converged = true;
                    break;
                }
                const Eigen::MatrixXd jacobian = au::JacobianSpace(robot.slist, joints);
                Eigen::Matrix<double, 6, 6> gram = jacobian * jacobian.transpose();
                gram.diagonal().array() += damping * damping;
                Eigen::VectorXd step = jacobian.transpose() * gram.ldlt().solve(error);
                if (step.norm() > max_joint_step)
                {
                    step *= max_joint_step / step.norm();
                }
                joints += step;
                if (use_limits)
                {
                    for (Eigen::Index i = 0; i < joints.size(); ++i)
                    {
                        joints(i) = std::clamp(joints(i), robot.joint_lower_limits(i), robot.joint_upper_limits(i));
                    }
                }
            }
            if (!converged)
            {
                const Eigen::VectorXd final_error = error_twist(joints);
                converged = final_error.head(3).norm() <= angular_tolerance &&
                            final_error.tail(3).norm() <= linear_tolerance;
            }
            return std::make_pair(joints, converged);
        },
        py::arg("robot"), py::arg("target_pose"), py::arg("max_iterations") = 100,
        py::arg("angular_tolerance") = 1e-3, py::arg("linear_tolerance") = 1e-3,
        py::arg("damping") = 1e-2, py::arg("max_joint_step") = 0.25,
        py::call_guard<py::gil_scoped_release>());
}
