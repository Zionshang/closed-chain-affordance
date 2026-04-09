#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <affordance_util/affordance_util.hpp>
#include <cc_affordance_planner/cc_affordance_planner.hpp>
#include <cc_affordance_planner/cc_affordance_planner_interface.hpp>

namespace py = pybind11;
namespace aff = affordance_util;
namespace cca = cc_affordance_planner;

namespace
{

aff::RobotDescription robot_description_from_config(const aff::RobotConfig &robot_config,
                                                    const std::optional<Eigen::VectorXd> &joint_states,
                                                    double gripper_state)
{
    aff::RobotDescription robot_description;
    robot_description.slist = robot_config.Slist;
    robot_description.M = robot_config.M;
    robot_description.joint_states =
        joint_states.value_or(Eigen::VectorXd::Zero(robot_config.Slist.cols()));
    robot_description.gripper_state = gripper_state;
    return robot_description;
}

std::string read_text_file(const std::string &file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

} // namespace

PYBIND11_MODULE(closed_chain_affordance, m)
{
    m.doc() = "Python bindings for the Closed-Chain Affordance planning framework";

    py::enum_<aff::Axis>(m, "Axis")
        .value("X", aff::Axis::X)
        .value("Y", aff::Axis::Y)
        .value("Z", aff::Axis::Z)
        .value("X_MINUS", aff::Axis::X_MINUS)
        .value("Y_MINUS", aff::Axis::Y_MINUS)
        .value("Z_MINUS", aff::Axis::Z_MINUS)
        .value("ORIGIN", aff::Axis::ORIGIN)
        .value("MANUAL", aff::Axis::MANUAL);

    py::enum_<aff::PoseSpecificationMethod>(m, "PoseSpecificationMethod")
        .value("PROVIDED", aff::PoseSpecificationMethod::PROVIDED)
        .value("FROM_FK", aff::PoseSpecificationMethod::FROM_FK)
        .value("FROM_FRAME_NAME", aff::PoseSpecificationMethod::FROM_FRAME_NAME);

    py::enum_<aff::GripperGoalType>(m, "GripperGoalType")
        .value("CONSTANT", aff::GripperGoalType::CONSTANT)
        .value("CONTINUOUS", aff::GripperGoalType::CONTINUOUS);

    py::enum_<aff::ScrewType>(m, "ScrewType")
        .value("ROTATION", aff::ScrewType::ROTATION)
        .value("TRANSLATION", aff::ScrewType::TRANSLATION)
        .value("SCREW", aff::ScrewType::SCREW)
        .value("UNSET", aff::ScrewType::UNSET);

    py::enum_<aff::VirtualScrewOrder>(m, "VirtualScrewOrder")
        .value("XYZ", aff::VirtualScrewOrder::XYZ)
        .value("YZX", aff::VirtualScrewOrder::YZX)
        .value("ZXY", aff::VirtualScrewOrder::ZXY)
        .value("XY", aff::VirtualScrewOrder::XY)
        .value("YZ", aff::VirtualScrewOrder::YZ)
        .value("ZX", aff::VirtualScrewOrder::ZX)
        .value("NONE", aff::VirtualScrewOrder::NONE);

    py::class_<aff::VecInfo>(m, "VecInfo")
        .def(py::init<>())
        .def_readwrite("axis", &aff::VecInfo::axis)
        .def_readwrite("location", &aff::VecInfo::location);

    py::class_<aff::PoseFrom>(m, "PoseFrom")
        .def(py::init<>())
        .def_readwrite("method", &aff::PoseFrom::method)
        .def_readwrite("frame_name", &aff::PoseFrom::frame_name)
        .def_readwrite("post_transform", &aff::PoseFrom::post_transform);

    py::class_<aff::ScrewInfoFrom>(m, "ScrewInfoFrom")
        .def(py::init<>())
        .def_readwrite("method", &aff::ScrewInfoFrom::method)
        .def_readwrite("frame_name", &aff::ScrewInfoFrom::frame_name)
        .def_readwrite("post_transform", &aff::ScrewInfoFrom::post_transform)
        .def_readwrite("axis_in_final_pose", &aff::ScrewInfoFrom::axis_in_final_pose);

    py::class_<aff::ScrewInfo>(m, "ScrewInfo")
        .def(py::init<>())
        .def_readwrite("type", &aff::ScrewInfo::type)
        .def_readwrite("axis", &aff::ScrewInfo::axis)
        .def_readwrite("location", &aff::ScrewInfo::location)
        .def_readwrite("screw", &aff::ScrewInfo::screw)
        .def_readwrite("pitch", &aff::ScrewInfo::pitch);

    py::class_<aff::RobotDescription>(m, "RobotDescription")
        .def(py::init<>())
        .def_readwrite("slist", &aff::RobotDescription::slist)
        .def_readwrite("M", &aff::RobotDescription::M)
        .def_readwrite("joint_states", &aff::RobotDescription::joint_states)
        .def_readwrite("gripper_state", &aff::RobotDescription::gripper_state);

    py::enum_<cca::EeOrientationConstraint>(m, "EeOrientationConstraint")
        .value("PRESERVE", cca::EeOrientationConstraint::PRESERVE)
        .value("DEFAULT", cca::EeOrientationConstraint::DEFAULT);

    py::enum_<cca::PlanningType>(m, "PlanningType")
        .value("APPROACH", cca::PlanningType::APPROACH)
        .value("AFFORDANCE", cca::PlanningType::AFFORDANCE)
        .value("EE_ORIENTATION_ONLY", cca::PlanningType::EE_ORIENTATION_ONLY)
        .value("CARTESIAN_GOAL", cca::PlanningType::CARTESIAN_GOAL);

    py::enum_<cca::MotionType>(m, "MotionType")
        .value("APPROACH", cca::MotionType::APPROACH)
        .value("AFFORDANCE", cca::MotionType::AFFORDANCE);

    py::enum_<cca::TrajectoryDescription>(m, "TrajectoryDescription")
        .value("FULL", cca::TrajectoryDescription::FULL)
        .value("PARTIAL", cca::TrajectoryDescription::PARTIAL)
        .value("UNSET", cca::TrajectoryDescription::UNSET);

    py::enum_<cca::UpdateMethod>(m, "UpdateMethod")
        .value("INVERSE", cca::UpdateMethod::INVERSE)
        .value("TRANSPOSE", cca::UpdateMethod::TRANSPOSE)
        .value("BEST", cca::UpdateMethod::BEST);

    py::class_<cca::Goal>(m, "Goal")
        .def(py::init<>())
        .def_readwrite("affordance", &cca::Goal::affordance)
        .def_readwrite("ee_orientation", &cca::Goal::ee_orientation)
        .def_readwrite("canonical_pose", &cca::Goal::canonical_pose)
        .def_readwrite("gripper", &cca::Goal::gripper);

    py::class_<cca::TaskDescription>(m, "TaskDescription")
        .def(py::init<>())
        .def(py::init<const cca::PlanningType &>(), py::arg("planning_type"))
        .def_readwrite("affordance_info", &cca::TaskDescription::affordance_info)
        .def_readwrite("goal", &cca::TaskDescription::goal)
        .def_readwrite("trajectory_density", &cca::TaskDescription::trajectory_density)
        .def_readwrite("motion_type", &cca::TaskDescription::motion_type)
        .def_readwrite("vir_screw_order", &cca::TaskDescription::vir_screw_order)
        .def_readwrite("gripper_goal_type", &cca::TaskDescription::gripper_goal_type)
        .def_readwrite("ee_orientation_constraint", &cca::TaskDescription::ee_orientation_constraint)
        .def_readwrite("affordance_info_from", &cca::TaskDescription::affordance_info_from)
        .def_readwrite("canonical_pose_from", &cca::TaskDescription::canonical_pose_from);

    py::class_<cca::PlannerConfig>(m, "PlannerConfig")
        .def(py::init<>())
        .def_readwrite("accuracy", &cca::PlannerConfig::accuracy)
        .def_readwrite("closure_err_threshold_ang", &cca::PlannerConfig::closure_err_threshold_ang)
        .def_readwrite("closure_err_threshold_lin", &cca::PlannerConfig::closure_err_threshold_lin)
        .def_readwrite("ik_max_itr", &cca::PlannerConfig::ik_max_itr)
        .def_readwrite("update_method", &cca::PlannerConfig::update_method);

    py::class_<cca::PlannerResult>(m, "PlannerResult")
        .def(py::init<>())
        .def_readwrite("success", &cca::PlannerResult::success)
        .def_readwrite("trajectory_description", &cca::PlannerResult::trajectory_description)
        .def_readwrite("joint_trajectory", &cca::PlannerResult::joint_trajectory)
        .def_readwrite("planning_time", &cca::PlannerResult::planning_time)
        .def_readwrite("update_method", &cca::PlannerResult::update_method)
        .def_readwrite("update_trail", &cca::PlannerResult::update_trail)
        .def_readwrite("includes_gripper_trajectory", &cca::PlannerResult::includes_gripper_trajectory)
        .def_readwrite("task_description", &cca::PlannerResult::task_description);

    py::class_<cca::CcAffordancePlannerInterface>(m, "CcAffordancePlannerInterface")
        .def(py::init<>())
        .def(py::init<const cca::PlannerConfig &>(), py::arg("planner_config"))
        .def("generate_joint_trajectory", &cca::CcAffordancePlannerInterface::generate_joint_trajectory,
             py::arg("robot_description"), py::arg("task_description"));

    m.def("axis_to_vec", &aff::axis_to_vec, py::arg("axis"));
    m.def("get_screw",
          py::overload_cast<const aff::ScrewInfo &>(&aff::get_screw),
          py::arg("screw_info"));
    m.def("fkin_space", &aff::FKinSpace, py::arg("M"), py::arg("slist"), py::arg("thetalist"));
    m.def("plan",
          [](const aff::RobotDescription &robot_description, const cca::TaskDescription &task_description,
             const cca::PlannerConfig &planner_config) {
              cca::CcAffordancePlannerInterface planner(planner_config);
              return planner.generate_joint_trajectory(robot_description, task_description);
          },
          py::arg("robot_description"), py::arg("task_description"),
          py::arg("planner_config") = cca::PlannerConfig());

    m.def(
        "build_robot_description_from_yaml",
        [](const std::string &config_file_path, const std::optional<Eigen::VectorXd> &joint_states,
           double gripper_state) {
            const aff::RobotConfig robot_config = aff::robot_builder(config_file_path);
            return robot_description_from_config(robot_config, joint_states, gripper_state);
        },
        py::arg("config_file_path"), py::arg("joint_states") = std::nullopt,
        py::arg("gripper_state") = std::numeric_limits<double>::quiet_NaN());

    m.def(
        "build_robot_description_from_urdf",
        [](const std::string &urdf_file_path, const std::string &urdf_config_file_path,
           const std::optional<Eigen::VectorXd> &joint_states, double gripper_state) {
            const std::string urdf_string = read_text_file(urdf_file_path);
            const aff::RobotConfig builder_info = aff::extract_info_for_urdf_robot_builder(urdf_config_file_path);
            const aff::RobotConfig robot_config = aff::robot_builder(urdf_string, builder_info);
            return robot_description_from_config(robot_config, joint_states, gripper_state);
        },
        py::arg("urdf_file_path"), py::arg("urdf_config_file_path"), py::arg("joint_states") = std::nullopt,
        py::arg("gripper_state") = std::numeric_limits<double>::quiet_NaN());
}
