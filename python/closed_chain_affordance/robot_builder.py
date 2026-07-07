"""Robot description builders from YAML config files and URDF strings.

The URDF parser is implemented with the standard-library ``xml.etree`` so that
no external URDF dependency (``urdfdom``/``yourdfpy``) is required. Rotation
handling leans on :mod:`scipy.spatial.transform`.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from functools import reduce

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .affordance_util import get_screw, get_screw_from_axis_location
from .enums import ScrewType
from .structs import JointData, RobotConfig, RobotDescription, ScrewInfo


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as handle:
        return handle.read()


def _parse_floats(text: str, count: int, default):
    """Parse a whitespace-separated list of floats, falling back to ``default``."""
    if text is None:
        return np.array(default, dtype=float)
    values = [float(token) for token in text.replace(",", " ").split()]
    if len(values) != count:
        return np.array(default, dtype=float)
    return np.array(values, dtype=float)


def _origin_to_matrix(origin_xyz, origin_rpy) -> np.ndarray:
    """URDF ``<origin>`` (xyz, rpy) -> 4x4 homogeneous transform.

    The URDF rpy convention is ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, which is
    an extrinsic XYZ Euler sequence.
    """
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("XYZ", origin_rpy).as_matrix()
    transform[:3, 3] = origin_xyz
    return transform


# --------------------------------------------------------------------------- #
# YAML robot builder
# --------------------------------------------------------------------------- #
def robot_builder_from_yaml(config_file_path: str) -> RobotConfig:
    """Build a :class:`RobotConfig` from a YAML file describing the robot screws."""
    with open(config_file_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        raise RuntimeError("Robot screw list cannot be built without a valid robot config yaml file")

    robot_config = RobotConfig()

    ref_frame_name = config["ref_frame"][0]["name"]
    robot_config.ref_frame_name = ref_frame_name

    joint_nodes = config["robot_joints"]
    joints_data = []
    for joint_node in joint_nodes:
        joint = JointData()
        joint.name = joint_node["name"]
        joint.screw_info.axis = np.asarray(joint_node["w"], dtype=float).reshape(3)
        joint.screw_info.location = np.asarray(joint_node["q"], dtype=float).reshape(3)
        joints_data.append(joint)

    ee_node = config["end_effector"][0]
    ee_frame_names = ee_node["frame_name"]
    ee_location = np.asarray(ee_node["q"], dtype=float).reshape(3)

    total_nof_joints = len(joints_data)
    slist = np.zeros((6, total_nof_joints))
    for i, joint in enumerate(joints_data):
        slist[:, i] = get_screw_from_axis_location(joint.screw_info.axis, joint.screw_info.location)
        robot_config.joint_names_robot.append(joint.name)

    robot_config.slist = slist
    robot_config.ee_frame_name = ee_frame_names
    m = np.eye(4)
    m[:3, 3] = ee_location
    robot_config.M = m
    return robot_config


def extract_info_for_urdf_robot_builder(config_file_path: str) -> RobotConfig:
    """Extract the info needed by the URDF builder from a YAML config file."""
    with open(config_file_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        raise RuntimeError("Unable to extract info for urdf robot_builder due to invalid yaml file")

    robot_config = RobotConfig()
    robot_config.ref_frame_name = config["ref_frame"][0]["name"]

    kinematic_chain = config["kinematic_chain"][0]
    robot_config.base_joint_name = kinematic_chain["base_joint_name"]
    robot_config.end_joint_name = kinematic_chain["end_joint_name"]

    robot_config.ee_frame_name = config["end_effector"][0]["frame_name"]
    return robot_config


# --------------------------------------------------------------------------- #
# URDF robot builder
# --------------------------------------------------------------------------- #
class _UrdfModel:
    """Minimal URDF model capturing the joints and links needed for screw extraction."""

    _MOTION_JOINT_TYPES = {
        "revolute": ScrewType.ROTATION,
        "continuous": ScrewType.ROTATION,
        "prismatic": ScrewType.TRANSLATION,
    }

    def __init__(self, urdf_string: str):
        root = ET.fromstring(urdf_string)

        self.joints = {}  # name -> dict(parent, child, type, origin(4x4), axis(3), limits)
        self.link_parent_joint = {}  # child link name -> joint name that produces it
        self.links = set()

        for link_el in root.findall("link"):
            if "name" in link_el.attrib:
                self.links.add(link_el.attrib["name"])

        for joint_el in root.findall("joint"):
            name = joint_el.attrib.get("name")
            jtype = joint_el.attrib.get("type", "fixed")
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            if parent_el is None or child_el is None:
                continue
            parent_link = parent_el.attrib["link"]
            child_link = child_el.attrib["link"]

            origin_el = joint_el.find("origin")
            if origin_el is not None:
                origin_xyz = _parse_floats(origin_el.attrib.get("xyz"), 3, [0.0, 0.0, 0.0])
                origin_rpy = _parse_floats(origin_el.attrib.get("rpy"), 3, [0.0, 0.0, 0.0])
                origin = _origin_to_matrix(origin_xyz, origin_rpy)
            else:
                origin = np.eye(4)

            axis_el = joint_el.find("axis")
            axis = _parse_floats(axis_el.attrib.get("xyz") if axis_el is not None else None, 3, [1.0, 0.0, 0.0])

            limit_el = joint_el.find("limit")
            if limit_el is not None:
                lower = float(limit_el.attrib.get("lower", 0.0))
                upper = float(limit_el.attrib.get("upper", 0.0))
            else:
                lower = 0.0
                upper = 0.0

            self.joints[name] = {
                "type": jtype,
                "parent": parent_link,
                "child": child_link,
                "origin": origin,
                "axis": axis,
                "lower": lower,
                "upper": upper,
            }
            self.links.add(parent_link)
            self.links.add(child_link)
            self.link_parent_joint[child_link] = name

        # The root link is the one that never appears as a joint child.
        child_links = {j["child"] for j in self.joints.values()}
        self.root_link = next((link for link in self.links if link not in child_links), None)

    def has_joint(self, name: str) -> bool:
        return name in self.joints

    def has_link(self, name: str) -> bool:
        return name in self.links

    def transform_ref_to_link(self, link_name: str, reference_frame: str) -> np.ndarray:
        """Transform from the reference frame to ``link_name`` (ref -> link)."""
        if link_name == reference_frame:
            return np.eye(4)

        origins = []
        current = link_name
        while current != reference_frame:
            parent_joint_name = self.link_parent_joint.get(current)
            if parent_joint_name is None:
                raise RuntimeError("Reached root link without finding reference frame")
            joint = self.joints[parent_joint_name]
            origins.append(joint["origin"])
            current = joint["parent"]

        # origins were collected from the target link upward; compose from the
        # reference-frame side outward.
        return reduce(np.matmul, reversed(origins), np.eye(4))

    def transform_ref_to_joint(self, joint_name: str, reference_frame: str) -> np.ndarray:
        """Transform from the reference frame to ``joint_name`` (ref -> joint frame)."""
        joint = self.joints[joint_name]
        return self.transform_ref_to_link(joint["parent"], reference_frame) @ joint["origin"]

    def chain_from_base_to_end(self, base_joint_name: str, end_joint_name: str) -> list:
        """Joint names ordered from base to end (inclusive)."""
        chain = []
        current_joint_name = end_joint_name
        while current_joint_name is not None:
            chain.insert(0, current_joint_name)
            if current_joint_name == base_joint_name:
                break
            joint = self.joints[current_joint_name]
            parent_link = joint["parent"]
            if parent_link == self.root_link and self.link_parent_joint.get(parent_link) is None:
                if current_joint_name != base_joint_name:
                    raise RuntimeError("Base joint not found on path from end effector frame")
                break
            current_joint_name = self.link_parent_joint.get(parent_link)
        if not chain or chain[0] != base_joint_name:
            raise RuntimeError("Base joint not found on path from end effector frame")
        return chain


def robot_builder_from_urdf(urdf_string: str, robot_config: RobotConfig) -> RobotConfig:
    """Build a :class:`RobotConfig` (screws + EE transform) from a URDF string."""
    out = RobotConfig()
    out.ref_frame_name = robot_config.ref_frame_name
    out.base_joint_name = robot_config.base_joint_name
    out.end_joint_name = robot_config.end_joint_name
    out.ee_frame_name = robot_config.ee_frame_name
    out.joint_names_robot = []

    base_joint_name = robot_config.base_joint_name
    end_joint_name = robot_config.end_joint_name
    ref_frame_name = robot_config.ref_frame_name
    ee_frame_name = robot_config.ee_frame_name

    model = _UrdfModel(urdf_string)
    if not model.has_joint(base_joint_name):
        raise RuntimeError("Robot URDF does not contain specified base joint")
    if not model.has_link(ref_frame_name):
        raise RuntimeError("Robot URDF does not contain specified reference frame")
    if not model.has_link(ee_frame_name):
        raise RuntimeError("Robot URDF does not contain specified ee frame")

    chain_list = model.chain_from_base_to_end(base_joint_name, end_joint_name)

    joint_pose_in_ref_frame = model.transform_ref_to_joint(base_joint_name, ref_frame_name)
    joints_data = []
    for joint_name in chain_list:
        joint = model.joints[joint_name]
        if joint_name != base_joint_name:
            joint_pose_in_ref_frame = joint_pose_in_ref_frame @ joint["origin"]

        jtype = joint["type"]
        if jtype == "fixed":
            continue

        if jtype not in _UrdfModel._MOTION_JOINT_TYPES:
            raise RuntimeError("Kinematic chain contains a joint type not accounted for.")

        screw_type = _UrdfModel._MOTION_JOINT_TYPES[jtype]
        screw_info = ScrewInfo()
        screw_info.type = screw_type
        screw_info.location = joint_pose_in_ref_frame[:3, 3].copy()
        world_axis = joint_pose_in_ref_frame[:3, :3] @ joint["axis"]
        screw_info.axis = world_axis
        screw_info.screw = get_screw(screw_info)

        joint_data = JointData(
            name=joint_name,
            screw_info=screw_info,
            limits_lower=joint["lower"],
            limits_upper=joint["upper"],
        )
        joints_data.append(joint_data)

    slist = np.zeros((6, len(joints_data)))
    for i, joint in enumerate(joints_data):
        slist[:, i] = get_screw(joint.screw_info)
        out.joint_names_robot.append(joint.name)

    out.slist = slist
    out.M = model.transform_ref_to_link(ee_frame_name, ref_frame_name)
    return out


# --------------------------------------------------------------------------- #
# Binding-level convenience builders -> RobotDescription
# --------------------------------------------------------------------------- #
def _robot_description_from_config(
    robot_config: RobotConfig, joint_states, gripper_state: float
) -> RobotDescription:
    robot_description = RobotDescription()
    robot_description.slist = robot_config.slist
    robot_description.M = robot_config.M
    if joint_states is None:
        robot_description.joint_states = np.zeros(robot_config.slist.shape[1])
    else:
        robot_description.joint_states = np.asarray(joint_states, dtype=float).reshape(-1)
    if gripper_state is None or (isinstance(gripper_state, float) and math.isnan(gripper_state)):
        robot_description.gripper_state = float("nan")
    else:
        robot_description.gripper_state = float(gripper_state)
    return robot_description


def build_robot_description_from_yaml(
    config_file_path: str, joint_states=None, gripper_state=float("nan")
) -> RobotDescription:
    """Build a :class:`RobotDescription` from a YAML robot config file."""
    robot_config = robot_builder_from_yaml(config_file_path)
    return _robot_description_from_config(robot_config, joint_states, gripper_state)


def build_robot_description_from_urdf(
    urdf_file_path: str,
    urdf_config_file_path: str,
    joint_states=None,
    gripper_state=float("nan"),
) -> RobotDescription:
    """Build a :class:`RobotDescription` from a URDF file + YAML builder config."""
    urdf_string = _read_text_file(urdf_file_path)
    builder_info = extract_info_for_urdf_robot_builder(urdf_config_file_path)
    robot_config = robot_builder_from_urdf(urdf_string, builder_info)
    return _robot_description_from_config(robot_config, joint_states, gripper_state)
