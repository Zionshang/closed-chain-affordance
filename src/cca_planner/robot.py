"""Minimal URDF loader that produces Torch planner tensors.

This module intentionally implements only the serial-chain information needed
by the planner: the home screw list, home TCP transform, and initial joints.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class RobotDescription:
    """Torch-native serial robot model."""

    slist: torch.Tensor
    M: torch.Tensor
    joint_states: torch.Tensor
    joint_names: tuple[str, ...]


def _vector(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if text is None:
        return default
    values = tuple(float(token) for token in text.replace(",", " ").split())
    return values if len(values) == 3 else default


def _origin_matrix(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """URDF fixed-axis RPY transform: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=dtype,
        device=device,
    )
    transform = torch.eye(4, dtype=dtype, device=device)
    transform[:3, :3] = rotation
    transform[:3, 3] = torch.tensor(xyz, dtype=dtype, device=device)
    return transform


class _UrdfModel:
    def __init__(self, text: str, *, dtype: torch.dtype, device: torch.device):
        root = ET.fromstring(text)
        self.joints: dict[str, dict] = {}
        self.link_parent_joint: dict[str, str] = {}
        self.links = {element.attrib["name"] for element in root.findall("link")}

        for element in root.findall("joint"):
            name = element.attrib.get("name")
            parent = element.find("parent")
            child = element.find("child")
            if not name or parent is None or child is None:
                continue
            parent_link = parent.attrib["link"]
            child_link = child.attrib["link"]
            origin_element = element.find("origin")
            xyz = _vector(
                origin_element.attrib.get("xyz") if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = _vector(
                origin_element.attrib.get("rpy") if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            axis_element = element.find("axis")
            axis = _vector(
                axis_element.attrib.get("xyz") if axis_element is not None else None,
                (1.0, 0.0, 0.0),
            )
            self.joints[name] = {
                "type": element.attrib.get("type", "fixed"),
                "parent": parent_link,
                "child": child_link,
                "origin": _origin_matrix(xyz, rpy, dtype=dtype, device=device),
                "axis": torch.tensor(axis, dtype=dtype, device=device),
            }
            self.links.update((parent_link, child_link))
            self.link_parent_joint[child_link] = name

        child_links = {joint["child"] for joint in self.joints.values()}
        self.root_link = next((link for link in self.links if link not in child_links), None)
        self.dtype = dtype
        self.device = device

    def transform_to_link(self, link: str, reference: str) -> torch.Tensor:
        if link == reference:
            return torch.eye(4, dtype=self.dtype, device=self.device)
        origins: list[torch.Tensor] = []
        current = link
        while current != reference:
            parent_joint = self.link_parent_joint.get(current)
            if parent_joint is None:
                raise ValueError(f"reference link {reference!r} is not an ancestor of {link!r}")
            joint = self.joints[parent_joint]
            origins.append(joint["origin"])
            current = joint["parent"]
        transform = torch.eye(4, dtype=self.dtype, device=self.device)
        for origin in reversed(origins):
            transform = transform @ origin
        return transform

    def chain(self, base_joint: str, end_joint: str) -> list[str]:
        if base_joint not in self.joints or end_joint not in self.joints:
            raise ValueError("configured base/end joint is missing from the URDF")
        names: list[str] = []
        current: str | None = end_joint
        while current is not None:
            names.insert(0, current)
            if current == base_joint:
                return names
            parent_link = self.joints[current]["parent"]
            current = self.link_parent_joint.get(parent_link)
        raise ValueError("base joint is not on the chain ending at end_joint")


def load_robot_from_urdf(
    urdf_path: str | Path,
    config_path: str | Path,
    *,
    joint_states=None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> RobotDescription:
    """Load a configured serial chain and return tensors ready for the planner."""
    device = torch.device("cpu" if device is None else device)
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not config:
        raise ValueError("robot config YAML is empty")

    reference = config["ref_frame"][0]["name"]
    chain_config = config["kinematic_chain"][0]
    base_joint = chain_config["base_joint_name"]
    end_joint = chain_config["end_joint_name"]
    ee_link = config["end_effector"][0]["frame_name"]

    model = _UrdfModel(Path(urdf_path).read_text(encoding="utf-8"), dtype=dtype, device=device)
    if reference not in model.links or ee_link not in model.links:
        raise ValueError("configured reference or end-effector link is missing from the URDF")

    screws: list[torch.Tensor] = []
    names: list[str] = []
    for name in model.chain(base_joint, end_joint):
        joint = model.joints[name]
        if joint["type"] == "fixed":
            continue
        if joint["type"] not in {"revolute", "continuous", "prismatic"}:
            raise ValueError(f"unsupported joint type {joint['type']!r} in configured chain")
        pose = model.transform_to_link(joint["parent"], reference) @ joint["origin"]
        axis = pose[:3, :3] @ joint["axis"]
        if joint["type"] == "prismatic":
            screw = torch.cat([torch.zeros(3, dtype=dtype, device=device), axis])
        else:
            screw = torch.cat([axis, torch.linalg.cross(pose[:3, 3], axis)])
        screws.append(screw)
        names.append(name)

    if not screws:
        raise ValueError("configured chain contains no movable joints")
    slist = torch.stack(screws, dim=-1)
    if joint_states is None:
        q = torch.zeros(len(screws), dtype=dtype, device=device)
    else:
        q = torch.as_tensor(joint_states, dtype=dtype, device=device).reshape(-1)
        if q.numel() != len(screws):
            raise ValueError(f"joint_states has {q.numel()} entries; expected {len(screws)}")
    return RobotDescription(
        slist=slist,
        M=model.transform_to_link(ee_link, reference),
        joint_states=q,
        joint_names=tuple(names),
    )


# Descriptive alias retained for the example and discoverability.
build_robot_description_from_urdf = load_robot_from_urdf
