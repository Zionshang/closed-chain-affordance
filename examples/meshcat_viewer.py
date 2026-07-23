from __future__ import annotations

import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import meshcat.geometry as g
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


class MeshcatViewer:
    def __init__(
        self,
        urdf_path: Path,
        package_map: dict[str, Path] | None = None,
        frame_delay: float = 0.25,
        root_path: str = "robot",
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        self.package_map = {name: Path(path).resolve() for name, path in (package_map or {}).items()}
        self.frame_delay = frame_delay
        self.root_path = root_path

        self.pin = pin
        self.MeshcatVisualizer = MeshcatVisualizer
        self._patched_urdf_path = self._create_patched_urdf()

        self.model = None
        self.data = None
        self.collision_model = None
        self.visual_model = None
        self.viz = None

    def open(self) -> None:
        self.model, self.collision_model, self.visual_model = self.pin.buildModelsFromUrdf(
            str(self._patched_urdf_path)
        )
        self.data = self.model.createData()
        self.viz = self.MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel(rootNodeName=self.root_path)

    def animate_trajectory(
        self,
        trajectory: list[np.ndarray],
        joint_names: list[str],
        loop: bool = True,
        fixed_joint_values: dict[str, float | np.ndarray] | None = None,
        trajectory_frame_name: str | None = None,
        trajectory_color: int = 0xD62728,
    ) -> None:
        if self.viz is None or self.model is None or self.data is None:
            raise RuntimeError("Call open() before animate_trajectory().")

        print("MeshCat viewer started.", flush=True)
        print(f"Open this URL in your browser: {self.viz.viewer.url()}", flush=True)
        print("Animating trajectory. Press Ctrl+C to stop.", flush=True)

        fixed_joint_values = fixed_joint_values or {}

        if trajectory_frame_name is not None:
            self._draw_frame_trajectory(
                trajectory=trajectory,
                joint_names=joint_names,
                fixed_joint_values=fixed_joint_values,
                frame_name=trajectory_frame_name,
                color=trajectory_color,
            )

        while True:
            for point in trajectory:
                q = self.configuration_from_values(
                    joint_names=joint_names,
                    joint_values=np.asarray(point, dtype=float),
                    fixed_joint_values=fixed_joint_values,
                )
                self.viz.display(q)
                time.sleep(self.frame_delay)
            if not loop:
                break

    def configuration_from_values(
        self,
        joint_names: list[str],
        joint_values: np.ndarray,
        fixed_joint_values: dict[str, float | np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call open() before building configurations.")

        q = self.pin.neutral(self.model)
        fixed_joint_values = fixed_joint_values or {}

        for joint_name, joint_value in zip(joint_names, joint_values):
            self._assign_joint_value(q, joint_name, joint_value)

        for joint_name, joint_value in fixed_joint_values.items():
            self._assign_joint_value(q, joint_name, joint_value)

        return q

    def _assign_joint_value(self, q: np.ndarray, joint_name: str, joint_value: float | np.ndarray) -> None:
        joint_id = self.model.getJointId(joint_name)
        if joint_id == 0:
            raise KeyError(f"Joint '{joint_name}' was not found in the Pinocchio model.")

        joint = self.model.joints[joint_id]
        idx_q = joint.idx_q
        nq = joint.nq
        value_array = np.atleast_1d(np.asarray(joint_value, dtype=float))
        if value_array.size != nq:
            raise ValueError(
                f"Joint '{joint_name}' expects {nq} configuration value(s), but got {value_array.size}."
            )
        q[idx_q : idx_q + nq] = value_array

    def _create_patched_urdf(self) -> Path:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        for mesh_node in root.findall(".//mesh"):
            filename = mesh_node.attrib.get("filename")
            if filename is None:
                continue
            mesh_node.attrib["filename"] = str(self._resolve_mesh_uri(filename))

        temp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".urdf",
            prefix=f"{self.urdf_path.stem}_pin_",
            delete=False,
        )
        tree.write(temp_file, encoding="utf-8", xml_declaration=True)
        temp_file.close()
        return Path(temp_file.name)

    def _draw_frame_trajectory(
        self,
        trajectory: list[np.ndarray],
        joint_names: list[str],
        fixed_joint_values: dict[str, float | np.ndarray],
        frame_name: str,
        color: int,
    ) -> None:
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= len(self.model.frames):
            raise KeyError(f"Frame '{frame_name}' was not found in the Pinocchio model.")

        positions = []
        for point in trajectory:
            q = self.configuration_from_values(
                joint_names=joint_names,
                joint_values=np.asarray(point, dtype=float),
                fixed_joint_values=fixed_joint_values,
            )
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            positions.append(self.data.oMf[frame_id].translation.copy())

        if not positions:
            return

        points = np.column_stack(positions)
        self.viz.viewer[self.root_path]["frame_trajectory"].set_object(
            g.Line(
                g.PointsGeometry(points),
                g.LineBasicMaterial(color=color, linewidth=2.0),
            )
        )

    def _resolve_mesh_uri(self, mesh_uri: str) -> Path:
        if mesh_uri.startswith("package://"):
            package_and_path = mesh_uri[len("package://") :]
            package_name, relative_path = package_and_path.split("/", 1)
            if package_name not in self.package_map:
                raise KeyError(f"Missing package mapping for '{package_name}' while resolving '{mesh_uri}'.")
            return (self.package_map[package_name] / relative_path).resolve()
        return (self.urdf_path.parent / mesh_uri).resolve()
