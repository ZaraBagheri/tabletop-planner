"""Robot/workspace profiles. A different arm still needs its own validated controller."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from .domain import TRAYS


@dataclass
class RobotConfig:
    arm_urdf: str = "franka_panda/panda.urdf"
    joint_indices: list = field(default_factory=lambda: list(range(7)))
    home: list = field(default_factory=lambda: [0, -.4, 0, -2.4, 0, 2.0, .8])
    end_effector_link: int = 11
    finger_joints: list = field(default_factory=lambda: [9, 10])
    workspace_lower: list = field(default_factory=lambda: [.23, -.36, .025])
    workspace_upper: list = field(default_factory=lambda: [.72, .36, .50])
    trays: dict = field(default_factory=lambda: {key: list(value) for key, value in TRAYS.items()})
    scene_positions: list = field(default_factory=lambda: [[.43, -.07], [.56, -.07], [.43, .07], [.56, .07]])
    scene_jitter: float = .018
    approach_height: float = .30
    release_height: float = .075
    gripper_open: float = .04
    gripper_closed: float = .018
    arm_force: float = 180
    gripper_force: float = 30
    tray_half_size: float = .075
    tray_membership_half_size: float = .053
    block_half_size: float = .02

    def __post_init__(self):
        def finite(values):
            return all(type(value) in (int, float) and math.isfinite(value) for value in values)
        if not isinstance(self.arm_urdf, str) or not self.arm_urdf or not self.joint_indices or len(self.home) != len(self.joint_indices):
            raise ValueError("Provide an arm URDF and matching joint_indices/home arrays")
        indices = self.joint_indices + self.finger_joints + [self.end_effector_link]
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("Joint and link indices must be nonnegative integers")
        if len(set(self.joint_indices + self.finger_joints)) != len(self.joint_indices + self.finger_joints):
            raise ValueError("Arm and finger joints must be distinct")
        if not finite(self.home) or len(self.workspace_lower) != 3 or len(self.workspace_upper) != 3:
            raise ValueError("Invalid home positions or workspace dimensions")
        if not finite(self.workspace_lower + self.workspace_upper) or any(a >= b for a, b in zip(self.workspace_lower, self.workspace_upper)):
            raise ValueError("Workspace bounds must be finite and increasing")
        sizes = [self.approach_height, self.release_height, self.gripper_open, self.gripper_closed,
                 self.arm_force, self.gripper_force, self.tray_half_size, self.tray_membership_half_size, self.block_half_size]
        if not finite(sizes) or any(value <= 0 for value in sizes):
            raise ValueError("Robot dimensions, forces, and gripper settings must be positive")
        if not self.gripper_closed < self.gripper_open or not self.release_height < self.approach_height:
            raise ValueError("Invalid gripper widths or approach/release heights")
        if not self.workspace_lower[2] <= self.release_height < self.approach_height <= self.workspace_upper[2]:
            raise ValueError("Motion heights must lie inside the workspace")
        if not self.block_half_size < self.tray_membership_half_size < self.tray_half_size:
            raise ValueError("Tray membership must leave space for the block inside the tray")
        if set(self.trays) != set(TRAYS) or any(len(p) != 3 or not finite(p) for p in self.trays.values()):
            raise ValueError("Provide finite left_tray and right_tray positions")
        if len(self.scene_positions) != 4 or any(len(p) != 2 or not finite(p) for p in self.scene_positions):
            raise ValueError("Provide four finite two-dimensional starting positions")
        if not finite([self.scene_jitter]) or self.scene_jitter < 0:
            raise ValueError("scene_jitter must be finite and nonnegative")
        for point in list(self.trays.values()) + self.scene_positions:
            if not all(self.workspace_lower[i] <= point[i] <= self.workspace_upper[i] for i in (0, 1)):
                raise ValueError("Tray and starting positions must lie inside the workspace")

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        try:
            return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Invalid robot profile: {exc}") from exc

    def to_dict(self):
        return asdict(self)

    def contains(self, position):
        return len(position) == 3 and all(math.isfinite(v) and low <= v <= high
                                        for v, low, high in zip(position, self.workspace_lower, self.workspace_upper))
