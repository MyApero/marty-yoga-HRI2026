import toml

from .lucy_body import LucyBody
from .coach_behaviour import CoachBehaviour


class LucyCoachRobot(CoachBehaviour, LucyBody):
    """The coach's behaviours on any Lucy robot; poses use the robot's own section when the pose file has one."""

    def __init__(self, url: str, robot_package: str, joint_map: dict, ros=None):
        self.robot_package = robot_package
        super().__init__(url, joint_map, ros=ros)

    def load_pose(self, file: str | dict):
        pose = toml.load(file) if isinstance(file, str) else file
        if self.robot_package in pose:
            return dict(pose[self.robot_package])
        return super().load_pose(pose)
