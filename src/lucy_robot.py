import toml

from .lucy_body import LucyBody
from .coach_behaviour import CoachBehaviour


def matching_robot(robots: dict, joints: set[str], preferred: str | None = None) -> str:
    def fit(name):
        mapped = [row["joint"] for rows in robots[name].values() for row in rows]
        return sum(joint in joints for joint in mapped) / len(mapped) if mapped else 0.0

    best = max(robots, key=lambda name: (fit(name), name == preferred))
    if fit(best) == 0:
        raise ValueError("no robot in lucy_robots.toml matches the joints Lucy is running")
    return best


class LucyCoachRobot(CoachBehaviour, LucyBody):
    """The coach's behaviours on any Lucy robot; poses use the robot's own section when the pose file has one."""

    def __init__(self, url: str, robot_package: str | None, joint_map: dict, ros=None, robots: dict | None = None):
        self.robot_package = robot_package
        super().__init__(url, joint_map, ros=ros)
        if robots:
            self.robot_package = matching_robot(robots, set(self.owners), robot_package)
            self.joint_map = robots[self.robot_package]
            print(f"Coaching {self.robot_package} through Lucy")

    def load_pose(self, file: str | dict):
        pose = toml.load(file) if isinstance(file, str) else file
        if self.robot_package in pose:
            return dict(pose[self.robot_package])
        return super().load_pose(pose)
