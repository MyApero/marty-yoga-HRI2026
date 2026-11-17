import math
import time

import pytest

from src.lucy_robot import LucyCoachRobot, matching_robot

JTC = "joint_trajectory_controller/JointTrajectoryController"
CONTROLLERS = [
    {"name": "arms_controller", "type": JTC, "state": "active",
     "claimed_interfaces": ["l_arm/position", "r_arm/position"]},
    {"name": "body_controller", "type": JTC, "state": "active", "claimed_interfaces": ["antenna/position"]},
    {"name": "arms_torque_enable_controller", "type": "forward_command_controller/ForwardCommandController",
     "state": "active", "claimed_interfaces": ["l_arm/torque_enable"]},
    {"name": "legs_controller", "type": JTC, "state": "inactive", "claimed_interfaces": []},
]


class FakeTopic:
    def __init__(self, name):
        self.name, self.sent = name, []

    def publish(self, message):
        self.sent.append(dict(message))

    def subscribe(self, callback):
        self.callback = callback


class FakeRos:
    is_connected = True


class FakeRobot(LucyCoachRobot):
    def __init__(self, joint_map):
        self.topics = {}
        super().__init__("ws://localhost:9090", "fake_urdf", joint_map, ros=FakeRos())

    def ros_topic(self, name, message_type):
        return self.topics.setdefault(name, FakeTopic(name))

    def list_controllers(self):
        return CONTROLLERS

    def call_service(self, name, service_type, request):
        self.service_calls = getattr(self, "service_calls", []) + [(name, request)]
        return {"success": True}


def settle(robot):
    deadline = time.time() + 2
    while not robot.is_empty and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)


@pytest.fixture
def robot():
    robot = FakeRobot({
        "left arm": [{"joint": "l_arm", "scale": 2.0, "offset": -90.0}],
        "eyes": [{"joint": "antenna", "scale": -1.0}],
    })
    settle(robot)
    for topic in robot.topics.values():
        topic.sent.clear()
    return robot


def test_a_gesture_sends_the_whole_controller_with_scale_and_offset_in_radians(robot):
    robot.move_joint("left arm", 10, 500)
    (message,) = robot.topics["/arms_controller/joint_trajectory"].sent
    assert message["joint_names"] == ["l_arm", "r_arm"]
    assert message["points"][0]["positions"] == pytest.approx([math.radians(-70), 0.0])
    assert message["points"][0]["time_from_start"] == {"sec": 0, "nanosec": 500_000_000}
    assert robot.topics["/body_controller/joint_trajectory"].sent == []


def test_later_moves_keep_the_other_joints_where_they_were_sent(robot):
    robot.move_joint("left arm", 10, 100)
    robot.move_joint("r_arm", 30, 100)
    last = robot.topics["/arms_controller/joint_trajectory"].sent[-1]
    assert last["points"][0]["positions"] == pytest.approx([math.radians(-70), math.radians(30)])


def test_only_active_trajectory_controllers_take_joints(robot):
    assert robot.owners == {"l_arm": "arms_controller", "r_arm": "arms_controller", "antenna": "body_controller"}
    assert "/legs_controller/joint_trajectory" not in robot.topics


def test_joints_the_robot_does_not_have_are_ignored(robot):
    robot.move_joint("left knee", 40, 100)
    assert all(not topic.sent for topic in robot.topics.values())


def test_progress_lights_fill_the_eye_ring_and_off_clears_it(robot):
    robot.set_light_marty("#FF8000", 5, 10)
    settle(robot)
    lit = robot.topics["/eye_leds"].sent[-1]["data"]
    assert lit == [0xFF8000] * 7 + [0] * 5
    robot.disco_off()
    assert robot.topics["/eye_leds"].sent[-1]["data"] == [0] * 12


def test_a_pose_with_a_section_for_the_robot_uses_its_joint_values(robot, tmp_path):
    pose = tmp_path / "pose.toml"
    pose.write_text("[marty]\nLeftArm = 50\n\n[fake_urdf]\nr_arm = 20\n")
    assert robot.load_pose(str(pose)) == {"r_arm": 20}


def test_a_pose_without_a_section_for_the_robot_falls_back_to_the_marty_gestures(robot, tmp_path):
    pose = tmp_path / "pose.toml"
    pose.write_text("[marty]\nLeftArm = 50\nEyes = 10\n")
    assert robot.load_pose(str(pose)) == {"left arm": 50, "eyes": 10}


def test_the_coach_mountain_pose_has_whole_degrees_for_the_reachy_humanoid():
    import toml

    values = toml.load("poses/mountain/pose.toml")["reachy_mini_so101_humanoid_urdf"]
    assert values and all(isinstance(v, int) for v in values.values())


def test_the_lucy_robot_maps_load_with_the_coach_toml_reader():
    import toml

    for robot, gestures in toml.load("lucy_robots.toml").items():
        assert all("joint" in row and "scale" in row for rows in gestures.values() for row in rows), robot


ROBOTS = {
    "marty_urdf": {"left arm": [{"joint": "arm_left_joint", "scale": 1.0}]},
    "reachy_mini_so101_humanoid_urdf": {"left arm": [{"joint": "l_arm_shoulder_lift", "scale": 1.0}],
                                        "eyes": [{"joint": "left_antenna", "scale": 1.0}]},
    "unitree_g1_urdf": {"left knee": [{"joint": "left_knee_joint", "scale": 1.0}],
                        "eyes": [{"joint": "waist_yaw_joint", "scale": 1.0}]},
    "unitree_r1_urdf": {"left knee": [{"joint": "left_knee_joint", "scale": 1.0}],
                        "eyes": [{"joint": "head_pitch_joint", "scale": 1.0}]},
}


def test_the_running_robot_wins_over_a_stale_config():
    joints = {"l_arm_shoulder_lift", "left_antenna"}
    assert matching_robot(ROBOTS, joints, preferred="marty_urdf") == "reachy_mini_so101_humanoid_urdf"


def test_robots_sharing_joint_names_are_told_apart_by_the_rest():
    assert matching_robot(ROBOTS, {"left_knee_joint", "waist_yaw_joint"}) == "unitree_g1_urdf"
    assert matching_robot(ROBOTS, {"left_knee_joint", "head_pitch_joint"}) == "unitree_r1_urdf"


def test_an_equal_fit_keeps_the_configured_robot():
    joints = {"left_knee_joint", "waist_yaw_joint", "head_pitch_joint"}
    assert matching_robot(ROBOTS, joints, preferred="unitree_r1_urdf") == "unitree_r1_urdf"


def test_a_robot_with_no_mapped_joint_is_refused():
    with pytest.raises(ValueError):
        matching_robot(ROBOTS, {"some_other_joint"})


def test_the_coach_switches_to_the_robot_lucy_runs():
    class Running(FakeRobot):
        def __init__(self):
            self.topics = {}
            LucyCoachRobot.__init__(self, "ws://localhost:9090", "marty_urdf", {}, ros=FakeRos(),
                                    robots={**ROBOTS, "fake_urdf": {"left arm": [{"joint": "l_arm", "scale": 1.0}]}})

    robot = Running()
    assert robot.robot_package == "fake_urdf"
    assert robot.targets("left arm", 10) == {"l_arm": pytest.approx(math.radians(10))}


def test_the_coach_takes_control_so_the_panel_stops_sending_its_sliders():
    robot = FakeRobot({})
    assert ("/lucy/control", {"client_id": "yoga_coach", "acquire": True}) in robot.service_calls
    assert robot.topics["/lucy/client_heartbeat"].sent[0] == {"data": "yoga_coach"}
