import math
import threading
import time
from urllib.parse import urlparse

import roslibpy

CLIENT_ID = "yoga_coach"
HEARTBEAT_PERIOD_SEC = 1.0
EYE_LED_COUNT = 12
TRAJECTORY_CONTROLLER = "joint_trajectory_controller/JointTrajectoryController"


def controller_joints(controllers: list[dict]) -> dict[str, str]:
    owners = {}
    for controller in controllers:
        if controller.get("type") != TRAJECTORY_CONTROLLER or controller.get("state") != "active":
            continue
        for interface in controller.get("claimed_interfaces", []):
            joint, _, kind = interface.rpartition("/")
            if kind == "position":
                owners[joint] = controller["name"]
    return owners


def eye_colours(colours: list[str]) -> list[int]:
    values = [int(colour.lstrip("#"), 16) for colour in colours[:EYE_LED_COUNT]]
    return values + [0] * (EYE_LED_COUNT - len(values))


def trajectory(joints: list[str], positions: list[float], duration_ms: float) -> dict:
    seconds = max(duration_ms, 1) / 1000
    stamp = {"sec": int(seconds), "nanosec": int((seconds % 1) * 1e9)}
    return {"joint_names": joints, "points": [{"positions": positions, "time_from_start": stamp}]}


class LucyBody:
    """Drives any Lucy robot over rosbridge with the martypy calls the coach makes."""

    def __init__(self, url: str, joint_map: dict, ros=None):
        address = urlparse(url)
        self.ros = ros or roslibpy.Ros(host=address.hostname, port=address.port or 9090)
        if not self.ros.is_connected:
            self.ros.run(timeout=5)
        self.joint_map = joint_map
        self.positions: dict[str, float] = {}
        self.ros_topic("/joint_states", "sensor_msgs/msg/JointState").subscribe(self._on_joint_states)
        self.owners = controller_joints(self.list_controllers())
        self.commands = {joint: 0.0 for joint in self.owners}
        self.trajectories = {
            name: self.ros_topic(f"/{name}/joint_trajectory", "trajectory_msgs/msg/JointTrajectory")
            for name in set(self.owners.values())
        }
        self.eye_leds = self.ros_topic("/eye_leds", "std_msgs/msg/UInt32MultiArray")
        self.heartbeat = self.ros_topic("/lucy/client_heartbeat", "std_msgs/msg/String")
        threading.Thread(target=self._beat, daemon=True).start()
        self.take_control()

    def ros_topic(self, name: str, message_type: str):
        return roslibpy.Topic(self.ros, name, message_type)

    def call_service(self, name: str, service_type: str, request: dict) -> dict:
        return roslibpy.Service(self.ros, name, service_type).call(roslibpy.ServiceRequest(request), timeout=5)

    def list_controllers(self) -> list[dict]:
        return self.call_service(
            "/controller_manager/list_controllers", "controller_manager_msgs/srv/ListControllers", {}
        )["controller"]

    def take_control(self) -> None:
        try:
            response = self.call_service(
                "/lucy/control", "lucy_msgs/srv/ClientControl", {"client_id": CLIENT_ID, "acquire": True}
            )
        except Exception as error:
            print(f"Lucy control not taken, the panel may override the coach: {error}")
            return
        if not response.get("success"):
            print(f"Lucy kept control elsewhere: {response.get('message')}")

    def _beat(self) -> None:
        while self.ros.is_connected:
            self.heartbeat.publish(roslibpy.Message({"data": CLIENT_ID}))
            time.sleep(HEARTBEAT_PERIOD_SEC)

    def _on_joint_states(self, message: dict) -> None:
        self.positions.update(zip(message["name"], message["position"]))

    def targets(self, name: str, angle: float) -> dict[str, float]:
        mapped = self.joint_map.get(name, [{"joint": name, "scale": 1.0}])
        return {row["joint"]: math.radians(angle * row["scale"] + row.get("offset", 0.0)) for row in mapped}

    def move_joint(self, name: str, angle: float, duration_ms: float, blocking: bool = False) -> None:
        self.move_joints(self.targets(name, angle), duration_ms)
        if blocking:
            time.sleep(duration_ms / 1000)

    def move_joints(self, targets: dict[str, float], duration_ms: float) -> None:
        moved = {joint: value for joint, value in targets.items() if joint in self.owners}
        self.commands.update(moved)
        for controller in {self.owners[joint] for joint in moved}:
            joints = [joint for joint, owner in self.owners.items() if owner == controller]
            positions = [self.commands[joint] for joint in joints]
            self.trajectories[controller].publish(roslibpy.Message(trajectory(joints, positions, duration_ms)))

    def stand_straight(self, duration_ms: float = 1000) -> None:
        rest = {}
        for name in self.joint_map:
            rest.update(self.targets(name, 0))
        self.move_joints(rest, duration_ms)
        time.sleep(duration_ms / 1000)

    def walk(self, *args, **kwargs) -> None:
        pass

    def disco_color_eyepicker(self, colours: list[str], add_on: str = "") -> None:
        self.eye_leds.publish(roslibpy.Message({"data": eye_colours(colours)}))

    def disco_off(self, add_on: str = "") -> None:
        self.disco_color_eyepicker([])

    def get_joints(self) -> dict:
        return {i: {"name": joint, "pos": math.degrees(value)} for i, (joint, value) in enumerate(self.positions.items())}

    def is_conn_ready(self) -> bool:
        return self.ros.is_connected
