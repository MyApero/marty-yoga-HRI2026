import base64
import math
import threading
import time
from urllib.parse import urlparse

import cv2
import roslibpy

from .voice_leds import FPS as VOICE_FPS, band_levels, eye_frame

CLIENT_ID = "yoga_coach"
HEARTBEAT_PERIOD_SEC = 1.0
LUCY_WAIT_SEC = 120.0
CONTROLLER_REFRESH_SEC = 2.0
EYE_LED_COUNT = 12
TRAJECTORY_CONTROLLER = "joint_trajectory_controller/JointTrajectoryController"
STREAM_TOPIC = "/yoga_coach/image/compressed"
STREAM_PERIOD_SEC = 1 / 30  # rosbridge carries ~30 fps of 960 px JPEGs on localhost, ~24 above that
STREAM_WIDTH = 960


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
        self._connect()
        self.joint_map = joint_map
        self.positions: dict[str, float] = {}
        self.ros_topic("/joint_states", "sensor_msgs/msg/JointState").subscribe(self._on_joint_states)
        self.owners = controller_joints(self.wait_for_controllers())
        self.commands: dict[str, float] = {}
        self.trajectories = {}
        self._bind_trajectories()
        self.refreshed_at = time.monotonic()
        self.eye_leds = self.ros_topic("/eye_leds", "std_msgs/msg/UInt32MultiArray")
        self.motion_interrupted = threading.Event()
        self.eye_state = eye_colours([])
        self.voice_take = 0
        self.voice_until = 0.0
        self.heartbeat = self.ros_topic("/lucy/client_heartbeat", "std_msgs/msg/String")
        self.stream = self.ros_topic(STREAM_TOPIC, "sensor_msgs/msg/CompressedImage")
        self.stream_frame = None
        self.stream_ready = threading.Condition()
        threading.Thread(target=self._stream_forever, daemon=True).start()
        self.active_client = None
        self.ros_topic("/lucy/active_client", "std_msgs/msg/String").subscribe(self._on_active_client)
        threading.Thread(target=self._beat, daemon=True).start()
        self.take_control()

    def _connect(self) -> None:
        try:
            if not self.ros.is_connected:
                self.ros.run(timeout=5)
        except Exception:
            pass
        self._wait(lambda: self.ros.is_connected, "Lucy's rosbridge")

    def wait_for_controllers(self) -> list[dict]:
        found = []

        def ready():
            try:
                found[:] = self.list_controllers()
            except Exception:
                return False
            return bool(controller_joints(found))

        self._wait(ready, "Lucy's controllers")
        return found

    def _wait(self, ready, what: str) -> None:
        deadline = time.monotonic() + LUCY_WAIT_SEC
        announced = False
        while not ready():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{what} not ready after {LUCY_WAIT_SEC:.0f} s")
            if not announced:
                print(f"Waiting for {what}...")
                announced = True
            time.sleep(1)

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

    def _on_active_client(self, message: dict) -> None:
        self.active_client = message["data"]

    def has_control(self) -> bool:
        return self.active_client == CLIENT_ID  # Lucy accepts anyone's commands; only send while in control

    def _beat(self) -> None:
        while True:
            self._beat_once()
            time.sleep(HEARTBEAT_PERIOD_SEC)

    def _beat_once(self) -> None:
        if not self.ros.is_connected:
            return
        self.heartbeat.publish(roslibpy.Message({"data": CLIENT_ID}))
        if self.active_client == "":
            self.take_control()

    def publish_frame(self, frame) -> None:
        # Only hands the frame over: encoding here cost the coach loop ~10 ms and halved it to 15 fps.
        with self.stream_ready:
            self.stream_frame = frame
            self.stream_ready.notify()

    def _stream_forever(self) -> None:
        next_at = 0.0
        while True:
            time.sleep(max(0.0, next_at - time.monotonic()))
            with self.stream_ready:
                self.stream_ready.wait_for(lambda: self.stream_frame is not None)
                frame, self.stream_frame = self.stream_frame, None
            # Schedule from the last slot, not from now, so frame jitter does not erode the rate.
            next_at = max(next_at + STREAM_PERIOD_SEC, time.monotonic() - STREAM_PERIOD_SEC)
            if self.ros.is_connected:
                try:
                    self._publish_jpeg(frame)
                except Exception as error:
                    print(f"Lucy stream frame dropped: {error}")

    def _publish_jpeg(self, frame) -> None:
        scale = STREAM_WIDTH / frame.shape[1]
        ok, jpeg = cv2.imencode(".jpg", cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        if not ok:
            return
        wall = time.time()
        header = {"stamp": {"sec": int(wall), "nanosec": int((wall % 1) * 1e9)}, "frame_id": CLIENT_ID}
        self.stream.publish(roslibpy.Message({"header": header, "format": "jpeg", "data": base64.b64encode(jpeg).decode()}))

    def _on_joint_states(self, message: dict) -> None:
        self.positions.update(zip(message["name"], message["position"]))

    def targets(self, name: str, angle: float) -> dict[str, float]:
        mapped = self.joint_map.get(name, [{"joint": name, "scale": 1.0}])
        return {row["joint"]: math.radians(angle * row["scale"] + row.get("offset", 0.0)) for row in mapped}

    def move_joint(self, name: str, angle: float, duration_ms: float, blocking: bool = False) -> None:
        self.move_joints(self.targets(name, angle), duration_ms)
        if blocking:
            self.motion_interrupted.clear()
            self.motion_interrupted.wait(duration_ms / 1000)

    def interrupt_motion(self) -> None:
        self.motion_interrupted.set()

    def _bind_trajectories(self) -> None:
        for name in set(self.owners.values()) - set(self.trajectories):
            self.trajectories[name] = self.ros_topic(f"/{name}/joint_trajectory", "trajectory_msgs/msg/JointTrajectory")

    def _refresh_controllers(self) -> None:
        if time.monotonic() - self.refreshed_at < CONTROLLER_REFRESH_SEC:
            return
        self.refreshed_at = time.monotonic()
        try:
            self.owners = controller_joints(self.list_controllers())
        except Exception:
            return
        self._bind_trajectories()

    def move_joints(self, targets: dict[str, float], duration_ms: float) -> None:
        if not self.has_control():
            return
        if any(joint not in self.owners for joint in targets):
            self._refresh_controllers()
        moved = {joint: value for joint, value in targets.items() if joint in self.owners}
        self.commands.update(moved)
        for controller in {self.owners[joint] for joint in moved}:
            joints = [joint for joint, owner in self.owners.items() if owner == controller]
            positions = [self.commands.get(joint, self.positions.get(joint, 0.0)) for joint in joints]
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
        self.eye_state = eye_colours(colours)
        if time.monotonic() < self.voice_until or not self.has_control():
            return  # the voice owns the eyes, or someone else has control; this state comes back later
        self.eye_leds.publish(roslibpy.Message({"data": self.eye_state}))

    def show_voice(self, audio, sample_rate: int, delay_s: float = 0.0) -> None:
        frames = [eye_frame(row) for row in band_levels(audio, sample_rate)]
        start = time.monotonic() + delay_s
        self.voice_take += 1
        self.voice_until = start + len(frames) / VOICE_FPS
        threading.Thread(target=self._play_voice, args=(self.voice_take, frames, start), daemon=True).start()

    def _play_voice(self, take: int, frames: list[list[int]], start: float) -> None:
        for i, colours in enumerate(frames):
            time.sleep(max(0.0, start + i / VOICE_FPS - time.monotonic()))
            if take != self.voice_take:
                return  # the next chunk took over
            if self.has_control():
                self.eye_leds.publish(roslibpy.Message({"data": colours}))
        time.sleep(max(0.0, self.voice_until - time.monotonic()))
        if take == self.voice_take and self.has_control():
            self.eye_leds.publish(roslibpy.Message({"data": self.eye_state}))

    def disco_off(self, add_on: str = "") -> None:
        self.disco_color_eyepicker([])

    def get_joints(self) -> dict:
        return {i: {"name": joint, "pos": math.degrees(value)} for i, (joint, value) in enumerate(self.positions.items())}

    def is_conn_ready(self) -> bool:
        return self.ros.is_connected
