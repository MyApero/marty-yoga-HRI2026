"""Announce an app to Lucy as a plugin: a button in the panel's command rail (docs/plugins.md in lucy_control_panel)."""

import json
import threading
import time

import roslibpy

PLUGINS_TOPIC = "/lucy/plugins"
ACTION_TOPIC = "/lucy/plugins/action"
MESSAGE_TYPE = "std_msgs/msg/String"
ANNOUNCE_PERIOD_SEC = 1.0  # the panel drops a plugin it has not heard from for 3.5 s


def parse_action(data: str, plugin_id: str) -> tuple[str, str | None] | None:
    """(action, option) when the message is for this plugin; anyone on the bus can publish, so check it all."""
    try:
        request = json.loads(data)
    except (TypeError, ValueError):
        return None
    if not isinstance(request, dict) or request.get("plugin") != plugin_id:
        return None
    action, option = request.get("action"), request.get("option")
    if not isinstance(action, str):
        return None
    return action, option if isinstance(option, str) else None


class LucyPlugin:
    def __init__(self, ros, plugin_id: str, manifest, on_action):
        """manifest() returns the current button description; on_action(action, option) runs on the ROS thread."""
        self.ros = ros
        self.plugin_id = plugin_id
        self.manifest = manifest
        self.on_action = on_action
        self.announcer = roslibpy.Topic(ros, PLUGINS_TOPIC, MESSAGE_TYPE)
        roslibpy.Topic(ros, ACTION_TOPIC, MESSAGE_TYPE).subscribe(self._on_message)
        threading.Thread(target=self._announce_forever, daemon=True).start()

    def announce(self) -> None:
        """Publish the manifest now, e.g. right after a state change, instead of waiting for the next beat."""
        if self.ros.is_connected:
            manifest = {"id": self.plugin_id, **self.manifest()}
            self.announcer.publish(roslibpy.Message({"data": json.dumps(manifest)}))

    def _announce_forever(self) -> None:
        while True:
            try:
                self.announce()
            except Exception as error:
                print(f"Lucy plugin announce failed: {error}")
            time.sleep(ANNOUNCE_PERIOD_SEC)

    def _on_message(self, message: dict) -> None:
        request = parse_action(message.get("data"), self.plugin_id)
        if request:
            self.on_action(*request)
