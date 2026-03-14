from dataclasses import dataclass, field


@dataclass
class SessionState:
    pose_name: str | None = None
    actual_run: list = field(default_factory=list)
    history: list = field(default_factory=list)
    name_files: str | None = None
    pose_ended: bool = True
    is_pose_ending: bool = False
