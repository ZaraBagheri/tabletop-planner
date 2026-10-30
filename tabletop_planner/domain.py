from dataclasses import dataclass
import json
from typing import Protocol

COLORS = ("red", "blue", "green", "yellow")
TRAYS = {"left_tray": [0.50, 0.25, 0.035], "right_tray": [0.50, -0.25, 0.035]}


@dataclass(frozen=True)
class Action:
    action: str
    target: str = ""

    def to_dict(self):
        return {"action": self.action, "target": self.target}

    def to_json(self):
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def parse(cls, text):
        """Reject prose, unknown fields/types and invalid schemas; never execute text."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Expected one JSON object") from exc
        if (not isinstance(data, dict) or set(data) != {"action", "target"}
                or not all(type(v) is str for v in data.values())):
            raise ValueError("Expected string action and target fields only")
        name, target = data["action"], data["target"]
        if name not in ("pick", "place", "done", "ask", "fail") or len(target) > 200:
            raise ValueError("Unknown action or excessively long target")
        if name == "done" and target != "":
            raise ValueError("done requires an empty target")
        if name != "done" and not target.strip():
            raise ValueError("This action requires a target")
        return cls(name, target)


@dataclass
class Episode:
    episode_id: str
    seed: int
    split: str
    kind: str
    instruction: str
    objects: list
    goals: list
    expected: str = "done"

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


class RobotBackend(Protocol):
    """A future hardware backend implements this contract without changing the planner."""
    def observe(self) -> dict: ...
    def execute(self, action: Action) -> dict: ...
    def close(self) -> None: ...


def apply_goal_spec(episode, path):
    """Attach independently supplied goal labels to a custom instruction; never choose actions."""
    from pathlib import Path
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec)-{"expected", "goals"}:
        raise ValueError("Goal file accepts only expected and goals fields")
    expected, goals = spec.get("expected", "done"), spec.get("goals", [])
    if expected not in ("done", "ask", "fail") or not isinstance(goals, list):
        raise ValueError("Invalid expected terminal action or goals list")
    if expected != "done" and goals:
        raise ValueError("ask/fail goal specifications must not include transfers")
    resolved = []
    for goal in goals:
        if not isinstance(goal, dict) or set(goal) not in ({"object_id", "zone"}, {"color", "zone"}):
            raise ValueError("Each goal needs object_id or color, and zone")
        if not all(isinstance(value, str) for value in goal.values()):
            raise ValueError("Goal identifiers and zones must be strings")
        if goal["zone"] not in TRAYS:
            raise ValueError("Unknown goal tray")
        key = "object_id" if "object_id" in goal else "color"
        matches = [obj for obj in episode.objects if obj["id" if key == "object_id" else "color"] == goal[key]]
        if len(matches) != 1:
            raise ValueError("Each goal must identify exactly one observed object")
        resolved.append({"object_id": matches[0]["id"], "zone": goal["zone"]})
    if len({goal["object_id"] for goal in resolved}) != len(resolved):
        raise ValueError("Repeated object goals are not supported")
    episode.goals, episode.expected = resolved, expected


def grade(episode, observation, terminal):
    """Hidden ground-truth task specification is used ONLY here and by the data oracle."""
    moves = [e for e in observation["history"] if e["ok"] and e["action"] in ("pick", "place")]
    if episode.expected in ("ask", "fail"):
        return terminal == episode.expected and not moves
    objects = {o["id"]: o for o in observation["objects"]}
    if terminal != "done" or observation["holding"] is not None:
        return False
    if not all(objects[g["object_id"]]["zone"] == g["zone"] for g in episode.goals):
        return False
    # Enforce requested order for goals that were not already satisfied at reset.
    initial = {o["id"]: o for o in episode.objects}
    required = [(g["object_id"], g["zone"]) for g in episode.goals
                if initial[g["object_id"]]["zone"] != g["zone"]]
    placements = [(e["object_id"], e["target"]) for e in moves if e["action"] == "place"]
    return placements == required
