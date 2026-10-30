from copy import deepcopy
from .config import RobotConfig


class SymbolicBackend:
    """Fast discrete backend for data generation; physical results use BulletBackend."""
    def __init__(self, episode, config=None):
        self.config = config or RobotConfig()
        self.trays = deepcopy(self.config.trays)
        self.objects = {o["id"]: deepcopy(o) for o in episode.objects}
        self.holding = None
        self.history = []

    def observe(self):
        return {"objects": deepcopy(list(self.objects.values())), "trays": deepcopy(self.trays),
                "holding": self.holding, "history": deepcopy(self.history)}

    def validate(self, action):
        try:
            type(action).parse(action.to_json())
        except (ValueError, AttributeError, TypeError):
            return "INVALID_ACTION"
        if action.action == "pick":
            if self.holding:
                return "GRIPPER_FULL"
            if action.target not in self.objects:
                return "OBJECT_NOT_FOUND"
        elif action.action == "place":
            if action.target not in self.trays:
                return "UNKNOWN_TRAY"
            if not self.holding:
                return "GRIPPER_EMPTY"
        return None

    def record(self, action, error=None, object_id=None):
        event = {**action.to_dict(), "ok": error is None}
        if error:
            event["error"] = error
        if object_id:
            event["object_id"] = object_id
        self.history.append(event)
        return deepcopy(event)

    def execute(self, action):
        error = self.validate(action)
        if error:
            return self.record(action, error)
        obj_id = self.holding if action.action == "place" else None
        if action.action == "pick":
            self.holding = action.target
            self.objects[action.target]["zone"] = "held"
            self.objects[action.target]["position"][2] = self.config.approach_height
        elif action.action == "place":
            self.objects[self.holding]["zone"] = action.target
            self.objects[self.holding]["position"] = list(self.trays[action.target])
            self.holding = None
        return self.record(action, object_id=obj_id)

    def close(self):
        pass
