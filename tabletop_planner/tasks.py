import random
from .domain import Action, COLORS, Episode, TRAYS
from .config import RobotConfig

# Train/validation share phrasing families but never episode seeds. Test families
# are absent from both. The yellow -> right tray task is held out from both, too.
PHRASES = {
    "train": ["Put the {color} block in the {side} tray.",
              "Move the {color} cube to the {side} tray.",
              "Please place the {color} block into the {side} tray.",
              "Transfer the {color} cube to the {side} tray.",
              "Pick up the {color} block and put it in the {side} tray.",
              "Place the {color} cube in the tray on the {side}."],
    "test": ["The {color} block belongs in the {side} tray; take it there.",
             "Could you deliver the {color} cube to the tray on the {side}?"],
}


def make_episode(index, split="train", seed=42, kind=None, config=None):
    config = config or RobotConfig()
    trays = config.trays
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]
    episode_seed = seed + offset + index
    rng = random.Random(episode_seed)
    kind = kind or ("single", "double", "missing", "ambiguous", "already_done")[index % 5]
    colors = list(COLORS)
    rng.shuffle(colors)
    # Two rows in the reachable central work area, with non-overlapping jitter.
    positions = config.scene_positions
    objects = [{"id": f"block_{i}", "color": color, "zone": "table",
                "position": [round(x+rng.uniform(-config.scene_jitter, config.scene_jitter), 3),
                             round(y+rng.uniform(-config.scene_jitter, config.scene_jitter), 3), config.block_half_size+.005]}
               for i, (color, (x, y)) in enumerate(zip(colors, positions))]
    eligible = [(o, zone) for o in objects for zone in trays
                if split == "test" or (o["color"], zone) != ("yellow", "right_tray")]
    obj, zone = rng.choice(eligible)
    if split == "test" and index % 2 == 0:
        obj, zone = next(o for o in objects if o["color"] == "yellow"), "right_tray"
    phrase_set = PHRASES["test" if split == "test" else "train"]

    def phrase(o, z):
        return rng.choice(phrase_set).format(color=o["color"], side=z.split("_")[0])

    instruction = phrase(obj, zone)
    goals = [{"object_id": obj["id"], "zone": zone}]
    expected = "done"
    if kind == "double":
        # Different objects and trays avoid stacking; stacking is out of scope.
        other, other_zone = rng.choice([(o, z) for o, z in eligible
                                      if o["id"] != obj["id"] and z != zone])
        instruction += " Then " + phrase(other, other_zone)
        goals.append({"object_id": other["id"], "zone": other_zone})
    elif kind == "missing":
        objects = [o for o in objects if o["id"] != obj["id"]]
        goals, expected = [], "fail"
    elif kind == "ambiguous":
        instruction = ("Put the block in the left tray." if split != "test"
                       else "Please take a block over to the tray on the left.")
        if index % 2:
            objects[0]["color"] = objects[1]["color"] = "red"
            instruction = ("Put the red block in the left tray." if split != "test"
                           else "Could you deliver the red cube to the tray on the left?")
        goals, expected = [], "ask"
    elif kind == "already_done":
        obj["zone"] = zone
        obj["position"] = list(trays[zone])
    elif kind != "single":
        raise ValueError("Unknown task kind")
    return Episode(f"{split}-{episode_seed}", episode_seed, split, kind,
                   instruction, objects, goals, expected)


def expert_action(episode, observation):
    """Oracle with task-spec access, only for synthetic data and controller checks."""
    if episode.expected == "ask":
        return Action("ask", "Which color block do you mean?")
    if episode.expected == "fail":
        return Action("fail", "The requested block is missing.")
    objects = {o["id"]: o for o in observation["objects"]}
    for goal in episode.goals:
        if objects[goal["object_id"]]["zone"] == goal["zone"]:
            continue
        if observation["holding"] == goal["object_id"]:
            return Action("place", goal["zone"])
        if observation["holding"]:
            return Action("fail", "Unexpected object in gripper.")
        return Action("pick", goal["object_id"])
    return Action("done")
