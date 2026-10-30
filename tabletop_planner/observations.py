"""Versioned model inputs shared by data generation and inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

LEGACY_FORMAT = "raw-v1"
DEFAULT_FORMAT = "semantic-v2"
FORMATS = (LEGACY_FORMAT, DEFAULT_FORMAT)


def system_prompt(observation_format=DEFAULT_FORMAT):
    if observation_format not in FORMATS:
        raise ValueError(f"Unknown observation format: {observation_format}")
    filename = "system_prompt_legacy.txt" if observation_format == LEGACY_FORMAT else "system_prompt.txt"
    return Path(__file__).with_name(filename).read_text(encoding="utf-8")


def planner_config(observation_format=DEFAULT_FORMAT, prompt=None):
    if observation_format not in FORMATS:
        raise ValueError(f"Unknown observation format: {observation_format}")
    prompt = system_prompt(observation_format) if prompt is None else prompt
    return {"observation_format": observation_format, "system_prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def load_planner_config(adapter=None, observation_format="auto"):
    metadata = Path(adapter)/"planner_config.json" if adapter else None
    if metadata and metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        config = planner_config(saved["observation_format"], saved["system_prompt"])
        if config != saved:
            raise ValueError("Adapter prompt metadata is inconsistent")
    else:
        config = planner_config(LEGACY_FORMAT if adapter else DEFAULT_FORMAT)
    return config if observation_format == "auto" else planner_config(observation_format)


def model_observation(observation, observation_format=DEFAULT_FORMAT):
    if observation_format == LEGACY_FORMAT:
        return deepcopy(observation)
    if observation_format != DEFAULT_FORMAT:
        raise ValueError(f"Unknown observation format: {observation_format}")
    # Whitelist semantic fields; simulator poses and hidden goal labels never enter the prompt.
    objects = [{key: obj[key] for key in ("id", "color", "zone")}
               for obj in observation["objects"]]
    history = [{key: event[key] for key in ("action", "target", "ok", "error", "object_id") if key in event}
               for event in observation.get("history", [])]
    return {"objects": sorted(objects, key=lambda obj: obj["id"]),
            "trays": sorted(observation["trays"]), "holding": observation["holding"],
            "history": history}


def messages_for(instruction, observation, observation_format=DEFAULT_FORMAT, prompt=None):
    prompt = system_prompt(observation_format) if prompt is None else prompt
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({"instruction": instruction,
             "observation": model_observation(observation, observation_format)}, separators=(",", ":"))}]


def config_from_rows(rows):
    """Identify old JSONL files and reject mixed schemas/prompts before training."""
    configs = []
    for row in rows:
        payload = json.loads(row["messages"][1]["content"])
        obs = payload["observation"]
        inferred = DEFAULT_FORMAT if isinstance(obs.get("trays"), list) else LEGACY_FORMAT
        fmt = row.get("observation_format", inferred)
        if fmt != inferred:
            raise ValueError("Training row observation format disagrees with its payload")
        model_observation(obs, fmt)
        configs.append(planner_config(fmt, row["messages"][0]["content"]))
    if not configs or any(config != configs[0] for config in configs[1:]):
        raise ValueError("Training and validation must share one observation format and system prompt")
    return configs[0]
