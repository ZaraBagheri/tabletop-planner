"""Generate frozen expert datasets using the same model inputs as inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile

from .backends import make_backend
from .config import RobotConfig
from .domain import Action, grade
from .observations import DEFAULT_FORMAT, messages_for, planner_config, system_prompt
from .tasks import expert_action, make_episode

SYSTEM_PROMPT = system_prompt()


def generate_dataset(output, train=800, val=100, test=100, seed=42, *, backend="symbolic",
                     config=None, observation_format=DEFAULT_FORMAT, recovery_rate=.2):
    if backend not in ("symbolic", "bullet", "mixed"):
        raise ValueError("Dataset backend must be symbolic, bullet, or mixed")
    if not 0 <= recovery_rate <= 1:
        raise ValueError("recovery_rate must be between zero and one")
    counts = {"train": train, "val": val, "test": test}
    if any(type(count) is not int or not 1 <= count < 1_000_000 for count in counts.values()):
        raise ValueError("Each split needs between 1 and 999999 episodes")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty dataset directory to preserve existing experiments")
    config = config or RobotConfig()
    inputs = planner_config(observation_format)
    manifest = {"seed": seed, "source": "synthetic expert", "backend": backend,
                "observation_format": observation_format, "prompt_sha256": inputs["prompt_sha256"],
                "recovery_rate": recovery_rate, "recovery_source": "simulated rejected pick/place actions in training only",
                "held_out_combination": {"color": "yellow", "zone": "right_tray"}, "splits": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tabletop-data-", dir=output.parent) as temporary:
        staging = Path(temporary)/"dataset"
        staging.mkdir()
        for split, count in counts.items():
            examples, recoveries, controller_failures = 0, 0, 0
            sources, actions = Counter(), Counter()
            with (staging/f"{split}.jsonl").open("w", encoding="utf-8") as data, (staging/f"{split}_episodes.jsonl").open("w", encoding="utf-8") as episodes:
                for i in range(count):
                    episode = make_episode(i, split, seed, config=config)
                    source = ("bullet" if i % 2 else "symbolic") if backend == "mixed" else backend
                    env = make_backend(episode, source, config=config)
                    rng = random.Random(episode.seed + 10_000_000)
                    rows, terminal = [], None
                    try:
                        for _ in range(12):
                            action = expert_action(episode, env.observe())
                            if split == "train" and action.action in ("pick", "place") and rng.random() < recovery_rate:
                                # Explicitly simulated execution rejection; never used by demo/evaluate.
                                env.record(action, "SIMULATED_EXECUTION_FAILURE")
                                recoveries += 1
                            messages = messages_for(episode.instruction, env.observe(), observation_format)
                            messages.append({"role": "assistant", "content": action.to_json()})
                            rows.append({"episode_id": episode.episode_id, "observation_format": observation_format,
                                         "backend": source, "messages": messages})
                            result = env.execute(action)
                            controller_failures += not result["ok"]
                            if action.action in ("done", "ask", "fail"):
                                terminal = action.action
                                break
                        if not grade(episode, env.observe(), terminal):
                            raise RuntimeError(f"Expert episode {episode.episode_id} did not satisfy its goals on {source}; dataset was not published")
                    finally:
                        env.close()
                    episodes.write(json.dumps(episode.to_dict()) + "\n")
                    for row in rows:
                        data.write(json.dumps(row) + "\n")
                        actions[json.loads(row["messages"][-1]["content"])["action"]] += 1
                    examples += len(rows)
                    sources[source] += 1
                    if (i+1) % 25 == 0 or i+1 == count:
                        print(f"generate split={split} episodes={i+1}/{count} examples={examples}", flush=True)
            path = staging/f"{split}.jsonl"
            manifest["splits"][split] = {"episodes": count, "examples": examples,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "backends": dict(sources),
                "action_counts": dict(actions), "recovery_observations": recoveries,
                "controller_failures": controller_failures}
        (staging/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staging/"planner_config.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        (staging/"robot_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        if output.exists():
            # rmdir refuses nonempty directories, including a concurrent publisher's output.
            output.rmdir()
        staging.rename(output)
    return manifest
