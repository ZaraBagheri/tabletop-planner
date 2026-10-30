"""Bounded execution, independent task grading, and paired backend comparisons."""
from collections import Counter
import json
from pathlib import Path
import time
from .backends import make_backend
from .config import RobotConfig
from .domain import Episode, grade

VALIDATION_ERRORS = {"INVALID_ACTION", "GRIPPER_FULL", "GRIPPER_EMPTY", "OBJECT_NOT_FOUND", "UNKNOWN_TRAY"}


def failure_categories(episode, final, terminal, trace, grade_task=True):
    categories = set()
    for event in trace:
        if event.get("error") == "INVALID_MODEL_OUTPUT":
            categories.add("malformed_output")
        result = event.get("result", {})
        if result and not result["ok"]:
            categories.add("invalid_action" if result.get("error") in VALIDATION_ERRORS else "execution_failure")
        if not grade_task or not result.get("ok"):
            continue
        action, obs = event["action"], event["observation"]
        if episode.expected in ("ask", "fail"):
            if action["action"] in ("pick", "place"):
                categories.add("unexpected_movement")
            continue
        remaining = [goal for goal in episode.goals
                     if next((obj["zone"] for obj in obs["objects"] if obj["id"] == goal["object_id"]), None) != goal["zone"]]
        if action["action"] == "pick":
            if not remaining:
                categories.add("unnecessary_movement")
            elif action["target"] != remaining[0]["object_id"]:
                categories.add("wrong_target")
                if any(goal["object_id"] == action["target"] for goal in remaining[1:]):
                    categories.add("wrong_order")
        if action["action"] == "place" and remaining:
            if action["target"] != remaining[0]["zone"]:
                categories.add("wrong_destination")
    if terminal == "budget_exhausted":
        categories.add("action_budget_exhausted")
    if grade_task:
        if episode.expected == "ask" and terminal != "ask":
            categories.add("failed_to_ask")
        elif episode.expected == "fail" and terminal != "fail":
            categories.add("failed_to_report_missing")
        elif episode.expected == "done":
            if terminal in ("ask", "fail"):
                categories.add("unexpected_refusal")
            if terminal == "done" and not grade(episode, final, terminal):
                categories.add("incorrect_completion")
                objects = {obj["id"]: obj for obj in final["objects"]}
                if any(objects.get(goal["object_id"], {}).get("zone") != goal["zone"] for goal in episode.goals):
                    categories.add("premature_completion")
    return sorted(categories)


def rollout(episode, backend, planner, max_actions=8, *, grade_task=True):
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    trace, terminal = [], None
    start = time.monotonic()
    for _ in range(max_actions):
        obs = backend.observe()
        decision_start = time.monotonic()
        try:
            action = planner.next_action(episode.instruction, obs)
        except ValueError as exc:
            trace.append({"observation": obs, "error": "INVALID_MODEL_OUTPUT", "detail": str(exc),
                          "raw_output": getattr(planner, "last_text", ""),
                          "planner_latency_s": round(time.monotonic()-decision_start, 4)})
            terminal = "invalid"
            break
        latency = time.monotonic()-decision_start
        result = backend.execute(action)
        event = {"observation": obs, "action": action.to_dict(), "result": result,
                 "planner_latency_s": round(latency, 4), "after": backend.observe()}
        raw = getattr(planner, "last_text", None)
        if isinstance(raw, str):
            event["raw_output"] = raw
        trace.append(event)
        if action.action in ("done", "ask", "fail"):
            terminal = action.action
            break
    terminal = terminal or "budget_exhausted"
    final = backend.observe()
    return {"episode_id": episode.episode_id, "kind": episode.kind, "instruction": episode.instruction,
            "passed": grade(episode, final, terminal) if grade_task else None, "graded": grade_task,
            "terminal": terminal, "actions": len(trace),
            "malformed_outputs": sum(t.get("error") == "INVALID_MODEL_OUTPUT" for t in trace),
            "invalid_actions": sum(not t["result"]["ok"] and t["result"].get("error") in VALIDATION_ERRORS for t in trace if "result" in t),
            "execution_failures": sum(not t["result"]["ok"] and t["result"].get("error") not in VALIDATION_ERRORS for t in trace if "result" in t),
            "failure_categories": failure_categories(episode, final, terminal, trace, grade_task),
            "elapsed_s": round(time.monotonic()-start, 4), "final": final, "trace": trace}


def _summarize(results, planner, backend):
    failures = Counter(category for result in results for category in result["failure_categories"])
    summary = {"planner": planner.name, "backend": backend, "episodes": len(results),
               "success_rate": sum(r["passed"] for r in results)/len(results),
               "malformed_outputs": sum(r["malformed_outputs"] for r in results),
               "invalid_actions": sum(r["invalid_actions"] for r in results),
               "execution_failures": sum(r["execution_failures"] for r in results),
               "mean_actions": sum(r["actions"] for r in results)/len(results),
               "mean_elapsed_s": sum(r["elapsed_s"] for r in results)/len(results),
               "failure_counts": dict(failures), "by_kind": {}}
    fmt = getattr(planner, "observation_format", None)
    if isinstance(fmt, str):
        summary["observation_format"] = fmt
    for kind in sorted({r["kind"] for r in results}):
        group = [r for r in results if r["kind"] == kind]
        summary["by_kind"][kind] = {"episodes": len(group), "passed": sum(r["passed"] for r in group),
            "success_rate": sum(r["passed"] for r in group)/len(group),
            "failure_counts": dict(Counter(c for r in group for c in r["failure_categories"]))}
    return summary


def evaluate(episodes_path, output, planner, backend_type="symbolic", limit=None, *, config=None, max_actions=8):
    if backend_type not in ("symbolic", "bullet", "both"):
        raise ValueError("Unknown evaluation backend")
    if limit is not None and limit < 1 or max_actions < 1:
        raise ValueError("Episode and action limits must be positive")
    records = [json.loads(line) for line in Path(episodes_path).read_text().splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("Evaluation requires at least one episode")
    # Freeze the records once so paired evaluation uses exactly the same episodes.
    config = config or RobotConfig()
    output = Path(output)
    outputs, outcomes = {}, {}
    for kind in (("symbolic", "bullet") if backend_type == "both" else (backend_type,)):
        directory = output/kind if backend_type == "both" else output
        directory.mkdir(parents=True, exist_ok=True)
        results = []
        with (directory/"rollouts.jsonl").open("w", encoding="utf-8") as file:
            for record in records:
                episode = Episode(**record)
                backend = make_backend(episode, kind, config=config)
                try:
                    result = rollout(episode, backend, planner, max_actions=max_actions)
                finally:
                    backend.close()
                file.write(json.dumps(result) + "\n")
                file.flush()
                results.append(result)
                print(f"{kind} {episode.episode_id}: {'PASS' if result['passed'] else 'FAIL'} {result['failure_categories']}", flush=True)
        summary = _summarize(results, planner, kind)
        summary["max_actions"] = max_actions
        (directory/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (directory/"robot_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        outputs[kind], outcomes[kind] = summary, results
    if backend_type != "both":
        return outputs[backend_type]
    agreement = Counter()
    for symbolic, bullet in zip(outcomes["symbolic"], outcomes["bullet"]):
        label = ("both_passed" if symbolic["passed"] and bullet["passed"] else
                 "symbolic_only" if symbolic["passed"] else "bullet_only" if bullet["passed"] else "both_failed")
        agreement[label] += 1
    summary = {"planner": planner.name, "episodes": len(records), "backends": outputs, "agreement": dict(agreement)}
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
