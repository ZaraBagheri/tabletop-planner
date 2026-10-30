import argparse
import json
from pathlib import Path

from .config import RobotConfig
from .observations import DEFAULT_FORMAT, FORMATS


def planner_from(args):
    from .planners import ModelPlanner
    kwargs = {} if args.observation_format == "auto" else {"observation_format": args.observation_format}
    return ModelPlanner(args.model, args.adapter, args.device, args.revision, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Tabletop Planner: train a language planner, test a robot in simulation")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("generate", help="Generate frozen expert demonstrations")
    data.add_argument("--output", default="data")
    data.add_argument("--train", type=int, default=800)
    data.add_argument("--val", type=int, default=100)
    data.add_argument("--test", type=int, default=100)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--backend", choices=("symbolic", "bullet", "mixed"), default="symbolic")
    data.add_argument("--robot-config", help="Robot/workspace profile JSON")
    data.add_argument("--observation-format", choices=FORMATS, default=DEFAULT_FORMAT)
    data.add_argument("--recovery-rate", type=float, default=.2, help="Fraction of training pick/place examples following simulated rejected actions")
    demo = commands.add_parser("demo", help="Execute one episode with the language model")
    demo.add_argument("--backend", choices=("symbolic", "bullet"), default="bullet")
    demo.add_argument("--gui", action="store_true")
    demo.add_argument("--record", action="store_true")
    demo.add_argument("--kind", choices=("single", "double", "missing", "ambiguous", "already_done"), default="double")
    demo.add_argument("--index", type=int, default=0)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--instruction", help="Run your own instruction in the generated scene")
    demo.add_argument("--goals", help="JSON file containing independent expected goals for a custom instruction")
    evaluate = commands.add_parser("evaluate", help="Evaluate on held-out episodes")
    evaluate.add_argument("--episodes", default="data/test_episodes.jsonl")
    evaluate.add_argument("--backend", choices=("symbolic", "bullet", "both"), default="symbolic")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--output", default="runs/evaluation")
    for command in (demo, evaluate):
        command.add_argument("--planner", choices=("model",), default="model", help=argparse.SUPPRESS)
        command.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
        command.add_argument("--adapter")
        command.add_argument("--device", default="auto")
        command.add_argument("--revision", default="main")
        command.add_argument("--observation-format", choices=("auto",)+FORMATS, default="auto")
        command.add_argument("--robot-config", help="Robot/workspace profile JSON")
        command.add_argument("--max-actions", type=int, default=8)
    training = commands.add_parser("train", help="Train LoRA adapters with completion-only loss")
    training.add_argument("--data", default="data")
    training.add_argument("--output", default="checkpoints/planner")
    training.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    training.add_argument("--revision", default="main")
    training.add_argument("--device", default="auto")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--accumulation", type=int, default=8)
    training.add_argument("--lr", type=float, default=2e-4)
    training.add_argument("--rank", type=int, default=8)
    training.add_argument("--max-length", type=int, default=1536)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--resume", help="Checkpoint directory containing training_state.pt")
    training.add_argument("--checkpoint-every", type=int, default=25)
    training.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    if args.command == "train":
        from .training import train
        report = train(args.data, args.output, args.model, args.revision, args.epochs,
                       args.batch_size, args.accumulation, args.lr, args.rank, args.max_length,
                       args.device, args.max_steps, args.seed, resume=args.resume,
                       checkpoint_every=args.checkpoint_every, log_every=args.log_every)
        print(json.dumps(report, indent=2))
        return
    profile = args.robot_config
    if args.command == "evaluate" and profile is None:
        candidate = Path(args.episodes).parent/"robot_config.json"
        profile = candidate if candidate.exists() else None
    try:
        config = RobotConfig.load(profile)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.command == "generate":
        from .data import generate_dataset
        print(json.dumps(generate_dataset(args.output, args.train, args.val, args.test, args.seed,
            backend=args.backend, config=config, observation_format=args.observation_format,
            recovery_rate=args.recovery_rate), indent=2))
        return
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.command == "evaluate":
        from .evaluation import evaluate as run_evaluation
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        print(json.dumps(run_evaluation(args.episodes, args.output, planner_from(args), args.backend,
                                       args.limit, config=config, max_actions=args.max_actions), indent=2))
        return
    from .tasks import make_episode
    from .domain import apply_goal_spec
    from .evaluation import rollout
    from .backends import make_backend
    if args.backend != "bullet" and (args.gui or args.record):
        parser.error("--gui and --record require --backend bullet")
    custom_instruction = args.instruction is not None
    if custom_instruction and not args.instruction.strip():
        parser.error("--instruction must not be empty")
    if args.goals and not custom_instruction:
        parser.error("--goals requires --instruction")
    episode = make_episode(args.index, "test", args.seed, args.kind, config=config)
    if custom_instruction:
        episode.instruction = args.instruction.strip()
        episode.kind = "custom"
    if args.goals:
        try:
            apply_goal_spec(episode, args.goals)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    graded = not custom_instruction or bool(args.goals)
    planner = planner_from(args)
    backend = make_backend(episode, args.backend, config=config, gui=args.gui, record=args.record)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        print(episode.instruction, flush=True)
        print("Scene: " + ", ".join(f"{obj['id']}={obj['color']} ({obj['zone']})" for obj in backend.observe()["objects"]), flush=True)
        result = rollout(episode, backend, planner, max_actions=args.max_actions, grade_task=graded)
        result["planner"], result["backend"] = planner.name, args.backend
        (output/"rollout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output/"robot_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        for row in result["trace"]:
            print(row.get("action", row.get("error")), row.get("result", ""))
        if args.record:
            backend.save_media(output)
        status = ("PASS" if result["passed"] else "FAIL") if graded else "FINISHED (ungraded custom instruction)"
        print(status, f"Output: {output.resolve()}")
        if result["failure_categories"]:
            print("Diagnostics:", ", ".join(result["failure_categories"]))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
