# Improved training and evaluation workflow

## Model inputs and compatibility

New datasets use `semantic-v2`: object ID, color, zone, tray names, gripper state,
and action history. Coordinates remain available to the controller and evaluator,
but are omitted from the language-model prompt. Consequently, coordinate-only
changes between symbolic and physics observations cannot change the model's input.
This does not guarantee correct decisions; the new representation needs training
and held-out task evaluation.

Each new adapter includes `planner_config.json`, which records its observation
format, exact system prompt, and prompt hash. Inference reloads that configuration.
Older adapters without this metadata automatically retain `raw-v1` and the original
prompt. Existing datasets and adapter files are not rewritten. Use
`--observation-format raw-v1` to reproduce an original base-model experiment;
an explicit format override is a different evaluation condition.

## Installation options

The main README installs both simulation and training dependencies with
`python -m pip install -e ".[sim,train]"`. Use `".[sim]"` for simulation dependencies
only or `".[train]"` for training/inference dependencies only. Running a language
model requires the training/inference dependencies even with a symbolic backend.

For platform-specific PyTorch builds, follow its [official installation instructions](https://pytorch.org/get-started/locally/).
The Linux CPU training check uses this explicit CPU wheel before installing the extras:

```shell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[sim,train]"
```

The project pins Transformers 4.51.3, PEFT 0.15.2, and Accelerate 1.6.0.
The installed PyTorch version is recorded in each training report. Training uses
LoRA on attention query/value projections, completion-only labels, gradient
accumulation, clipping, and gradient checkpointing. Prompt/padding tokens are
excluded from loss, and oversized examples are rejected instead of truncated.

## Generate a new, frozen dataset

With the virtual environment active, these commands work in macOS/Linux terminals
and Windows PowerShell. Install `".[sim,train]"` first.

```shell
python -m tabletop_planner generate --train 800 --val 100 --test 100 --seed 2026 --backend mixed --output data/v2
```

The output directory must be new or empty. `--backend mixed` alternates symbolic
and actual PyBullet episodes. `--backend bullet` uses physics for every episode;
`symbolic` is faster. The expert chooses dataset labels only, and never substitutes
for the model in a demo or evaluation. Episodes must pass independent grading
before their demonstrations are published. All physical grasps still use the
explicit attachment aid.

By default, 20% of training pick/place examples follow a **simulated rejected
action**, labeled `SIMULATED_EXECUTION_FAILURE`. This is controlled failure
augmentation, not recorded hardware failures. Validation/test do not receive these
injections. `--recovery-rate 0` disables them. Real controller errors are counted
separately. Ambiguous examples include both unspecified colors and duplicate-color
objects. The manifest records source backends, action counts, failures, and hashes.

## Train, stop, and resume

```shell
python -m tabletop_planner train --data data/v2 --output checkpoints/planner_v2 --epochs 3 --max-steps 60 --checkpoint-every 10 --log-every 5
```

The terminal shows the device, dataset sizes, validation batches, training batches,
optimizer updates, and elapsed time. Checkpoints contain adapter weights, tokenizer,
input configuration, optimizer state, batch position, and Python/PyTorch random
states. They are saved after complete optimizer updates under
`checkpoints/planner_v2/checkpoints/step-000010`, etc. Checkpoint directories are
published only after saving completes; `latest_checkpoint.json` identifies the
latest completed periodic checkpoint.

During training, **Ctrl+C requests a graceful stop**. Wait for the message confirming
that the current optimizer update and saving have finished. A request during
validation stops that validation without reporting a partial loss. Startup/model
loading occurs before this handler is installed. A crash or forced process kill
can recover only from an already completed checkpoint.

After a graceful stop or the step limit, resume from the run directory:

```shell
python -m tabletop_planner train --data data/v2 --output checkpoints/planner_v2 --resume checkpoints/planner_v2 --epochs 3 --max-steps 120
```

To recover after a crash, give `--resume` the completed step directory instead.
`--max-steps` is the **total** update target, including saved updates. `--epochs`
is also a total limit. Batch size, accumulation, learning rate, rank, context
length, seed, and base revision are restored from the checkpoint; device and
logging/checkpoint intervals may be selected for the resumed run. Dataset hashes
and the input configuration must match. Exact resume equivalence is tested on CPU;
changing hardware can change numerical results.

Older adapter-only runs, including `checkpoints/planner_120`, lack optimizer state
and cannot be resumed exactly. They remain available for inference. Train a fresh
adapter for the new observation format.

## Compare on the same tasks in both environments

```shell
python -m tabletop_planner evaluate --episodes data/v2/test_episodes.jsonl --backend both --limit 20 --output runs/base_v2
python -m tabletop_planner evaluate --adapter checkpoints/planner_v2 --episodes data/v2/test_episodes.jsonl --backend both --limit 20 --output runs/finetuned_v2
```

Each command freezes one episode list and reuses it for symbolic and physics runs.
The output contains `symbolic/` and `bullet/` summaries and rollouts, plus a combined
`summary.json` reporting both-pass, symbolic-only, physics-only, and both-fail counts.
The saved robot profile is loaded from the dataset directory unless explicitly
overridden with `--robot-config`. `--max-actions` controls the shared action budget.

Diagnostics distinguish malformed outputs, invalid action arguments, execution
failures, wrong targets/destinations/order, unnecessary movements, incorrect or
premature completion, unexpected refusals, and failure to ask/report missing
objects. Failure-category counts represent episodes exhibiting each category;
several categories can apply to one episode. Diagnostics use hidden goals only
for evaluation and never repair or replace the model's actions.

## Grade a custom instruction

```shell
python -m tabletop_planner demo --adapter checkpoints/planner_v2 --gui --instruction "Put the red block in the left tray." --goals configs/red_left_goals.json --output runs/custom_v2
```

The independently supplied JSON contains the expected terminal action and ordered
goals, for example:

```json
{"expected":"done","goals":[{"color":"red","zone":"left_tray"}]}
```

A goal can identify an object by `color` or `object_id`, and must match exactly one
object in the generated scene. Goals are evaluator labels and do not enter the
model prompt. Without `--goals`, custom commands remain ungraded. For ask/fail
checks, use the corresponding `expected` value and an empty goals list.

## Configure the workspace and plan hardware work

`configs/panda.json` exposes tray and starting positions, workspace bounds, home
joints, joint/link indices, gripper settings, motion heights, and forces. Pass it
with `--robot-config` to generation, demos, or evaluation. The controller checks
workspace bounds and the URDF joint limits. See [HARDWARE.md](HARDWARE.md) before
working on a different robot. The supplied controller remains validated only for
the simulated Panda configuration; a JSON profile alone does not implement a
physical robot driver or make different kinematics compatible.
