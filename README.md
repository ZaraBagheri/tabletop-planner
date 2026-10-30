# Tabletop Planner

**Connecting language models and robot task planning.**

Fine-tune a language model to plan pick-and-place tasks, then evaluate its actions with a simulated robot arm. This personal project explores the full workflow: generating demonstrations, training adapters, running a planner, and measuring whether its decisions complete a task. No physical robot or paid LLM API is required.

## How it works

> “Put the yellow block in the right tray. Then move the blue block to the left tray.”

The planner reads the instruction and current scene, then returns exactly one action:

```json
{"action":"pick","target":"block_2"}
```

The simulator executes it, returns an updated observation, and the planner decides again. Allowed actions are `pick`, `place`, `done`, `ask`, and `fail`. `ask` and `fail` end the episode; resuming after clarification is future work. Invalid outputs are rejected without an expert fallback.

I chose [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and fine-tuned it with **LoRA** using PyTorch, Hugging Face Transformers, and PEFT. LoRA trains added parameters while keeping the base weights frozen; loss is calculated only on the expected assistant response.

| Component | Responsibility |
|---|---|
| Language model | Choose objects, destinations, action order, and terminal responses. |
| PyBullet Panda controller | Execute arm movements, picking, and placing. |
| Independent evaluator | Check actual object locations and transfer order. |

New models receive IDs, colors, zones, gripper state, and action history; coordinates stay with the controller. Each adapter saves its prompt and input format. Older adapters retain their original inputs automatically. Demo and evaluation runs keep model weights fixed; pass `--adapter` to load trained weights, or omit it to use the pretrained model.

## Current results and scope

The local development comparison used the same 20 generated episodes for both adapters:

| Adapter | Symbolic | PyBullet |
|---|---:|---:|
| Earlier fine-tuned adapter, 60 updates | 6/20 | 7/20 |
| v2 adapter, three epochs / 165 updates | 6/20 | 6/20 |

**The model remains experimental: the workflow changes did not improve overall planning accuracy.** The runs used different training datasets and input formats. Conditions, failures, and limitations are recorded in [VALIDATION.md](docs/VALIDATION.md).

The simulator supplies object state and assists grasps with a fixed attachment. Motion follows overhead waypoints. Camera perception, general collision-free planning, and physical robot integration remain future work. This project studies high-level task planning; it does not learn grasping or motor control.

## Setup

Use **Python 3.11 or 3.12**. From the project folder, create and activate a virtual environment.

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For Python 3.11, substitute `python3.11` or `py -3.11`. If `.venv` exists, activate it without recreating it. Windows Command Prompt uses `.venv\Scripts\activate.bat`.

Install dependencies on any platform:

```shell
python -m pip install --upgrade pip
python -m pip install -e ".[sim,train]"
```

The commands below work once the environment is active. Platform-specific PyTorch options are in the [detailed workflow](docs/IMPROVEMENTS.md#installation-options).

## Generate, train, and evaluate

`tasks.py` and `data.py` generate synthetic examples of single transfers, ordered transfers, missing objects, ambiguity, and already-completed goals. An expert function labels the correct next action, and training learns from those answers automatically.

Episodes stay within one split. Test instruction templates are held out, and yellow-to-right-tray goals are excluded from training/validation. Object IDs are shuffled relative to colors. `mixed` data combines symbolic and physics episodes; training-only augmentation adds simulated rejected actions. See [DATA_CARD.md](docs/DATA_CARD.md).

```shell
# Optional: try the pretrained model in the simulator
python -m tabletop_planner demo --gui

# Generate training, validation, and test episodes
python -m tabletop_planner generate --train 800 --val 100 --test 100 --backend mixed --output data/v2

# Start with a bounded training run
python -m tabletop_planner train --data data/v2 --output checkpoints/planner_v2 --epochs 3 --max-steps 60 --checkpoint-every 25

# Compare the pretrained model and trained adapter on identical tasks
python -m tabletop_planner evaluate --episodes data/v2/test_episodes.jsonl --backend both --limit 20 --output runs/base_v2
python -m tabletop_planner evaluate --adapter checkpoints/planner_v2 --episodes data/v2/test_episodes.jsonl --backend both --limit 20 --output runs/finetuned_v2

# Watch the trained planner
python -m tabletop_planner demo --adapter checkpoints/planner_v2 --gui
```

Use new or empty data and training output directories. The first model load downloads its weights. Training selects CUDA, Apple MPS, or CPU automatically and displays progress; runtime depends on the device and dataset.

Checkpoints save adapter weights, optimizer state, batch position, and random states. **Ctrl+C during training requests a graceful stop and save; wait for confirmation.** Continue a saved run with `--resume checkpoints/planner_v2`. `--max-steps` is the total update target, including previous updates. The [workflow guide](docs/IMPROVEMENTS.md) includes complete resume commands.

Paired evaluation writes separate backend summaries and rollouts, plus an agreement summary. A `done` response passes only when object placement and action order are correct. Diagnostics distinguish malformed outputs, invalid actions, execution failures, wrong targets, and incorrect completion.

## Give your own instruction

```shell
python -m tabletop_planner demo --adapter checkpoints/planner_v2 --gui --instruction "Put the red block in the left tray." --goals configs/red_left_goals.json --output runs/custom_demo
```

Each run starts a fresh scene. The default scene has red, blue, green, and yellow blocks and two trays. The goals file independently defines success and is not shown to the model. Custom instructions without `--goals` remain ungraded.

For headless recording, replace `--gui` with `--record --output runs/learned_demo`. Outputs include `replay.gif`, `final.png`, and `rollout.json`.

## Change the configuration or code

Training options include `--epochs`, `--lr`, `--rank`, `--batch-size`, `--accumulation`, and `--device`; list them with `python -m tabletop_planner train --help`. New experiments start from the base model unless `--resume` is supplied.

| Change | Files to start with |
|---|---|
| Tasks and demonstrations | [tasks.py](tabletop_planner/tasks.py), [data.py](tabletop_planner/data.py) |
| Fine-tuning and checkpoints | [training.py](tabletop_planner/training.py) |
| Model inputs and inference | [observations.py](tabletop_planner/observations.py), [system_prompt.txt](tabletop_planner/system_prompt.txt), [planners.py](tabletop_planner/planners.py) |
| Robot/workspace settings | [config.py](tabletop_planner/config.py), [panda.json](configs/panda.json) |
| Execution and grading | [bullet_backend.py](tabletop_planner/bullet_backend.py), [symbolic.py](tabletop_planner/symbolic.py), [domain.py](tabletop_planner/domain.py), [evaluation.py](tabletop_planner/evaluation.py) |
| Command-line options | [cli.py](tabletop_planner/cli.py) |

Select a robot profile with `--robot-config configs/panda.json`. A physical arm needs its own driver and calibration; see [HARDWARE.md](docs/HARDWARE.md). Further experiments should examine failure cases, broaden training coverage, and use untouched evaluation data.

## Tests and license

```shell
python -m unittest discover -s tests -v
python -m tests.ml_smoke
```

Physics tests run when PyBullet is installed. The offline training check uses a tiny random model to verify masking, checkpointing, and resume mechanics; it does not establish planning capability.

Project code is [MIT-licensed](LICENSE). Qwen weights, [PyBullet](https://github.com/bulletphysics/bullet3), robot assets, and dependencies retain their own licenses.
