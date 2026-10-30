# Initial adapter — development smoke run

## Status

This card describes the archived 20-step example adapter. The newer local
semantic-v2 experiment and its unsuccessful planning-accuracy comparison are
recorded separately in [VALIDATION.md](VALIDATION.md).

**This adapter is not a reliable robot planner and is not a finished model release.**
It is included in the downloadable project archive to reproduce a real training,
saving, loading and evaluation run. Its folder is ignored by Git by default.

The project implementation supports a full training experiment, but the included
adapter received only 20 optimizer steps. Archived controller recordings do not
establish its learned capability. Current demos use the language model exclusively.

## Model and training

- Base: Qwen/Qwen2.5-0.5B-Instruct, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Method: LoRA, rank 8, alpha 16, dropout 0.05, attention q_proj/v_proj modules.
- Trainable parameters: 540,672.
- Objective: next-action completion-only causal language-model loss.
- Data: 30 generated training episodes / 66 action examples, 5 validation
  episodes / 11 examples; seed 42. Only 20 batches were optimized in this short run.
- Learning rate 0.0002, batch size 1, gradient accumulation 1, maximum length 1536.
- Validation loss: 0.9698 before, 0.4258 after.

## Closed-loop development check

Both models received the same prompt and the same ten symbolic test episodes.
Greedy free-form decoding was used; exact JSON output was required, with no
automatic repair, action substitution, expert fallback or constrained decoder.

| Model | Cases | Completed correctly |
|---|---:|---:|
| Original Qwen2.5-0.5B-Instruct | 10 | 0 |
| Same model + 20-step LoRA adapter | 10 | 0 |

The original model repeatedly produced fenced JSON despite the requested exact
schema; some proposed actions were also semantically invalid. The short-trained
model produced some executable actions but still made wrong-target/goal decisions
and invalid terminal responses. Raw traces are included in the corresponding
`examples/*model_smoke/` directories. Inspect them rather than inferring task
competence from lower validation loss.

This is a negative development result, not a benchmark claim about Qwen models.
It uses a small synthetic task set, a short training run, a specific prompt and a
strict output contract. No trained-model physical-simulation result is claimed.

## Reproduce the adapter check

After installing training dependencies:

```bash
python -m tabletop_planner evaluate --observation-format raw-v1 --episodes examples/training_smoke_data/test_episodes.jsonl --output runs/recheck_base
python -m tabletop_planner evaluate --planner model --adapter examples/training_smoke_adapter --episodes examples/training_smoke_data/test_episodes.jsonl --output runs/recheck_adapter
```

Base weights are not bundled and will be downloaded on first use. The adapter and
tokenizer are under `examples/training_smoke_adapter/` in the archive. If working
from a Git checkout without that ignored folder, reproduce the short training
command from README.md using the bundled smoke dataset.

## Next experiment

Train the full multi-epoch configuration and use validation episodes to develop
settings. Keep a new, untouched final evaluation set for reporting results; these
ten cases are now development evidence. Separate syntax adherence, next-action
accuracy and complete-task success. If adding constrained decoding or JSON repair,
evaluate it as a separately labeled condition for both models.

No perception, motion-control or grasp-policy weights are learned. The model only
predicts symbolic task actions from structured scene observations. Hardware transfer
has not been tested. Base model and library licenses continue to apply.
