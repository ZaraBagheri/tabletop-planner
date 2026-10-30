# Validation log — 2026-10-28

## Code and simulator checks

- **15 automated tests passed**, including two real PyBullet tests.
- Core checks cover strict action parsing, gripper preconditions, invalid actions,
  false completion, requested ordering, ambiguity before motion, immutable
  observations, absence of hidden goals in model input, reproducible/disjoint
  dataset splits, held-out task combination, and bounded rollout length.
- A physics test verifies two transfers from actual body poses; another verifies
  that a missing-object request does not move the arm.
- Editable package installation and the `tabletop` CLI entry point were exercised.
- Python compilation and Git whitespace checks passed.
- A headless Panda GIF was generated and a final rendered frame visually inspected.
  The desktop GUI was not tested in this environment.

## Archived scripted baselines — runtime planner removed

These results record the earlier rule planner, which has since been removed from
the code and CLI. They are preserved as historical controller evidence, not as
current model results or an available planning mode. Current physics unit tests
replay fixed action fixtures solely to check the controller.

Generated with seed 42, 400 train episodes, 50 validation episodes, and 100 test
episodes. The task types are balanced.

| Planner | Backend | Cases | Passed | Notes |
|---|---|---:|---:|---|
| Rules | Symbolic | 100 | 100 | 20 per task type |
| Rules | PyBullet | 20 | 20 | 4 per task type; 8 cases require transfers |

The other physics cases require asking, reporting a missing target, or recognizing
an already-completed task. These checks do not establish general-purpose robot
reliability. All manipulation uses an explicit assisted grasp, and generated paths
use fixed overhead waypoints. This is not a collision-free-planning benchmark.

Raw results are in `examples/symbolic_baseline/` and `examples/physics_baseline/`.
The baseline covers the generated language grammar and is intentionally strong.

## Training checks

The fully offline tiny-random-model check passed: completion masking, padding
masking, two optimizer steps, nonzero learned LoRA-B weights, adapter save and
reload. This verifies software mechanics, not planning skill.

A real short LoRA run was also completed on Qwen2.5-0.5B-Instruct:

- Base revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Dataset: 30 train episodes / 66 action examples; 5 validation episodes / 11 examples.
- 20 optimizer steps, batch size 1, no accumulation, rank 8, learning rate 0.0002.
- 540,672 trainable parameters; CPU execution, approximately 259 seconds for the
  measured training/validation section (model loading is excluded).
- Validation completion loss: **0.9698 before → 0.4258 after**.
- Versions: PyTorch 2.15.0+cu132 (executed on CPU), Transformers 4.51.3, PEFT 0.15.2.

The full training report and dataset hashes are preserved in
`examples/training_smoke_report.json`. A decrease in loss does not by itself prove
better task execution. See the measured model comparison in MODEL_CARD.md.

## Remaining limits

- The full default multi-epoch training experiment has not been run.
- No physical robot, perception model, realistic grasp learner, or hardware transfer.
- Only Linux/Python 3.12 CPU execution was tested locally; Windows, macOS/MPS and
  CUDA training are not validated here. Python 3.11 is recommended for PyBullet wheels.
- The GitHub Actions workflow is supplied but has not run on a hosted repository.
- No statistical reliability claim is justified by these small, synthetic test sets.

## Update — 2026-10-30: input formats, checkpoints, and paired evaluation

- **34 automated tests passed**, including real PyBullet generation and paired
  evaluation checks. Tests cover coordinate-invariant prompts, preserved legacy
  input metadata, protected/atomic dataset publication, training-only recovery
  augmentation, independent custom goals, and separate failure categories.
- The offline tiny-random-model training check passed. With gradient accumulation,
  resumed CPU weights exactly matched uninterrupted training, both within an epoch
  and from the final-batch periodic checkpoint. A simulated Ctrl+C saved progress;
  resuming that checkpoint also matched uninterrupted weights. Changed resume data
  was rejected. These tests validate training mechanics, not planning capability.
- The wheel was built locally and inspected: it contains the new modules and both
  prompt files. Existing `checkpoints/planner_120` adapter files were unchanged.
- Current checks ran on macOS/Python 3.12. Windows setup is documented but was not
  executed here. The real local Qwen experiment uses Apple MPS; numerical resume
  equivalence is asserted only by the CPU test.

### Local development comparison, first 60 updates

A separate `semantic-v2` dataset was generated with seed 2026: 100 training
episodes / 220 action examples, 10 validation episodes / 22 examples, and 20 test
episodes. Each split mixes symbolic and PyBullet sources. There were 33 simulated
recovery observations in training and no actual controller failures during data
collection. Validation/test had no injected failures.

The new Qwen adapter trained for 60 updates with accumulation 4, reducing validation
loss from 0.9804 to 0.09245. The original adapter had 60 updates with accumulation 8
on a different, larger symbolic dataset. Thus this is a comparison of complete
experiment conditions, not an isolated ablation of the observation format.

Both adapters were evaluated on the same first ten test episodes in both backends:

| Adapter | Input | Symbolic | PyBullet |
|---|---|---:|---:|
| Original local 60-step adapter | raw-v1 | 3/10 | 4/10 |
| New 60-step adapter | semantic-v2 | 4/10 | 4/10 |

The new run's outcomes agreed across backends for all ten cases; that does not
establish reliable planning. Ordered transfers, already-completed goals, and
refusals still failed. The custom red-to-left task also failed by asking an
unnecessary question. No successful model grasp is claimed for that custom run.

Artifacts: `runs/data_v2/`, `checkpoints/planner_v2_smoke/`,
`runs/eval_original_paired/`, `runs/eval_v2_paired/`, and
`runs/custom_v2_check/`. These local artifacts are ignored by Git. The ten cases
used to inspect failures are development evidence, not an untouched final test set.

### Completed three-epoch local experiment

The saved 60-step v2 run was resumed on Apple MPS, preserving its optimizer and
batch position, and completed all three epochs at **165 optimizer updates**.
The final validation loss was **0.09645** (initially 0.9804). The earlier 60-step
validation loss was lower, so longer training did not consistently help even that
metric. No hyperparameters or prompt were changed between the two stages.

Both the completed model and original adapter were then evaluated on all 20
frozen test episodes, with four episodes per task type and an eight-action budget:

| Adapter | Symbolic | PyBullet |
|---|---:|---:|
| Original local 60-step raw-v1 adapter | 6/20 | 7/20 |
| Completed 165-step semantic-v2 adapter | 6/20 | 6/20 |

The new model produced identical backend success/failure outcomes across all 20
cases and no malformed JSON, but it still made invalid actions and serious
semantic errors. It passed missing-object cases and two single transfers, while
all ordered transfers, ambiguity cases, and already-completed cases failed.
There is **no measured overall planning-accuracy improvement** in this experiment.
The input contract and workflow changes should not be described as a successful
final learned planner.

On the original red-to-left custom instruction, the completed model picked the
correct red `block_1` and placed it in `left_tray`, then unnecessarily moved blue
`block_0` there too. Independent grading correctly rejected the episode. This is
an improvement over the false missing-object response on that particular decision,
but the complete task still failed and is not a successful demonstration.

Final artifacts: `checkpoints/planner_v2/`, `runs/eval_v2_final/`,
`runs/eval_original_final/`, and `runs/custom_v2_final/`. The first ten test cases
had already been inspected during development; the remaining ten were newly
checked after the fixed three-epoch run. Results are small synthetic development
evidence, not a general reliability claim. Existing adapter weights and earlier
results were retained. Further work needs broader training coverage and model
selection based on validation task success, rather than assuming additional
updates or lower completion loss will solve grounding.

The final media check exposed a source-built PyBullet compatibility issue: camera
pixels were tuples instead of NumPy arrays. The conversion was fixed, a tuple-pixel
PNG/GIF export regression test passed, and an actual rendered simulator frame was
visually inspected. These rendering checks are independent of learned planning.
