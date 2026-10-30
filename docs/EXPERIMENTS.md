# Reproducible first experiment

1. Generate and freeze a dataset; preserve its manifest and source commit.
2. Evaluate the untouched base model on the frozen test file.
3. Train LoRA using only train and validation. Choose settings using validation,
   not test performance. Preserve the adapter and training report.
4. Evaluate the fine-tuned model using the same test episodes and action budget.
5. Use `evaluate --backend both` to run both model configurations on the same fixed
   episodes in symbolic and physics environments. Keep assisted
   grasping enabled consistently; do not compare different execution rules.
6. Inspect every failure category: malformed output, wrong object, wrong order,
   premature completion, missed grasp, missed tray, and failure to ask/refuse.
7. Add held-out natural human paraphrases and expand sample counts before making
   stronger generalization claims.

## Result table to fill with measured results

| Planner | Backend | N | Task success | Invalid JSON | Invalid actions | Mean actions | Mean time |
|---|---|---:|---:|---:|---:|---:|---:|
| Original model | Symbolic | | | | | | |
| LoRA model | Symbolic | | | | | | |
| Original model | PyBullet | | | | | | |
| LoRA model | PyBullet | | | | | | |

Also report per-task counts. The evaluator requires correct final states and order;
it does not treat a fluent answer as success. For ask/fail tasks, the current grader
checks terminal action type and absence of movement, not semantic quality of text.

Training loss can improve while task success does not. Include negative results.
Investigate paraphrase robustness and extensions beyond the generated grammar,
with appropriate checks and comparisons between base and fine-tuned models.

Suggested extensions after the first measured release: noisy observations, more
natural instructions, clarification/resumption, a learned vision front end, and
a hardware backend. Change one major factor at a time.

The [updated workflow](IMPROVEMENTS.md) documents mixed-source datasets,
versioned model inputs, resumable checkpoints, and independently labeled custom
instructions. Preserve the input format when comparing adapters: an old adapter
automatically retains its original prompt. A format change requires retraining.
