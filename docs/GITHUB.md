# Publish a reviewable repository

The project is prepared for GitHub but has not been uploaded to an account.

Before publishing, review the code, run the tests, and make sure you can explain
the model inputs, completion masking, LoRA parameters, split design, simulator
assumptions, and success checks. Add your own experiments and learning notes.

From the extracted project folder:

```bash
git init -b main
git add .
git status
git commit -m "Add tabletop planner simulation and LoRA training pipeline"
```

Use your own configured Git author identity. Create an empty GitHub repository,
then use the remote URL GitHub provides:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

The ignore file excludes environments, regenerated datasets, full runs, checkpoints,
secrets and weight binaries. Keep the small example media, validation evidence and
source code. Publish a trained adapter separately with its base revision, data
description, limitations and license information; never commit base-model weights.

The bundled smoke adapter, if present, is a limited training check rather than a
finished model. Do not label it as a production-ready robot planner. Add a model
card for a final trained release with your actual evaluation results.

A strong repository opening includes a short demo, one-command start, a clear
architecture explanation, measured comparisons, and a candid limitations section.
Attribute AI assistance according to your university/employer's expectations.
Only claim work and results you can reproduce and explain.
