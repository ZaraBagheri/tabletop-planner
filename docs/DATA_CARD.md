# Synthetic tabletop next-action dataset

## Source and purpose

Generated from the code in `tasks.py` and a deterministic task oracle. Symbolic,
PyBullet, and mixed source modes are available. No human
demonstrations, personal data, scraped text, API-generated labels, or real robot
trajectories are used. Each row contains system/user/assistant messages and an
episode identifier, observation-format version, and backend. The assistant completion is the next JSON action.

The oracle has a hidden structured task specification; that specification is not
part of model input. Inputs include language instructions and current observations.
Physics evaluation independently measures the resulting object poses.

## Distribution

Four block colors, shuffled object IDs, jittered starting positions, and two trays.
Five evenly cycled task types: single transfer, ordered double transfer, missing
object, ambiguous reference, and already-satisfied task. A double task uses two
different objects and two different trays. No stacking is generated.

Train/validation share six sentence-template families but have disjoint episode
seeds. Test uses two additional template families and new seeds. Yellow-to-right
goals are excluded from train/validation, and deliberately included in some test
episodes. Not every test episode is a held-out-combination task: report the scope
of any generalization claim accordingly.

All steps from an episode stay in its split. `manifest.json` records generation
seed, example/episode counts, source backends, action counts, recoveries, and hashes.
Generation requires a new/empty directory. Training verifies train/validation
episode disjointness and rejects changed data on resume.

`semantic-v2` model inputs omit coordinates and whitelist semantic object state;
`raw-v1` remains available for older experiments. New adapters save the format and
exact prompt. Training-only recovery augmentation simulates rejected pick/place
actions; these injections are distinct from actual physics execution failures.
Every generated episode must satisfy its goals before publication. Physics data
still uses assisted grasps. See [IMPROVEMENTS.md](IMPROVEMENTS.md).

## Limitations

- Synthetic language is simple, repetitive and generated from a small set of templates.
- Coordinate-invariant observations can repeat across different episode seeds;
  seed-disjoint splits do not guarantee unique semantic states. Test phrasing
  remains held out from training and validation.
- The split jointly changes phrasing, layouts and some combinations; it does not
  isolate the contribution of each factor. Add separate test suites for that.
- Static ask/fail labels teach basic behavior; they do not assess question quality
  or recovery across a user's clarification.
- Semantic observations remove coordinate variation, but neither source mode
  establishes robustness to perception errors or untested physical conditions.
- No continuous trajectories, force control, real camera images, or real grasps.
- Small test counts or one run are insufficient to establish broad reliability.

For a stronger study, add independently written paraphrases, more objects and
layouts, separate held-out axes, and failure trajectories gathered from simulation.
