# BlackBox

BlackBox is an experiment in visual memory for AI agents.

The basic problem is simple: a camera sees an object, then stops seeing it. What should an agent believe happened next?

Most vision demos are good at describing the current frame. BlackBox is focused on the harder part: keeping track of what was seen, what changed, what is only inferred, and when the system should admit that it is uncertain.

A typical example is a set of keys on a desk. If the keys disappear while someone is interacting with a nearby backpack, BlackBox should remember the last direct observation, keep multiple possible explanations alive, and avoid claiming the keys are inside the backpack unless the evidence actually supports that.

## Current pipeline

```text
video
  -> detection
  -> tracking
  -> persistent object identity
  -> state-change events
  -> object relations
  -> unseen-change opportunities
  -> belief distributions
  -> grounded answers / re-observation
```

The project currently includes:

- OpenCV video ingestion
- an injectable detector interface and YOLO adapter
- deterministic IoU tracking
- conservative object re-identification
- event extraction for appearance, movement, disappearance, and interaction
- lightweight spatial relations
- uncertainty-aware world-state projection
- competing location and change hypotheses
- evidence-grounded queries
- entropy-based re-observation planning
- a swappable unseen-change model
- controlled calibration experiments with Brier score, NLL, ECE, and reliability diagrams
- a first learned logistic-regression baseline with episode-level train/validation/test splits

## Why the uncertainty matters

A disappearance is not the same thing as a physical departure. A nearby object is not automatically a container. A track ID is not automatically a persistent real-world identity.

BlackBox keeps those distinctions explicit. Observed and inferred information carry different provenance, and weak visual evidence is not silently upgraded into certainty.

## Quick start

Python 3.12+ is recommended.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Process a video:

```bash
python scripts/process_video.py path/to/video.mp4
```

Run the controlled calibration benchmark:

```bash
python scripts/run_calibration_experiment.py
```

Train and evaluate the first learned change model:

```bash
python scripts/run_learned_experiment.py
```

Experiment outputs are written under `runs/` and are ignored by git.

## Research direction

The current research question is:

> How should a persistent visual agent's belief about an object's state change as opportunities for unobserved change accumulate, and can better-calibrated uncertainty improve when and where the agent chooses to look again?

The present experiments are deliberately small and controlled. Synthetic results are useful for validating the evaluation machinery, but they are not evidence that the model is calibrated in the real world. Real recorded episodes and stronger baselines are the next step.

## Status

This is an active research prototype, not a finished product. The architecture is intentionally modular so the detector, tracker, identity resolver, belief model, and re-observation policy can be replaced independently as the experiments get stronger.

The next milestones are real-world episode collection, learned belief models, and a cleaner end-to-end demo of the full keys/backpack scenario.
