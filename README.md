# BlackBox

BlackBox is a persistent visual-memory system for AI agents.

Most vision systems answer questions about the current frame. BlackBox is designed for the harder case: an object was seen, then disappeared, other things changed while it was unseen, and the system still needs to reason about where it may be now without pretending it knows more than it observed.

A representative example is a set of keys on a desk. If the keys disappear while someone interacts with a nearby bag, BlackBox preserves the last direct observation, tracks what changed afterward, ranks plausible location hypotheses, and keeps inferred information separate from observed facts.

## Demo

The current end-to-end demo supports a fixed-camera video workflow:

1. register visual targets such as `My Keys -> keys` and `My Bag -> bag`
2. upload a short scene
3. process detections, tracks, identities, state changes, relations, and uncertainty
4. ask a grounded question such as `Where are my keys?`
5. inspect the leading hypothesis, alternatives, evidence, caveats, and suggested re-observation target

In the current hero recording, BlackBox ranks the bag as the leading location association for the missing keys while explicitly marking the answer as inferred rather than directly observed.

## Architecture

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

The repository includes:

- OpenCV video ingestion and annotated-video output
- YOLO plus target-aware open-vocabulary detection
- deterministic IoU tracking and conservative object re-identification
- same-class detection deduplication for open-vocabulary targets
- event extraction for appearance, movement, disappearance, reappearance, and interaction
- lightweight spatial relations
- uncertainty-aware world-state projection
- competing location and change hypotheses
- evidence-grounded queries
- entropy-based re-observation planning
- a swappable unseen-change model
- controlled calibration experiments using Brier score, NLL, ECE, and reliability diagrams
- a learned logistic-regression baseline with episode-level train/validation/test splits
- a FastAPI backend and Next.js frontend for the demo workflow

## Why uncertainty matters

A disappearance is not the same thing as a physical departure. A nearby object is not automatically a container. A track ID is not automatically a persistent real-world identity.

BlackBox keeps those distinctions explicit. Observed and inferred information carry different provenance, and weak evidence is not silently upgraded into certainty.

Location weights in the current prototype are heuristic scores, not calibrated real-world probabilities. Spatial association is evidence for a hypothesis, not proof of containment or transfer.

## Run the app

Python 3.12+ is recommended.

Backend:

```bash
python -m pip install -e ".[dev,open-vocab]"
python -m uvicorn blackbox.api:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Run tests:

```bash
pytest -q
```

## Research direction

The current research question is:

> How should a persistent visual agent's belief about an object's state change as opportunities for unobserved change accumulate, and can better-calibrated uncertainty improve when and where the agent chooses to look again?

The current synthetic experiments validate the evaluation machinery and compare baseline belief-update rules. They are not evidence that the model is calibrated in the real world. Controlled recorded episodes are the next empirical step.

## Status

BlackBox is an active prototype with a working end-to-end hero demo. The current priority is packaging that demo for hackathon submissions while preserving the research architecture underneath it.

The detector, tracker, identity resolver, belief model, and re-observation policy are intentionally modular so each can be improved independently without changing the core product idea.
