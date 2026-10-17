# Hackathon submission kit

Use this as the shared core for BlackBox hackathon submissions. Venue-specific sponsor integrations should be added as thin adapters around this core instead of changing the product story.

## One-line pitch

BlackBox is a persistent visual-memory layer for cameras that remembers what was seen, reasons about what may have changed while objects were unseen, and answers with explicit uncertainty instead of invented certainty.

## Short description

Most computer-vision systems describe the current frame. BlackBox focuses on what happens after an object leaves view.

It ingests video, tracks user-selected visual concepts, records state-change events, preserves observed versus inferred provenance, and maintains competing hypotheses about an object's current state. A user can later ask a question such as "Where are my keys?" and BlackBox returns a ranked, evidence-backed answer with alternatives, caveats, and a recommendation for what to inspect next.

The current hero scenario starts with keys visible beside a bag. The keys become occluded while the user interacts with the bag, the bag moves, and the keys remain unseen. BlackBox ranks the bag as the leading association while explicitly refusing to claim that the keys are definitely inside it.

## Problem

Cameras generate streams of observations, but most AI vision demos do not maintain a trustworthy memory of what happened between frames.

A useful visual agent needs to distinguish:

- what it directly observed
- what it inferred
- what could have changed while an object was unseen
- how confident it should remain as evidence becomes stale
- when another observation would be useful

BlackBox treats this as a persistent world-state and belief problem rather than a one-frame captioning problem.

## Technical architecture

```text
camera / uploaded video
        |
        v
target-aware detection
        |
        v
tracking + persistent identity
        |
        v
state-change event extraction
        |
        v
relations + unseen-change opportunities
        |
        v
uncertainty-aware world-state projection
        |
        v
ranked hypotheses + grounded query engine
        |
        v
re-observation recommendation
```

Current implementation:

- OpenCV video ingestion
- YOLO for fixed-class perception
- OWLv2-style open-vocabulary target detection
- same-class detection deduplication
- IoU tracking
- conservative object identity resolution
- event-sourced world-state projection
- observed/inferred provenance
- heuristic location distributions
- evidence-grounded queries
- entropy-based re-observation planning
- FastAPI backend
- Next.js frontend
- calibration experiment harness with Brier score, NLL, and ECE

## Demo script

Target length: about 2 minutes.

### 0:00-0:15 — Problem

Show the keys and bag in the original scene.

Narration:

"Vision models are good at telling you what is visible right now. But if an object disappears, what should an agent believe happened next?"

### 0:15-0:35 — Register memory targets

Show:

- My Keys -> keys
- My Bag -> bag

Narration:

"BlackBox lets the user register concepts worth remembering. These labels define what the system should track; they do not pretend to establish perfect physical identity."

### 0:35-1:00 — Process the scene

Show the BlackBox view while the keys disappear and the bag is handled or moved.

Point out:

- detections / tracking
- timeline events
- keys become unobserved
- bag remains part of the scene history

Narration:

"BlackBox converts the video into structured observations and state-change events. A disappearance is treated as an inference event, not proof that an object left the scene."

### 1:00-1:30 — Ask the question

Ask:

`Where are my keys?`

Show the current hero answer.

Narration:

"BlackBox ranks the bag as the leading hypothesis, but the answer remains explicitly inferred. It preserves alternatives such as the last-seen area and outside the observed scene instead of fabricating certainty."

### 1:30-1:50 — Evidence and uncertainty

Show:

- inferred badge
- leading heuristic weight
- concise evidence
- caveats
- Next check

Narration:

"The system keeps provenance and uncertainty visible. Spatial association is evidence, not proof of containment or transfer. If uncertainty remains high, BlackBox also recommends where to look next."

### 1:50-2:00 — Close

Narration:

"BlackBox is a memory layer for cameras: not just what the camera sees now, but what it saw, what changed, and what the agent can still reasonably believe."

## Screenshots to capture

Capture clean images of:

1. the full app after the hero scene processes
2. the tracked-target panel showing My Keys and My Bag
3. the State panel with keys unobserved and bag state visible
4. the "Where are my keys?" answer with the leading bag hypothesis
5. alternatives + caveats + Next check
6. the timeline around disappearance / interaction
7. one annotated-video frame with target boxes

Avoid screenshots containing terminal clutter, browser developer tools, session errors, or model download logs.

## Claims we can make

Safe claims for the current prototype:

- working end-to-end video-to-memory demo
- target-aware visual tracking
- persistent event history
- observed-versus-inferred provenance
- ranked location hypotheses
- explicit uncertainty and alternatives
- evidence-grounded answers
- re-observation recommendations
- modular belief model and calibration evaluation infrastructure

## Claims we should not make

Do not claim:

- that 58% is a calibrated real-world probability
- that spatial association proves the keys are inside the bag
- perfect object identity or general-purpose re-identification
- production-grade surveillance or whole-house tracking
- real-world calibration based only on synthetic experiments
- that persistent visual memory itself is a novel research concept

Use "heuristic weight" for the current location score.

## Submission assets still needed

- final clean screen recording
- 3-6 polished screenshots
- architecture diagram
- public-facing repository cleanup if the repo will be shared
- venue-specific sponsor integration
- final title/tagline and submission copy
- optional short GIF of the hero flow

## Venue adaptation rule

Keep the same BlackBox core.

For each hackathon, change only the integration emphasis:

- OpenCV: agentic vision and computer-vision pipeline
- NVIDIA x Nebius: NVIDIA model / infrastructure plus agentic inference over incomplete visual evidence
- AMD: meaningful AMD-backed perception/inference or benchmarking
- Amazon: expose BlackBox memory/query capabilities through the required agent or MCP-style interface
- IBM Bob: document and demonstrate Bob-assisted engineering work completed during the hackathon window

Do not invent sponsor usage. Only describe integrations that are actually implemented and demonstrated.
