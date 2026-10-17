# Real-world episode recording guide

The first real dataset should stay small and controlled. Use a fixed phone or webcam, stable lighting, and one target object per clip.

## Minimum episode set

Record at least 24 clips:

- 6 unchanged while unseen
- 6 moved to another visible area
- 6 moved near or into a backpack/container
- 6 removed from the observed scene

Try to vary whether a person moves nearby objects, whether the backpack moves, and how long the target stays unseen.

## Filming rules

1. Keep the camera fixed for the entire clip.
2. Start with the target visible for at least two seconds.
3. Make the disappearance/occlusion easy to timestamp.
4. Do not intentionally hide the ground-truth outcome from the annotator.
5. End with enough footage to establish the true final state.
6. Keep raw recordings outside git. Only commit manifests and small derived metadata.

## Manifest format

Create a JSON file such as `data/episodes/manifest.json`:

```json
[
  {
    "episode_id": "keys-001",
    "video_path": "data/episodes/raw/keys-001.mp4",
    "target_class": "keys",
    "changed_while_unobserved": true,
    "final_location_label": "inside_backpack",
    "evaluation_frame": 145,
    "notes": "Backpack moved after keys disappeared."
  }
]
```

The manifest is ground truth. BlackBox predictions should be written separately so evaluation never overwrites the annotation.

## First hero demo

For the polished demo:

1. Keys visible on desk.
2. Backpack nearby.
3. Person reaches across the scene and occludes the keys.
4. Keys are moved while unseen.
5. Backpack moves or exits.
6. Ask BlackBox where the keys are.
7. Show last direct observation, candidate hypotheses, uncertainty, and recommended re-observation target.

A good demo should include at least one moment where BlackBox explicitly says it is uncertain rather than inventing a precise answer.


## Choosing the evaluation frame

Set `evaluation_frame` to the frame immediately before the clip reveals the ground-truth outcome. BlackBox is scored using only evidence available up to that frame.

This matters because the final seconds of a controlled clip may deliberately reveal where the object actually ended up. Those reveal frames are for annotation, not for the model's prediction.

After each episode has been processed into `runs/<episode_id>/observations.jsonl`, evaluate the real-world set with:

```bash
python scripts/evaluate_recorded_outputs.py \
  --manifest data/episodes/manifest.json \
  --runs-dir runs/episodes
```

The evaluator reports Brier score, NLL, and ECE against the recorded ground truth. If the requested target class was never detected, the episode fails explicitly instead of silently inventing a prediction.
