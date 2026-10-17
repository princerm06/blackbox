# Hero demo recording checklist

Use this for the first end-to-end BlackBox demo. Keep the setup simple so failures are attributable to the system rather than the recording.

## Setup

- Fix the phone or webcam in one position. Do not hand-hold it.
- Frame a desk or table with the entire backpack visible.
- Use steady indoor lighting.
- Put the keys in a visually clear area with contrast against the desk.
- Keep the backpack close enough to the keys that an interaction between them is visible.
- Record in landscape if using a phone.
- Do not zoom or move the camera during a take.

Register these targets before processing:

    My Keys = keys
    Blue Backpack = backpack

The display names are for the interface. The prompts are visual concepts, not proof of a specific physical instance.

## Take 1 — observed baseline

Purpose: prove the system can directly observe both objects.

1. Start recording.
2. Leave the keys visible on the desk for 3 seconds.
3. Leave the backpack visible nearby for the entire take.
4. Do not touch either object.
5. Record about 8–10 seconds total.

Expected result:

- keys and backpack are detected
- the query for My Keys is marked observed
- no inferred transfer or containment claim

## Take 2 — disappearance with weak evidence

Purpose: show BlackBox becoming uncertain without inventing a location.

1. Start with keys visible for 3 seconds.
2. Reach across the keys so they become occluded.
3. While they are hidden, move the keys to another part of the desk.
4. Move your hand away but keep the keys out of the camera's visible region until the final reveal.
5. Keep the backpack present but do not interact with it.
6. Pause for 3–5 seconds.
7. At the end, reveal the keys only for ground-truth annotation.

Expected behavior before the reveal:

- keys are not currently observed
- last-seen evidence is preserved
- uncertainty increases
- the system should not claim the backpack contains the keys

## Take 3 — keys + backpack hero scenario

Purpose: create the main demo.

1. Start recording with keys clearly visible on the desk for 3 seconds.
2. Keep the backpack next to the keys.
3. Reach toward the backpack and keys so the keys become occluded.
4. While occluded, move the keys into or immediately beside the backpack.
5. Continue interacting with the backpack for 1–2 seconds.
6. Move the backpack to the other side of the frame.
7. Optionally move the backpack out of frame for the stronger variant.
8. Leave the keys unseen for 3–5 seconds.
9. Stop the demo evaluation point here.
10. Only after that point, reveal the keys' real location for annotation.

The query should be asked using only evidence available before step 10.

Expected behavior:

- last direct observation of My Keys remains explicit
- backpack interaction is evidence, not proof of transfer
- location candidates are ranked
- the answer remains marked inferred
- uncertainty/caveats remain visible
- a re-observation recommendation may point toward the backpack or last-seen region

## Take 4 — object removed from scene

Purpose: test the outside-scene hypothesis.

1. Start with keys visible for 3 seconds.
2. Occlude the keys with your hand.
3. Remove the keys completely from the camera's observed area.
4. Do not move the backpack.
5. Leave the scene stable for another 5 seconds.
6. End the evaluation period.
7. Reveal the ground truth after the evaluation point if needed.

Expected behavior:

- the system should retain multiple hypotheses
- outside observed scene can become a candidate
- it should not treat disappearance itself as proof that the object left

## Recording order

Record these four clips first:

    hero-01-observed.mp4
    hero-02-moved-desk.mp4
    hero-03-backpack.mp4
    hero-04-removed.mp4

Do not record the full 24-episode research set until these four clips expose what the detector, tracker, and belief system actually struggle with. Fix obvious perception failures first, then record the controlled dataset.

## Evaluation point

For every clip with a final reveal, record the frame immediately before the reveal as `evaluation_frame`.

That frame is the cutoff BlackBox is allowed to use when it predicts whether the object changed while unseen. The reveal exists only to establish ground truth.

## Demo walkthrough

For the polished hero clip:

1. Add `My Keys` and `Blue Backpack` as tracked targets.
2. Upload `hero-03-backpack.mp4`.
3. Switch to **BlackBox view**.
4. Let the annotated video reach the disappearance/interaction.
5. Ask: `Where are my keys?`
6. Show the observed/inferred badge and confidence.
7. Show alternatives and caveats.
8. Click a timeline event to jump to the supporting video moment.
9. Click **Last seen** on the keys object.
10. If BlackBox recommends re-observation, show that recommendation.
11. Reveal the actual final location only after the system's answer is visible.

The strongest demo is one where BlackBox is useful while still clearly distinguishing what it saw from what it inferred.
