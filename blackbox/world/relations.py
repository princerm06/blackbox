from itertools import combinations

from blackbox.schemas.identity import IdentityAssignment
from blackbox.schemas.perception import BoundingBox
from blackbox.schemas.relations import ObjectRelation, RelationType


def _center(bbox: BoundingBox) -> tuple[float, float]:
    return (
        (bbox.x1 + bbox.x2) / 2.0,
        (bbox.y1 + bbox.y2) / 2.0,
    )


def _normalized_center_distance(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)

    dx = second_x - first_x
    dy = second_y - first_y
    distance = (dx * dx + dy * dy) ** 0.5

    reference_scale = max(
        first.width,
        first.height,
        second.width,
        second.height,
        1.0,
    )

    return distance / reference_scale


def infer_spatial_relations(
    assignments: list[IdentityAssignment],
    frame_index: int,
    timestamp_ms: int,
    max_normalized_distance: float = 1.5,
) -> list[ObjectRelation]:
    if max_normalized_distance <= 0.0:
        raise ValueError(
            "max_normalized_distance must be positive"
        )

    relations: list[ObjectRelation] = []

    for first, second in combinations(assignments, 2):
        distance = _normalized_center_distance(
            first.observation.bbox,
            second.observation.bbox,
        )

        if distance > max_normalized_distance:
            continue

        confidence = max(
            0.0,
            1.0 - (distance / max_normalized_distance),
        )

        first_id, second_id = sorted(
            (first.object_id, second.object_id)
        )

        evidence_track_ids = (
            first.track_id,
            second.track_id,
        )

        relations.append(
            ObjectRelation(
                first_object_id=first_id,
                second_object_id=second_id,
                relation_type=RelationType.SPATIALLY_ASSOCIATED,
                confidence=confidence,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                evidence_track_ids=evidence_track_ids,
            )
        )

    return relations
