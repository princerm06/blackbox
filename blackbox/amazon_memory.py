"""Amazon hackathon conversational memory adapter.

This module intentionally operates over a preprocessed BlackBox hero-run snapshot.
It does not claim fresh cloud vision inference. The snapshot mirrors output produced
by the full local BlackBox pipeline and lets the Alexa+/MCP layer exercise the
memory/reasoning interaction reliably.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryFact:
    statement: str
    provenance: str


HERO_FACTS = (
    MemoryFact("My Keys were directly observed on the desk.", "observed"),
    MemoryFact("My Keys later became unobserved.", "observed"),
    MemoryFact("My Bag was nearby around the disappearance window.", "observed"),
    MemoryFact("My Bag moved after the keys became unobserved.", "observed"),
    MemoryFact("A person was present during part of the uncertainty window.", "observed"),
)

LOCATION_HYPOTHESES = (
    ("associated with My Bag", 0.58),
    ("associated with person", 0.19),
    ("last-seen area", 0.15),
    ("outside observed scene", 0.07),
)


def where_are_my_keys() -> dict[str, object]:
    """Return the ranked BlackBox location memory for the hero run."""
    return {
        "answer": "The leading hypothesis is that My Keys is associated with My Bag.",
        "heuristic_weight": 0.58,
        "provenance": "inferred",
        "alternatives": [
            {"label": label, "weight": weight}
            for label, weight in LOCATION_HYPOTHESES[1:]
        ],
        "evidence": [fact.statement for fact in HERO_FACTS],
        "caveats": [
            "BlackBox did not directly observe the keys enter the bag.",
            "Spatial association does not prove containment or transfer.",
            "The displayed weights are heuristic scores, not calibrated probabilities.",
        ],
        "next_check": "Inspect My Bag first to reduce uncertainty.",
    }


def what_changed() -> dict[str, object]:
    """Summarize temporal state changes from the hero run."""
    return {
        "changes": [
            {"time_s": 20.3, "object": "My Keys", "change": "became unobserved"},
            {"time_s": 23.4, "object": "My Bag", "change": "moved"},
            {"time_s": 24.4, "object": "person", "change": "became unobserved"},
            {"time_s": 31.9, "object": "My Bag", "change": "remained observed"},
        ],
        "summary": (
            "The keys became unobserved, the bag moved afterward, and the person "
            "later left the observed scene."
        ),
    }


def what_do_you_know() -> dict[str, object]:
    """Separate directly observed facts from inference and unknowns."""
    return {
        "observed": [
            "My Keys were visible before the disappearance event.",
            "My Bag was spatially associated with the keys around that time.",
            "My Bag moved after My Keys became unobserved.",
            "A person was present during part of the uncertainty interval.",
        ],
        "inferred": [
            "My Bag is the leading association for the missing keys.",
        ],
        "unknown": [
            "Whether the keys entered the bag.",
            "Whether the person moved the keys while they were occluded.",
            "The keys' exact current physical location.",
        ],
    }


def recommended_check() -> dict[str, object]:
    """Return BlackBox's next re-observation recommendation."""
    return {
        "target": "My Bag",
        "score": 0.82,
        "reason": (
            "The bag is the leading associated object and checking it should reduce "
            "uncertainty most efficiently."
        ),
    }
