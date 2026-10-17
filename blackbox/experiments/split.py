import random

from blackbox.experiments.schema import UnseenEpisodeCase


def split_cases(
    cases: tuple[UnseenEpisodeCase, ...],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[
    tuple[UnseenEpisodeCase, ...],
    tuple[UnseenEpisodeCase, ...],
    tuple[UnseenEpisodeCase, ...],
]:
    """Episode-level train/validation/test split.

    Splitting episodes instead of frames prevents near-duplicate observations
    from leaking across evaluation partitions.
    """
    if not cases:
        raise ValueError("cases must not be empty")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1"
        )

    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(
        len(shuffled) * validation_fraction
    )

    train = tuple(shuffled[:train_end])
    validation = tuple(shuffled[train_end:validation_end])
    test = tuple(shuffled[validation_end:])

    if not train or not test:
        raise ValueError("split produced an empty train or test set")

    return train, validation, test
