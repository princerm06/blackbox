from dataclasses import dataclass
import re


_TARGET_ID_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = _TARGET_ID_PATTERN.sub("-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("target name must contain letters or numbers")
    return slug


@dataclass(frozen=True)
class TrackingTarget:
    """User-requested object concept to follow.

    A target prompt describes a visual concept. It does not, by itself,
    establish that a detection is a specific physical instance.
    """

    target_id: str
    display_name: str
    prompt: str
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id must not be empty")
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")


class TargetRegistry:
    """Small in-memory registry for the objects a session wants tracked."""

    def __init__(self) -> None:
        self._targets: dict[str, TrackingTarget] = {}

    def register(
        self,
        display_name: str,
        prompt: str | None = None,
        target_id: str | None = None,
    ) -> TrackingTarget:
        clean_name = display_name.strip()
        clean_prompt = (prompt or display_name).strip()
        resolved_id = target_id.strip() if target_id else _slugify(clean_name)

        if resolved_id in self._targets:
            raise ValueError(f"target_id already registered: {resolved_id}")

        target = TrackingTarget(
            target_id=resolved_id,
            display_name=clean_name,
            prompt=clean_prompt,
        )
        self._targets[resolved_id] = target
        return target

    def unregister(self, target_id: str) -> TrackingTarget:
        try:
            return self._targets.pop(target_id)
        except KeyError as error:
            raise KeyError(f"unknown target_id: {target_id}") from error

    def set_enabled(
        self,
        target_id: str,
        enabled: bool,
    ) -> TrackingTarget:
        try:
            current = self._targets[target_id]
        except KeyError as error:
            raise KeyError(f"unknown target_id: {target_id}") from error

        updated = TrackingTarget(
            target_id=current.target_id,
            display_name=current.display_name,
            prompt=current.prompt,
            enabled=enabled,
        )
        self._targets[target_id] = updated
        return updated

    def targets(self) -> tuple[TrackingTarget, ...]:
        return tuple(self._targets.values())

    def enabled_targets(self) -> tuple[TrackingTarget, ...]:
        return tuple(
            target for target in self._targets.values()
            if target.enabled
        )

    def prompts(self) -> tuple[str, ...]:
        return tuple(target.prompt for target in self.enabled_targets())
