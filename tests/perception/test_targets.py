import pytest

from blackbox.perception.targets import TargetRegistry


def test_register_target_uses_readable_slug() -> None:
    registry = TargetRegistry()

    target = registry.register("My Keys", prompt="keys")

    assert target.target_id == "my-keys"
    assert target.display_name == "My Keys"
    assert target.prompt == "keys"
    assert registry.prompts() == ("keys",)


def test_disabled_target_is_not_prompted() -> None:
    registry = TargetRegistry()
    target = registry.register("Black Wallet", prompt="wallet")

    registry.set_enabled(target.target_id, False)

    assert registry.prompts() == ()
    assert registry.targets()[0].enabled is False


def test_duplicate_target_id_is_rejected() -> None:
    registry = TargetRegistry()
    registry.register("My Keys", prompt="keys")

    with pytest.raises(ValueError, match="already registered"):
        registry.register("Other Keys", prompt="keys", target_id="my-keys")


def test_unregister_removes_target() -> None:
    registry = TargetRegistry()
    target = registry.register("Blue Backpack", prompt="backpack")

    removed = registry.unregister(target.target_id)

    assert removed == target
    assert registry.targets() == ()
