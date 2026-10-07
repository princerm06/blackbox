import importlib.util
from pathlib import Path


def load_script_module():
    script_path = Path("scripts/process_video_targets.py")
    spec = importlib.util.spec_from_file_location(
        "process_video_targets",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_target_with_explicit_prompt() -> None:
    module = load_script_module()

    assert module.parse_target("My Keys=keys") == ("My Keys", "keys")


def test_parse_target_uses_same_text_without_separator() -> None:
    module = load_script_module()

    assert module.parse_target("wallet") == ("wallet", "wallet")
