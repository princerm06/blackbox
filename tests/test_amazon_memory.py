from blackbox.amazon_memory import (
    recommended_check,
    what_changed,
    what_do_you_know,
    where_are_my_keys,
)


def test_where_are_my_keys_preserves_uncertainty():
    result = where_are_my_keys()
    assert result["provenance"] == "inferred"
    assert result["heuristic_weight"] == 0.58
    assert any("did not directly observe" in item for item in result["caveats"])


def test_what_changed_has_temporal_events():
    result = what_changed()
    assert len(result["changes"]) >= 3
    assert any(item["object"] == "My Keys" for item in result["changes"])


def test_what_do_you_know_separates_provenance():
    result = what_do_you_know()
    assert result["observed"]
    assert result["inferred"]
    assert result["unknown"]
    assert "exact current physical location" in result["unknown"][-1]


def test_recommended_check_targets_bag():
    result = recommended_check()
    assert result["target"] == "My Bag"
    assert 0.0 <= result["score"] <= 1.0
