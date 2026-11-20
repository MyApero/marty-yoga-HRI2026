import json

from src.lucy_plugin import parse_action


def message(**fields):
    return json.dumps(fields)


def test_an_action_for_this_plugin_is_read_with_its_option():
    assert parse_action(message(plugin="yoga_coach", action="coach", option="chair"), "yoga_coach") == ("coach", "chair")
    assert parse_action(message(plugin="yoga_coach", action="stop"), "yoga_coach") == ("stop", None)


def test_other_plugins_and_malformed_messages_are_ignored():
    assert parse_action(message(plugin="other_app", action="stop"), "yoga_coach") is None
    assert parse_action(message(plugin="yoga_coach", action=3), "yoga_coach") is None
    assert parse_action("not json", "yoga_coach") is None
    assert parse_action(None, "yoga_coach") is None
    assert parse_action(message(plugin="yoga_coach", action="coach", option=["chair"]), "yoga_coach") == ("coach", None)
