import pytest

from studio.workflow import fill, ltx_frames

TEMPLATE = {"3": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
            "9": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
            "15": {"class_type": "EmptyLatent", "inputs": {"width": 640, "height": 352, "length": 97}}}


def test_ltx_frames_are_8n_plus_1_at_24_fps():
    assert ltx_frames(5) == 121
    assert ltx_frames(4) == 97
    assert all((ltx_frames(s) - 1) % 8 == 0 for s in (1, 2.5, 7, 10))


def test_fill_sets_each_slot_and_leaves_the_template_unchanged():
    slots = {"prompt": "3.text", "image": "9.image", "frames": "15.length"}
    filled = fill(TEMPLATE, slots, {"prompt": "a slow push-in", "image": "still.png", "frames": 121})
    assert filled["3"]["inputs"]["text"] == "a slow push-in"
    assert filled["9"]["inputs"]["image"] == "still.png"
    assert filled["15"]["inputs"]["length"] == 121
    assert TEMPLATE["3"]["inputs"]["text"] == "old"


def test_fill_refuses_a_slot_that_points_nowhere():
    with pytest.raises(KeyError, match="99.text"):
        fill(TEMPLATE, {"prompt": "99.text"}, {"prompt": "x"})


def test_fill_refuses_a_value_without_a_slot():
    with pytest.raises(KeyError, match="seed"):
        fill(TEMPLATE, {"prompt": "3.text"}, {"seed": 1})
