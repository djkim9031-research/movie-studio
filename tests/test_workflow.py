import json
from pathlib import Path

import pytest

from studio.workflow import fill, ltx_frames, strip_prompt_enhancer

WORKFLOWS = Path(__file__).resolve().parent.parent / "config" / "workflows"

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


ENHANCED = {
    "376": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "a pond at dawn"}},
    "380": {"class_type": "TextGenerateLTX2Prompt", "inputs": {"prompt": ["376", 0], "clip": ["393", 0]}},
    "381": {"class_type": "PreviewAny", "inputs": {"source": ["382", 0]}},
    "382": {"class_type": "ComfySwitchNode", "inputs": {"switch": ["383", 0], "on_false": ["376", 0],
                                                         "on_true": ["380", 0]}},
    "383": {"class_type": "PrimitiveBoolean", "inputs": {"value": False}},
    "387": {"class_type": "CLIPLoader", "inputs": {"clip_name": "gemma4-12b.safetensors"}},
    "393": {"class_type": "CLIPLoader", "inputs": {"clip_name": "gemma4_e2b.safetensors"}},
    "364": {"class_type": "CLIPTextEncode", "inputs": {"text": ["382", 0], "clip": ["387", 0]}},
}


def test_strip_prompt_enhancer_feeds_the_prompt_straight_to_the_text_encoder():
    stripped = strip_prompt_enhancer(ENHANCED)
    assert stripped["364"]["inputs"]["text"] == ["376", 0]
    assert set(stripped) == {"376", "387", "364"}
    assert ENHANCED["364"]["inputs"]["text"] == ["382", 0]


def test_strip_prompt_enhancer_leaves_a_workflow_without_one_alone():
    assert strip_prompt_enhancer(TEMPLATE) == TEMPLATE


def test_the_exported_ltx_workflow_keeps_no_trace_of_the_enhancer():
    workflow = strip_prompt_enhancer(json.loads((WORKFLOWS / "ltx25_i2v.json").read_text()))
    assert not any(n["class_type"] in ("TextGenerateLTX2Prompt", "ComfySwitchNode") for n in workflow.values())
    assert all("gemma4_e2b" not in str(n["inputs"]) for n in workflow.values())
    linked = {v[0] for n in workflow.values() for v in n["inputs"].values() if isinstance(v, list)}
    assert linked <= set(workflow)


def test_every_ltx_slot_points_at_an_input_of_the_stripped_workflow():
    workflow = strip_prompt_enhancer(json.loads((WORKFLOWS / "ltx25_i2v.json").read_text()))
    slots = json.loads((WORKFLOWS / "ltx25_i2v.slots.json").read_text())
    filled = fill(workflow, slots, {"prompt": "p", "image": "i.png", "seconds": 5, "width": 1280, "height": 704,
                                    "seed": 7, "prefix": "studio/x"})
    assert filled["398:362"]["inputs"]["value"] == 5 and filled["398:372"]["inputs"]["value"] == 1280
