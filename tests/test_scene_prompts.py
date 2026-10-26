import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from studio import scene_prompts
from studio.refs import Reference, Report


def report(people=6, extras=("saturn_cassini",)):
    r = Report()
    for i in range(people):
        r.used.append(Reference("person", Path(f"person{i + 1}.png"), Path(f"person{i + 1}.png"), f"p{i}.jpg"))
    for name in extras:
        r.used.append(Reference("extra", Path("extra.png"), Path("extra_square.png"), f"{name}.png"))
    return r


class FakeClaude:
    def __init__(self, reply, stop_reason="end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, route, *, film, **params):
        self.calls.append((route, film, params))
        return SimpleNamespace(stop_reason=self.stop_reason, content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text=json.dumps(self.reply))])


REPLY = {"still_request": "Create a photograph of the person shown in images 1-6 ...",
         "motion_prompt": "A shot in deep space. The camera stays still ..."}


def test_write_sends_only_text_and_returns_both_prompts():
    claude = FakeClaude(REPLY)
    prompts = scene_prompts.write(claude, film="scene-space", user_prompt="me floating near Saturn, thin smile",
                                  report=report(), orientation="landscape")
    assert prompts.still_request.startswith("Create a photograph") and prompts.motion_prompt.startswith("A shot")
    route, film, params = claude.calls[0]
    assert (route, film) == ("scene_prompts", "scene-space")
    for message in params["messages"]:
        assert all(block["type"] == "text" for block in message["content"])
    assert all(block["type"] == "text" for block in params["system"])
    assert params["output_config"]["format"]["type"] == "json_schema"
    sent = params["messages"][0]["content"][0]["text"]
    assert "me floating near Saturn, thin smile" in sent and "landscape" in sent


def test_roles_number_the_references_as_the_image_model_sees_them():
    text = scene_prompts.roles(report(people=6, extras=("saturn_cassini", "IMG_0042")))
    assert "Images 1-6: six photos of the same person" in text
    assert 'Image 7: a picture without people, file name "saturn cassini"' in text
    assert 'Image 8: a picture without people, file name "IMG 0042"' in text


def test_roles_cope_with_one_photo_and_no_person():
    assert "Image 1: one photo of the person" in scene_prompts.roles(report(people=1, extras=()))
    assert "no photo of a person" in scene_prompts.roles(report(people=0, extras=("beach",)))


def test_extra_file_names_are_reduced_to_plain_words():
    text = scene_prompts.roles(report(people=1, extras=('evil"name\n<script>' + "x" * 80,)))
    assert '"' not in text.split("file name ")[1].strip()[1:-2] and "<" not in text and len(text) < 300


def test_a_reply_cut_off_at_the_token_limit_is_an_error():
    with pytest.raises(scene_prompts.PromptError, match="cut off"):
        scene_prompts.write(FakeClaude(REPLY, stop_reason="max_tokens"), film="f", user_prompt="x",
                            report=report(), orientation="landscape")
