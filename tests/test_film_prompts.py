import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from studio import film_prompts
from tests.test_scene_prompts import report


class FakeClaude:
    def __init__(self, reply, stop_reason="end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, route, *, film, **params):
        self.calls.append((route, film, params))
        return SimpleNamespace(stop_reason=self.stop_reason, content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text=json.dumps(self.reply))])


def shot(n):
    return {"still_request": f"Create a photograph of the person shown in images 1-6 ... shot {n}",
            "motion_prompt": f"A courtyard in front of a palace ... shot {n}",
            "sound_prompt": f"Generate the sound of hooves on stone ... shot {n}"}


def plan(count):
    return {"title": "Ride at the Palace", "shots": [shot(n) for n in range(1, count + 1)]}


def write(claude, shots=4):
    return film_prompts.write(claude, film="film-ride", user_prompt="me riding a horse in front of the palace",
                              report=report(), orientation="landscape", shots=shots, seconds=5)


def test_write_sends_only_text_and_returns_the_asked_shots():
    claude = FakeClaude(plan(4))
    made = write(claude)
    assert made.title == "Ride at the Palace" and len(made.shots) == 4
    assert made.shots[3].sound_prompt.endswith("shot 4")
    route, film, params = claude.calls[0]
    assert (route, film) == ("film_prompts", "film-ride")
    for message in params["messages"]:
        assert all(block["type"] == "text" for block in message["content"])
    assert all(block["type"] == "text" for block in params["system"])
    assert params["output_config"]["format"]["type"] == "json_schema"
    sent = params["messages"][0]["content"][0]["text"]
    assert "me riding a horse in front of the palace" in sent
    assert "4 shots of 5 seconds" in sent and "Images 1-6: six photos of the same person" in sent


def test_a_plan_with_another_number_of_shots_is_an_error():
    with pytest.raises(film_prompts.PromptError, match="3 shots instead of 4"):
        write(FakeClaude(plan(3)))


def test_an_empty_prompt_in_a_shot_is_refused():
    bad = plan(4)
    bad["shots"][1]["sound_prompt"] = " "
    with pytest.raises(ValidationError):
        write(FakeClaude(bad))


def test_a_reply_cut_off_at_the_token_limit_is_an_error():
    with pytest.raises(film_prompts.PromptError, match="cut off"):
        write(FakeClaude(plan(4), stop_reason="max_tokens"))
