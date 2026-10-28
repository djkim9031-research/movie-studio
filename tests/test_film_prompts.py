import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from studio import film_prompts, scene_prompts
from tests.test_scene_prompts import report


class FakeClaude:
    """Replies with `reply`, or with the next of a list of replies on each call."""

    def __init__(self, reply, stop_reason="end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, route, *, film, **params):
        self.calls.append((route, film, params))
        reply = self.reply[len(self.calls) - 1] if isinstance(self.reply, list) else self.reply
        return SimpleNamespace(stop_reason=self.stop_reason, content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text=json.dumps(reply))])


def shot(n):
    return {"still_request": f"Create a photograph of the person shown in images 1-6 ... shot {n}",
            "motion_prompt": f"A courtyard in front of a palace ... shot {n}",
            "sound_prompt": f"Generate the sound of hooves on stone ... shot {n}"}


def plan(count):
    return {"title": "Ride at the Palace", "shots": [shot(n) for n in range(1, count + 1)]}


def write(claude, shots=4, user_prompt="me riding a horse in front of the palace"):
    return film_prompts.write(claude, film="film-ride", user_prompt=user_prompt, report=report(),
                              orientation="landscape", shots=shots, seconds=5)


def broken(n, key, text):
    made = copy.deepcopy(plan(4))
    made["shots"][n - 1][key] = text
    return made


def sent(claude, call):
    return claude.calls[call][2]["messages"][0]["content"][0]["text"]


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


def test_a_sound_prompt_that_names_sounds_to_leave_out_goes_back_with_the_plan():
    bad = broken(2, "sound_prompt", "Generate the sound of a low roar over the harbor. Avoid music, voices and speech.")
    claude = FakeClaude([bad, plan(4)])
    made = write(claude)
    assert made.shots[1].sound_prompt == plan(4)["shots"][1]["sound_prompt"] and len(claude.calls) == 2
    repair = sent(claude, 1)
    assert "shot 2 sound prompt" in repair and all(w in repair for w in ("avoid", "music", "voices", "speech"))
    assert "Avoid music, voices and speech." in repair and "me riding a horse in front of the palace" in repair


def test_a_person_looking_into_the_camera_goes_back():
    claude = FakeClaude([broken(3, "still_request", "Create a photograph ... the person looking straight into the "
                                                     "camera, the palace behind"), plan(4)])
    write(claude)
    assert "shot 3 still request" in sent(claude, 1) and "looking straight into the camera" in sent(claude, 1)


def test_a_person_facing_the_camera_goes_back():
    claude = FakeClaude([broken(1, "motion_prompt", "The person stands with the face toward the camera while the "
                                                    "horse walks"), plan(4)])
    write(claude)
    assert "shot 1 motion prompt" in sent(claude, 1) and "face toward the camera" in sent(claude, 1)


def test_a_strong_expression_goes_back():
    claude = FakeClaude([broken(4, "still_request", "Create a photograph ... with a shocked look, eyes wide and the "
                                                    "mouth open"), plan(4)])
    write(claude)
    repair = sent(claude, 1)
    assert "shot 4 still request" in repair and all(w in repair for w in ("shocked", "eyes wide", "mouth open"))


def test_a_shockwave_far_shots_and_gull_cries_pass():
    fine = broken(3, "still_request", "Create a photograph ... a dome-shaped shockwave on the horizon, the person "
                                      "turned three quarters, eyes on the horizon, a tense face")
    fine["shots"][3]["sound_prompt"] = "Generate the sound of distant gulls crying and the low hum of traffic."
    claude = FakeClaude(fine)
    write(claude)
    assert len(claude.calls) == 1


def test_words_the_user_asked_for_pass():
    fine = broken(2, "sound_prompt", "Generate the sound of a street musician playing music on the steps.")
    claude = FakeClaude(fine)
    write(claude, user_prompt="me riding past a street musician playing music on the palace steps")
    assert len(claude.calls) == 1


def test_a_plan_still_broken_after_two_repairs_is_an_error():
    bad = broken(2, "sound_prompt", "Generate the sound of hooves. No music.")
    claude = FakeClaude([bad, bad, bad])
    with pytest.raises(film_prompts.PromptError, match="after 2 repairs: shot 2 sound prompt"):
        write(claude)
    assert len(claude.calls) == 3


def test_the_film_has_its_own_rules_and_the_scene_flow_keeps_its_likeness_rule():
    assert scene_prompts.STILL_RULES not in film_prompts.SYSTEM
    assert "face toward the camera" not in film_prompts.SYSTEM
    assert "the face toward the camera or at three quarters" in scene_prompts.STILL_RULES
