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


WARDROBE = "a light blue hanbok, a black gat and white socks with black shoes"
PLACE = "the stone courtyard in front of the palace gate"
LIGHT = "soft morning light from the left under a pale sky"


def shot(n):
    return {"view": "gate",
            "still_request": f"Create a photograph of the person shown in images 1-6 (all 6 show the same person), "
                             f"dressed exactly as in the character sheet in image 7, in the place shown in image 8: "
                             f"{PLACE}, {LIGHT}. The person wears {WARDROBE} ... shot {n}",
            "motion_prompt": f"In {PLACE}, the person in {WARDROBE} rides slowly toward the gate ... shot {n}",
            "sound_prompt": f"Generate the sound of hooves on stone ... shot {n}",
            "foley": True, "end_state": f"The rider halts at the gate ... shot {n}",
            "event": "", "event_box": [], "event_pictures": []}


def plan(count):
    return {"title": "Ride at the Palace",
            "bible": {"wardrobe": WARDROBE, "place": PLACE, "light": LIGHT,
                      "bed": "Generate the sound of a quiet palace courtyard: a light wind and distant birds.",
                      "views": [{"id": "gate", "request": f"Create a photograph of the place shown in image 1, "
                                                          f"looking at the gate: {PLACE}, {LIGHT}, no people"}]},
            "set_request": f"Create a photograph of the place shown in image 1 ... {PLACE}, {LIGHT}, no people",
            "sheet_request": f"Create a character sheet of the person shown in images 1-6 ... {WARDROBE}",
            "shots": [shot(n) for n in range(1, count + 1)]}


def locked(text):
    """`text` with the plan's locked words, so a test can change one rule at a time."""
    return f"{text}, {PLACE}, {LIGHT}, wearing {WARDROBE}"


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
    assert "me riding a horse in front of the palace" in sent and "4 shots of 5 seconds" in sent
    assert made.set_request.startswith("Create a photograph of the place")
    assert made.sheet_request.startswith("Create a character sheet")


def test_the_references_are_numbered_for_the_set_the_sheet_the_views_and_the_shots():
    claude = FakeClaude(plan(4))
    film_prompts.write(claude, film="film-ride", user_prompt="me riding a horse in front of the palace",
                       report=report(extras=("palace_gate", "crowd")), orientation="landscape", shots=4, seconds=5)
    text = sent(claude, 0)
    assert 'Set picture: image 1: a picture of the place without people, file name "palace gate".' in text
    assert "Character sheet: images 1-6: six photos of the same person (square face crops)." in text
    assert "Each view: image 1: the set picture." in text
    assert "Each shot: images 1-6: the person's photos; image 7: the character sheet; image 8: the shot's view." in text
    assert "A shot without the person: image 1: the shot's view." in text
    assert "Event pictures: none." in text
    assert "crowd" not in text  # a picture of other people is not used


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
    claude = FakeClaude([broken(3, "still_request", locked("Create a photograph ... the person looking straight "
                                                            "into the camera, the palace behind")), plan(4)])
    write(claude)
    assert "shot 3 still request" in sent(claude, 1) and "looking straight into the camera" in sent(claude, 1)


def test_a_person_facing_the_camera_goes_back():
    claude = FakeClaude([broken(1, "motion_prompt", locked("The person stands with the face toward the camera "
                                                           "while the horse walks")), plan(4)])
    write(claude)
    assert "shot 1 motion prompt" in sent(claude, 1) and "face toward the camera" in sent(claude, 1)


def test_a_strong_expression_goes_back():
    claude = FakeClaude([broken(4, "still_request", locked("Create a photograph ... with a shocked look, eyes "
                                                           "wide and the mouth open")), plan(4)])
    write(claude)
    repair = sent(claude, 1)
    assert "shot 4 still request" in repair and all(w in repair for w in ("shocked", "eyes wide", "mouth open"))


def test_a_shockwave_far_shots_and_gull_cries_pass():
    fine = broken(3, "still_request", locked("Create a photograph ... the glow of a dome-shaped shockwave on "
                                             "the sky, the person turned three quarters, eyes on the horizon, a tense "
                                             "face"))
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


def test_the_sheet_may_face_the_camera_but_keeps_a_neutral_face():
    bad = copy.deepcopy(plan(4))
    bad["sheet_request"] = f"Create a character sheet ... the front view facing the camera, a shocked face, {WARDROBE}"
    claude = FakeClaude([bad, plan(4)])
    write(claude)
    repair = sent(claude, 1)
    assert "sheet request: strong expression (shocked)" in repair and "looks at the camera" not in repair


def test_event_pictures_are_numbered_for_painting_events():
    claude = FakeClaude(plan(4))
    film_prompts.write(claude, film="film-ride", user_prompt="me riding a horse in front of the palace",
                       report=report(extras=("palace_gate", "fireworks", "smoke")), orientation="landscape", shots=4,
                       seconds=5, events={"fireworks.png", "smoke.png"})
    text = sent(claude, 0)
    assert 'Set picture: image 1: a picture of the place without people, file name "palace gate".' in text
    assert 'Event pictures: 1: file name "fireworks"; 2: file name "smoke".' in text


def test_locked_words_must_be_copied_where_the_rules_say():
    bad = copy.deepcopy(plan(4))
    bad["shots"][1]["still_request"] = bad["shots"][1]["still_request"].replace(LIGHT, "warm evening light")
    bad["shots"][2]["motion_prompt"] = "The person rides slowly toward the gate"
    bad["sheet_request"] = "Create a character sheet of the person shown in images 1-6 in a blue robe"
    bad["bible"]["views"][0]["request"] = "Create a photograph of the place shown in image 1 looking at the gate"
    claude = FakeClaude([bad, plan(4)])
    write(claude)
    repair = sent(claude, 1)
    assert "shot 2 still request misses the locked light" in repair
    assert "shot 3 motion prompt misses the locked wardrobe, place" in repair
    assert "sheet request misses the locked wardrobe" in repair
    assert 'view "gate" request misses the locked place, light' in repair


def test_every_shot_names_a_view_of_the_bible():
    claude = FakeClaude([broken(3, "view", "reverse"), plan(4)])
    write(claude)
    assert 'shot 3 names view "reverse", which the bible does not have (gate)' in sent(claude, 1)


def test_an_event_is_painted_into_its_box_and_never_drawn_with_the_still():
    bad = copy.deepcopy(plan(4))
    phrase = "a small orange flash on the horizon line in the right third, about 4% of the frame width"
    bad["shots"][1].update(event=phrase, event_box=[0.70, 0.30, 0.74, 0.36], event_pictures=[1])
    bad["shots"][1]["still_request"] += f", {phrase}"
    bad["shots"][2].update(event="a towering wave across the bay, about 60% of the frame width",
                           event_box=[0.2, 0.3, 0.8, 0.6], event_pictures=[1])
    bad["shots"][3].update(event="a column of smoke, about 20% of the frame width", event_box=[0.6, 0.1, 0.65, 0.4],
                           event_pictures=[3])
    claude = FakeClaude([bad, plan(4)])
    film_prompts.write(claude, film="film-ride", user_prompt="me riding a horse in front of the palace",
                       report=report(extras=("palace_gate", "fireworks")), orientation="landscape", shots=4,
                       seconds=5, events={"fireworks.png"})
    repair = sent(claude, 1)
    assert "shot 2 still request describes the event; the still is drawn without it" in repair
    assert "shot 3 event is 60% of the frame width; show it at most 30% or imply it" in repair
    assert "shot 4 event says 20% but its box is 5% of the frame width" in repair
    assert "shot 4 event pictures [3]: there is 1" in repair


def test_the_bed_is_a_sound_prompt_too():
    bad = copy.deepcopy(plan(4))
    bad["bible"]["bed"] = "Generate the sound of a courtyard. Avoid music and voices."
    claude = FakeClaude([bad, plan(4)])
    write(claude)
    assert "bed names sounds to leave out (avoid, music, voices)" in sent(claude, 1)
