"""The shot plan of a short film, written by Claude from the user's words: per shot a still request for FLUX 3 Klein's
prompt upsampler, a motion prompt for Wan 2.2 and a sound prompt for PrismAudio. Only text is sent: the references go
by number and role, never pixels."""

from typing import Annotated

from pydantic import BaseModel, StringConstraints

from studio.refs import Report
from studio.scene_prompts import STILL_RULES, PromptError, roles

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Shot(BaseModel):
    still_request: Text
    motion_prompt: Text
    sound_prompt: Text


class FilmPlan(BaseModel):
    title: Text
    shots: list[Shot]


SHOT = {"type": "object",
        "properties": {key: {"type": "string"} for key in ("still_request", "motion_prompt", "sound_prompt")},
        "required": ["still_request", "motion_prompt", "sound_prompt"], "additionalProperties": False}
SCHEMA = {"type": "object",
          "properties": {"title": {"type": "string"}, "shots": {"type": "array", "items": SHOT}},
          "required": ["title", "shots"], "additionalProperties": False}

SYSTEM = """You plan a short film for a personal project, rendered by open models on a local GPU: a few shots of one \
scene, each a still that becomes a silent clip, with its sound made afterwards from the finished clip.

For each shot you write three prompts:
1. still_request: an instruction for the prompt upsampler of FLUX 3 Image Klein. The upsampler sees the reference \
images, which you never see, and expands your instruction into the full caption the image model draws from, together \
with the same references. Each still is drawn on its own: the image model never sees the other shots.
2. motion_prompt: the prompt for Wan 2.2 image-to-video. The still is the first frame of the clip; the prompt says \
what moves and what the camera does. Wan makes no sound.
3. sound_prompt: the prompt for PrismAudio, which watches the finished clip and makes its sound.

You get the user's description, the number of shots and their length, the orientation and the numbered references. \
Never guess how a person looks (age, gender, ethnicity, hair, build): call them "the person" and let the references \
carry their looks.

Shots:
- Tell the user's scene as a short sequence with a beginning, a middle and an end: each shot a new moment or a new \
angle of the same scene, in order.
- Keep the person, the clothes, the place and the light the same in every shot unless the user's description \
changes them, and write them out in full in every still request.
- Vary the framing from shot to shot (medium shot, medium close-up; facing the camera, three quarters), always with \
the face large and visible.
- title: two to six words.

""" + STILL_RULES + """
Motion prompt:
- One paragraph in the present tense, 60 to 120 words, for the shot's length: the setting and the person as they \
appear in the still (clothes and place, no face description), then the motion in order, then the camera. No sound.
- Motion is small and slow. A wish for movement becomes a gentle version of it that keeps the person in frame and the \
face toward the camera; no turn or roll that hides the face.
- The first frame's expression holds for the whole shot; say so. The eyes stay open and blink once; the mouth stays \
closed. Do not name any other facial movement: named movements get performed.
- Hands only where the shot needs them; otherwise the arms rest.
- The camera stays still and the framing stays the same, unless the user asks for a camera move; then one slow move.
- Distant objects stay still; natural motion of the surroundings (water, leaves, dust, a horse's mane) may go on.

Sound prompt:
- One to three sentences, 30 to 70 words, written as instructions: "Generate the sound of ...", then what to \
include, then what to avoid.
- Only sounds the shot shows or implies, the nearest and loudest first, then the ambience of the place.
- No music and no voices of any kind (speech, singing, shouts) unless the user asks for them.
"""


def write(claude, *, film: str, user_prompt: str, report: Report, orientation: str, shots: int,
          seconds: int) -> FilmPlan:
    shape = "landscape (16:9)" if orientation == "landscape" else "portrait (9:16)"
    text = (f"Scene from the user: {user_prompt}\nFilm: {shots} shots of {seconds} seconds each\n"
            f"Orientation: {shape}\nReferences, numbered as the image model receives them:\n{roles(report)}\n"
            "Write the title and the shots.")
    message = claude.create("film_prompts", film=film, max_tokens=16000,
                            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                            messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                            output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    if message.stop_reason == "max_tokens":
        raise PromptError("the shot plan was cut off at the token limit")
    reply = next((block.text for block in message.content if block.type == "text"), None)
    if reply is None:
        raise PromptError(f"no text in the shot plan reply (stop reason {message.stop_reason})")
    made = FilmPlan.model_validate_json(reply)
    if len(made.shots) != shots:
        raise PromptError(f"the plan has {len(made.shots)} shots instead of {shots}")
    return made
