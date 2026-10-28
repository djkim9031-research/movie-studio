"""The shot plan of a short film, written by Claude from the user's words: per shot a still request for FLUX 3 Klein's
prompt upsampler, a motion prompt for Wan 2.2 and a sound prompt for PrismAudio. Only text is sent: the references go
by number and role, never pixels."""

import re
from typing import Annotated

from pydantic import BaseModel, StringConstraints

from studio.refs import Report
from studio.scene_prompts import PromptError, roles

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
with the same references. Each still is drawn on its own: the image model never sees the other shots, and whatever a \
still request leaves open is invented anew.
2. motion_prompt: the prompt for Wan 2.2 image-to-video. The still is the first frame of the clip; the prompt says \
what moves and what the camera does. Wan makes no sound.
3. sound_prompt: the prompt for PrismAudio, which watches the finished clip and makes its sound.

You get the user's description, the number of shots and their length, the orientation and the numbered references. \
Never guess how a person looks (age, gender, ethnicity, hair, build): call them "the person" and let the references \
carry their looks.

Shots:
- Tell the user's scene as a short sequence with a beginning, a middle and an end: each shot a new moment or a new \
angle of the same scene, in order.
- Stage each shot from the story, as a film crew would. When something happens away from the person (in the sky, on \
the horizon, down the street), the person looks at it, not at the camera: show the event in a wide shot or over the \
person's shoulder, the person seen from behind or in profile, and their reaction in a closer shot at three quarters, \
the eyes on the event off-frame. The person never looks into the lens unless the user asks.
- Keep the person, the clothes, the place and the light the same in every shot unless the user's description \
changes them, written in the same words in every still request: the clothes named plainly (garment, color, \
material), the view described the same way each time.
- Distance and scale: something far away stays small in the frame, on the horizon; its size shows in what it dwarfs \
(ships, the far shore, the skyline) and in how high it climbs, not in how much of the frame it fills. Something that \
comes closer grows from shot to shot: far and small first, larger only later.
- title: two to six words.

Still request:
- Open with "Create a photograph of the person shown in images 1-N (all N show the same person)", or "in image 1" \
for one photo, then the scene. Name every other reference by number and say what to take from it ("the balcony from \
images 8-10").
- Ask to keep the person's exact face, identity, skin tone and hair. A picture of other people with blurred faces is \
there for the place or a prop: take only that from it, never a body, a face or clothes.
- Frame for the shot's moment: in a wide or over-the-shoulder shot the person is small, seen from behind or in \
profile, looking at the event; in a reaction shot, a medium shot or medium close-up at three quarters, the face large \
and unobstructed (a helmet gets a clear visor, all hair inside), the eyes on the event off-frame.
- The expression is understated, as in life: calm, absorbed, puzzled, tense, worried, grave. Strong feeling shows in \
the body (the mug lowered, a step back, a hand tight on the railing), never in a strong expression word or in eyes, \
brows or mouth; say the mouth is closed.
- Dress and equip the person as the scene requires, in plain words, the same in every shot.
- One key light with a stated direction. When a reference shows a lit object that appears in the scene, light the \
person from the same direction as that object is lit in its image.
- Far things stay small at a stated place (on the horizon, in the right third); near things that matter stay whole \
inside the frame.
- Ask for a real photograph: a plausible camera position and lens, natural colors, fine skin and fabric detail. No \
quality words ("8k", "masterpiece", "ultra-detailed"), no art style unless the user asks for one.
- No other people, text or logos unless asked.
- One paragraph of English prose, 120 to 220 words.

Motion prompt:
- One paragraph in the present tense, 60 to 120 words, for the shot's length: the setting and the person as they \
appear in the still (clothes and place, no face description), then the motion in order, then the camera. No sound.
- Motion is slow and believable and follows the story: the person may turn toward the event, lower the mug, step \
back or grip the railing. In a reaction shot the head moves little.
- The face stays understated and changes slowly. Do not name facial movements: named movements get performed. The \
mouth stays closed.
- A far event moves as something that size would, seen from far away: slowly. A flash brightens on the horizon, \
smoke climbs, a wave line rises and nears; the skyline, the shore and the rest of the view stay fixed.
- Hands only where the shot needs them; otherwise the arms rest.
- The camera stays still and the framing stays the same, unless the user asks for a camera move; then one slow move.

Sound prompt:
- One to three sentences, 30 to 70 words: "Generate the sound of ...", then the sounds in the shot.
- Only sounds the shot shows or implies, the nearest and loudest first, then the ambience of the place. Far events \
sound far: low, rolling and muffled rather than sharp.
- Name only what is heard. Never name a sound to exclude it, with "avoid", "no" or "without": the sound model takes \
every sound it reads as a request. Music and voices of any kind (speech, singing, shouts) stay out by not being \
named, unless the user asks for them.
"""

# What the code checks in a plan. A negation still names the sound it excludes, and PrismAudio makes what it reads;
# BFL's upsampler turns a strong expression word into wide eyes, raised brows and an open mouth. Words from the
# user's own description pass.
NEGATIONS = {"avoid", "avoids", "avoiding", "without", "no", "not", "never", "except"}
VOICES = {"speech", "speak", "speaks", "speaking", "voice", "voices", "vocal", "vocals", "talk", "talks", "talking",
          "dialogue", "conversation", "chatter", "sing", "sings", "singing", "song", "songs", "music", "musical",
          "melody", "shout", "shouts", "shouting", "scream", "screams", "screaming", "yell", "yells", "yelling",
          "whisper", "whispers", "whispering", "laugh", "laughs", "laughing", "laughter"}
STRONG = {"shocked", "alarmed", "frightened", "terrified", "horrified", "panicked", "panicking", "panic", "scared",
          "screaming", "gasping", "gasps", "wide-eyed", "agape"}
STRONG_FACE = re.compile(r"\b(eyes (go |are |grow )?wide|wide(ned)? eyes|eyes widen\w*|raised (eye)?brows|"
                         r"mouth (falls |drops |hangs )?open|open(s|ing)? (the |his |her |their )?mouth|open mouth)\b")
GAZE = re.compile(r"\b(look(s|ing)?|gaz(e|es|ing)|star(e|es|ing)|glanc(e|es|ing)|eyes)\b[^.;]{0,30}?"
                  r"\b(at|into|toward|towards|to)( the)? (camera|lens|viewer)\b"
                  r"|\bfac(e|es|ing)( straight)?( at| toward| towards)?( the)? (camera|lens|viewer)\b|\beye contact\b")
REPAIRS = 2
REPAIR = """

A plan you wrote for this film breaks these rules:
{problems}
Write the whole plan again: fix these and keep everything else. The plan:
{plan}"""


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())


def problems(made: FilmPlan, user_prompt: str) -> list[str]:
    """Where a plan breaks the rules the code can check, one line per shot and prompt."""
    asked, lowered = set(_words(user_prompt)), user_prompt.lower()
    found = []
    for n, shot in enumerate(made.shots, start=1):
        for label, text in (("still request", shot.still_request), ("motion prompt", shot.motion_prompt)):
            gaze = GAZE.search(text.lower())
            if gaze and not asked & {"camera", "lens"}:
                found.append(f'shot {n} {label}: the person looks at the camera ("{gaze.group(0)}"); they look at '
                             "what happens")
            strong = [w for w in dict.fromkeys(_words(text)) if w in STRONG and w not in asked]
            strong += [m.group(0) for m in STRONG_FACE.finditer(text.lower()) if m.group(0) not in lowered]
            if strong:
                found.append(f"shot {n} {label}: strong expression ({', '.join(strong)}); keep it understated and "
                             "show it in the body")
        named = [w for w in dict.fromkeys(_words(shot.sound_prompt)) if w in NEGATIONS | VOICES and w not in asked]
        if named:
            found.append(f"shot {n} sound prompt names sounds to leave out ({', '.join(named)}); name only what is "
                         "heard")
    return found


def _ask(claude, film: str, text: str, shots: int) -> FilmPlan:
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


def write(claude, *, film: str, user_prompt: str, report: Report, orientation: str, shots: int,
          seconds: int) -> FilmPlan:
    """The plan, sent back with what it breaks at most REPAIRS times."""
    shape = "landscape (16:9)" if orientation == "landscape" else "portrait (9:16)"
    text = (f"Scene from the user: {user_prompt}\nFilm: {shots} shots of {seconds} seconds each\n"
            f"Orientation: {shape}\nReferences, numbered as the image model receives them:\n{roles(report)}\n"
            "Write the title and the shots.")
    made = _ask(claude, film, text, shots)
    for _ in range(REPAIRS):
        found = problems(made, user_prompt)
        if not found:
            return made
        made = _ask(claude, film, text + REPAIR.format(problems="\n".join(f"- {p}" for p in found),
                                                       plan=made.model_dump_json(indent=1)), shots)
    found = problems(made, user_prompt)
    if found:
        raise PromptError(f"the shot plan still breaks the rules after {REPAIRS} repairs: " + "; ".join(found))
    return made
