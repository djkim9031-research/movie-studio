"""The shot plan of a short film, written by Claude from the user's words: per shot a still request for FLUX 3 Klein's
prompt upsampler, a motion prompt for Wan 2.2 and a sound prompt for PrismAudio. Only text is sent: the references go
by number and role, never pixels."""

import re
from typing import Annotated

from pydantic import BaseModel, StringConstraints

from studio.refs import Report
from studio.scene_prompts import NUMBERS, PromptError, _plain

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Shot(BaseModel):
    view: Text
    still_request: Text
    motion_prompt: Text
    sound_prompt: Text
    foley: bool
    end_state: Text
    event: str = ""
    event_box: list[float] = []
    event_pictures: list[int] = []


class View(BaseModel):
    id: Text
    request: Text


class Bible(BaseModel):
    wardrobe: Text
    place: Text
    light: Text
    bed: Text
    views: list[View]


class FilmPlan(BaseModel):
    title: Text
    bible: Bible
    set_request: Text
    sheet_request: Text
    shots: list[Shot]


TEXT = {"type": "string"}
SHOT = {"type": "object",
        "properties": {"view": TEXT, "still_request": TEXT, "motion_prompt": TEXT, "sound_prompt": TEXT,
                       "foley": {"type": "boolean"}, "end_state": TEXT, "event": TEXT,
                       "event_box": {"type": "array", "items": {"type": "number"}},
                       "event_pictures": {"type": "array", "items": {"type": "integer"}}},
        "required": ["view", "still_request", "motion_prompt", "sound_prompt", "foley", "end_state", "event",
                     "event_box", "event_pictures"], "additionalProperties": False}
VIEW = {"type": "object", "properties": {"id": TEXT, "request": TEXT}, "required": ["id", "request"],
        "additionalProperties": False}
BIBLE = {"type": "object",
         "properties": {"wardrobe": TEXT, "place": TEXT, "light": TEXT, "bed": TEXT,
                        "views": {"type": "array", "items": VIEW}},
         "required": ["wardrobe", "place", "light", "bed", "views"], "additionalProperties": False}
SCHEMA = {"type": "object",
          "properties": {"title": TEXT, "bible": BIBLE, "set_request": TEXT, "sheet_request": TEXT,
                         "shots": {"type": "array", "items": SHOT}},
          "required": ["title", "bible", "set_request", "sheet_request", "shots"], "additionalProperties": False}

SYSTEM = """You plan a short film for a personal project, rendered by open models on a local GPU: a few shots of one \
scene. The look comes first: the set (one wide picture of the place), a character sheet of the person, and a picture \
of the place from each camera direction the shots need are drawn before any shot, and every shot's still is drawn \
from them, so the place and the person look the same in every shot. Each still then becomes a silent clip; an event \
seen in a shot is painted into its place on the still first; the sound is made afterwards.

You write:
1. bible: the film's fixed words, copied verbatim into the prompts as the rules below say.
2. set_request, sheet_request and one request per view: instructions for the prompt upsampler of FLUX 3 Image Klein. \
The upsampler sees the reference images, which you never see, and expands each instruction into the full caption the \
image model draws from, together with the same references.
3. per shot, still_request: the same kind of instruction for the shot's still, drawn from the person's photos, the \
character sheet and the shot's view. Each still is drawn on its own: the image model never sees the other shots, and \
whatever a still request leaves open is invented anew.
4. per shot, motion_prompt: the prompt for Wan 2.2 image-to-video. The still is the first frame of the clip; the \
prompt says what moves and what the camera does. Wan makes no sound.
5. per shot, sound_prompt: the prompt for PrismAudio, which watches the finished clip and makes the shot's own sound \
when foley is true.

You get the user's description, the number of shots and their length, the orientation and the numbered references. \
Never guess how a person looks (age, gender, ethnicity, hair, build): call them "the person" and let the references \
carry their looks.

The bible:
- wardrobe: the person's clothes in one plain phrase (garment, color, material, shoes). Copy it verbatim into the \
sheet request and into every still request and motion prompt that shows the person.
- place: the place and what is in view, in one sentence. Copy it verbatim into the set request, every view request, \
every still request and every motion prompt.
- light: time of day, sun direction and sky, in one phrase. Copy it verbatim into the set request, every view request \
and every still request.
- bed: a sound prompt for the place's ambience under the whole film, one sentence, only what is heard.
- views: one to four camera directions the shots need (for example toward the harbor, the reverse toward the \
building, along the railing), each with a short id and a request: a still request for the place seen from that \
direction, drawn from the set picture alone ("Create a photograph of the place shown in image 1, seen from ..."), \
with no people and none of the story's events.

Shots:
- Tell the user's scene as a short sequence with a beginning, a middle and an end: each shot a new moment or a new \
angle of the same scene, in order.
- Stage each shot from the story, as a film crew would. When something happens away from the person (in the sky, on \
the horizon, down the street), the person looks at it, not at the camera: show the event in a wide shot or over the \
person's shoulder, the person seen from behind or in profile, and their reaction in a closer shot at three quarters, \
the eyes on the event off-frame. The person never looks into the lens unless the user asks.
- Each shot names the view it is seen in (view, one of the bible's ids).
- end_state: one sentence on how the shot ends (where the person is and looks, what they hold, the event's stage); \
the next shot begins there.
- Distance and scale: something far away stays small in the frame, on the horizon; its size shows in what it dwarfs \
(ships, the far shore, the skyline) and in how high it climbs, not in how much of the frame it fills. Something that \
comes closer grows from shot to shot: far and small first, larger only later.
- Events are not drawn with the shot. The shot's still is drawn without the event, then the event is painted into its \
box from the event pictures. So the still request never describes the event, only its light on the scene (a glow, a \
haze) once it has happened. event says what is seen, where and how big, with its width as a percentage of the frame \
("about 4% of the frame width"); event_box is that area as fractions of the frame [left, top, right, bottom]; \
event_pictures lists the numbers of the event pictures to paint from. A shot without an event leaves event empty and \
event_box and event_pictures as empty lists. A far event stays under 10% of the frame width; spectacle that would need \
more than 30% (a wave reaching the city, a collapse) is implied instead: the person's reaction, the light, the sound.
- foley: true when the shot shows or implies its own sound (a mug touching the railing, the boom reaching the \
balcony); otherwise the bed carries the shot.
- title: two to six words.

The look:
- set_request: one wide photograph of the place at the scene's time of day and light, with no people in it, from \
where most shots look: everything the shots show (floor, railing, furniture, the view to the horizon, landmarks) at \
its place and size, and none of the story's events. Open with "Create a photograph of the place shown in images \
1-N" (or "in image 1") and say what to take from those pictures. One paragraph, 120 to 220 words.
- sheet_request: a character sheet of the person in the film's clothes: three full-body views side by side on a \
plain light gray background (front, three quarters, back), standing straight with the arms relaxed, a neutral \
expression with the mouth closed, soft even light. Open with "Create a character sheet of the person shown in images \
1-N (all N show the same person)" (or "in image 1") and ask to keep their exact face, identity, skin tone and hair. \
One paragraph, 80 to 160 words.

Still request:
- Open with "Create a photograph of the person shown in images 1-N (all N show the same person), dressed exactly as \
in the character sheet in image S, in the place shown in image V", with the numbers given for the shots (V is the \
shot's view). A shot without the person is drawn from its view alone, so it opens with "Create a photograph of \
the place shown in image 1".
- Keep the place exactly as in image V and say only what this moment changes: where the camera stands within the \
view, the light, the person.
- Ask to keep the person's exact face, identity, skin tone and hair.
- Frame for the shot's moment: in a wide or over-the-shoulder shot the person is small, seen from behind or in \
profile, looking at the event; in a reaction shot, a medium shot or medium close-up at three quarters, the face large \
and unobstructed (a helmet gets a clear visor, all hair inside), the eyes on the event off-frame.
- The expression is understated, as in life: calm, absorbed, puzzled, tense, worried, grave. Strong feeling shows in \
the body (the mug lowered, a step back, a hand tight on the railing), never in a strong expression word or in eyes, \
brows or mouth; say the mouth is closed.
- One key light with a stated direction, the same as in the set picture unless the event changes it.
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
- A far event moves as something that size would, seen from far away: slowly. A painted-in flash brightens, smoke \
climbs, a wave line rises and nears, each staying near its box; the skyline, the shore and the rest of the view stay \
fixed.
- Hands only where the shot needs them; otherwise the arms rest.
- The camera stays still and the framing stays the same, unless the user asks for a camera move; then one slow move.

Sound prompt:
- One to three sentences, 30 to 70 words: "Generate the sound of ...", then the sounds in the shot. The bed already \
carries the place's ambience, so the shot's own sounds come first.
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


def _stem(ref) -> str:
    return _plain(ref.source.rsplit(".", 1)[0])


def look_roles(report: Report, events: frozenset[str] | set[str] = frozenset()) -> str:
    """The references of each picture the film draws, numbered as the image model receives them, and the event
    pictures that events are painted from. Other pictures of people are not used."""
    people = report.people
    places = [r for r in report.used if r.kind == "extra" and not r.faces_blurred and r.source not in events]
    shown = [r for r in report.used if r.kind == "extra" and r.source in events]
    named = "; ".join(f'image {i}: a picture of the place without people, file name "{_stem(r)}"'
                      for i, r in enumerate(places, start=1))
    photos = (f"images 1-{people}: {NUMBERS[people]} photos of the same person (square face crops)" if people > 1
              else "image 1: one photo of the person (a square face crop)")
    mine = f"images 1-{people}: the person's photos" if people > 1 else "image 1: the person's photo"
    event = "; ".join(f'{i}: file name "{_stem(r)}"' for i, r in enumerate(shown, start=1)) or "none"
    return (f"References, numbered as the image model receives them.\nSet picture: {named}.\n"
            f"Character sheet: {photos}.\nEach view: image 1: the set picture.\n"
            f"Each shot: {mine}; image {people + 1}: the character sheet; image {people + 2}: the shot's view.\n"
            "A shot without the person: image 1: the shot's view.\n"
            f"Event pictures: {event}.")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())


def _strong(text: str, asked: set[str], lowered: str) -> list[str]:
    found = [w for w in dict.fromkeys(_words(text)) if w in STRONG and w not in asked]
    return found + [m.group(0) for m in STRONG_FACE.finditer(text.lower()) if m.group(0) not in lowered]


def _missing(text: str, bible: Bible, names: tuple[str, ...]) -> list[str]:
    return [n for n in names if getattr(bible, n).lower() not in text.lower()]


def _event_problems(n: int, shot: Shot, events: int) -> list[str]:
    if not shot.event:
        return [f"shot {n} has an event box or pictures but no event"] if shot.event_box or shot.event_pictures else []
    found = []
    if shot.event.lower() in shot.still_request.lower():
        found.append(f"shot {n} still request describes the event; the still is drawn without it")
    box = shot.event_box
    if not (len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        found.append(f"shot {n} event box {box} is not [left, top, right, bottom] as fractions of the frame")
    else:
        width = round((box[2] - box[0]) * 100)
        stated = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", shot.event)]
        if width > 30:
            found.append(f"shot {n} event is {width}% of the frame width; show it at most 30% or imply it")
        if not stated:
            found.append(f"shot {n} event does not give its width as a percentage of the frame")
        elif abs(stated[0] - width) > 2:
            found.append(f"shot {n} event says {stated[0]:g}% but its box is {width}% of the frame width")
    if any(not 1 <= p <= events for p in shot.event_pictures):
        found.append(f"shot {n} event pictures {shot.event_pictures}: there {'is' if events == 1 else 'are'} "
                     f"{events}")
    return found


def problems(made: FilmPlan, user_prompt: str, events: int = 0) -> list[str]:
    """Where a plan breaks the rules the code can check, one line per shot and prompt."""
    asked, lowered = set(_words(user_prompt)), user_prompt.lower()
    found = []
    b = made.bible
    ids = [v.id for v in b.views]
    if not 1 <= len(ids) <= 4:
        found.append(f"the bible has {len(ids)} views; give 1 to 4")
    if len(set(ids)) != len(ids):
        found.append("the bible repeats a view id")
    for label, text, names in (("set request", made.set_request, ("place", "light")),
                               ("sheet request", made.sheet_request, ("wardrobe",)),
                               *((f'view "{v.id}" request', v.request, ("place", "light")) for v in b.views)):
        missing = _missing(text, b, names)
        if missing:
            found.append(f"{label} misses the locked {', '.join(missing)}")
    strong = _strong(made.sheet_request, asked, lowered)
    if strong:
        found.append(f"sheet request: strong expression ({', '.join(strong)}); keep the face neutral")
    named = [w for w in dict.fromkeys(_words(b.bed)) if w in NEGATIONS | VOICES and w not in asked]
    if named:
        found.append(f"bed names sounds to leave out ({', '.join(named)}); name only what is heard")
    for n, shot in enumerate(made.shots, start=1):
        person = shot.still_request.strip().lower().startswith("create a photograph of the person")
        wardrobe = ("wardrobe",) if person else ()
        for label, text, names in (("still request", shot.still_request, (*wardrobe, "place", "light")),
                                   ("motion prompt", shot.motion_prompt, (*wardrobe, "place"))):
            missing = _missing(text, b, names)
            if missing:
                found.append(f"shot {n} {label} misses the locked {', '.join(missing)}")
            gaze = GAZE.search(text.lower())
            if gaze and not asked & {"camera", "lens"}:
                found.append(f'shot {n} {label}: the person looks at the camera ("{gaze.group(0)}"); they look at '
                             "what happens")
            strong = _strong(text, asked, lowered)
            if strong:
                found.append(f"shot {n} {label}: strong expression ({', '.join(strong)}); keep it understated and "
                             "show it in the body")
        if shot.view not in ids:
            found.append(f'shot {n} names view "{shot.view}", which the bible does not have ({", ".join(ids)})')
        found += _event_problems(n, shot, events)
        named = [w for w in dict.fromkeys(_words(shot.sound_prompt)) if w in NEGATIONS | VOICES and w not in asked]
        if named:
            found.append(f"shot {n} sound prompt names sounds to leave out ({', '.join(named)}); name only what is "
                         "heard")
    return found


def _ask(claude, film: str, text: str, shots: int) -> FilmPlan:
    message = claude.create("film_prompts", film=film, max_tokens=64000,
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
          seconds: int, events: frozenset[str] | set[str] = frozenset()) -> FilmPlan:
    """The plan, sent back with what it breaks at most REPAIRS times."""
    shape = "landscape (16:9)" if orientation == "landscape" else "portrait (9:16)"
    text = (f"Scene from the user: {user_prompt}\nFilm: {shots} shots of {seconds} seconds each\n"
            f"Orientation: {shape}\n{look_roles(report, events)}\n"
            "Write the title, the bible, the set request, the sheet request and the shots.")
    made = _ask(claude, film, text, shots)
    for _ in range(REPAIRS):
        found = problems(made, user_prompt, len(events))
        if not found:
            return made
        made = _ask(claude, film, text + REPAIR.format(problems="\n".join(f"- {p}" for p in found),
                                                       plan=made.model_dump_json(indent=1)), shots)
    found = problems(made, user_prompt, len(events))
    if found:
        raise PromptError(f"the shot plan still breaks the rules after {REPAIRS} repairs: " + "; ".join(found))
    return made
