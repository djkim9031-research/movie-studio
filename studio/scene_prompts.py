"""The two prompts of a scene, written by Claude from the user's words: the still request for FLUX 3 Klein's prompt
upsampler and the motion prompt for LTX-2.5. Only text is sent: the references go by number and role, never pixels."""

import re

from pydantic import BaseModel

from studio.refs import Report


class PromptError(RuntimeError):
    pass


class ScenePrompts(BaseModel):
    still_request: str
    motion_prompt: str


SCHEMA = {"type": "object",
          "properties": {"still_request": {"type": "string"}, "motion_prompt": {"type": "string"}},
          "required": ["still_request", "motion_prompt"], "additionalProperties": False}

SYSTEM = """You write the two prompts of one shot for a personal short film, rendered by open models on a local GPU.

1. still_request: an instruction for the prompt upsampler of FLUX 3 Image Klein. The upsampler sees the reference \
images, which you never see, and expands your instruction into the full caption the image model draws from, together \
with the same references.
2. motion_prompt: the prompt for LTX-2.5 image-to-video. The still is the first frame of a 5-second clip with sound; \
the prompt says what moves, what the camera does and what is heard.

You get the user's description of the scene, the orientation and the numbered references. Never guess how a person \
looks (age, gender, ethnicity, hair, build): call them "the person" and let the references carry their looks.

Still request:
- Open with "Create a photograph of the person shown in images 1-N (all N show the same person)", or "in image 1" \
for one photo, then the scene. Name every other reference by number and say what to take from it ("Saturn from image \
7").
- Ask to keep the person's exact face, identity, skin tone and hair.
- Frame the face large and clearly visible: a medium close-up or medium shot, the face toward the camera or at three \
quarters, nothing over it. When the scene puts something on the head, keep the face unobstructed (a helmet gets a \
clear visor with the sun visor raised and all hair inside).
- Dress and equip the person as the scene really requires, with the details that make it believable (gloves with a \
spacesuit, a harness on a cliff).
- One key light with a stated direction. When a reference shows a lit object that appears in the scene, light the \
person from the same direction as that object is lit in its image.
- Objects that matter stay whole inside the frame (a planet, a landmark), at a stated position and size.
- Ask for a real photograph: a plausible camera position and lens, natural colors, fine skin and fabric detail. No \
quality words ("8k", "masterpiece", "ultra-detailed"), no art style unless the user asks for one.
- No other people, text or logos unless asked.
- One paragraph of English prose, 120 to 220 words.

Motion prompt:
- One paragraph in the present tense, 100 to 180 words, for 5 seconds: the setting and the person as they appear in \
the still (clothes and place, no face description), then the motion in order, then the camera, then the sound.
- Motion is small and slow. A wish for movement ("float around", "dance") becomes a gentle version of it that keeps \
the person in frame and the face toward the camera; no turn or roll that hides the face.
- The first frame's expression holds for the whole shot; say so ("holding the same thin, closed-mouth smile as in \
the first frame"). The eyes stay open, look into the camera and blink once; the mouth stays closed. Do not name any \
other facial movement (eyebrows, frowning, laughing): named movements get performed.
- No speech unless the user asks for spoken words; then quote the exact line.
- Hands only where the scene needs them; otherwise the arms rest or float loosely at the sides, mostly out of frame.
- The camera stays still and the framing stays the same, unless the user asks for a camera move; then one slow move.
- Distant objects stay still; natural motion of the surroundings (water, leaves, dust) may go on.
- Sound: the close, quiet sound of the place (breathing inside a helmet, wind, waves), no music.

Example. Scene: "me floating in space with Saturn in the background, thin smile, float around". Images 1-6: six \
photos of the same person; image 7: a picture without people, file name "saturn cassini". Landscape.
still_request: "Create a photograph of the person shown in images 1-6 (all six show the same person) floating \
weightlessly in outer space, with Saturn from image 7 in the background. Keep the person's exact face, identity, skin \
tone and hair. They wear a modern white EVA spacesuit with small mission patches and a clear bubble helmet with the \
gold sun visor raised, so the face is fully visible and unobstructed, with all hair inside the helmet. Medium \
close-up from the chest up: the person on the left third of the frame, body tilted slightly as if drifting, face \
turned three-quarters toward the camera with a calm, relaxed expression and a slight smile. Saturn exactly as in \
image 7, its banded globe and tilted rings with the planet's shadow falling across the rings, sits in the right half \
of the background, farther away, about a third of the frame wide and entirely inside the frame. Harsh, direct \
sunlight from the same direction that lights Saturn in image 7 gives crisp highlights on the face and suit and deep \
shadows on the other side, with no studio fill light. Deep black space with a few faint stars. A real photograph \
taken by another astronaut with a 35 mm lens, natural colors, fine skin and fabric detail."
motion_prompt: "A shot in deep space. A person in a white EVA spacesuit and thick white spacesuit gloves floats \
weightlessly in zero gravity, seen from the chest up, the face visible through a clear bubble helmet with the gold \
sun visor raised, lit by harsh direct sunlight. They drift slowly around in a small, lazy loop within the frame, \
gently bobbing up and down, the body tilting by only a few degrees, the arms floating loosely down at the sides, \
mostly out of frame. The face stays turned toward the camera and fully visible through the helmet glass the whole \
time, relaxed and still, holding the same thin, closed-mouth smile as in the first frame; the eyes stay open, look \
into the camera and blink once. Small highlights glide across the curved helmet glass as they move. In the \
background, Saturn with its tilted rings hangs motionless against black space with a few faint stars. The camera \
stays still and the framing stays a medium close-up. The sound is close and quiet: slow breathing inside the helmet, \
the faint hum of the suit's life support and a soft crackle of radio static, with no speech."
"""

NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def _plain(name: str) -> str:
    return " ".join(re.sub(r"[^A-Za-z0-9]+", " ", name).split())[:40].strip()


def roles(report: Report) -> str:
    """The references by number, as the image model receives them."""
    people = report.people
    if people == 0:
        lines = ["There is no photo of a person among the references."]
    elif people == 1:
        lines = ["Image 1: one photo of the person (a square face crop)."]
    else:
        lines = [f"Images 1-{people}: {NUMBERS[people]} photos of the same person (square face crops)."]
    extras = [r for r in report.used if r.kind == "extra"]
    for number, ref in enumerate(extras, start=people + 1):
        stem = ref.source.rsplit(".", 1)[0]
        lines.append(f'Image {number}: a picture without people, file name "{_plain(stem)}".')
    return "\n".join(lines)


def write(claude, *, film: str, user_prompt: str, report: Report, orientation: str) -> ScenePrompts:
    shape = "landscape (16:9)" if orientation == "landscape" else "portrait (9:16)"
    text = (f"Scene from the user: {user_prompt}\nOrientation: {shape}\n"
            f"References, numbered as the image model receives them:\n{roles(report)}\n"
            "Write the still_request and the motion_prompt.")
    message = claude.create("scene_prompts", film=film, max_tokens=8000,
                            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                            messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                            output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    if message.stop_reason == "max_tokens":
        raise PromptError("the prompt reply was cut off at the token limit")
    reply = next((block.text for block in message.content if block.type == "text"), None)
    if reply is None:
        raise PromptError(f"no text in the prompt reply (stop reason {message.stop_reason})")
    return ScenePrompts.model_validate_json(reply)
