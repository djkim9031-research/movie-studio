# Decisions

## 2026-10-26

- First film: a 30-60 s pilot, then a 1-2 min short. English dialogue.
- Cast: real people, with their consent, from their photos, and fictional characters. Real voices are cloned from
  the people's own recordings.
- Open-weight models for picture and sound. The Claude API is the only paid service. This GPU first; rented GPUs
  later for big renders and LoRA training.
- Photos, voices and frames of real people never leave this machine. Claude sees text and frames of fictional
  characters; real-person shots are gated locally by face match, voice match and word error rate.
- Image and video models run in ComfyUI inside Docker, driven over its local HTTP API. Audio models run in their
  own Python environments.
- First picture comparison: LTX-2.5 against Wan 2.2. MiniMax H3 waits for a license.
- Build order: foundation; picture runtime; LTX-2.5 vs Wan 2.2; keyframes and likeness; voices; talking shots;
  sound, music and mix; the Claude director; the pilot; the 1-2 min film. Later: speed work, LoRA training,
  rented GPUs.
- Claude API spend is capped at $40 a month and $20 a film, enforced by the ledger before every call.
- FFmpeg is the BtbN n8.1 static build. n9.0 and newer builds need driver 610+ for NVENC; this machine has 580.
  The sha256 in config/studio.yaml was copied from that release's checksums.sha256 on 2026-10-26; the install
  trusts only the pinned value.

## 2026-10-26: picture runtime

- LTX-2.5 distilled int8 only; NVFP4 is compared in the next increment.
- ComfyUI v0.42.1 in a CUDA 13.0 cuDNN runtime image. A devel image comes with the speed work, when kernels are
  compiled.
- Packages come from `docker/comfyui/requirements.lock` with their hashes, frozen from the image that passed the
  first renders. A package change means a new lock and the renders again.
- `--offline`: ComfyUI's paid partner nodes are off and the web page cannot reach the internet.
- The container runs on an internal Docker network with no route out and no DNS. This machine reaches it at
  http://192.168.131.2:8188. Two limits remain: the container can reach services listening on this machine's side
  of that network (ssh, NFS, NoMachine; none relays traffic without a login), and ComfyUI has no login, so any
  account on this machine can open it while it runs.
- Models are mounted read-only. ComfyUI's input, output, temp and user folders stay writable so the web UI can take
  uploads; studio copies files in and out of them without following links.
- The client polls `/history/{prompt_id}` instead of a websocket: no missed events, but no live progress.
- The stock template's prompt enhancer is removed from the exported workflow in code; Claude writes the prompts.
  Its model, Gemma 4 E2B, is downloaded only so the template runs unchanged in the web UI.
- ComfyUI reuses a node's result while its inputs are unchanged. A clip taken whole from an earlier run counts as a
  failure; a repeat of the same picture, prompt and seed needs a new seed or a restart.

## 2026-10-26: scene flow

- One command pair turns photos, other pictures and a prompt into stills and then clips: `studio scene stills` and
  `studio scene clips`. Stills come first and the user picks which to animate: likeness varies by seed and cannot
  be scored automatically.
- Stills: FLUX 3 Image Klein distilled in BF16 with BFL's prompt upsampler, as BFL's multi-image workflow (16 steps,
  no guidance, references at 1 MP). The upsampler sees the references; Claude does not.
- References: the person in most of the pictures, found with YuNet and grouped with SFace (average linkage at
  cosine 0.363, never two faces of one picture), gives up to 6 face crops ranked by sharpness, size and angle;
  near-copies are skipped and other faces blurred. Pictures without people go in whole, padded to a square only for
  the upsampler, whose image batch crops everything to the first image's square. At most 10 references.
- Face recognition runs only on the input pictures. The FLUX license forbids biometric processing of the model and
  its outputs, so generated images are never scored for likeness.
- Claude writes the still request and the motion prompt from the user's words (route scene_prompts, structured
  output). It gets the prompt, the number of photos of the person and the file names of the other pictures, never
  an image. Its instructions hold what the first space scene taught: references by number, a large visible face,
  light matching the lit references, a real-photo look; in motion a still camera, small drift, the face toward the
  camera, the first frame's expression held, no speech, no named facial movements.
- Clips: LTX-2.5 distilled int8, 5 s at 1280x704, as make-video.
- Each run removes its copies from ComfyUI's input and output folders, without following links.
- The face models sit in `models/faces`, outside ComfyUI's model folders.
- Pictures come as a folder, files, or both: every picture directly in a folder (JPEG, PNG, WebP; at most 60) in
  name order, each once. Faces are found one picture at a time and only the pictures used are read again, so a full
  folder never sits in memory. HEIC is reported, not read: Pillow needs an extra library for it.

## 2026-10-26: identity program

- Target: a film of up to 5 minutes in which each real person looks the same from every angle. Work is measured
  first: a profile of the person from their real photos, and frame-by-frame scores of generated images and clips.
- Face recognition runs only on files whose whole lineage allows it: camera originals in `cast/ALIAS/{photos,video,
  bench}`, and files made by model sets marked `biometric_outputs: allowed`. Every generated file gets a provenance
  sidecar (sha256, model sets and revisions, inputs, settings); files without one are refused.
- FLUX 3 Klein is marked forbidden: its license bars biometric processing of the model or anything it produces, and
  using its outputs to improve another model. Its images never enter a training set.
- LTX-2.5 is marked allowed. The LTX-2.x license has no biometric clause; Lightricks' use policy bars outputs in
  "biometric identification" uses. Privately scoring the user's own consented likeness is the user's decision.
- Scores leave out the frames copied from a conditioning image (and two on each side), smooth over 0.3 s so a blink
  is not a drop, count a dip only when it lasts 0.25 s below the person's genuine 5th percentile, and report faces
  too turned or too small as unscored rather than failed.
- Next: a second scorer (AuraFace), head-angle calibration and a real target from the user's own phone clips, a
  fixed bench of start photos and motions, then video settings and engines compared on it.

## 2026-10-27: films and their sound

- `studio film make` turns a picture folder and a prompt into a short film: Claude plans 3 to 6 shots (route
  film_prompts, text only, structured output), each shot gets one FLUX 3 Klein still and one Wan 2.2 clip
  (4-step lightx2v 1022 LoRAs, 5 s, 16 fps), and FFmpeg joins them with hard cuts. Every still is drawn before the
  first clip, so the GPU swaps models once.
- Each still request repeats the person, clothes, place and light in full: the image model sees one shot at a time.
- Sound is a second step, `studio film sound`, on a finished film: PrismAudio (FunAudioLLM, MIT weights) makes a
  44.1 kHz stereo track per shot from the clip and the shot's sound prompt; the tracks are fitted to the shots,
  joined, normalized in two passes to -16 LUFS / -1.5 dBTP and encoded as AAC 48 kHz.
- PrismAudio's parts: VideoPrism LvT (Apache-2.0, JAX), T5Gemma (Gemma Terms of Use, gated on Hugging Face),
  Synchformer (MIT) and the Stable Audio Open VAE (Stability AI Community License: non-commercial use free).
- Not used: MMAudio, the best measured (FoleyBench), because it needs Apple's DFN5B CLIP, licensed for scientific
  research only; HunyuanVideo-Foley, because its license bars use and display of its outputs in the EU, UK and
  South Korea.
- PrismAudio runs in its own Python environment (torch and JAX for this GPU) from a pinned source commit, offline,
  while ComfyUI is down. Upstream pins torch 2.6 and flash-attn, which do not run on this card; the runner keeps
  the source unchanged and replaces what does not run here.
- The mix (2026-10-27): PrismAudio normalizes each track's peak on its own, which left shots 12 dB apart, one shot
  cancelling 5 dB in mono and the film's true peak over -1.5 dBTP after AAC. Each track is now made 0.5 s longer at
  both ends (the shot's first and last frames held), every shot gets one gain to the same level, a track below 0
  correlation is narrowed (inverted below -0.5), shots crossfade over 0.5 s centred on each cut, and a 4x
  oversampled limiter at -2 dBTP plus a measured second encode keep the AAC under -1.5 dBTP.
- The runner rounds the audio latent length up, so every whole-second shot gets a full track. Upstream rounds to
  nearest, which leaves about half the lengths short (of the film's 2 to 10 s: 2, 4, 6, 8 and 10 s; 10 s by
  15 ms); for the other lengths the two are the same.
- No music for now.
- T5Gemma access was granted on 2026-10-27; the set is pinned at ddd0c8a6. On this card a 5 s shot takes about
  50 s (VideoPrism's JAX compile is most of it) and 13 GB of GPU memory at peak.

## 2026-10-27: GPU sharing

- GPU work starts only when agent-vllm is stopped and no other compute process holds the GPU. The display's own
  programs do not count: Xorg and gnome-shell are graphics processes, and NoMachine's screen encoder opens a CUDA
  context for NVENC while a client is connected, so `gpu.display_processes` names it. Its memory counts against the
  free-memory floor like the display's.
- A busy GPU stops a command with one FAIL line that names each process.

## 2026-10-27: film direction

- The first meteor film showed the film's rules working against the story: the scene flow's still rules (the face
  large and toward the camera, for likeness) kept the person facing the lens with the event behind them, the motion
  rules froze the first frame's expression, and BFL's upsampler turned "shocked" and "frightened" into wide eyes,
  raised brows and an open mouth. The film now has its own rules; the scene flow keeps its likeness rule.
- Shots are staged from the story: the person looks at what happens, in wide or over-the-shoulder shots from behind
  and reaction shots at three quarters, never into the lens unless asked. Expressions are understated and shown in
  the body; the mouth stays closed.
- Clothes and the view are written in the same words in every still request, and a picture of other people serves
  only for the place or a prop: from one such picture the upsampler described three different outfits across four
  shots. Far events stay small on the horizon and grow from shot to shot.
- Sound prompts name only what is heard. Every prompt of that film ended with "Avoid music, voices, speech...", and
  PrismAudio makes what it reads.
- The code checks a plan for gaze at the camera, strong expression words and negated or voice words in sound
  prompts (words from the user's own description pass), sends it back with the problems at most twice, then stops.

## 2026-10-28: look first

- meteor-v2 still looked generated and inconsistent: every still was drawn from words and loose references, so the
  place, the light and the clothes were invented anew in each shot. A film now draws its look first: set pictures
  (the place at the scene's light, no people, from the pictures of the place) and character sheets (the person in the
  film's clothes, front, three quarters and back, from the photos). Every shot's still is drawn from the photos, the
  first set picture and the first sheet, so the place and the clothes come from pictures, not words.
- Several candidates of each picture and several takes of each still; the film uses the first of each until the user
  picks.
- The set pictures and sheets are FLUX outputs: they reach the image model as references only, resized and padded,
  never through face detection.
- Pictures of the event (`--event`) go only to the shots whose still request names them (with those numbered before
  them, so the numbers hold), never to the set: the image model copies from every reference it gets. Pictures of
  other people are not used.
- One still takes at most 10 references, so a film checks that per still and allows up to 30 pictures in all.

## 2026-10-28: the harness after the research

The research (`~/studio-data/docs/research/reports/Realistic AI video pipelines 2026.md`) found the gap to praised AI
films mostly between shots and after generation: professionals keep 2-5% of what they generate, automated pipelines
gain most from explicit continuity state and little from critic loops, and the craft stages (selection, edit, sound)
carry the rest. The user approved its first five requirements.

- Continuity as data: the plan holds a bible (the wardrobe, the place and the light as locked phrases, an ambience
  prompt, one to four views). The code checks that each locked phrase is copied where the rules say, and checks every
  upsampler caption before drawing: a dropped phrase goes back in as a short sentence (`caption_fixed` in film.json).
  To check the caption, the upsampler now runs on its own before the image is drawn from the caption.
- Views: each view gets its own pictures, drawn from the first set picture alone, and every shot references its
  view. Re-framing with Qwen-Image-Edit-2511 and a multi-angle LoRA, the community default, is a later bench entry.
- Events: a shot's still is drawn without its event; the event is painted into its box from the event pictures, the
  rest of the still kept pixel for pixel. Events are capped at 30% of the frame width; bigger spectacle is implied.
- Takes: four takes of each shot's picked still instead of two takes of every still; `film pick` and `film takes`
  change the cut. Each take gets numbers that sort without faces (motion, still ends, color drift); the user picks.
- Sound in layers: one bed under the whole film, Foley only for shots with their own sound (8 s windows), a speech
  check on every generated track (made again once, then dropped), and one loudness for the whole film instead of
  one per shot, which had pushed quiet shots into the limiter (1.4 to 3.2 LU low).
- PrismAudio's model card, unchanged since 2026-05-31, says MIT but limits the code, weights and data to research and
  educational use; commercial use needs the authors' permission. Our earlier notes said only MIT. A personal film
  fits; anything shared further needs another sound model or the authors' word.
