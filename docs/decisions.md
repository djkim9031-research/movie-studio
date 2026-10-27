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
- No music for now.
