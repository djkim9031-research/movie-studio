# movie-studio

Short films from a written story. The Claude API writes the prompts and reviews shots; open-weight models render
pictures, voices, music and sound on a local GPU; FFmpeg assembles the result.

What works today:
- a photoreal scene of a person from their photos and a prompt, as stills and then 5-second clips with sound;
- a short film from the same folder and prompt in one command: Claude plans 3 to 6 shots, each becomes a still and a
  Wan 2.2 clip, joined into one MP4. Sound is added by a second command once its worker is set up.

The build order of the rest is in [docs/decisions.md](docs/decisions.md).

## The scene flow

```
your photos + pictures of places or things + a prompt
  |
  1. references (CPU)    YuNet finds the faces, SFace tells people apart; the person in most photos gives up to
  |                      6 face crops (1024 px, other faces blurred, metadata dropped); pictures without people
  |                      are used whole
  2. prompts (Claude)    your words -> a still request and a motion prompt; Claude gets text only
  |
  3. stills (FLUX 3      BFL's prompt upsampler looks at the references and writes the full caption; the distilled
  |   Image Klein)       model draws 1920x1088 in 16 steps, one seed per still
  |
  4. clips (LTX-2.5)     the stills you pick become 5 s clips with sound at 1280x704, with the motion prompt
  v
projects/scenes/NAME/review.html: every still with its clips, and every prompt used
```

Both GPU steps run in ComfyUI inside Docker. The still graph is built by `studio/stills.py` from BFL's
`flux3_klein_bf16_mi2i` workflow; the clip graph is `config/workflows/ltx25_i2v.json`. Each still's exact graph is
saved next to it as `stillN.graph.json`.

## What you need

- Linux with an NVIDIA GPU of 48 GB or more and driver 580 or newer. Tested on an RTX PRO 5000 72 GB (Blackwell):
  stills peaked at 38 GiB of GPU memory and clips at 43 GiB.
- Docker 24 or newer with the NVIDIA Container Toolkit 1.17 or newer (tested with 1.20.1).
- [uv](https://docs.astral.sh/uv/), the tool that installs this project's Python packages and runs its commands.
  Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- About 150 GB of free disk for scenes: 81 GB of model weights, an 11 GB image, room for results. Films add
  69 GB for Wan 2.2, and their sound about 20 GB more (weights and the sound worker's environment).
- An Anthropic API key. Claude Opus 5.5 writes the two prompts of a scene for about $0.03.
- A Hugging Face read token, after accepting the licenses of
  [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) and
  [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
  on huggingface.co with the same account.

## Setup

1. Clone and install:

   ```sh
   git clone https://github.com/djkim9031-research/movie-studio.git
   cd movie-studio
   uv sync
   uv run studio fetch-ffmpeg
   ```

   `uv sync` creates the project's own Python environment in `.venv`: Python 3.12 and the exact package versions
   in `uv.lock`, nothing installed system-wide. `uv run studio ...` runs the project's `studio` command in that
   environment, so run every command below from the `movie-studio` folder. To type `studio ...` without `uv run`,
   run `source .venv/bin/activate` once per terminal.

2. Save the two keys outside the repository, readable only by you. Type each key at the hidden prompt instead of
   pasting it into a command line, so it never lands in your shell history:

   ```sh
   mkdir -p -m 700 ~/studio-data/secrets
   read -rs -p "Anthropic API key: " key; echo
   (umask 077; printf 'ANTHROPIC_API_KEY=%s\n' "$key" > ~/studio-data/secrets/anthropic.env); unset key
   read -rs -p "Hugging Face token: " key; echo
   (umask 077; printf 'HF_TOKEN=%s\n' "$key" > ~/studio-data/secrets/hf.env); unset key
   ```

   `studio` refuses a key file that others can read. Data lives in `~/studio-data`; change `data_root` in
   `config/studio.yaml` to put it elsewhere.

3. Check the machine and the Claude key (a few cents):

   ```sh
   uv run studio doctor --claude
   ```

4. Download the models at their pinned versions; each file is checked by size and sha256:

   ```sh
   uv run studio models fetch ltx25-int8
   uv run studio models fetch flux3-klein
   uv run studio models fetch face-detect
   uv run studio models fetch face-id
   ```

5. Build the ComfyUI image (pinned ComfyUI commit, CUDA 13.0, PyTorch and every other package from a hashed lock):

   ```sh
   uv run studio gpu build
   ```

## Make a scene

1. Put the pictures for the scene in one folder, for example `~/pictures/space`:
   - 6 to 10 photos of the person: sharp, face clearly visible, a few angles and kinds of light, no beauty filter.
     Photos with other people in them are fine; their faces are blurred out of the crops.
   - Optional pictures of places or things the scene should show. A real photo beats a description: for a planet,
     a NASA photo (public domain) gives the right rings and light.

   Every JPEG, PNG or WebP picture directly in the folder is read, up to 60; subfolders and other files are left
   alone. The person in most of the photos is the subject, and their 6 best faces are used. Export HEIC photos
   (iPhone) as JPEG first.

2. Make the stills (4 by default, about a minute each):

   ```sh
   uv run studio scene stills --name space \
     --prompt "me floating in space with Saturn in the background, thin smile, float around" \
     ~/pictures/space
   ```

   Picture files work too, instead of the folder or next to it.

   Open the review page it prints, `~/studio-data/projects/scenes/space/review.html`, in a browser and note the
   numbers of the stills that look most like the person.

3. Turn the chosen stills into clips (about 40 seconds each):

   ```sh
   uv run studio scene clips --name space 2 3
   ```

   Reload the review page: each clip sits under its still.

Options: `--count N` stills, `--portrait` for 9:16, `--seconds N` (up to 10), `--seed N` to repeat a clip,
`--keep-up` to leave ComfyUI running between commands. The prompt works best with the place, the expression to keep
and the kind of motion; Claude turns it into the detailed prompts the models need.

## Make a film

```
your photos + pictures of places or things + a prompt
  |
  1. references (CPU)    as for a scene: up to 6 face crops of the person, other pictures whole
  |
  2. shot plan (Claude)  your words -> a title and 3-6 shots, each with a still request, a motion prompt and a
  |                      sound prompt; Claude gets text only
  3. stills (FLUX 3      one 1920x1088 still per shot from the references
  |   Image Klein)
  4. clips (Wan 2.2)     each still becomes a 5 s clip at 1280x704, 16 fps (4-step lightx2v LoRAs); every still is
  |                      drawn before the first clip, so the GPU swaps models once
  5. film (FFmpeg)       each clip cut to whole seconds, joined with hard cuts, H.264
  v
projects/films/NAME/film.mp4 and review.html
```

1. Download the video models once (69 GB):

   ```sh
   uv run studio models fetch wan22-i2v
   uv run studio models fetch wan22-lightx2v-1022
   ```

2. Put the pictures in one folder as for a scene, and make the film:

   ```sh
   uv run studio film make --name ride --shots 4 \
     --prompt "me wearing a light blue hanbok and a gat, riding a horse in front of the palace" \
     ~/pictures/palace
   ```

   Four shots take about 25 minutes: a minute and a half per still, three minutes per clip, and the model loads.
   Claude's shot plan costs about $0.10.

3. Open `~/studio-data/projects/films/ride/review.html`: the film at the top, then each shot's still, clip and
   prompts.

Options: `--shots N` (3 to 6), `--seconds N` per shot (Wan 2.2 is trained on 5; longer shots are possible but may
repeat motion), `--portrait`, `--keep-up`. A name is used once; pick a new one for another take.

The film is silent: Wan 2.2 makes no sound. The shot plan already holds a sound prompt per shot, and
`uv run studio film sound --name ride` turns it into film-sound.mp4 once the sound worker is set up (see Sound).

### Files of a film

`~/studio-data/projects/films/NAME/`:

- `refs/`: the references as used
- `stills/shotN.png` and `stills/shotN.graph.json`: each shot's still and the ComfyUI graph that drew it
- `clips/shotN.mp4`: each shot's clip, silent
- `film.mp4`: the shots joined; `film-sound.mp4` once sound is added
- `film.json`: the prompt, the title, every prompt Claude wrote, seeds, timings and model versions
- `review.html`: the page to watch it all; it shows real people, so keep it to yourself

Every still, clip and film has a `.prov.json` record beside it: its sha256, the models that made it and its inputs.

### Sound

Wan makes no sound, so a film gets its sound in a second step: PrismAudio watches each finished clip and makes a
44.1 kHz stereo track from the shot's sound prompt (hooves, wind, cloth, the ambience of the place; no music or
voices). The tracks are fitted to the shots, joined, brought to -16 LUFS and written as `film-sound.mp4`.

PrismAudio runs in its own Python environment (Python 3.11 with PyTorch and JAX for this GPU), from a pinned source
commit, with downloads turned off. Set it up once:

1. Request access to [google/t5gemma-l-l-ul2-it](https://huggingface.co/google/t5gemma-l-l-ul2-it) with the
   Hugging Face account of your token. PrismAudio reads the sound prompt with T5Gemma, and Google reviews requests
   by hand.

2. Once access is granted, download the weights (11 GB) and build the worker (7 GB of packages):

   ```sh
   uv run studio models fetch prismaudio
   uv run studio models fetch videoprism-lvt
   uv run studio models fetch t5gemma
   workers/prismaudio/setup.sh
   ```

Then add sound to a film:

```sh
uv run studio film sound --name ride
```

It stops ComfyUI first: the worker needs the GPU, about 13 GB, and takes about 50 s per shot. If a setup step is
missing, it says which.

## What leaves the machine

- Photos, crops, stills and clips stay on this machine. ComfyUI runs on an internal Docker network with no route out
  and no DNS; its copies of your pictures and results are removed after each run.
- Claude receives your prompt text, the number of photos of the person and the file names of the other pictures.
  No image is ever sent.
- The sound worker runs on this machine with downloads turned off; it reads the clips and writes the tracks in the
  film's folder.
- Model downloads come from Hugging Face at pinned commits.

## Licenses and limits

- FLUX 3 Image Klein: FLUX Kommunity License (non-commercial use; commercial output for qualifying users under
  US$5M revenue). It forbids biometric processing of the model and its outputs, so face recognition runs only on
  your input photos, never on generated images, and likeness is judged by eye.
- Wan 2.2 and the lightx2v LoRAs: Apache-2.0.
- Film sound: PrismAudio (MIT) with VideoPrism (Apache-2.0), Synchformer (MIT), T5Gemma (Gemma Terms of Use) and
  the Stable Audio Open VAE (Stability AI Community License: non-commercial use free; commercial use needs
  Stability AI's terms). Why not MMAudio or HunyuanVideo-Foley: [docs/decisions.md](docs/decisions.md).
- LTX-2.5: LTX-2.x Community License. YuNet (MIT) and SFace (Apache-2.0) come from the OpenCV model zoo. BFL's
  upsampler instructions in `config/prompts/` are MIT; see `config/prompts/LICENSE.flux3-workflows`.
- Use photos only of people who agree to it.
- Nothing is trained: the stills copy features from the reference photos while they are drawn, so likeness varies
  by seed. Make several stills and pick. In clips, small slow motion keeps the face closest to the still.

## Files of a scene

`~/studio-data/projects/scenes/NAME/`:

- `refs/`: the face crops and the other pictures as used
- `stills/stillN.png` and `stills/stillN.graph.json`: each still and the exact ComfyUI graph that drew it
- `clips/stillN-seedS-VARIANT-Ls.mp4`: the clips (seed S, video variant, L seconds); LTX clips have sound, Wan
  clips do not
- `scene.json`: the prompt, Claude's two prompts, the upsampler's caption, seeds, timings and model versions
- `review.html`: the page to compare them; it shows real people, so keep it to yourself

## Other commands

- `studio make-video PICTURE --prompt TEXT [--seconds 5] [--seed N] [--name NAME] [--keep-up]`: one clip with
  sound from any picture (LTX-2.5); saves `projects/videos/NAME-seedN.mp4` and the settings beside it
- `studio gpu build|up|down|status`: the ComfyUI container. With it up, the ComfyUI web page is at
  http://192.168.131.2:8188 on this machine; from another machine, `ssh -L 8188:192.168.131.2:8188 <this machine>`
  and then http://localhost:8188
- `studio models verify|fetch NAME`: pinned model weights, checked by size and sha256
- `studio render-test`: a fixed 5 s clip from the test still, with time and GPU memory
- `studio doctor [--claude]`: GPU, driver, disks and FFmpeg; with `--claude`, the API key and prompt cache
- `studio cost [--month YYYY-MM]`: Claude API spend by scene and route, against the caps in `config/studio.yaml`
- `studio smoke-mux`: a 10 s test pattern with a tone at -16 LUFS
- `studio fetch-ffmpeg`: install the pinned FFmpeg build

## Layout

- `studio/`: the Python package behind the `studio` command
- `config/studio.yaml`: paths, Claude routes and prices, spend caps, pinned models, the pinned FFmpeg and ComfyUI
- `config/workflows/`, `config/prompts/`: the clip graph and the upsampler instructions
- `docker/comfyui/`: the ComfyUI image and its hashed package lock
- `tests/`: `uv run pytest`
- `scripts/hygiene/`: the commit and push checks of this repository, for contributors
