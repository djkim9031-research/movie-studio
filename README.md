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

- Linux with an NVIDIA GPU of 72 GB or more (the only size tested) and driver 580 or newer. The GPU steps start only
  when 60 GiB of GPU memory is free (`gpu.min_free_gib` in `config/studio.yaml`). On an RTX PRO 5000 72 GB
  (Blackwell), stills peaked at 42 GiB, scene clips at 43 GiB and the sound worker at 13 GiB; film clips used 70 GiB,
  as ComfyUI keeps models on the GPU while there is room.
- RAM: tested with 124 GB. At peak the sound worker used 16 GiB and ComfyUI 6 GiB, besides the page cache that holds
  the model files.
- Docker 24 or newer, usable without sudo, with the NVIDIA Container Toolkit 1.17 or newer (tested with 1.20.1).
- [uv](https://docs.astral.sh/uv/), the tool that installs this project's Python packages and runs its commands.
  Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Disk: 81 GB of model weights for scenes, 73 GB more for films and 19 GB more for their sound (weights and the sound
  worker's environment), an 11 GB image in Docker's folder, and room for results. `studio doctor` and
  `studio models fetch` keep 300 GB free on the disk of your home folder (`disk.home_min_free_gb` in
  `config/studio.yaml`), so with that default start with about 475 GB free, or lower the floor.
- An Anthropic API key. Claude Opus 5.5 writes the two prompts of a scene for about $0.03 and a film's shot plan for
  about $0.08.
- A Hugging Face read token, after accepting the licenses of
  [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) and
  [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
  on huggingface.co with the same account. For film sound, also request access to
  [google/t5gemma-l-l-ul2-it](https://huggingface.co/google/t5gemma-l-l-ul2-it); Google reviews requests by hand, so
  ask early.

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

   `ltx25-int8` is used only by scenes and `make-video`; skip it if you only make films.

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

For a film with sound, set up once: Setup above, step 1 below and the sound worker (Sound, below). Then each film is
two commands: `studio film make` (step 2) for the picture and `studio film sound` for its sound.

```
your photos + pictures of the place + a prompt (+ pictures of the event)
  |
  1. references (CPU)    as for a scene: up to 6 face crops of the person, other pictures whole
  |
  2. shot plan (Claude)  your words -> a title, a set request, a character sheet request and 3-6 shots, each with
  |                      a still request, a motion prompt and a sound prompt; Claude gets text only, and a plan
  |                      that breaks a checked rule goes back to it, at most twice
  3. look (FLUX 3        3 set pictures (the place, no people) and 3 character sheets (you in the film's clothes:
  |   Image Klein)       front, three quarters, back), 1920x1088
  4. stills (FLUX 3      3 stills per shot, drawn from your photos, the first sheet, the first set picture and the
  |   Image Klein)       event pictures
  5. takes (Wan 2.2)     2 clips of 5 s per still at 1280x704, 16 fps (4-step lightx2v LoRAs); every still is drawn
  |                      before the first clip, so the GPU swaps models once
  6. film (FFmpeg)       each shot's first still's first take, cut to whole seconds, joined with hard cuts, H.264
  v
projects/films/NAME/film.mp4 and review.html (every candidate and take; the ones in the film marked)
```

1. Download the video models once (73 GB):

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

   The folder needs at least one picture of the place without people: the set picture is drawn from it. Pictures of
   other people are not used. Four shots take about 2 hours with the defaults: 18 stills at two and a half minutes,
   24 clips at three, and the model loads; `--candidates 1 --takes 1` takes about half an hour. Claude's shot plan
   costs about $0.08, up to three times that when it is sent back. Claude stages the shots from your words: the
   person looks at what happens (wide shots from behind, reaction shots at three quarters), not into the lens unless
   you ask.

3. Open `~/studio-data/projects/films/ride/review.html`: the film at the top (with sound, once added), then each
   shot's still, clip and prompts.

Options: `--shots N` (3 to 6), `--seconds N` per shot (Wan 2.2 is trained on 5; longer shots are possible but may
repeat motion), `--candidates N` (1 to 6: set pictures, sheets and stills per shot, default 3), `--takes N` (1 to 4
clips per still, default 2), `--event PICTURE` (a picture of what happens, for example a real blast over the sea; it
goes to every shot and never to the set; repeat for more), `--portrait`, `--keep-up`. A name is used once; pick a new
one for another film.

The film is silent: Wan 2.2 makes no sound. The shot plan already holds a sound prompt per shot, and
`uv run studio film sound --name ride` turns it into film-sound.mp4 once the sound worker is set up (see Sound).

### Files of a film

`~/studio-data/projects/films/NAME/`:

- `refs/`: the references as used
- `look/setN.png` and `look/sheetN.png`: the set picture and character sheet candidates; the stills are drawn from
  the first of each
- `stills/shotN-C.png` and `stills/shotN-C.graph.json`: candidate C of shot N's still and the ComfyUI graph that
  drew it
- `clips/shotN-C-T.mp4`: take T of still C, silent; the film uses each shot's `shotN-1-1.mp4`
- `sound/shotN.wav`: each shot's sound track, once sound is added; 0.5 s longer at both ends for the crossfades
- `film.mp4`: the shots joined; `film-sound.mp4` once sound is added
- `film.json`: the prompt, the title, every prompt Claude wrote, seeds, timings and model versions
- `review.html`: the page to watch it all; it shows real people, so keep it to yourself

Every still, clip and film has a `.prov.json` record beside it: its sha256, the models that made it and its inputs.

### Sound

Wan makes no sound, so a film gets its sound in a second step: PrismAudio watches each finished clip and makes a
44.1 kHz stereo track from the shot's sound prompt (hooves, wind, cloth, the ambience of the place; no music or
voices). Each track is made 0.5 s longer at both ends (from the shot's first and last frames held), so the mix can
crossfade without moving any sound:

- every shot is brought to one level;
- a track whose left and right would cancel in mono has its stereo narrowed, or its right channel inverted when it
  is mostly out of phase;
- neighbouring shots crossfade over 0.5 s centred on each cut;
- the film is brought to -16 LUFS with its true peak under -1.5 dBTP after AAC, and written as `film-sound.mp4`.

`film sound` prints each shot's level and stereo fix and the film's loudness and true peak; `film.json` keeps them.

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

   `setup.sh` builds the worker in `~/studio-data/workers/prismaudio`. If you moved `data_root`, run it as
   `STUDIO_DATA=/your/data/root workers/prismaudio/setup.sh`.

Then add sound to a film:

```sh
uv run studio film sound --name ride
```

It stops ComfyUI first: the worker needs the GPU, about 13 GB, and takes about a minute per shot. If a setup step is
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
