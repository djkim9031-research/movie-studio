# movie-studio

Short films from a written story. The Claude API writes the prompts and reviews shots; open-weight models render
pictures, voices, music and sound on a local GPU; FFmpeg assembles the result.

What works today: a photoreal scene of a person from their photos and a prompt, as stills and then 5-second clips with
sound. The build order of the rest is in [docs/decisions.md](docs/decisions.md).

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
- About 150 GB of free disk: 81 GB of model weights, an 11 GB image, room for results.
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

## What leaves the machine

- Photos, crops, stills and clips stay on this machine. ComfyUI runs on an internal Docker network with no route out
  and no DNS; its copies of your pictures and results are removed after each run.
- Claude receives your prompt text, the number of photos of the person and the file names of the other pictures.
  No image is ever sent.
- Model downloads come from Hugging Face at pinned commits.

## Licenses and limits

- FLUX 3 Image Klein: FLUX Kommunity License (non-commercial use; commercial output for qualifying users under
  US$5M revenue). It forbids biometric processing of the model and its outputs, so face recognition runs only on
  your input photos, never on generated images, and likeness is judged by eye.
- LTX-2.5: LTX-2.x Community License. YuNet (MIT) and SFace (Apache-2.0) come from the OpenCV model zoo. BFL's
  upsampler instructions in `config/prompts/` are MIT; see `config/prompts/LICENSE.flux3-workflows`.
- Use photos only of people who agree to it.
- Nothing is trained: the stills copy features from the reference photos while they are drawn, so likeness varies
  by seed. Make several stills and pick. In clips, small slow motion keeps the face closest to the still.

## Files of a scene

`~/studio-data/projects/scenes/NAME/`:

- `refs/`: the face crops and the other pictures as used
- `stills/stillN.png` and `stills/stillN.graph.json`: each still and the exact ComfyUI graph that drew it
- `clips/stillN-seedS.mp4`: the clips, with sound
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
