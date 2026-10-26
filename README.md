# movie-studio

Short films from a written story. The Claude API writes the story bible, screenplay and shot list and reviews
shots; open-weight models render pictures, voices, music and sound on a local GPU; FFmpeg assembles one MP4.

Status: foundation. The build order is in [docs/decisions.md](docs/decisions.md).

## Layout

- `studio/`: the Python package behind the `studio` command
- `config/studio.yaml`: paths, Claude routes and prices, spend caps, GPU and disk thresholds, the pinned FFmpeg
- `tests/`: `uv run pytest`
- `scripts/hygiene/`: commit and push checks for this repository

Data stays outside the repository in `~/studio-data`: models, cast photos and voices, projects, the spend ledger
and secrets.

## Setup

1. `uv sync`
2. `uv run studio fetch-ffmpeg`
3. Put `ANTHROPIC_API_KEY=...` in `~/studio-data/secrets/anthropic.env`, mode 0600.
4. `uv run studio doctor --claude`

## Commands

- `studio doctor [--claude]`: GPU, driver, disks and FFmpeg; with `--claude`, the API key and prompt cache
- `studio smoke-mux`: a 10 s test pattern with a tone at -16 LUFS
- `studio cost [--month YYYY-MM]`: Claude API spend by film and route
- `studio fetch-ffmpeg`: install the pinned FFmpeg build

## Rules

- Photos, voices and frames of real people stay on this machine. The Claude API sees text, and frames of fictional
  characters only.
- GPU work starts only when no other compute process holds the GPU.
- Each clone runs `scripts/hygiene/setup-repo.sh` once; `scripts/hygiene/selftest.sh` checks the hooks.
