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
