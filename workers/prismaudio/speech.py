"""How much of a track sounds like speech: reads {"wav": path, "start": s, "end": s or null} on stdin and prints
{"speech": share}, the share of that part of the track inside the speech segments Silero VAD finds (16 kHz mono, threshold 0.5). Runs on CPU in the worker
environment; the model ships inside the silero-vad package, so nothing is downloaded. The WAV is read with the wave
module: torchaudio 2.9 and later need torchcodec to read files, which the worker leaves out."""

import json
import sys
import wave

import numpy as np
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

RATE = 16000


def main() -> None:
    job = json.load(sys.stdin)
    torch.set_num_threads(4)
    with wave.open(job["wav"], "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{job['wav']} is not 16-bit PCM")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    pcm = pcm.reshape(-1, channels).mean(axis=1)
    end = job.get("end")
    pcm = pcm[int(job.get("start", 0) * rate):None if end is None else int(end * rate)]
    mono = torch.from_numpy(np.ascontiguousarray(pcm))
    mono = torchaudio.functional.resample(mono, rate, RATE)
    stamps = get_speech_timestamps(mono, load_silero_vad(), sampling_rate=RATE, threshold=0.5)
    share = sum(s["end"] - s["start"] for s in stamps) / max(len(mono), 1)
    print(json.dumps({"speech": round(share, 3)}))


if __name__ == "__main__":
    main()
