# Capture guide

Photos and video of one person, so the models learn their face from every side instead of inventing the sides they
never saw. Plan about 1.5 to 2 hours with a tripod and a swivel chair.

## Before you start

- **Consent.** The person agrees to be filmed for this use. Write one line in `consent.txt`: alias, date, what the
  pictures are for, and that they stay on this machine.
- **No cloud copies.** Phone photo apps upload to iCloud or Google Photos by default. Pause the sync for the session
  or use a camera, copy the files to the workstation by cable, then delete them from the phone.
- **Phone settings.**
  - Photos as JPEG ("Most Compatible"), not HEIC.
  - Video 4K at 30 fps with HDR off.
  - The 2x lens, so close faces are not distorted.
  - Exposure and focus locked (press and hold on the face).
  - A clean lens.
- **Light.** Soft, even light first: window light or an overcast day, no hard shadows. Two more looks for the photos
  later: warm indoor light and one hard light from the side.
- **Look.** Hair, glasses and make-up as in the film. A plain background, nobody else in the frame, no filter or
  beauty mode.

## 1. Bench clips (10 minutes)

These set the target the generated clips are measured against.

- Tripod at eye level, 1 to 1.5 m away, head and shoulders in a landscape frame.
- Three motions, two takes each, 5 to 6 seconds per take:
  1. **Still:** look into the lens, hold a thin closed-mouth smile, blink naturally.
  2. **Turn:** start facing the lens, turn the head slowly about 30 degrees to your left, hold for a second, turn
     back.
  3. **Float:** a gentle sway and tilt of head and shoulders, face toward the lens.
- Three photos in the same setup: facing the lens, turned 30 degrees left, turned 30 degrees right.

## 2. Head turns (15 minutes)

- On a swivel chair, tripod 1 to 1.5 m away, head and shoulders in frame.
- Turn slowly through a full circle over 20 to 30 seconds. Do it three times: camera at eye level, a little above
  (looking down at you), a little below (looking up).
- Once more with a smile, and once while saying a sentence.

## 3. Photos (20 to 30 minutes): 60 to 100

- Angles: front, three-quarter left and right, profile left and right, back, chin up, chin down.
- Crops: close-up (the face fills about half the frame), waist up, full body.
- Expressions: neutral, closed smile, open smile, laugh, serious, talking.
- The three light looks from above.
- Each outfit the film uses: full body from the front, the side and the back.

## 4. Motion clips (20 to 30 minutes): 20 to 40 clips of 3 to 5 seconds

Walk toward the camera and across it, sit down, stand up, laugh, turn around, wave, look around, talk. Keep the
camera still: a tripod, or a second person holding it steady.

## Where the files go

On the workstation, under the person's alias:

```
~/studio-data/cast/ALIAS/consent.txt
~/studio-data/cast/ALIAS/photos/   every photo
~/studio-data/cast/ALIAS/video/    head turns and motion clips
~/studio-data/cast/ALIAS/bench/    the six bench clips and three bench photos
```

Keep the originals; the tools choose, crop and caption. Everything in `cast/` is readable only by you and never
leaves the machine.
