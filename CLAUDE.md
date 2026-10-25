# Roti-Rolling Robot — Gemma Hardware Hackathon

## ⚠ THE ROBOT CHANGED (confirmed Oct 24) — read `Instructions.md`
The arm is a **Seeed reBot B601-DM**, not the YAM. Every YAM/i2rt reference below is
historical. `Instructions.md` (project root) is the source of truth for the arm model
(mujoco_menagerie `seeed_rebot_devarm` in `third_party/menagerie/`), the IK / roller
frame conventions, and the list of things to measure on real hardware. Don't clone i2rt.
Note: our `scripts/run_sim.py` is the decision loop; the Windows sim viewer of the same
name goes in as `scripts/view_arm.py`.

## Session handoff — read this first

**The whole project now runs on a single machine: a teammate's MacBook (Apple M4,
24 GB unified memory).** Gemma, camera, decision loop, safety wrapper, MuJoCo sim and
voice all run on the Mac. No second machine, no network link in the loop.

**Event is Saturday Oct 24, 2026, 10:00 AM — i.e. today.** Act accordingly: prefer the
shortest path to a working demo over anything general-purpose.

### How we got here (Oct 24, early morning)
- Scoped on a Windows laptop; moved to a Linux laptop (Dell, Fedora 38, i7, 11 GB RAM,
  GTX 1050 Mobile ~4 GB with no NVIDIA driver). Measured there, **CPU only**:
  `gemma4:e2b` 36 s/decision, `gemma4:e4b` 43 s/decision (warm) — both return valid tool
  calls, both far too slow for the demo. NVIDIA driver install was prepared (580.196.07
  `.run` in `~/nvidia/` on that box) but **not done** — no longer needed.
- No teammate has a bigger NVIDIA GPU. The M4 MacBook runs E4B on its GPU → it became
  the single machine.
- `docs/SETUP.md` and `docs/PACKING.md` predate this switch — treat their machine/install
  sections as historical.

### Mac status
Reported done by the team (not yet verified — run the checks below):
1. Ollama installed; `gemma4:e4b` pulled (the only model — no E2B fallback).
2. Python 3.11 venv: `uv venv --python 3.11 .venv` +
   `uv pip install ollama opencv-python pillow numpy mujoco`.

Verify with `python scripts/check_env.py` (venv active), then
`python scripts/smoke_tool_call.py - gemma4:e4b` twice — the 2nd latency is the real one.
**E4B is the brain.** Smoke test: 8–9 s warm. The loop (`brain/gemma.py`) gets **0.5–3 s** with 512 px frames + `"think": false` (thinking burned ~180 tokens for the same answer; `GEMMA_THINK=1` re-enables). No E2B.

## Context
- Event: Google Gemma Hardware Hackathon @ AGI House, 1868 Floribunda Ave,
  Hillsborough CA — Saturday Oct 24, 2026, 10:00 AM.
- HARD RULE: must use a Gemma-based model and run 100% on-device. No cloud APIs
  anywhere in the loop (no OpenAI/Anthropic/Gemini API calls in the product code).
  Claude Code is a dev tool, not part of the product — that's fine.
- Track: Physical AI, IoT & Robotics. Judging is science-fair style at the workbench;
  top 6 teams demo on stage.
- Team of 3 (4 allowed): software engineer (Gemma pipeline), mechatronics engineer
  (arm, primitives, roller mount), EE (camera, lighting, power, safety stop, telemetry).
- Resource hub: https://www.agihouse.org/blog/google-gemma-hardware-builder-resource-hub
- Robot guide: `docs/Ultimate Bots robots at the Gemma Edge & Physical AI Build Day — Oct 24.pdf`

## The project
A robot that rolls Indian roti/chapati dough round, closing the loop with vision.
- Gemma = the brain. It looks at the dough from an overhead camera, judges
  shape/thickness, and picks the next action via function calling at ~1 Hz.
- The arm executes motion primitives taught by demonstration (YAM leader-follower teleop
  + trajectory record & replay), parameterized by stroke angle / offset / pressure.
- NOT training an end-to-end policy (ACT/diffusion) — too risky for a one-day build.
- Single-handle roller (fondant/pastry roller) held by one gripper. Change stroke
  direction instead of rotating the dough.
- No knife / no chopping on real hardware (safety).

### Demo moment
A judge squashes the dough into a lopsided blob → robot inspects it, says its plan out
loud ("thicker on the left, rolling at 45°"), and rolls it round.

## Plan A and Plan B — decided, don't relitigate

**Plan A — the YAM loop in MuJoCo simulation.** The primary deliverable. It needs no
robot, so it cannot be blocked. Real dough on a real board, real overhead webcam, real
Gemma judgment — MuJoCo shows the stroke the arm would make. This is also the ticket to
the Ultimate Bots table: the guide says *"build and show it in sim, and the handler may
put it on the real robot."*

**Plan B — YAM rolls the real dough.** The stretch goal. Handler's call.

Same brain, same vision code, same tool schemas; only the executor backend differs.
**Plan B is a flag flip, not a rewrite** — keep the executor interface clean.

**Note: Plan A still needs the camera, board, dough and roller.** Only the arm is virtual.

**Focus is Plan A, entirely on the Mac.** Plan B caveat: i2rt drives the real YAM over
Linux SocketCAN, which macOS doesn't have. If Plan B happens, the real backend runs on a
Linux host (the handler's machine or the Fedora laptop) and the Mac sends it stroke
commands over an Ethernet cable — the executor interface stays the same.

### Explicitly out of scope
Booster K1, "coach mode", Pixel 11 Pro, Raspberry Pi 5, Coral, Arduino, and the loaner
Jetson Orin Nano. All were considered and dropped. Don't reintroduce them.

## Robot access — the real constraints
From the Ultimate Bots guide (now read, PDF in `docs/`):
- Four platforms on site: **Booster K1** (preferred and most available), Unitree G1,
  Agibot X2, and **one pair of YAM Ultra arms**.
- *"Booster K1s can get dedicated access, other on-robot deployments are done with a
  handler."* So **YAM is the handler's call**, worked as a team with the handler.
- K1 is the only one you can get to yourself — but its arms are **4 DoF**, so it cannot
  roll dough. That's why Plan A/B both target YAM.
- Go to the Ultimate Bots table with the team and the idea **right after the keynote**.
- YAM arms bring no compute, no camera, no speaker. We supply all three.

## Architecture
```
overhead USB webcam ──► frame grabber ──► Gemma 4 (vision + tool calling, local)
                                              │ tool call (JSON)
                                              ▼
                              safety wrapper (clamp speed/force, rate limit,
                                              workspace bounds, stop)
                                              ▼
                     primitive executor ──► sim backend (MuJoCo YAM)  [Plan A]
                                       └──► real backend (i2rt YAM SDK)  [Plan B]
```
- Tools exposed to Gemma:
  - `roll(direction_deg, pressure, length_cm)`
  - `rotate_dough(deg)` (optional — prefer changing stroke direction)
  - `dust_flour()`
  - `inspect()` — re-capture and re-assess
  - `done(reason)`
  - `stop()`
- Both backends share one interface so sim ↔ real is a flag.
- Always keep a typed-command fallback and a hard stop key. Never let model output
  reach the robot without the safety wrapper.

## Setup state (Mac)

### Model
- **Gemma 4 E4B on the M4 GPU via Ollama** is the target: the robot guide says *"E4B
  calls tools noticeably better; run it on a 16 GB Jetson or a laptop GPU."* Ollama tag
  `gemma4:e4b`, 6.6 GB download, 4.5 B effective params (8 B with embeddings), 128 K
  context, vision + tools.
- No E2B fallback — `gemma4:e4b` is the only model used.
- **Ollama must be ≥ 0.22.0** — the Gemma 4 tool-call parser was broken in earlier
  builds. Verify tool calls actually parse before building on them.
- Running a laptop locally satisfies the on-device rule; the Jetson is a loaner, not a
  requirement (per the robot guide: *"your code runs on your laptop or Jetson kit"*).
- Venue Wi-Fi is shared by 100+ builders — download everything beforehand.

### Python
Python 3.11 venv in `.venv/` via uv: `ollama opencv-python pillow numpy mujoco`.

### MuJoCo + YAM model
- `pip install mujoco` works natively on Apple Silicon. **On macOS the viewer needs
  `mjpython`** instead of `python` (e.g. `mjpython -m mujoco.viewer --mjcf=<file>`).
- YAM MJCF/URDF models come from https://github.com/i2rt-robotics/i2rt — clone it for
  the model files; the i2rt package itself (CAN) isn't needed for the sim.
- Read its `control_with_mujoco` example for how joints are driven. `minimum_gello` and
  `record_replay_trajectory` are Plan B (Linux) only.

### Webcam
- **UVC-class** USB camera (Logitech C920/C922) — no driver needed on macOS.
- macOS camera permission is per app: allow the terminal under System Settings →
  Privacy & Security → Camera, then restart the terminal. Camera access does **not**
  work from an SSH session.
- Index 0 is usually the built-in FaceTime camera; the USB one is usually 1. Disable
  iPhone Continuity Camera so indices don't shift.
- Tape it down once; moving it invalidates calibration.

### Voice
macOS `say "..."` — built-in, offline TTS. Use it for the spoken plan.

### Demo hygiene
Plugged in; `caffeinate -dims` during the demo; Do Not Disturb on.

## Layout
```
brain/        # Gemma client, prompts, tool schemas, decision loop
primitives/   # recorded trajectories + parameterized replay
robot/        # executor interface, sim backend, i2rt real backend, safety wrapper
vision/       # camera capture, board calibration, optional circularity sanity check
data/dough_photos/  # overhead photos of dough at different stages for offline testing
docs/         # SETUP.md, PACKING.md, Ultimate Bots PDF
scripts/      # run_sim.py, run_real.py, test_on_photos.py
              # check_env.py — env check; smoke_tool_call.py — Gemma image + tool-call test
              #   (OLLAMA_HOST_URL env var points it at another Ollama host)
```

## Next steps, in order
1. Verify the Mac (`check_env.py`, then `smoke_tool_call.py` latency) — confirm
   `gemma4:e4b` does **vision + tool calling offline** at usable speed.
1b. Clone i2rt, find the YAM model (`find i2rt -name "*.xml" | grep -i yam`), open it with
   `mjpython -m mujoco.viewer --mjcf=<file>`.
2. Take 20–30 overhead dough photos into `data/dough_photos/` (lopsided, oval, thick
   spots, done).
3. `scripts/test_on_photos.py` — run Gemma on each photo, print its judgment + chosen
   tool call. Iterate on the prompt here, not on the robot.
4. Decision loop + safety wrapper against a stub executor.
5. Load the YAM model in MuJoCo; wire the sim backend. **This is Plan A — the deliverable.**
6. Pitch at the Ultimate Bots table after the keynote (safe: soft dough, no knife, low
   speed, kill switch, already working in sim).
7. If YAM time comes: record primitives with the handler, calibrate camera to board,
   swap the executor flag, tune pressure/height on real dough.
8. Spoken narration, on-screen step count + latency. Freeze code, rehearse blob → round 10x.

## References
- i2rt Python API (YAM arms): https://github.com/i2rt-robotics/i2rt
- Ollama gemma4 tags: https://ollama.com/library/gemma4
- Booster K1 SDK (out of scope, kept for reference): `pip install boosteros`
