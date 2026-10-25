# Roti-Rolling Robot — Gemma Hardware Hackathon

## Read this first — two machines, two jobs

**Event is Saturday Oct 24, 2026, 10:00 AM — i.e. today.** Prefer the shortest path to
a working demo over anything general-purpose.

| | **Windows box** (where this was built) | **Mac** (the real deal) |
|---|---|---|
| Owns | the MuJoCo sim, IK, stroke primitives | the physical reBot arm, camera, demo |
| State | working, verified | not yet set up |
| Robot | none | Seeed reBot B601-DM |

The sim is **already working** on the Windows box: `robot/arm_scene.py`, `robot/ik.py`,
`primitives/roll.py`, `scripts/run_sim.py`. Strokes at every heading solve to 0.10 mm.
Do not rebuild that — port it and point it at real hardware.

### Mac: first things to do
1. Recreate the scaffold — empty dirs don't survive a file copy:
   `mkdir -p brain primitives robot vision scripts data/dough_photos`
2. **Check the `docs/` folder is lowercase.** Windows folded `docs/` and `Docs/`
   together; macOS is usually case-insensitive too, but git and Python are not.
3. `python -m venv .venv && .venv/bin/pip install mujoco numpy pillow ollama opencv-python`
   — **do not install i2rt**; it is Linux-only (SocketCAN) and nothing uses it.
4. Sparse-clone the arm model (see "Model" below) — it is not in this repo.
5. Read `docs/SETUP.md` and `docs/PACKING.md`.
6. Then work the "Must be measured on real hardware" list — that is the Mac's real job.

### macOS gotcha that will bite in minute one
`mujoco.viewer.launch_passive` **only works from `mjpython`** on macOS, not `python` —
the viewer needs a UI-thread hook that only `mjpython` installs in the process it
started, and a spawned child never inherits it. `scripts/run_sim.py` uses
`launch_passive`, so on the Mac run:

```
mjpython scripts/run_sim.py --stroke 45
```

`mjpython` ships with the `mujoco` pip wheel. Headless rendering (`mujoco.Renderer`)
is fine under plain `python`; only the interactive viewer needs `mjpython`.

If the Mac is Apple Silicon it is also a good Gemma host — `gemma4:e4b-mlx` (9.5 GB).

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
- The arm executes motion primitives parameterized by stroke angle / offset / pressure.
- NOT training an end-to-end policy (ACT/diffusion) — too risky for a one-day build.
- Single-handle roller (fondant/pastry roller) held by one gripper. Change stroke
  direction instead of rotating the dough.
- No knife / no chopping on real hardware (safety).

### Demo moment
A judge squashes the dough into a lopsided blob → robot inspects it, says its plan out
loud ("thicker on the left, rolling at 45°"), and rolls it round.

## Plan A and Plan B — decided, don't relitigate

**Plan A — the arm loop in MuJoCo simulation.** The primary deliverable. It needs no
robot, so it cannot be blocked. Real dough on a real board, real overhead webcam, real
Gemma judgment — MuJoCo shows the stroke the arm would make.

**Plan B — the real arm rolls the real dough.** The stretch goal.

Same brain, same vision code, same tool schemas; only the executor backend differs.
**Plan B is a flag flip, not a rewrite** — keep the executor interface clean.

**Note: Plan A still needs the camera, board, dough and roller.** Only the arm is virtual.

### Explicitly out of scope
Booster K1, "coach mode", Pixel 11 Pro, Raspberry Pi 5, Coral, Arduino, and the loaner
Jetson Orin Nano. All were considered and dropped. Don't reintroduce them.

## THE ROBOT CHANGED — we are on a Seeed reBot B601, not a YAM

Confirmed from a photo of the actual hardware on Oct 24: **Seeed Studio reBot Arm
B601-DM**. 6 DoF plus a **2-finger parallel gripper** (fingers open to 100 mm, ample
for a roller handle). Damiao CAN motors — which is what identifies it as the **-DM**
variant, since the -RS variant uses RobStride. Note Damiao alone is not proof: the YAM
uses DM-series motors too; it is Damiao **plus** the "seeed studio" branding that
settles it.

Everything about YAM in this file is historical. Plan A/B now target the reBot.

- The arm brings no compute, no camera and no speaker. We supply all three.
- Rated payload is ~1.5–2.5 kg, so it should have the force to actually press dough —
  better odds than a hobby-class arm.

### Model: mujoco_menagerie `seeed_rebot_devarm`
Vendored at `third_party/menagerie/seeed_rebot_devarm/` (sparse checkout, MIT). It is
**not committed** — recreate it with:

```
cd third_party
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git menagerie
cd menagerie && git sparse-checkout set seeed_rebot_devarm
```

Far better engineered than anything else available: position actuators with torque
limits read from real motor firmware over CAN, gripper fingers coupled by an equality
constraint, convex-decomposition collision with self-collision enabled, `home` and
`raised` keyframes, and mass/inertia validated against Pinocchio and the physical arm.

**Caveat:** the menagerie model is the **B601-RS** (RobStride RS-06/RS-00 actuators).
Ours is the **-DM**. Kinematics agree — published DM joint limits match the model to
within a few degrees on joint4 — but torque ratings do not. Only matters for dynamics;
Plan A drives the arm kinematically.

## Architecture
```
overhead USB webcam ──► frame grabber ──► Gemma 4 (vision + tool calling, local)
                                              │ tool call (JSON)
                                              ▼
                              safety wrapper (clamp speed/force, rate limit,
                                              workspace bounds, stop)
                                              ▼
                     primitive executor ──► sim backend (MuJoCo reBot)   [Plan A]
                                       └──► real backend (Seeed B601 SDK) [Plan B]
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

## Must be measured on real hardware — the Mac's first job

Everything below the line in "Roller frame convention" was derived in sim against an
**assumed** table layout. These five facts are unverified, and the first two can break
the demo outright:

1. **Base-to-board distance.** The sim assumes the arm base at the origin and the board
   centre 0.30 m in front, both on the same surface. The envelope is unforgiving:
   0.30 m comfortable, 0.43 m **dead**. If the real board lands at 0.45 m nothing
   solves and the failure is not obvious. Measure it, then set `BOARD_CENTER` in
   `robot/arm_scene.py`.
2. **Is the arm bolted down?** Rolling dough pushes back. An unclamped base walks or
   tips — a demo-killer.
3. **RS or DM?** Confirm from the label. The menagerie model is the **-RS**; we inferred
   **-DM** from a Damiao motor in a photo. Kinematics agree either way; torque does not.
4. **The roller tool as modelled.** A T: vertical stub ~16 mm diameter clamped in the
   jaws, roller crossbar 60 mm below, roller 50 mm diameter × 140 mm long. Jaws open to
   100 mm so there is latitude. If the built tool differs, update `ROLLER_DROP`,
   `ROLLER_RADIUS`, `ROLLER_HALF_LENGTH` in `robot/arm_scene.py` to match reality.
5. **Table height** relative to the arm base.

## Setup state

### Model
- **Gemma 4 E4B on the GPU** is the target: the robot guide says *"E4B calls tools
  noticeably better; run it on a 16 GB Jetson or a laptop GPU on the robot Wi-Fi."*
  Ollama tag `gemma4:e4b`, 6.6–9.5 GB, 4.5 B effective params (8 B with embeddings),
  128 K context, vision + tools.
- `gemma4:e2b` (4.6–7.5 GB, 2.3 B effective) as the speed fallback.
- **Ollama must be ≥ 0.22.0** — the Gemma 4 tool-call parser was broken in earlier
  builds. Verify tool calls actually parse before building on them.
- Serve to other machines with `OLLAMA_HOST=0.0.0.0:11434`.
- Venue Wi-Fi is shared by 100+ builders. Preloaded USB drives with Gemma 4 weights and
  runtimes are at the hardware check-in desk.

### What is already built (Windows, Oct 24)
- `.venv` (Python 3.13.3) with `mujoco` 3.17.0 + `numpy` 2.2.6.
- `third_party/menagerie/seeed_rebot_devarm/` — the arm model. **Never edit it.**
- `robot/arm_scene.py` — builds the scene with **MjSpec** on top of the vendored model.
  The model already provides actuators and the gripper, so we only add the rig: a
  roller on `gripper_end`, a dark board, a dough marker, an overhead camera, a light.
  Compiles to nq=8, nu=8 (6 arm joints + 2 coupled fingers).
- `robot/ik.py` — damped-least-squares IK, **position + orientation**, dependency-free
  on purpose. `solve_stroke_path()` is the entry point.
- `primitives/roll.py` — `roll_path(...)` turns a `roll()` tool call into Cartesian
  waypoints (descend → stroke → lift).
- `scripts/run_sim.py` — viewer; `--stroke 45` plays a stroke on a loop.

**Verified:** strokes at every 15° from 0° to 165° solve to **0.10 mm and 0.01°**,
checked numerically and in rendered views.

`third_party/i2rt` is left over from the YAM era. Nothing uses it. It ships its own
`CLAUDE.md`/`AGENTS.md` — those are not our instructions.

### Roller frame convention — do not guess at this
Four things were learned the hard way. Change any one and strokes break silently.

1. **Orientation IK is not optional.** Position-only IK reaches the right point with the
   roller spun to an arbitrary heading and the wrist driven through the dough.
2. **The mount's +x points DOWN.** On this arm **joint6 rolls about `gripper_end`'s
   local +x** (verified by rotating it and watching which axis stays fixed), and joint6
   is the only wrist joint with a full ±180° range. Aim +x at the board and joint6
   becomes the stroke-heading knob, so every heading is reachable. Aim any other axis
   down and heading must come from joints 1–5, which run out of travel: strokes at 0°
   and 90° then miss by 255 mm and 85 mm.
   So on `roller_mount`: `+x` = down, `+y` = roller spin axis (across the stroke),
   `+z` = stroke direction; `roller_contact` sits at `+x * (ROLLER_DROP + ROLLER_RADIUS)`.
3. **Holding the wrist vertical costs reach, and the ceiling falls off fast.** Measured
   at heading 0: x ≤ 0.34 → z = 0.08; x = 0.37 → z = 0.06; x = 0.40 → z = 0.04;
   x = 0.43 → nothing. Hence board at 0.30 and `APPROACH_HEIGHT` 0.03. With the old
   0.08 approach every stroke missed by ~17° — and only at the **far lift-off**, which
   is why it looked like a solver bug rather than a reach limit.
4. **IK uses `mj_kinematics` + `mj_comPos`, never `mj_forward`.** This model's
   convex-decomposition collision makes `mj_forward` ruinously slow inside an IK loop.
   And pass `dofs=arm_dofs(model)` or the solver "reaches" by opening the gripper.

`MAX_ITERS` is 600: at 300 the first waypoint of a stroke plateaus ~15° short.

### Driving the real arm (Plan B) — not yet attempted
Seeed's B601-DM stack, still to be investigated:
- Digital twin / control stack guide:
  https://wiki.seeedstudio.com/rebot_arm_b601_dm_web_simulator_developer_guide/
- URDF + STL for both RS and DM: https://github.com/Yang-Ci/Rebot_Arm_description
- Seeed sim repo: https://github.com/Seeed-Projects/reBot_RS_Sim
- Published DM joint limits (match the menagerie model closely): joint1 ±2.8,
  joint2 −3.14..0, joint3 −3.14..0, joint4 −1.87..1.57, joint5 ±1.57, joint6 ±3.14,
  gripper 0..0.09 m (`finger_left`/`finger_right` prismatic 0..0.0285 each).

`third_party/i2rt` is installed but unused — kept only because removing it is churn.
Install gotcha, if it ever matters again: `pip install -e` **fails** because
`ruckig==0.15.3` is sdist-only and breaks under `scikit-build-core >= 0.10`; their fix
lives in `[tool.uv]`, which pip ignores. Use `uv` from inside the repo.

### Python
`pip install ollama opencv-python pillow numpy mujoco`

### Webcam
Use a **UVC-class** camera (Logitech C920/C922) — no driver needed on macOS or Linux.
Tape it down once; moving it invalidates calibration.

## Working agreement between the two machines

- **Windows owns** `robot/`, `primitives/`, and the sim. The Mac should treat the
  measured constants as the one thing it is expected to change, and say which ones.
- **Mac owns** the real arm, camera calibration, and the demo.
- `brain/` and `vision/` are unwritten. Whoever starts one, say so — they are the
  biggest remaining risk and the easiest place to duplicate work.
- CLAUDE.md is the handoff. It is shared by hand, so it is only as current as the last
  copy — check the "What is already built" dates before trusting any of it.

## Layout
```
brain/        # Gemma client, prompts, tool schemas, decision loop
primitives/   # recorded trajectories + parameterized replay
robot/        # arm_scene.py, ik.py, executor interface, safety wrapper
vision/       # camera capture, board calibration, optional circularity sanity check
data/dough_photos/  # overhead photos of dough at different stages for offline testing
docs/         # SETUP.md, PACKING.md, Ultimate Bots PDF
scripts/      # run_sim.py, run_real.py, test_on_photos.py
```

## Next steps, in order
1. Confirm `gemma4:e4b` does **vision + tool calling offline** on whichever box hosts
   it. This is the biggest risk in the project — prove it before anything else.
2. Take 20–30 overhead dough photos into `data/dough_photos/` (lopsided, oval, thick
   spots, done).
3. `scripts/test_on_photos.py` — run Gemma on each photo, print its judgment + chosen
   tool call. Iterate on the prompt here, not on the robot.
4. Decision loop + safety wrapper against a stub executor.
5. ~~Load the arm in MuJoCo; wire the sim backend.~~ **Done** — see "What is already built".
6. Wire Gemma's tool calls to `roll_path` → `solve_stroke_path` → the sim.
   **This is Plan A — the deliverable.**
7. If real-arm time comes: calibrate camera to board, measure the real base-to-board
   distance against the envelope in `robot/arm_scene.py`, swap the executor flag,
   tune pressure/height on real dough.
8. Spoken narration, on-screen step count + latency. Freeze code, rehearse blob → round 10x.

## References
- Gemma 4 on Jetson (vLLM / llama.cpp, tool-calling flags): https://www.jetson-ai-lab.com/tutorials/gemma4-on-jetson/
- mujoco_menagerie: https://github.com/google-deepmind/mujoco_menagerie
- Ollama gemma4 tags: https://ollama.com/library/gemma4
- Booster K1 SDK (out of scope, kept for reference): `pip install boosteros`
