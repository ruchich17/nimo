# Setup checklist — Fri night, Oct 23 2026

Sources: [AGI House Builder Resource Hub](https://www.agihouse.org/blog/google-gemma-hardware-builder-resource-hub) (Oct 20, 2026), plus the per-device guides linked below.

## 0. Hardware reality check — this laptop

| | |
|---|---|
| CPU | Intel i7-6500U — 2 cores / 4 threads, Skylake, 2015 |
| RAM | 15.9 GB |
| GPU | AMD Radeon R5 M330 + Intel HD 520 — **neither can run an LLM** |
| Disk | C: 37.7 GB free · E: 64.5 GB free |
| Camera | HP Truevision HD (built-in only — **no USB webcam**) |
| Installed | Python 3.13.3, pip 25.0.1, git 2.51, node 22.15 |
| Missing | Ollama, WSL distro, Docker, any CUDA |

**This machine cannot host Gemma for the live ~1 Hz loop.** It is fine as the
orchestrator (camera, safety wrapper, primitive executor, telemetry UI, robot SDK)
and fine for slow offline prompt iteration on E2B tonight.

It *is* adequate for Jetson setup duty: the quick start wants a laptop with
≥25 GB free (C: has 37.7 GB) and a 16 GB+ USB flash drive.

## 1. Event facts that drive the plan

- **Hardware loans are first come, first served**, returned at the end. Arrive early.
- **Teams are 3–4 people.** We are 3 — a fourth is allowed.
- **Venue Wi-Fi is shared by 100+ builders.** Download every model tonight.
- **USB drives preloaded with Gemma 4 weights and runtimes** are at the hardware
  check-in desk — a good backup, not a substitute for downloading tonight.
- Rule restated by the organisers: Gemma-based model, entirely on-device, no cloud
  anywhere in the loop.
- Teams **build in simulation first** and bring the idea to the Ultimate Bots table
  after the keynote.

## 2. Two separate questions — don't conflate them

### Which robot? → **YAM.**
It is the only platform that can roll dough. K1's arms are 4 DoF (confirmed in the
Ultimate Bots guide). This also means **YAM brings no compute, no camera and no
speaker** — the arms are just arms. The brain, the overhead webcam and the narration
are all ours to supply.

### Where does the brain run? → **the teammate's Linux + NVIDIA box.**
That one machine solves three problems at once and is the **primary dev box today**:

1. Runs **Gemma 4 E4B on GPU** — the variant that calls tools noticeably better.
   Serve with `OLLAMA_HOST=0.0.0.0:11434`; other machines point at it over the LAN.
   Still 100 % local, no cloud.
2. **It is the only machine that can run i2rt** (see §5) — Linux, SocketCAN, Python 3.11.
3. Runs **MuJoCo** for the sim-first work the handler wants to see.

Claude Code's CLI is fully supported on Linux (Ubuntu 20.04+ / Debian 10+). Only *this
desktop app* is Mac/Windows-only — the teammate works on their box directly.

This Windows laptop becomes the second screen: telemetry, offline photo testing, notes.

**Backups, in order:** an Apple Silicon Mac ≥16 GB (`gemma4:e4b-mlx`) → the loaner
Jetson Orin Nano Super (E2B only; **Ollama does not work with Gemma 4 on Orin Nano**,
so llama.cpp or vLLM; follow the
[hackathon guide](https://www.jetson-ai-lab.com/tutorials/hackathon-guide/) "Case 2";
reach it over USB-C with `ssh USER@192.168.55.1`).

## 3. Plan 1 and Plan 2 share one brain

Per the Ultimate Bots guide: *"Booster K1s can get dedicated access, other on-robot
deployments are done with a handler."* YAM is the handler's call, worked **as a team
with the handler**, and there is only one pair. So:

**Plan A — the YAM loop in simulation.** The primary deliverable, and the one that
cannot fail: it needs no robot. Real dough on a real board, real overhead camera, real
Gemma judgment — MuJoCo shows the stroke the arm would make. Also the ticket to the
table: *"build and show it in sim, and the handler may put it on the real robot."*

**Plan B — YAM rolls the real dough.** The stretch. Handler's call, worked as a team
with the handler, one pair of arms for the event.

Same brain, same vision code, same tool schemas — only the executor backend differs
(the swappable interface CLAUDE.md specifies). Plan B is a flag flip, not a rewrite.

**Note: Plan A still needs the camera, the board, the dough and the roller.** Only the
arm is virtual. K1 / coach mode / Pixel / Raspberry Pi are all out of scope.

## 4. Install tonight (this laptop)

```powershell
winget install Ollama.Ollama
```

Need **Ollama ≥ 0.22.0** — the Gemma 4 tool-call parser was broken in earlier builds.

Keep models off the small C: drive, then restart the shell:

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_MODELS','E:\ollama-models','User')
```

```bash
ollama pull gemma4:e2b
```

`gemma4:e2b` — 4.6–7.5 GB, 2.3 B effective params (5.1 B with embeddings), 128 K
context, vision + tools. Pull `gemma4:e4b` (6.6–9.5 GB, 4.5 B effective / 8 B with
embeddings) onto the *good* machine, not this one.

```bash
pip install ollama opencv-python pillow numpy mujoco
```

MuJoCo installs cleanly on Windows via pip — the sim path works here.

Also download tonight, for the Jetson/LiteRT path:
- [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main)
- [litert-community/embeddinggemma-2-740m-litert-lm](https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm)

## 5. Known blocker: i2rt is Linux-only

The [YAM SDK](https://github.com/i2rt-robotics/i2rt) expects Debian/Ubuntu:
`apt install build-essential python3-dev linux-headers-$(uname -r)`, Python **3.11**
(`uv venv --python 3.11`), and CAN via
`sudo ip link set can0 up type can bitrate 1000000` — Linux SocketCAN, which does not
exist on Windows, and USB-CAN passthrough into WSL2 is painful.

- MuJoCo YAM sim + `control_with_mujoco` → works on Windows.
- `minimum_gello` teleop, `record_replay_trajectory`, real CAN → needs a Linux box.

**Resolved:** the teammate's Linux + NVIDIA box covers all of this. Do the i2rt install,
teleop recording and real-CAN work there. WSL on this laptop is no longer needed.

## 6. Must bring / still missing

- [ ] **Overhead USB webcam** + gooseneck or tripod. **Hard requirement for Plan 1** —
      YAM has no camera, and the built-in laptop camera cannot look down at a board.
      (The hub says USB webcams are provided *for the Arduino boards* — don't count on
      borrowing one.) Not needed for Plan 2; K1 has its own head camera.
- [x] Teammate's Linux + NVIDIA box confirmed as Gemma host and i2rt machine.
- [x] Ultimate Bots robot guide read and saved to `Docs/`.
- [ ] USB flash drive 16 GB+ — only if we fall back to the loaner Jetson.
- [ ] microSD 64 GB+ and a USB-C power supply, if we have spares.
- [ ] Per CLAUDE.md: dough balls (oiled, ziplocks), flour, board, non-slip mat or clamps,
      single-handle roller, light, Ethernet dongle, extension cord.

## Sources

- [AGI House Builder Resource Hub](https://www.agihouse.org/blog/google-gemma-hardware-builder-resource-hub)
- [Ollama — gemma4 library](https://ollama.com/library/gemma4)
- [Gemma 4 on Jetson — Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/gemma4-on-jetson/)
- [Jetson AI Lab hackathon guide](https://www.jetson-ai-lab.com/tutorials/hackathon-guide/)
- [Gemma 4 VLA demo on Orin Nano Super — NVIDIA on Hugging Face](https://huggingface.co/blog/nvidia/gemma4)
- [E4B CUDA OOM on Orin Nano — NVIDIA forums](https://forums.developer.nvidia.com/t/gemma4-e4b-on-jetson-orin-nano-fails-due-to-cuda-out-of-memory-issue/365641)
- [LiteRT demo app (Flutter)](https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/flutter)
- [LiteRT demo — Jetson](https://github.com/google-ai-edge/litert-samples/tree/main/samples/platforms/nvidia)
- [Mastering Edge AI on Raspberry Pi with LiteRT and Gemma](https://developers.googleblog.com/mastering-edge-ai-on-raspberry-pi-with-litert-and-gemma/)
- [Compile SLMs for Google Tensor](https://developers.google.com/edge/tensor-sdk/slm-compilation)
- [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt)
