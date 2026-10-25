# Packing list — Sat Oct 24, AGI House, Hillsborough

Plan A = YAM loop in MuJoCo (real dough, real camera, virtual arm).
Plan B = YAM rolls the real dough (handler's call).

Venue provides: loaner hardware (first come, first served), USB drives preloaded with
Gemma 4 weights and runtimes at hardware check-in. Venue Wi-Fi is shared by 100+ builders.

---

## If you bring only five things

1. **Power strip** (6-outlet, long cord)
2. **USB webcam** — UVC-class, e.g. Logitech C920/C922
3. **Camera mount** — gooseneck clamp or mini tripod
4. **Clip-on LED light**
5. **Dough, flour, roller, board**

---

## Power — the thing everyone underestimates

- [ ] **Power strip / surge protector, 6 outlets, 6–10 ft cord.** 100+ builders, few
      outlets. This is the highest-value item on the list.
- [ ] Laptop charger — **one per person**
- [ ] Multi-port USB-C GaN charger (65 W+) — charges three devices off one outlet
- [ ] Power bank
- [ ] Phone chargers

## Network

- [ ] **Ethernet dongle** — explicitly called for in the Ultimate Bots guide. Match the
      port: USB-A or USB-C.
- [ ] **Ethernet cable, Cat6, ~3 m** — the dongle is useless without one
- [ ] **USB flash drive, 16 GB+, USB 3.0** — to copy the preloaded Gemma weights at the
      check-in desk rather than pulling 7–10 GB over shared Wi-Fi
- [ ] Powered USB hub — a laptop runs out of ports fast with webcam + dongle + drive

## Vision rig — the part that decides whether the demo works

- [ ] **USB webcam, 1080p, UVC-compliant.** Logitech C920/C922 is the safe pick — it is
      what NVIDIA's own Gemma 4 demo used, and UVC means **it works on the Linux box with
      no driver**. Avoid anything needing a vendor driver.
- [ ] **Gooseneck clamp mount or mini tripod** to get it looking straight down
- [ ] **USB extension cable, ~3 m active** — stock webcam cables are ~1.5 m and will not
      reach from an overhead mount to the laptop
- [ ] **Clip-on LED / ring light.** Venue lighting is mixed and overhead mounts cast
      shadows onto the exact thing you are photographing. Biggest single vision variable.
- [ ] **Gaffer or painter's tape** — fix the camera once, never move it. Calibration and
      repeatability depend on this.
- [ ] **Scale reference in frame** — a coin, a ruler, or a circle of known diameter drawn
      on the board. Makes `length_cm` and "how round is it" far more reliable.
- [ ] Dark matte board or dark cloth under the dough — pale dough on a dark background
      is the highest-contrast setup for shape judgment

## Dough and kitchen

- [ ] **15–20 dough balls**, pre-made, oiled, in ziplocks. More than feels necessary —
      judges handle them, they dry out, and you will do many runs.
- [ ] **Flour** in a shaker + a small bowl
- [ ] **Single-handle roller** (fondant/pastry roller) — plus a spare if you have one
- [ ] **Board** + non-slip mat or clamps
- [ ] **Tray or baking sheet** to contain the mess on a shared workbench
- [ ] Water spray bottle (rehydrating dough that has dried out)
- [ ] **Paper towels, wet wipes, trash bags** — you are at a shared table
- [ ] One pre-rolled perfect roti, for the before/after comparison
- [ ] ❌ **No knife.** Safety rule, and it will not pass the handler's check.

## Demo and judging (science-fair style at the workbench)

- [ ] **Printed one-pager for the table** — architecture diagram, the hard rule
      (Gemma, 100 % on-device), and the demo ask. Judges walk past; this works while you
      are mid-run. Cheapest win on the list.
- [ ] HDMI cable + adapter — top 6 demo on stage; do not get caught without one
- [ ] Sticky notes and a marker
- [ ] Kill-switch key clearly labelled on the demo laptop

## Personal

- [ ] Water bottle, snacks, a layer (venues run cold), phone

## Software to have downloaded before arriving

- [ ] `gemma4:e4b` on the Linux + NVIDIA box (6.6–9.5 GB)
- [ ] `gemma4:e2b` as the fast fallback
- [ ] i2rt cloned and `uv pip install -e .` done
- [ ] MuJoCo running the YAM model **once**, successfully
- [ ] 20–30 overhead dough photos in `data/dough_photos/` for offline prompt iteration
