# Agile Quadrotor Cable MPC

**MPPI-MPC framework for agile quadrotor navigation with cable-suspended payloads through narrow apertures.**

Full 2D rigid-body quadrotor dynamics coupled with an XPBD deformable cable model and per-node collision avoidance inside a sampling-based Model Predictive Path Integral (MPPI) controller.

---

## Overview

This repository contains the simulation code for three experiments that evaluate how a quadrotor with a cable-suspended payload can traverse narrow gaps where the gap height is smaller than the cable length. The key insight is that aggressive forward acceleration causes the cable to swing backward, reducing its vertical projection and enabling traversal — a maneuver that is **structurally infeasible** for rigid-link cable models.



### The Problem

A quadrotor carries a payload on a cable of length *L* = 1.0 m and must fly through a gap of height *G* < *L*. Under the standard rigid-link assumption, the cable occupies a straight segment between the drone and the payload — if *G* < *L*, no configuration fits. By modeling the cable as a **deformable body** (XPBD particle chain), the controller can exploit the cable's natural swing-back dynamics to thread it through the aperture.

### What This Code Does

- Simulates the full coupled dynamics: quadrotor (rigid-body or point-mass) + XPBD cable + obstacle geometry
- Runs K = 256–512 parallel MPPI rollouts per control step, each propagating the complete physics
- Checks collisions of **every cable node** at every timestep (not just endpoints)
- Produces trajectory plots and animated GIFs of each experiment

---

## Experiments

### Experiment 1 — Full Quadrotor, G = 0.8 m

| | |
|---|---|
| **Drone model** | 2D rigid body (m = 1.5 kg, I = 0.06 kg·m², arm = 0.20 m) |
| **Gap** | 0.8 m (G/L = 0.8) |
| **Control inputs** | Rotor thrusts T₁, T₂ ∈ [0, 15] N |
| **Result** | ✅ Zero collisions — cable and drone body clear the gap |

The controller discovers an aggressive tilting strategy (peak pitch ≈ 48°) that generates forward acceleration and swing-back. The drone body (29.9 cm vertical extent at max tilt) fits comfortably within the 0.8 m gap with clearances of 31.9 cm (bottom) and 13.9 cm (top).

```
📁 Experiment 1 - Quad 0.8m gap/
├── mppi_quad2d.py              # Main simulation script
├── mppi_quad2d_analysis.png    # 6-panel trajectory + control plots
└── mppi_quad2d.gif             # Animated traversal
```

### Experiment 2 — Full Quadrotor, G = 0.4 m

| | |
|---|---|
| **Drone model** | 2D rigid body (same parameters) |
| **Gap** | 0.4 m (G/L = 0.4) |
| **Control inputs** | Rotor thrusts T₁, T₂ ∈ [0, 15] N |
| **Result** | ⚠️ Cable clears — but drone body collides with gap walls |

The cable nodes pass through the gap with clearances of 1.5 cm (bottom) and 3.4 cm (top). However, the pitch angle required for adequate swing-back (≈ 57°) gives the drone body a vertical extent of 33.4 cm — consuming 83.5% of the 0.4 m gap. The rotor tips collide with the obstacle walls. This reveals the **attitude–aperture coupling problem**: generating forward thrust requires pitching, which increases the drone's physical footprint.

```
📁 Experiment 2 - Quad 0.4m gap/
├── mppi_quad2d_04gap.py
├── mppi_quad2d_04gap_analysis.png
└── mppi_quad2d_04gap.gif
```

### Experiment 3 — Point Mass, G = 0.8 m and G = 0.4 m

| | |
|---|---|
| **Drone model** | Point mass (zero physical extent) |
| **Gap** | 0.8 m and 0.4 m |
| **Control inputs** | Direct accelerations aₓ, aᵧ |
| **Result** | ✅ Zero collisions in both gaps |

The point-mass model succeeds in both gaps because it has no physical extent and no attitude dynamics. This demonstrates that the point-mass abstraction is **useful but dangerous**: it produces feasible-looking trajectories that are unexecutable by any real quadrotor with nonzero arm length in tight gaps.

```
📁 Experiment 3 - Point mass/
├── mppi_vec.py                 # 0.8m gap
├── mppi_vec_04gap.py           # 0.4m gap
├── mppi_xpbd_analysis.png
└── mppi_xpbd.gif
```

---

## MPPI Tuning Across Experiments

The MPPI hyperparameters require scenario-specific adjustment as the gap narrows:

| Parameter | Exp. 1 (Quad 0.8m) | Exp. 2 (Quad 0.4m) | Exp. 3 (Point mass) |
|---|---|---|---|
| Samples K | 512 | 512 | 256 |
| Noise σ | 2.5 N | 3.5 N | 5.0 / 4.0 m/s² |
| Temperature λ | 10 | 15 | 20 |
| Progress weight | 2.0 | 4.0 | 2.0 |
| Terminal weight | 25 | 35 | 15 |
| Corner radius | 0.20 m | 0.15 m | — |
| Corner weight | 3000 | 1500 | — |

**Why Experiment 2 needs different tuning:** The feasible configuration space in the 0.4 m gap is extremely narrow. Higher noise (σ = 3.5) explores more aggressively, higher temperature (λ = 15) prevents the optimizer from collapsing onto a single rollout, and stronger progress/terminal weights push the drone through the repulsive barrier. The corner radius is reduced because the four gap corners are only 0.4 m apart — overlapping repulsive fields at larger radii would completely block the aperture.

---

## Dynamics Models

### Full 2D Quadrotor (Experiments 1 & 2)

State: `[x, y, θ, vx, vy, ω]` — Control: `[T₁, T₂]`

```
m · ẍ  = -(T₁ + T₂) · sin(θ)
m · ÿ  =  (T₁ + T₂) · cos(θ) - m·g
I · θ̈  =  (T₂ - T₁) · d
```

### Point Mass (Experiment 3)

State: `[x, y, vx, vy]` — Control: `[aₓ, aᵧ]`

```
ẍ = aₓ       (bounded by ±12 m/s²)
ÿ = aᵧ       (bounded by ±10 m/s²)
```

### XPBD Cable (All Experiments)

8-node particle chain with Verlet integration and iterative constraint projection (20 Gauss-Seidel iterations per physics step, 100 Hz physics within 10 Hz control).

---

## Requirements

```
python >= 3.8
numpy
matplotlib
```

## Usage

Each experiment is self-contained in a single Python file:

```bash
# Experiment 1: Full quadrotor, 0.8m gap
python "Experiment 1 - Quad 0.8m gap/mppi_quad2d.py"

# Experiment 2: Full quadrotor, 0.4m gap
python "Experiment 2 - Quad 0.4m gap/mppi_quad2d_04gap.py"

# Experiment 3: Point mass
python "Experiment 3 - Point mass/mppi_vec.py"         # 0.8m gap
python "Experiment 3 - Point mass/mppi_vec_04gap.py"    # 0.4m gap
```

Each script runs the full MPPI-MPC loop, prints live progress, reports collision analysis, and saves trajectory plots (`.png`) and animations (`.gif`).

**Typical runtimes** (single CPU core, NumPy):
- Full quadrotor (K=512): ~1 s/step, ~60–100 s total
- Point mass (K=256): ~0.6 s/step, ~20 s total

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{agile-quadrotor-cable-mpc,
  author = {},
  title  = {Agile Quadrotor Cable MPC: MPPI-based Navigation Through Narrow Apertures with Deformable Cable Dynamics},
  year   = {2025},
  url    = {https://github.com/Julestevez/agile-quadrotor-cable-MPC}
}
```

## License

MIT
