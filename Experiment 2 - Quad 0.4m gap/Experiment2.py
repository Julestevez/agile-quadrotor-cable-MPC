"""
MPPI-MPC + XPBD Cable: Narrow Window Traversal (Full 2D Quadrotor Dynamics)
=============================================================================

UPGRADE from point-mass to full 2D rigid-body quadrotor:

  State:  [x, y, θ, vx, vy, ω]
    - (x, y): CoM position
    - θ: pitch angle (positive = nose up / CCW)
    - (vx, vy): translational velocity
    - ω: angular velocity

  Control: [T1, T2]
    - T1: left rotor thrust (along body z-axis)
    - T2: right rotor thrust (along body z-axis)

  Dynamics (Newton-Euler, 2D):
    m * ax = -(T1 + T2) * sin(θ)
    m * ay = (T1 + T2) * cos(θ) - m*g
    I * α  = (T2 - T1) * d

  where d = arm length (rotor to CoM distance).

  The cable attaches at the drone CoM (simplified — no offset).

APPROACH:
- Controller: MPPI (sampling-based MPC, no gradients needed)
- Cable model: XPBD/Verlet (8-node deformable cable)
- Drone: full 2D rigid body with [T1, T2] control
- All K rollouts simulated in parallel via vectorized NumPy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import time

# ============================================================
# PARAMETERS
# ============================================================
L_CABLE = 1.0
N_NODES = 8
SEG_LEN = L_CABLE / (N_NODES - 1)
OBS_W = 1.0
OBS_BOT = 1.4
OBS_TOP = 1.8
GAP = OBS_TOP - OBS_BOT  # 0.4m

START = np.array([-2.2, 1.5])
END = np.array([2.5, 1.5])
GRAV = 9.81

# Timing
DT_PHYS = 0.01       # physics step (100 Hz)
DT_CTRL = 0.1        # control step (10 Hz)
N_SUB = int(DT_CTRL / DT_PHYS)  # 10 sub-steps
T_TOTAL = 10.0
N_CTRL_MAX = int(T_TOTAL / DT_CTRL)  # 60 steps

# ---- Quadrotor physical parameters (2D) ----
MASS = 1.5            # kg (total quadrotor mass)
INERTIA = 0.06        # kg·m² (pitch moment of inertia, larger = more stable)
ARM_LEN = 0.20        # m (rotor to CoM distance, ~40cm tip-to-tip)

# Thrust limits per rotor
T_MIN = 0.0           # N (no negative thrust)
T_MAX = 15.0          # N (max per rotor; hover ≈ mg/2 = 7.36 N each)

# Hover thrust per rotor (for reference)
T_HOVER = MASS * GRAV / 2.0  # ≈ 7.36 N

# Max pitch angle for safety (soft constraint in cost)
THETA_MAX = np.radians(60)  # 60 degrees

# XPBD iterations (per physics step)
XPBD_ITERS = 20

# MPPI parameters
K_SAMPLES = 512       # number of rollout samples (more for full dynamics)
HORIZON = 15          # control steps ahead (1.5s)
SIGMA_T1 = 3.5        # exploration noise std for T1 (more for tight gap)
SIGMA_T2 = 3.5        # exploration noise std for T2
LAMBDA = 15.0         # temperature

print("=" * 65)
print("MPPI-MPC + XPBD Cable (Full 2D Quadrotor Dynamics)")
print("=" * 65)
print(f"Quadrotor: m={MASS}kg, I={INERTIA}kg·m², arm={ARM_LEN}m")
print(f"Thrust: [{T_MIN:.1f}, {T_MAX:.1f}] N per rotor, hover={T_HOVER:.2f}N")
print(f"Cable: {L_CABLE}m, {N_NODES} nodes, {XPBD_ITERS} XPBD iters")
print(f"Gap: {GAP}m < Cable: {L_CABLE}m → swing required")
print(f"MPPI: K={K_SAMPLES}, H={HORIZON}, σ_T={SIGMA_T1:.1f}, λ={LAMBDA}")
print(f"Control: {1/DT_CTRL:.0f}Hz, Physics: {1/DT_PHYS:.0f}Hz, {N_SUB} sub-steps")
print(f"Total: {T_TOTAL}s, {N_CTRL_MAX} control steps max")
print()

# ============================================================
# BATCHED 2D QUADROTOR DYNAMICS
# ============================================================
def batched_quadrotor_step(pos, vel, theta, omega, T1, T2, dt):
    """
    One physics step for K rollouts of the 2D quadrotor.

    pos:   (K, 2) — [x, y] position of CoM
    vel:   (K, 2) — [vx, vy] velocity
    theta: (K,)   — pitch angle
    omega: (K,)   — angular velocity
    T1:    (K,)   — left rotor thrust
    T2:    (K,)   — right rotor thrust
    dt:    scalar — timestep

    Returns updated pos, vel, theta, omega
    """
    # Total thrust
    T_total = T1 + T2  # (K,)

    # Translational acceleration in world frame
    #   ax = -(T1+T2) * sin(θ) / m
    #   ay = (T1+T2) * cos(θ) / m - g
    sin_th = np.sin(theta)  # (K,)
    cos_th = np.cos(theta)

    ax = -T_total * sin_th / MASS           # (K,)
    ay = T_total * cos_th / MASS - GRAV     # (K,)

    # Angular acceleration
    #   α = (T2 - T1) * d / I
    alpha = (T2 - T1) * ARM_LEN / INERTIA   # (K,)

    # Semi-implicit Euler integration
    vel[:, 0] += ax * dt
    vel[:, 1] += ay * dt
    pos += vel * dt

    omega += alpha * dt
    theta += omega * dt

    return pos, vel, theta, omega


# ============================================================
# BATCHED XPBD SIMULATION
# ============================================================
def batched_xpbd_step(cable_pos, cable_prev, drone_pos, dt):
    """
    One physics step for K rollouts simultaneously.

    cable_pos:  (K, N_NODES, 2) current positions
    cable_prev: (K, N_NODES, 2) previous positions
    drone_pos:  (K, 2) drone CoM position
    """
    # Verlet integration
    vel = cable_pos - cable_prev
    new_prev = cable_pos.copy()
    cable_pos = cable_pos + vel * 0.995 + np.array([0, -GRAV]) * dt**2

    # Pin node 0 to drone CoM
    cable_pos[:, 0, :] = drone_pos

    # XPBD constraint projection
    for _ in range(XPBD_ITERS):
        for i in range(N_NODES - 1):
            delta = cable_pos[:, i+1, :] - cable_pos[:, i, :]
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            dist = np.maximum(dist, 1e-6)
            corr = (dist - SEG_LEN) / dist * 0.5 * delta

            if i == 0:
                cable_pos[:, i+1, :] -= corr * 2.0
            else:
                cable_pos[:, i, :] += corr
                cable_pos[:, i+1, :] -= corr

    return cable_pos, new_prev


# ============================================================
# BATCHED FULL SIMULATION
# ============================================================
def batched_simulate(drone_pos, drone_vel, drone_theta, drone_omega,
                     cable_pos, cable_prev, controls):
    """
    Simulate K rollouts for H control steps.

    drone_pos:   (K, 2)
    drone_vel:   (K, 2)
    drone_theta: (K,)
    drone_omega: (K,)
    cable_pos:   (K, N_NODES, 2)
    cable_prev:  (K, N_NODES, 2)
    controls:    (K, H, 2) — [T1, T2] per step

    Returns:
        drone_traj:  (K, H+1, 2)
        cable_traj:  (K, H+1, N_NODES, 2)
        theta_traj:  (K, H+1)
    """
    K, H = controls.shape[0], controls.shape[1]

    drone_traj = np.zeros((K, H + 1, 2))
    cable_traj = np.zeros((K, H + 1, N_NODES, 2))
    theta_traj = np.zeros((K, H + 1))

    drone_traj[:, 0] = drone_pos.copy()
    cable_traj[:, 0] = cable_pos.copy()
    theta_traj[:, 0] = drone_theta.copy()

    dp = drone_pos.copy()
    dv = drone_vel.copy()
    dth = drone_theta.copy()
    dom = drone_omega.copy()
    cp = cable_pos.copy()
    cprev = cable_prev.copy()

    for k in range(H):
        T1 = np.clip(controls[:, k, 0], T_MIN, T_MAX)  # (K,)
        T2 = np.clip(controls[:, k, 1], T_MIN, T_MAX)

        for _ in range(N_SUB):
            dp, dv, dth, dom = batched_quadrotor_step(
                dp, dv, dth, dom, T1, T2, DT_PHYS)
            cp, cprev = batched_xpbd_step(cp, cprev, dp, DT_PHYS)

        drone_traj[:, k + 1] = dp.copy()
        cable_traj[:, k + 1] = cp.copy()
        theta_traj[:, k + 1] = dth.copy()

    return drone_traj, cable_traj, theta_traj


# ============================================================
# BATCHED COST FUNCTION
# ============================================================
def batched_cost(drone_traj, cable_traj, theta_traj, controls, target):
    """
    Compute cost for all K rollouts simultaneously.
    Returns: costs (K,)
    """
    K, Hp1 = drone_traj.shape[0], drone_traj.shape[1]

    costs = np.zeros(K)

    # 1. Progress: sum of distances to target
    dist_to_target = np.linalg.norm(drone_traj - target, axis=2)  # (K, H+1)
    costs += 4.0 * np.sum(dist_to_target * DT_CTRL, axis=1)

    # 2. Terminal cost
    costs += 35.0 * dist_to_target[:, -1]**2

    # 3. Collision penalty for ALL cable nodes (increased margin)
    for t in range(Hp1):
        nodes = cable_traj[:, t, :, :]
        px = nodes[:, :, 0]
        py = nodes[:, :, 1]

        in_column = np.abs(px) < OBS_W / 2

        bot_viol = np.maximum(OBS_BOT + 0.06 - py, 0)  # 6cm hard margin
        bot_pen = np.sum(bot_viol**2 * in_column, axis=1)

        top_viol = np.maximum(py - OBS_TOP + 0.06, 0)   # 6cm hard margin
        top_pen = np.sum(top_viol**2 * in_column, axis=1)

        costs += 5000.0 * (bot_pen + top_pen)

    # 4. Soft proximity penalty (wider margin, stronger)
    for t in range(Hp1):
        nodes = cable_traj[:, t, :, :]
        px = nodes[:, :, 0]
        py = nodes[:, :, 1]
        in_col = np.abs(px) < OBS_W / 2 + 0.3

        margin_bot = np.maximum(OBS_BOT + 0.15 - py, 0)  # 15cm soft margin
        margin_top = np.maximum(py - OBS_TOP + 0.15, 0)
        costs += 400.0 * np.sum((margin_bot**2 + margin_top**2) * in_col, axis=1) * DT_CTRL

    # 4b. Gap centering: when near/in gap, penalize deviation from gap center
    gap_center = (OBS_BOT + OBS_TOP) / 2.0  # = 1.6m
    for t in range(Hp1):
        dx_drone = drone_traj[:, t, 0]
        dy_drone = drone_traj[:, t, 1]
        near_gap = np.abs(dx_drone) < OBS_W / 2 + 0.5  # approaching or in gap
        dev_from_center = (dy_drone - gap_center)**2
        costs += 5.0 * dev_from_center * near_gap

    # 4c. Corner proximity penalty for ALL cable nodes
    #     The 4 corners of the gap are the most dangerous points.
    #     Penalize any cable node that gets close to any corner.
    corners = np.array([
        [-OBS_W/2, OBS_BOT],  # bottom-left inner corner
        [+OBS_W/2, OBS_BOT],  # bottom-right inner corner  ← payload passes here
        [-OBS_W/2, OBS_TOP],  # top-left inner corner
        [+OBS_W/2, OBS_TOP],  # top-right inner corner
    ])  # (4, 2)
    CORNER_RADIUS = 0.15  # smaller radius for tight 0.4m gap
    for t in range(Hp1):
        nodes = cable_traj[:, t, :, :]  # (K, N_NODES, 2)
        for ci in range(4):
            # Distance from each node to this corner
            diff = nodes - corners[ci]  # (K, N_NODES, 2)
            dist_to_corner = np.linalg.norm(diff, axis=2)  # (K, N_NODES)
            # Penalty: inverse-ish, only within CORNER_RADIUS
            penetration = np.maximum(CORNER_RADIUS - dist_to_corner, 0)  # (K, N_NODES)
            costs += 1500.0 * np.sum(penetration**2, axis=1)

    # 5. Control effort (penalize deviation from hover thrust)
    dT1 = controls[:, :, 0] - T_HOVER
    dT2 = controls[:, :, 1] - T_HOVER
    costs += 0.01 * np.sum(dT1**2 + dT2**2, axis=1) * DT_CTRL

    # 5b. Thrust differential penalty (penalize asymmetry → torque)
    dT_diff = controls[:, :, 0] - controls[:, :, 1]
    costs += 0.02 * np.sum(dT_diff**2, axis=1) * DT_CTRL

    # 6. Drone out of bounds
    for t in range(Hp1):
        too_low = np.maximum(0.2 - drone_traj[:, t, 1], 0)
        too_high = np.maximum(drone_traj[:, t, 1] - 2.8, 0)
        costs += 500.0 * (too_low**2 + too_high**2)

    # 7. Pitch angle penalty (soft limit to prevent flipping)
    for t in range(Hp1):
        th = theta_traj[:, t]
        pitch_viol = np.maximum(np.abs(th) - THETA_MAX, 0)
        costs += 500.0 * pitch_viol**2

    # 8. Angular velocity penalty (smooth flight, heavy damping)
    for t in range(1, Hp1):
        dtheta = np.abs(theta_traj[:, t] - theta_traj[:, t-1]) / DT_CTRL
        costs += 5.0 * dtheta**2 * DT_CTRL

    # 9. Pitch from vertical penalty (prefer upright flight)
    for t in range(Hp1):
        costs += 3.0 * theta_traj[:, t]**2

    return costs


# ============================================================
# MPPI CONTROLLER (vectorized, thrust inputs)
# ============================================================
class MPPIVec:
    def __init__(self):
        # Nominal control sequence: start at hover thrust
        self.U = np.full((HORIZON, 2), T_HOVER)

    def get_action(self, drone_pos, drone_vel, drone_theta, drone_omega,
                   cable_pos, cable_prev):
        K = K_SAMPLES
        H = HORIZON

        # Sample noise around nominal sequence
        noise = np.random.randn(K, H, 2) * np.array([SIGMA_T1, SIGMA_T2])

        # Build control sequences: U + noise
        U_samples = self.U[None, :, :] + noise   # (K, H, 2)
        U_samples[:, :, 0] = np.clip(U_samples[:, :, 0], T_MIN, T_MAX)
        U_samples[:, :, 1] = np.clip(U_samples[:, :, 1], T_MIN, T_MAX)

        # Broadcast initial state to K rollouts
        dp = np.tile(drone_pos, (K, 1))
        dv = np.tile(drone_vel, (K, 1))
        dth = np.full(K, drone_theta)
        dom = np.full(K, drone_omega)
        cp = np.tile(cable_pos[None, :, :], (K, 1, 1))
        cprev = np.tile(cable_prev[None, :, :], (K, 1, 1))

        # Simulate all K rollouts in parallel
        d_traj, c_traj, th_traj = batched_simulate(
            dp, dv, dth, dom, cp, cprev, U_samples)

        # Compute costs
        costs = batched_cost(d_traj, c_traj, th_traj, U_samples, END)

        # MPPI weights
        min_c = np.min(costs)
        w = np.exp(-(costs - min_c) / LAMBDA)
        w /= np.sum(w) + 1e-10

        # Weighted noise update
        weighted = np.sum(w[:, None, None] * noise, axis=0)  # (H, 2)
        self.U += weighted
        self.U[:, 0] = np.clip(self.U[:, 0], T_MIN, T_MAX)
        self.U[:, 1] = np.clip(self.U[:, 1], T_MIN, T_MAX)

        # Extract first action
        action = self.U[0].copy()

        # Shift warm start
        self.U[:-1] = self.U[1:]
        self.U[-1] = T_HOVER  # default to hover

        return action


# ============================================================
# SINGLE-STEP XPBD (for real system, non-batched)
# ============================================================
def xpbd_step_single(cable_pos, cable_prev, drone_pos, dt):
    vel = cable_pos - cable_prev
    new_prev = cable_pos.copy()
    cable_pos = cable_pos + vel * 0.995 + np.array([0, -GRAV]) * dt**2
    cable_pos[0] = drone_pos
    for _ in range(XPBD_ITERS):
        for i in range(N_NODES - 1):
            d = cable_pos[i+1] - cable_pos[i]
            dn = np.linalg.norm(d)
            if dn > 1e-6:
                c = (dn - SEG_LEN) / dn * 0.5
                m = d * c
                if i == 0:
                    cable_pos[i+1] -= m * 2
                else:
                    cable_pos[i] += m
                    cable_pos[i+1] -= m
    return cable_pos, new_prev


# ============================================================
# MAIN LOOP
# ============================================================

# Initialize drone state
drone_pos = START.copy()
drone_vel = np.zeros(2)
drone_theta = 0.0    # pitch angle
drone_omega = 0.0    # angular velocity

# Initialize cable
cable_pos = np.zeros((N_NODES, 2))
cable_prev = np.zeros((N_NODES, 2))
for i in range(N_NODES):
    cable_pos[i] = START + np.array([0, -L_CABLE * i / (N_NODES - 1)])
    cable_prev[i] = cable_pos[i].copy()

# Settle cable under gravity
print("Settling cable...")
for _ in range(500):
    vel_c = cable_pos - cable_prev
    cable_prev = cable_pos.copy()
    cable_pos += vel_c * 0.995 + np.array([0, -GRAV]) * DT_PHYS**2
    cable_pos[0] = START
    for _ in range(XPBD_ITERS):
        for i in range(N_NODES - 1):
            d = cable_pos[i+1] - cable_pos[i]
            dn = np.linalg.norm(d)
            if dn > 1e-6:
                c = (dn - SEG_LEN) / dn * 0.5
                m = d * c
                if i == 0:
                    cable_pos[i+1] -= m * 2
                else:
                    cable_pos[i] += m
                    cable_pos[i+1] -= m

# Init controller
mppi = MPPIVec()

# Storage
N_CTRL = N_CTRL_MAX
hist_drone = np.zeros((N_CTRL + 1, 2))
hist_theta = np.zeros(N_CTRL + 1)
hist_omega = np.zeros(N_CTRL + 1)
hist_cable = np.zeros((N_CTRL + 1, N_NODES, 2))
hist_ctrl = np.zeros((N_CTRL, 2))  # [T1, T2]
hist_vel = np.zeros((N_CTRL + 1, 2))

hist_drone[0] = drone_pos.copy()
hist_theta[0] = drone_theta
hist_omega[0] = drone_omega
hist_cable[0] = cable_pos.copy()
hist_vel[0] = drone_vel.copy()

print("\nRunning MPPI-MPC (full 2D quadrotor)...")
t0_total = time.time()

for step in range(N_CTRL):
    t_sim = step * DT_CTRL
    dist = np.linalg.norm(drone_pos - END)

    # Early stop
    if dist < 0.2 and np.linalg.norm(drone_vel) < 0.3 and t_sim > 2.0:
        print(f"  Step {step} t={t_sim:.1f}s: TARGET REACHED")
        hist_drone[step+1:] = drone_pos
        hist_theta[step+1:] = drone_theta
        hist_omega[step+1:] = drone_omega
        hist_cable[step+1:] = cable_pos
        hist_ctrl[step:] = T_HOVER
        hist_vel[step+1:] = drone_vel
        N_CTRL = step + 1
        break

    # MPPI
    t0 = time.time()
    action = mppi.get_action(drone_pos, drone_vel, drone_theta, drone_omega,
                             cable_pos, cable_prev)
    dt_mppi = time.time() - t0

    hist_ctrl[step] = action

    # Apply to real system: full 2D quadrotor dynamics
    T1_applied = np.clip(action[0], T_MIN, T_MAX)
    T2_applied = np.clip(action[1], T_MIN, T_MAX)

    for _ in range(N_SUB):
        # Quadrotor dynamics
        T_total = T1_applied + T2_applied
        ax = -T_total * np.sin(drone_theta) / MASS
        ay = T_total * np.cos(drone_theta) / MASS - GRAV
        alpha = (T2_applied - T1_applied) * ARM_LEN / INERTIA

        drone_vel[0] += ax * DT_PHYS
        drone_vel[1] += ay * DT_PHYS
        drone_pos += drone_vel * DT_PHYS

        drone_omega += alpha * DT_PHYS
        drone_theta += drone_omega * DT_PHYS

        # XPBD cable step
        cable_pos, cable_prev = xpbd_step_single(
            cable_pos, cable_prev, drone_pos, DT_PHYS)

    hist_drone[step + 1] = drone_pos.copy()
    hist_theta[step + 1] = drone_theta
    hist_omega[step + 1] = drone_omega
    hist_cable[step + 1] = cable_pos.copy()
    hist_vel[step + 1] = drone_vel.copy()

    if step % 10 == 0:
        print(f"  Step {step:3d} t={t_sim:.1f}s: "
              f"pos=({drone_pos[0]:+.2f},{drone_pos[1]:.2f}) "
              f"θ={np.degrees(drone_theta):+.1f}° "
              f"ω={np.degrees(drone_omega):+.1f}°/s "
              f"T=({T1_applied:.1f},{T2_applied:.1f})N "
              f"d={dist:.2f} dt={dt_mppi*1000:.0f}ms")

t_total_run = time.time() - t0_total
print(f"\nTotal: {t_total_run:.1f}s | Per step: {t_total_run/N_CTRL*1000:.0f}ms")

# Trim
hist_drone = hist_drone[:N_CTRL + 1]
hist_theta = hist_theta[:N_CTRL + 1]
hist_omega = hist_omega[:N_CTRL + 1]
hist_cable = hist_cable[:N_CTRL + 1]
hist_ctrl = hist_ctrl[:N_CTRL]
hist_vel = hist_vel[:N_CTRL + 1]

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

nc = 0; cb = 999; ct = 999
worst_bot_step = -1; worst_top_step = -1
print("\nDetailed clearance near gap (|x| < OBS_W/2):")
for k in range(len(hist_cable)):
    for i in range(N_NODES):
        px, py = hist_cable[k, i]
        if abs(px) < OBS_W / 2:
            margin_bot = py - OBS_BOT
            margin_top = OBS_TOP - py
            if margin_bot < cb:
                cb = margin_bot; worst_bot_step = k
            if margin_top < ct:
                ct = margin_top; worst_top_step = k
            if py < OBS_BOT or py > OBS_TOP:
                nc += 1; break

# Print timesteps near min clearance
print(f"  Worst bot clearance: {cb:.4f}m at step {worst_bot_step} (t={worst_bot_step*DT_CTRL:.2f}s)")
print(f"  Worst top clearance: {ct:.4f}m at step {worst_top_step} (t={worst_top_step*DT_CTRL:.2f}s)")

# Show drone y and all node y for worst steps
for label, ws in [("BOT", worst_bot_step), ("TOP", worst_top_step)]:
    if ws >= 0:
        dy = hist_drone[ws, 1]
        nodes_y = hist_cable[ws, :, 1]
        nodes_x = hist_cable[ws, :, 0]
        in_gap = np.abs(nodes_x) < OBS_W / 2
        print(f"  {label} worst step {ws}: drone_y={dy:.3f}, "
              f"nodes_y_in_gap={nodes_y[in_gap]}, "
              f"θ={np.degrees(hist_theta[ws]):.1f}°")

print(f"Collisions: {nc} frames" if nc else "✓ ZERO collisions!")
if cb < 999:
    print(f"Min clearance: bot={cb:.4f}m, top={ct:.4f}m")
print(f"Final pos: ({hist_drone[-1,0]:.2f}, {hist_drone[-1,1]:.2f})")
print(f"Final dist: {np.linalg.norm(hist_drone[-1]-END):.3f}m")
print(f"Max pitch: {np.degrees(np.max(np.abs(hist_theta))):.1f}°")
print(f"Max |ω|: {np.degrees(np.max(np.abs(hist_omega))):.1f}°/s")

# ============================================================
# PLOTS (3x2 grid: trajectory, thrusts, pitch, velocity, payload, clearance)
# ============================================================
print("\nPlotting...")
ta = np.linspace(0, N_CTRL * DT_CTRL, N_CTRL + 1)
tc = np.linspace(0, (N_CTRL - 1) * DT_CTRL, N_CTRL)

fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# --- (0,0) Trajectory ---
ax = axes[0, 0]
ax.set_aspect('equal')
ax.set_xlim(-2.8, 3.2); ax.set_ylim(-0.2, 3.2)
ax.set_title('MPPI + XPBD: Full 2D Quadrotor', fontsize=12, fontweight='bold')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.grid(True, alpha=0.12)
ax.add_patch(patches.Rectangle((-OBS_W/2, 0), OBS_W, OBS_BOT,
             facecolor='#CC0000', edgecolor='darkred', zorder=10))
ax.add_patch(patches.Rectangle((-OBS_W/2, OBS_TOP), OBS_W, 1.0,
             facecolor='#CC0000', edgecolor='darkred', zorder=10))
ax.plot(hist_drone[:, 0], hist_drone[:, 1], '-', color='green', lw=1, alpha=0.5, label='Drone CoM')
pay = hist_cable[:, -1, :]
ax.plot(pay[:, 0], pay[:, 1], '-', color='#FFD700', lw=1.5, alpha=0.7, label='Payload')

# Draw drone body at selected timesteps
for k in np.linspace(0, N_CTRL, min(22, N_CTRL), dtype=int):
    if k >= len(hist_cable): break
    nd = hist_cable[k]
    ax.plot(nd[:, 0], nd[:, 1], 'k-', lw=0.5, alpha=0.4)
    ax.plot(nd[-1, 0], nd[-1, 1], 'o', color='#FFD700', markeredgecolor='k', ms=3.5, zorder=5)

    # Draw drone body (oriented bar)
    cx, cy = hist_drone[k]
    th = hist_theta[k]
    dx = ARM_LEN * np.cos(th)
    dy = ARM_LEN * np.sin(th)
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
            '-', color='#2471a3', lw=2.5, alpha=0.6, solid_capstyle='round')
    ax.plot(cx, cy, 's', color='blue', ms=2.5, zorder=6)
ax.legend(fontsize=8, loc='upper left')

# --- (0,1) Thrust commands ---
ax = axes[0, 1]
ax.set_title('Rotor Thrusts')
ax.plot(tc, hist_ctrl[:, 0], 'b-', lw=1.2, label='$T_1$ (left)')
ax.plot(tc, hist_ctrl[:, 1], 'r-', lw=1.2, label='$T_2$ (right)')
ax.axhline(T_HOVER, color='gray', ls='--', alpha=0.5, label=f'hover ({T_HOVER:.1f}N)')
ax.axhline(T_MAX, color='k', ls=':', alpha=0.3)
ax.axhline(T_MIN, color='k', ls=':', alpha=0.3)
ax.set_xlabel('Time [s]'); ax.set_ylabel('Thrust [N]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

# --- (1,0) Pitch angle ---
ax = axes[1, 0]
ax.set_title('Pitch Angle θ')
ax.plot(ta, np.degrees(hist_theta), 'purple', lw=1.5)
ax.axhline(0, color='k', alpha=0.2)
ax.axhline(np.degrees(THETA_MAX), color='r', ls='--', alpha=0.3, label=f'limit ±{np.degrees(THETA_MAX):.0f}°')
ax.axhline(-np.degrees(THETA_MAX), color='r', ls='--', alpha=0.3)
ax.set_xlabel('Time [s]'); ax.set_ylabel('θ [°]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

# --- (1,1) Velocity ---
ax = axes[1, 1]
ax.set_title('Drone Velocity')
ax.plot(ta, hist_vel[:, 0], 'b-', lw=1.2, label='$v_x$')
ax.plot(ta, hist_vel[:, 1], 'r-', lw=1.2, label='$v_y$')
speed = np.linalg.norm(hist_vel, axis=1)
ax.plot(ta, speed, 'k--', lw=1, alpha=0.5, label='|v|')
ax.axhline(0, color='k', alpha=0.2)
ax.set_xlabel('Time [s]'); ax.set_ylabel('[m/s]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

# --- (2,0) Drone position ---
ax = axes[2, 0]
ax.set_title('Drone Position')
ax.plot(ta, hist_drone[:, 0], 'b-', lw=1.2, label='x')
ax.plot(ta, hist_drone[:, 1], 'r-', lw=1.2, label='y')
ax.axhline(END[0], color='b', ls='--', alpha=0.3)
ax.axhline(END[1], color='r', ls='--', alpha=0.3)
ax.set_xlabel('Time [s]'); ax.set_ylabel('[m]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

# --- (2,1) Payload relative ---
ax = axes[2, 1]
ax.set_title('Payload Relative to Drone')
ax.plot(ta, pay[:, 0] - hist_drone[:, 0], 'b-', lw=1.2, label='Δx')
ax.plot(ta, pay[:, 1] - hist_drone[:, 1], 'r-', lw=1.2, label='Δy')
ax.axhline(0, color='k', alpha=0.2)
ax.axhline(-L_CABLE, color='r', ls='--', alpha=0.3)
ax.set_xlabel('Time [s]'); ax.set_ylabel('[m]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('/home/claude/mppi_quad2d_04gap_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: mppi_quad2d_04gap_analysis.png")

# ============================================================
# GIF with drone body orientation
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_aspect('equal')
ax2.set_xlim(-2.8, 3.2); ax2.set_ylim(-0.2, 3.2)
ax2.set_title('MPPI-MPC + XPBD Cable (Full 2D Quadrotor)')
ax2.add_patch(patches.Rectangle((-OBS_W/2, 0), OBS_W, OBS_BOT, color='#CC0000', zorder=10))
ax2.add_patch(patches.Rectangle((-OBS_W/2, OBS_TOP), OBS_W, 1.0, color='#CC0000', zorder=10))

ln, = ax2.plot([], [], 'k-', lw=1.5)
body_line, = ax2.plot([], [], '-', color='#2471a3', lw=4, solid_capstyle='round', zorder=20)
dr, = ax2.plot([], [], 's', color='blue', ms=5, zorder=21)
pl, = ax2.plot([], [], 'o', color='#FFD700', markeredgecolor='k', ms=6, zorder=20)
tr, = ax2.plot([], [], '-', color='#FFD700', alpha=0.4, lw=1)
txx, tyy = [], []
nf = len(hist_cable)

def anim(f):
    k = min(f, nf - 1)
    nd = hist_cable[k]
    cx, cy = hist_drone[k]
    th = hist_theta[k]
    dx = ARM_LEN * np.cos(th)
    dy = ARM_LEN * np.sin(th)

    ln.set_data(nd[:, 0], nd[:, 1])
    body_line.set_data([cx - dx, cx + dx], [cy - dy, cy + dy])
    dr.set_data([cx], [cy])
    pl.set_data([nd[-1, 0]], [nd[-1, 1]])
    txx.append(nd[-1, 0]); tyy.append(nd[-1, 1])
    tr.set_data(txx, tyy)
    return ln, body_line, dr, pl, tr

ani = animation.FuncAnimation(fig2, anim, frames=nf, interval=100, blit=True)
ani.save('/home/claude/mppi_quad2d_04gap.gif', writer='pillow', fps=10)
print("Saved: mppi_quad2d_04gap.gif")
plt.close('all')
print("\n✓ Done!")
