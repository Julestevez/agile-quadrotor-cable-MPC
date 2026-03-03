"""
MPPI-MPC + XPBD Cable: Narrow Window Traversal (Batched/Vectorized)
====================================================================

KEY IDEA: Instead of simulating K rollouts one-by-one in Python loops,
we simulate ALL K rollouts simultaneously using batched NumPy operations.

State for K rollouts:
  drone_pos:  (K, 2)
  drone_vel:  (K, 2)  
  cable_pos:  (K, N_NODES, 2)
  cable_prev: (K, N_NODES, 2)

This makes MPPI tractable in pure Python/NumPy.

APPROACH:
- Controller: MPPI (sampling-based MPC, no gradients needed)
- Cable model: XPBD/Verlet (same physics as hallo_dron.py)
- Drone: point mass with acceleration control (ax, ay)
- All K rollouts simulated in parallel via vectorized NumPy

SIMPLIFICATIONS:
- Drone as point mass (acceleration control, not thrust/pitch)
  → Assumes perfect low-level attitude controller
- 2D planar
- N_NODES=8 cable nodes, 20 XPBD iterations (vs 15/50 in original)
- Control at 10 Hz, physics at 100 Hz (10 sub-steps)
- Horizon: 15 steps = 1.5s lookahead
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
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
GAP = OBS_TOP - OBS_BOT  # 0.8m

START = np.array([-2.2, 1.5])
END = np.array([2.5, 1.5])
G = 9.81

# Timing
DT_PHYS = 0.01       # physics step (100 Hz)
DT_CTRL = 0.1        # control step (10 Hz)
N_SUB = int(DT_CTRL / DT_PHYS)  # 10 sub-steps
T_TOTAL = 6.0
N_CTRL = int(T_TOTAL / DT_CTRL)  # 60 steps

# Acceleration limits
AX_MAX = 12.0
AY_MAX = 10.0

# XPBD iterations (per physics step)
XPBD_ITERS = 20

# MPPI parameters
K_SAMPLES = 256       # number of rollout samples
HORIZON = 15          # control steps ahead (1.5s)
SIGMA_AX = 5.0        # exploration noise std
SIGMA_AY = 4.0
LAMBDA = 20.0         # temperature

print("=" * 65)
print("MPPI-MPC + XPBD Cable (Vectorized)")
print("=" * 65)
print(f"Cable: {L_CABLE}m, {N_NODES} nodes, {XPBD_ITERS} XPBD iters")
print(f"Gap: {GAP}m < Cable: {L_CABLE}m → swing required")
print(f"MPPI: K={K_SAMPLES}, H={HORIZON}, σ=({SIGMA_AX},{SIGMA_AY}), λ={LAMBDA}")
print(f"Control: {1/DT_CTRL:.0f}Hz, Physics: {1/DT_PHYS:.0f}Hz, {N_SUB} sub-steps")
print(f"Total: {T_TOTAL}s, {N_CTRL} control steps")
print()

# ============================================================
# BATCHED XPBD SIMULATION
# ============================================================
def batched_xpbd_step(cable_pos, cable_prev, drone_pos, dt):
    """
    One physics step for K rollouts simultaneously.
    
    cable_pos:  (K, N_NODES, 2) current positions
    cable_prev: (K, N_NODES, 2) previous positions
    drone_pos:  (K, 2) drone position
    
    Returns updated cable_pos, cable_prev
    """
    K = cable_pos.shape[0]
    
    # Verlet integration
    vel = cable_pos - cable_prev        # (K, N, 2)
    new_prev = cable_pos.copy()
    cable_pos = cable_pos + vel * 0.995 + np.array([0, -G]) * dt**2
    
    # Pin node 0 to drone
    cable_pos[:, 0, :] = drone_pos      # (K, 2)
    
    # XPBD constraint projection
    for _ in range(XPBD_ITERS):
        for i in range(N_NODES - 1):
            delta = cable_pos[:, i+1, :] - cable_pos[:, i, :]   # (K, 2)
            dist = np.linalg.norm(delta, axis=1, keepdims=True)  # (K, 1)
            dist = np.maximum(dist, 1e-6)
            corr = (dist - SEG_LEN) / dist * 0.5 * delta        # (K, 2)
            
            if i == 0:
                # Node 0 is fixed to drone
                cable_pos[:, i+1, :] -= corr * 2.0
            else:
                cable_pos[:, i, :] += corr
                cable_pos[:, i+1, :] -= corr
    
    return cable_pos, new_prev


def batched_simulate(drone_pos, drone_vel, cable_pos, cable_prev, controls):
    """
    Simulate K rollouts for H control steps.
    
    drone_pos:  (K, 2)
    drone_vel:  (K, 2)
    cable_pos:  (K, N_NODES, 2)
    cable_prev: (K, N_NODES, 2)
    controls:   (K, H, 2) — accelerations (ax, ay) per step
    
    Returns:
        drone_traj:  (K, H+1, 2)
        cable_traj:  (K, H+1, N_NODES, 2)
    """
    K, H = controls.shape[0], controls.shape[1]
    
    drone_traj = np.zeros((K, H + 1, 2))
    cable_traj = np.zeros((K, H + 1, N_NODES, 2))
    
    drone_traj[:, 0] = drone_pos.copy()
    cable_traj[:, 0] = cable_pos.copy()
    
    dp = drone_pos.copy()
    dv = drone_vel.copy()
    cp = cable_pos.copy()
    cprev = cable_prev.copy()
    
    for k in range(H):
        ax = np.clip(controls[:, k, 0], -AX_MAX, AX_MAX)  # (K,)
        ay = np.clip(controls[:, k, 1], -AY_MAX, AY_MAX)
        
        for _ in range(N_SUB):
            dv[:, 0] += ax * DT_PHYS
            dv[:, 1] += ay * DT_PHYS
            dp += dv * DT_PHYS
            cp, cprev = batched_xpbd_step(cp, cprev, dp, DT_PHYS)
        
        drone_traj[:, k + 1] = dp.copy()
        cable_traj[:, k + 1] = cp.copy()
    
    return drone_traj, cable_traj

# ============================================================
# BATCHED COST FUNCTION
# ============================================================
def batched_cost(drone_traj, cable_traj, controls, target):
    """
    Compute cost for all K rollouts simultaneously.
    
    Returns: costs (K,)
    """
    K, Hp1 = drone_traj.shape[0], drone_traj.shape[1]
    H = Hp1 - 1
    
    costs = np.zeros(K)
    
    # 1. Progress: sum of distances to target over trajectory
    dist_to_target = np.linalg.norm(drone_traj - target, axis=2)  # (K, H+1)
    costs += 2.0 * np.sum(dist_to_target * DT_CTRL, axis=1)
    
    # 2. Terminal cost
    costs += 15.0 * dist_to_target[:, -1]**2
    
    # 3. Collision penalty for ALL cable nodes
    for t in range(Hp1):
        nodes = cable_traj[:, t, :, :]  # (K, N_NODES, 2)
        px = nodes[:, :, 0]  # (K, N_NODES)
        py = nodes[:, :, 1]
        
        in_column = np.abs(px) < OBS_W / 2  # (K, N_NODES) bool
        
        # Bottom violation
        bot_viol = np.maximum(OBS_BOT + 0.04 - py, 0)  # (K, N_NODES)
        bot_pen = np.sum(bot_viol**2 * in_column, axis=1)  # (K,)
        
        # Top violation
        top_viol = np.maximum(py - OBS_TOP + 0.04, 0)
        top_pen = np.sum(top_viol**2 * in_column, axis=1)
        
        costs += 5000.0 * (bot_pen + top_pen)
    
    # 4. Soft proximity penalty (encourage staying away from obstacles)
    for t in range(Hp1):
        nodes = cable_traj[:, t, :, :]
        px = nodes[:, :, 0]
        py = nodes[:, :, 1]
        in_col = np.abs(px) < OBS_W / 2 + 0.2
        
        margin_bot = np.maximum(OBS_BOT + 0.08 - py, 0)
        margin_top = np.maximum(py - OBS_TOP + 0.08, 0)
        costs += 200.0 * np.sum((margin_bot + margin_top) * in_col, axis=1) * DT_CTRL
    
    # 5. Control effort
    costs += 0.001 * np.sum(controls[:, :, 0]**2 + controls[:, :, 1]**2, axis=1) * DT_CTRL
    
    # 6. Drone out of bounds
    for t in range(Hp1):
        too_low = np.maximum(0.2 - drone_traj[:, t, 1], 0)
        too_high = np.maximum(drone_traj[:, t, 1] - 2.8, 0)
        costs += 500.0 * (too_low**2 + too_high**2)
    
    return costs

# ============================================================
# MPPI CONTROLLER (vectorized)
# ============================================================
class MPPIVec:
    def __init__(self):
        self.U = np.zeros((HORIZON, 2))  # nominal control sequence
    
    def get_action(self, drone_pos, drone_vel, cable_pos, cable_prev):
        K = K_SAMPLES
        H = HORIZON
        
        # Sample noise
        noise = np.random.randn(K, H, 2) * np.array([SIGMA_AX, SIGMA_AY])
        
        # Build control sequences: U + noise
        U_samples = self.U[None, :, :] + noise  # (K, H, 2)
        U_samples[:, :, 0] = np.clip(U_samples[:, :, 0], -AX_MAX, AX_MAX)
        U_samples[:, :, 1] = np.clip(U_samples[:, :, 1], -AY_MAX, AY_MAX)
        
        # Broadcast initial state to K rollouts
        dp = np.tile(drone_pos, (K, 1))                      # (K, 2)
        dv = np.tile(drone_vel, (K, 1))                      # (K, 2)
        cp = np.tile(cable_pos[None, :, :], (K, 1, 1))       # (K, N, 2)
        cprev = np.tile(cable_prev[None, :, :], (K, 1, 1))   # (K, N, 2)
        
        # Simulate all K rollouts in parallel
        d_traj, c_traj = batched_simulate(dp, dv, cp, cprev, U_samples)
        
        # Compute costs
        costs = batched_cost(d_traj, c_traj, U_samples, END)
        
        # MPPI weights
        min_c = np.min(costs)
        w = np.exp(-(costs - min_c) / LAMBDA)
        w /= np.sum(w) + 1e-10
        
        # Weighted noise update
        weighted = np.sum(w[:, None, None] * noise, axis=0)  # (H, 2)
        self.U += weighted
        self.U[:, 0] = np.clip(self.U[:, 0], -AX_MAX, AX_MAX)
        self.U[:, 1] = np.clip(self.U[:, 1], -AY_MAX, AY_MAX)
        
        # Extract first action
        action = self.U[0].copy()
        
        # Shift warm start
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0
        
        return action

# ============================================================
# MAIN LOOP
# ============================================================

# Initialize
drone_pos = START.copy()
drone_vel = np.zeros(2)
cable_pos = np.zeros((N_NODES, 2))
cable_prev = np.zeros((N_NODES, 2))
for i in range(N_NODES):
    cable_pos[i] = START + np.array([0, -L_CABLE * i / (N_NODES - 1)])
    cable_prev[i] = cable_pos[i].copy()

# Settle cable
print("Settling cable...")
for _ in range(500):
    vel = cable_pos - cable_prev
    cable_prev = cable_pos.copy()
    cable_pos += vel * 0.995 + np.array([0, -G]) * DT_PHYS**2
    cable_pos[0] = START
    for _ in range(XPBD_ITERS):
        for i in range(N_NODES - 1):
            d = cable_pos[i+1] - cable_pos[i]
            dn = np.linalg.norm(d)
            if dn > 1e-6:
                c = (dn - SEG_LEN) / dn * 0.5
                m = d * c
                if i == 0: cable_pos[i+1] -= m * 2
                else: cable_pos[i] += m; cable_pos[i+1] -= m

# Init controller
mppi = MPPIVec()

# Storage
hist_drone = np.zeros((N_CTRL + 1, 2))
hist_cable = np.zeros((N_CTRL + 1, N_NODES, 2))
hist_ctrl = np.zeros((N_CTRL, 2))
hist_drone[0] = drone_pos.copy()
hist_cable[0] = cable_pos.copy()

print("\nRunning MPPI-MPC...")
t0_total = time.time()

for step in range(N_CTRL):
    t_sim = step * DT_CTRL
    dist = np.linalg.norm(drone_pos - END)
    
    # Early stop
    if dist < 0.2 and np.linalg.norm(drone_vel) < 0.3 and t_sim > 2.0:
        print(f"  Step {step} t={t_sim:.1f}s: TARGET REACHED")
        hist_drone[step+1:] = drone_pos
        hist_cable[step+1:] = cable_pos
        hist_ctrl[step:] = 0
        N_CTRL = step + 1
        break
    
    # MPPI
    t0 = time.time()
    action = mppi.get_action(drone_pos, drone_vel, cable_pos, cable_prev)
    dt_mppi = time.time() - t0
    
    hist_ctrl[step] = action
    
    # Apply to real system
    for _ in range(N_SUB):
        drone_vel[0] += action[0] * DT_PHYS
        drone_vel[1] += action[1] * DT_PHYS
        drone_pos += drone_vel * DT_PHYS
        
        # XPBD step
        vel_c = cable_pos - cable_prev
        cable_prev = cable_pos.copy()
        cable_pos += vel_c * 0.995 + np.array([0, -G]) * DT_PHYS**2
        cable_pos[0] = drone_pos
        for _ in range(XPBD_ITERS):
            for i in range(N_NODES - 1):
                d = cable_pos[i+1] - cable_pos[i]
                dn = np.linalg.norm(d)
                if dn > 1e-6:
                    c = (dn - SEG_LEN) / dn * 0.5
                    m = d * c
                    if i == 0: cable_pos[i+1] -= m * 2
                    else: cable_pos[i] += m; cable_pos[i+1] -= m
    
    hist_drone[step + 1] = drone_pos.copy()
    hist_cable[step + 1] = cable_pos.copy()
    
    if step % 10 == 0:
        print(f"  Step {step:3d} t={t_sim:.1f}s: "
              f"pos=({drone_pos[0]:+.2f},{drone_pos[1]:.2f}) "
              f"vel=({drone_vel[0]:+.2f},{drone_vel[1]:+.2f}) "
              f"act=({action[0]:+.1f},{action[1]:+.1f}) "
              f"d={dist:.2f} dt={dt_mppi*1000:.0f}ms")

t_total = time.time() - t0_total
print(f"\nTotal: {t_total:.1f}s | Per step: {t_total/N_CTRL*1000:.0f}ms")

# Trim
hist_drone = hist_drone[:N_CTRL + 1]
hist_cable = hist_cable[:N_CTRL + 1]
hist_ctrl = hist_ctrl[:N_CTRL]

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

nc = 0; cb = 999; ct = 999
for k in range(len(hist_cable)):
    for i in range(N_NODES):
        px, py = hist_cable[k, i]
        if abs(px) < OBS_W / 2:
            cb = min(cb, py - OBS_BOT)
            ct = min(ct, OBS_TOP - py)
            if py < OBS_BOT or py > OBS_TOP:
                nc += 1; break

print(f"Collisions: {nc} frames" if nc else "✓ ZERO collisions!")
if cb < 999:
    print(f"Min clearance: bot={cb:.4f}m, top={ct:.4f}m")
print(f"Final pos: ({hist_drone[-1,0]:.2f}, {hist_drone[-1,1]:.2f})")
print(f"Final dist: {np.linalg.norm(hist_drone[-1]-END):.3f}m")

# ============================================================
# PLOTS
# ============================================================
print("\nPlotting...")
ta = np.linspace(0, N_CTRL*DT_CTRL, N_CTRL+1)
tc = np.linspace(0, (N_CTRL-1)*DT_CTRL, N_CTRL)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]; ax.set_aspect('equal')
ax.set_xlim(-2.8, 3.2); ax.set_ylim(-0.2, 3.2)
ax.set_title('MPPI-MPC + XPBD: Narrow Window', fontsize=12)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.grid(True, alpha=0.12)
ax.add_patch(patches.Rectangle((-OBS_W/2, 0), OBS_W, OBS_BOT,
             facecolor='#CC0000', edgecolor='darkred', zorder=10))
ax.add_patch(patches.Rectangle((-OBS_W/2, OBS_TOP), OBS_W, 1.0,
             facecolor='#CC0000', edgecolor='darkred', zorder=10))
ax.plot(hist_drone[:,0], hist_drone[:,1], '-', color='green', lw=1, alpha=0.5, label='Drone')
pay = hist_cable[:,-1,:]
ax.plot(pay[:,0], pay[:,1], '-', color='#FFD700', lw=1.5, alpha=0.7, label='Payload')
for k in np.linspace(0, N_CTRL, min(22, N_CTRL), dtype=int):
    if k >= len(hist_cable): break
    nd = hist_cable[k]
    ax.plot(nd[:,0], nd[:,1], 'k-', lw=0.5, alpha=0.4)
    ax.plot(nd[-1,0], nd[-1,1], 'o', color='#FFD700', markeredgecolor='k', ms=3.5, zorder=5)
    ax.plot(nd[0,0], nd[0,1], 's', color='blue', ms=3, zorder=6)
ax.legend(fontsize=8, loc='upper left')

ax = axes[0,1]; ax.set_title('Control (Acceleration)')
ax.plot(tc, hist_ctrl[:,0], 'b-', lw=1.2, label='$a_x$')
ax.plot(tc, hist_ctrl[:,1], 'r-', lw=1.2, label='$a_y$')
ax.axhline(0, color='k', alpha=0.2)
ax.set_xlabel('Time [s]'); ax.set_ylabel('m/s²')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

ax = axes[1,0]; ax.set_title('Drone Position')
ax.plot(ta, hist_drone[:,0], 'b-', lw=1.2, label='x')
ax.plot(ta, hist_drone[:,1], 'r-', lw=1.2, label='y')
ax.axhline(END[0], color='b', ls='--', alpha=0.3)
ax.axhline(END[1], color='r', ls='--', alpha=0.3)
ax.set_xlabel('Time [s]'); ax.set_ylabel('[m]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

ax = axes[1,1]; ax.set_title('Payload Relative to Drone')
ax.plot(ta, pay[:,0]-hist_drone[:,0], 'b-', lw=1.2, label='Δx')
ax.plot(ta, pay[:,1]-hist_drone[:,1], 'r-', lw=1.2, label='Δy')
ax.axhline(0, color='k', alpha=0.2); ax.axhline(-L_CABLE, color='r', ls='--', alpha=0.3)
ax.set_xlabel('Time [s]'); ax.set_ylabel('[m]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('/home/claude/mppi_xpbd_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: mppi_xpbd_analysis.png")

# GIF
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_aspect('equal'); ax2.set_xlim(-2.8, 3.2); ax2.set_ylim(-0.2, 3.2)
ax2.set_title('MPPI-MPC + XPBD Cable')
ax2.add_patch(patches.Rectangle((-OBS_W/2, 0), OBS_W, OBS_BOT, color='#CC0000', zorder=10))
ax2.add_patch(patches.Rectangle((-OBS_W/2, OBS_TOP), OBS_W, 1.0, color='#CC0000', zorder=10))
ln, = ax2.plot([], [], 'k-', lw=1.5)
dr, = ax2.plot([], [], 's', color='blue', ms=7, zorder=20)
pl, = ax2.plot([], [], 'o', color='#FFD700', markeredgecolor='k', ms=6, zorder=20)
tr, = ax2.plot([], [], '-', color='#FFD700', alpha=0.4, lw=1)
txx, tyy = [], []
nf = len(hist_cable)
def anim(f):
    k = min(f, nf-1); nd = hist_cable[k]
    ln.set_data(nd[:,0], nd[:,1])
    dr.set_data([nd[0,0]], [nd[0,1]])
    pl.set_data([nd[-1,0]], [nd[-1,1]])
    txx.append(nd[-1,0]); tyy.append(nd[-1,1])
    tr.set_data(txx, tyy)
    return ln, dr, pl, tr
ani = animation.FuncAnimation(fig2, anim, frames=nf, interval=100, blit=True)
ani.save('/home/claude/mppi_xpbd.gif', writer='pillow', fps=10)
print("Saved: mppi_xpbd.gif")
plt.close('all')
print("\n✓ Done!")
