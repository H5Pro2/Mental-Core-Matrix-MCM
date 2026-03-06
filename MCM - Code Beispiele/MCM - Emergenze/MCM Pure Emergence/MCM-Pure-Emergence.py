# Pure Emergence Simulation (MCM - "Nature Pure" mode)
# - No artificial "stability" rules are used during the simulation.
# - Particles follow simple, local physics with short-range attraction and repulsion + noise.
# - Clusters are detected **post-hoc** (passive observation) and persistence over time is used to count emergent structures.
# - This script runs multiple independent simulation runs and reports the fraction of runs that produce at least one persistent cluster.
#
# Save / modify parameters below as needed. The simulation does NOT alter particles when detecting clusters.

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# -------------------- Parameters --------------------
GRID_SIZE = 80.0            # continuous domain [0, GRID_SIZE] in both x and y (reflective walls)
N = 150                     # number of particles
steps = 350                 # time steps per run
dt = 0.05
runs = 30                   # independent runs to estimate probability
seed_base = 1000

# Interaction parameters (physically motivated; no "if stable -> create entity" rules)
epsilon = 1.0               # interaction strength
sigma = 1.0                 # characteristic interaction distance
r_cut = 3.0                 # interaction cutoff (beyond which force ~ 0)
noise_scale = 0.6           # random agitation (thermal-like)
damping = 0.05              # viscous damping (dissipation)

# Clustering detection (post-hoc, passive)
cluster_eps = 1.6           # neighbor distance threshold for forming a connected component (units same as sigma)
cluster_min_size = 5        # minimal cluster size to be considered a candidate
persist_frames = 12         # minimal number of consecutive frames a cluster must exist to be considered persistent

# Limits to keep runtime reasonable
max_total_particles = 1000  # not used to alter sim - just safety

# -------------------- Helper functions --------------------
def pairwise_dists(x, y):
    """Return pairwise distance matrix for points (x,y)."""
    dx = x[:,None] - x[None,:]
    dy = y[:,None] - y[None,:]
    return np.hypot(dx, dy)

def compute_forces(x, y):
    """
    Compute pairwise forces using a smooth short-range attractive/repulsive potential
    that is zero beyond r_cut. This is purely local physics (no external stabilization)
    """
    dx = x[:,None] - x[None,:]
    dy = y[:,None] - y[None,:]
    D = np.hypot(dx, dy) + 1e-12
    # normalized displacement vectors
    nx = dx / D
    ny = dy / D
    # simple potential: attractive for r between 0.6*sigma and 2.5*sigma, repulsive at very short distances
    r = D
    forces_x = np.zeros_like(dx)
    forces_y = np.zeros_like(dy)
    mask = (r > 1e-9) & (r < r_cut)
    # smooth attractive well: -A * (r - r0) inside range -> linear attractive toward r0
    r0 = 1.2 * sigma
    A = epsilon * 0.8
    # short-range repulsion (prevent collapse)
    rep_strength = epsilon * 2.5
    # compute scalar radial force magnitude
    fr = np.zeros_like(r)
    # attraction proportional to (r - r0) for r in (0.6*sigma, 2.5*sigma)
    inside_attract = mask & (r >= 0.6*sigma) & (r <= 2.5*sigma)
    fr[inside_attract] = -A * (r[inside_attract] - r0) / (r_cut)
    # strong short-range repulsion for r < 0.6*sigma
    short = mask & (r < 0.6*sigma)
    fr[short] = rep_strength * (0.6*sigma - r[short]) / (0.6*sigma)
    # apply smoothing factor near cutoff to avoid jumps
    near_cut = mask & (r > (r_cut - 0.8))
    fr[near_cut] *= 0.8 * (r_cut - r[near_cut]) / 0.8
    # convert to x,y components
    forces_x = (fr * nx)
    forces_y = (fr * ny)
    # net force per particle = sum over interactions (Newton's 3rd law implicit by pairwise signs)
    Fx = np.sum(forces_x, axis=1)
    Fy = np.sum(forces_y, axis=1)
    return Fx, Fy

def detect_clusters(x, y, eps, min_size):
    """
    Simple connected-component clustering: build graph where edges exist if distance <= eps.
    Return a list of clusters (each a list of indices) - only clusters with size >= min_size.
    """
    Np = len(x)
    if Np == 0:
        return []
    D = pairwise_dists(x, y)
    adj = D <= eps
    visited = np.zeros(Np, dtype=bool)
    clusters = []
    for i in range(Np):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            comp.append(v)
            neighbors = np.where(adj[v])[0]
            for nb in neighbors:
                if not visited[nb]:
                    stack.append(nb)
        if len(comp) >= min_size:
            clusters.append(comp)
    return clusters

# -------------------- Simulation loop over runs --------------------
start_time = time.time()
persistent_runs = 0
all_run_stats = []

for run in range(runs):
    np.random.seed(seed_base + run)
    # initialize particle positions and velocities uniformly, avoiding exact overlap
    x = np.random.uniform(0.0, GRID_SIZE, N)
    y = np.random.uniform(0.0, GRID_SIZE, N)
    vx = np.random.normal(scale=0.6, size=N)
    vy = np.random.normal(scale=0.6, size=N)
    # small initial energy variation (used only for interpretation, not for rules)
    energy = np.clip(np.random.normal(loc=1.0, scale=0.2, size=N), 0.1, 3.0)
    # record cluster existence over time (list of lists of cluster centers)
    cluster_history = []  # list per time: list of (cx, cy, size)
    for t in range(steps):
        # compute forces (pairwise)
        Fx, Fy = compute_forces(x, y)
        # stochastic agitation (thermal)
        Fx += np.random.normal(scale=noise_scale, size=N)
        Fy += np.random.normal(scale=noise_scale, size=N)
        # integrate velocities and positions (simple Euler with damping)
        vx += (Fx * dt) - (damping * vx * dt)
        vy += (Fy * dt) - (damping * vy * dt)
        x += vx * dt
        y += vy * dt
        # reflective walls: reflect velocity and clamp positions
        for arrp, arrv in ((x, vx), (y, vy)):
            below = arrp < 0.0
            if np.any(below):
                arrp[below] = -arrp[below]
                arrv[below] = -arrv[below] * 0.6  # some energy lost on bounce
            above = arrp > GRID_SIZE
            if np.any(above):
                arrp[above] = 2*GRID_SIZE - arrp[above]
                arrv[above] = -arrv[above] * 0.6
        # detect clusters at this frame (passive observation)
        clusters = detect_clusters(x, y, cluster_eps, cluster_min_size)
        centers = []
        for comp in clusters:
            cx = float(np.mean(x[comp])); cy = float(np.mean(y[comp])); sz = len(comp)
            centers.append((cx, cy, sz))
        cluster_history.append(centers)
    # analyze persistence: look for cluster centers that persist across consecutive frames
    persistent_structures = 0
    # naive persistence check: for each cluster center at frame t, see if a cluster exists within eps at frames t+1..t+persist_frames-1
    for t in range(len(cluster_history) - persist_frames):
        for center in cluster_history[t]:
            cx, cy, sz = center
            persists = True
            for dt_frame in range(1, persist_frames):
                found_similar = False
                for other in cluster_history[t + dt_frame]:
                    ox, oy, osz = other
                    if math.hypot(cx - ox, cy - oy) <= cluster_eps:
                        found_similar = True
                        break
                if not found_similar:
                    persists = False
                    break
            if persists:
                persistent_structures += 1
    # count unique persistent structures: naive dedup by centers
    # simple heuristic: if >=1 persistent structure found, consider run successful for emergence
    run_success = persistent_structures >= 1
    all_run_stats.append({
        "run": run,
        "persistent_structures": persistent_structures,
        "success": run_success,
        "clusters_final": len(cluster_history[-1])
    })
    if run_success:
        persistent_runs += 1

elapsed = time.time() - start_time
fraction = persistent_runs / runs
print(f"Runs: {runs}, Persistent runs: {persistent_runs}, Fraction: {fraction:.3f}, elapsed {elapsed:.1f}s")

# Show summary statistics
print("Per-run summary (first 10):")
for s in all_run_stats[:10]:
    print(s)

# Plot one representative run (last run) for visualization
rep_x, rep_y = None, None
np.random.seed(seed_base + runs - 1)
x = np.random.uniform(0.0, GRID_SIZE, N)
y = np.random.uniform(0.0, GRID_SIZE, N)
vx = np.random.normal(scale=0.6, size=N)
vy = np.random.normal(scale=0.6, size=N)
for t in range(steps):
    Fx, Fy = compute_forces(x, y)
    Fx += np.random.normal(scale=noise_scale, size=N)
    Fy += np.random.normal(scale=noise_scale, size=N)
    vx += (Fx * dt) - (damping * vx * dt)
    vy += (Fy * dt) - (damping * vy * dt)
    x += vx * dt; y += vy * dt
    # reflective bounds
    for arrp, arrv in ((x, vx), (y, vy)):
        below = arrp < 0.0
        if np.any(below):
            arrp[below] = -arrp[below]; arrv[below] = -arrv[below] * 0.6
        above = arrp > GRID_SIZE
        if np.any(above):
            arrp[above] = 2*GRID_SIZE - arrp[above]; arrv[above] = -arrv[above] * 0.6
rep_x, rep_y = x.copy(), y.copy()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Representative final particle positions")
plt.scatter(rep_x, rep_y, s=8)
plt.xlim(0, GRID_SIZE); plt.ylim(0, GRID_SIZE)
plt.subplot(1,2,2)
plt.title("Histogram of per-run persistent structures")
vals = [s['persistent_structures'] for s in all_run_stats]
plt.hist(vals, bins=20)
plt.xlabel("persistent structures per run"); plt.ylabel("count")
plt.tight_layout()
plt.show()

# Output the key result for the user
result = {
    "runs": runs,
    "persistent_runs": persistent_runs,
    "fraction": fraction,
    "per_run_stats": all_run_stats
}
result

