import numpy as np
from scipy.io.wavfile import write
import os
# ============================================================
#  MCM-INFINITY v4
#  Self-organizing energy universe with:
#   - dynamic field density (weights, birth/death)
#   - moving samples in 3D (positions + velocities)
#   - soft energy limiting (tanh)
#   - local coupling + diffusive wave events
#   - activity-driven eigen-time
#   - audio projection: freq = energy, amp = weight
#
#  NO archetypes, NO types, NO groups.
#  Everything emerges from energy, coupling, noise, and time.
# ============================================================

# -----------------------------
# 1. GLOBAL PARAMETERS
# -----------------------------
N_SLOTS = 512             # numerical upper bound of field samples
INIT_ACTIVE_FRAC = 0.5    # initial active fraction

INIT_DOMAIN_SIZE = 30.0   # initial spatial spread

# Interaction / coupling
COUPLING_RADIUS = 3.0
COUPLING_RADIUS_SQ = COUPLING_RADIUS ** 2
COUPLING_STRENGTH_E = 0.10   # energy equalization strength
COUPLING_STRENGTH_W = 0.05   # weight coupling strength

# Additional "wave" dynamics
WAVE_THRESHOLD = 2.0         # energy above this triggers a small wave
WAVE_STRENGTH = 0.12         # strength of wave injection to neighbors
WAVE_DECAY = 0.4             # radial decay factor for wave influence

# Noise levels
NOISE_LEVEL_E = 0.10         # energy noise
NOISE_LEVEL_V = 0.05         # velocity noise
NOISE_LEVEL_W = 0.02         # weight noise

VEL_DAMPING = 0.10           # velocity damping

# Eigen-time
DT_MAX = 0.15
BETA = 8.0                   # activity → time compression

MAX_STEPS = 8000             # technical upper bound

# Weight dynamics
WEIGHT_DECAY = 0.03
WEIGHT_GROWTH_FACTOR = 0.10
WEIGHT_REBIRTH_THRESHOLD = 0.02
WEIGHT_REBIRTH_PROB = 0.01

# Audio
AUDIO_SR = 22050
MAX_AUDIO_DURATION = 180.0

FREQ_MIN = 90.0
FREQ_MAX = 1200.0

np.random.seed(42)


# -----------------------------
# 2. INITIALIZATION
# -----------------------------
def init_system():
    """
    Initialize positions, velocities, energies, and weights.
    Positions: random in large volume
    Velocities: small random
    Energies: random in [-3, 3]
    Weights: some active, some near zero
    """
    positions = np.random.uniform(-INIT_DOMAIN_SIZE,
                                  INIT_DOMAIN_SIZE,
                                  size=(N_SLOTS, 3))
    velocities = np.random.normal(0.0, 0.2, size=(N_SLOTS, 3))
    energies = np.random.uniform(-3.0, 3.0, size=N_SLOTS)

    weights = np.zeros(N_SLOTS, dtype=np.float64)
    active_count = int(INIT_ACTIVE_FRAC * N_SLOTS)
    active_idx = np.random.choice(N_SLOTS, size=active_count, replace=False)
    weights[active_idx] = np.random.uniform(0.3, 1.0, size=active_count)

    return positions, velocities, energies, weights


# -----------------------------
# 3. SOFT ENERGY LIMITING
# -----------------------------
def soft_limit_energy(E, E_max=3.0):
    """
    Soft saturation of energy:
        E_eff = E_max * tanh(E / E_max)
    Keeps dynamics bounded without hard clipping.
    """
    return E_max * np.tanh(E / E_max)


# -----------------------------
# 4. STATE UPDATE (v4 core)
# -----------------------------
def update_state(positions, velocities, energies, weights, dt):
    """
    Update:
    - energies: local averaging + noise + wave events
    - weights: growth/decay + noise + rebirth
    - velocities: drift toward local clusters + noise + damping
    - positions: free flight, no walls
    """
    N = energies.shape[0]
    new_E = energies.copy()
    new_V = velocities.copy()
    new_W = weights.copy()

    # Precompute pairwise diffs for efficiency
    diff_all = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N,N,3)
    dist_sq_all = np.sum(diff_all * diff_all, axis=2)                     # (N,N)

    for i in range(N):
        if weights[i] <= 0.0:
            # "dead" sample: minimal dynamics, can be reborn later
            continue

        dist_sq = dist_sq_all[i]          # (N,)
        neigh_mask = (dist_sq > 0.0) & (dist_sq < COUPLING_RADIUS_SQ) & (weights > 0.0)

        if np.any(neigh_mask):
            # local energy coupling (weighted mean)
            neighbor_E = energies[neigh_mask]
            neighbor_W = weights[neigh_mask]
            neighbor_mean_E = np.average(neighbor_E, weights=neighbor_W)
            dE = COUPLING_STRENGTH_E * (neighbor_mean_E - energies[i])
            new_E[i] += dt * dE

            # weight coupling: energy differences feed into local weight
            local_energy_diff = np.abs(neighbor_mean_E - energies[i])
            new_W[i] += dt * (WEIGHT_GROWTH_FACTOR * local_energy_diff)

            # movement: drift toward local weighted center
            neighbor_pos = positions[neigh_mask]
            neighbor_mean_pos = np.average(neighbor_pos, axis=0, weights=neighbor_W)
            direction = neighbor_mean_pos - positions[i]
            accel = 0.03 * direction * local_energy_diff
            new_V[i] += dt * accel

            # wave events: if energy high, inject waves into neighbors
            if np.abs(energies[i]) > WAVE_THRESHOLD:
                # distance of neighbors
                local_dist_sq = dist_sq[neigh_mask]
                local_dist = np.sqrt(local_dist_sq) + 1e-8
                wave_factor = np.exp(-WAVE_DECAY * local_dist)
                # signed energy wave: keeps sign of local energy
                new_E[neigh_mask] += dt * WAVE_STRENGTH * energies[i] * wave_factor

    # global noise
    new_E += np.sqrt(dt) * NOISE_LEVEL_E * np.random.randn(N)
    new_V += np.sqrt(dt) * NOISE_LEVEL_V * np.random.randn(N, 3)
    new_W += np.sqrt(dt) * NOISE_LEVEL_W * np.random.randn(N)

    # natural weight decay
    new_W -= dt * WEIGHT_DECAY * new_W

    # rebirth of near-dead samples
    dead_mask = new_W < WEIGHT_REBIRTH_THRESHOLD
    dead_indices = np.where(dead_mask)[0]
    if dead_indices.size > 0:
        rebirth_flags = np.random.rand(dead_indices.size) < WEIGHT_REBIRTH_PROB
        for idx, flag in zip(dead_indices, rebirth_flags):
            if flag:
                # re-spawn at random place with fresh energy and weight
                new_W[idx] = np.random.uniform(0.3, 1.0)
                new_E[idx] = np.random.uniform(-3.0, 3.0)
                new_V[idx] = np.random.normal(0.0, 0.2, size=3)
                positions[idx] = np.random.uniform(-INIT_DOMAIN_SIZE,
                                                   INIT_DOMAIN_SIZE,
                                                   size=3)

    # clamp weights
    new_W = np.clip(new_W, 0.0, 1.0)

    # velocity damping
    new_V *= (1.0 - VEL_DAMPING * dt)

    # free movement, no walls
    new_pos = positions + new_V * dt

    # soft energy limiting
    new_E = soft_limit_energy(new_E, E_max=3.0)

    return new_pos, new_V, new_E, new_W


# -----------------------------
# 5. ACTIVITY & EIGENTIME
# -----------------------------
def compute_activity(E, E_prev, W, W_prev):
    """
    Field activity:
    - weighted energy change
    - plus weight change
    """
    dE = np.mean(np.abs(E - E_prev) * 0.5 * (W + W_prev))
    dW = np.mean(np.abs(W - W_prev))
    return dE + dW


def compute_dt(activity):
    return DT_MAX / (1.0 + BETA * activity)


# -----------------------------
# 6. MAIN SIMULATION
# -----------------------------
def run_universe_v4():
    positions, velocities, E, W = init_system()
    E_prev = E.copy()
    W_prev = W.copy()

    times = []
    E_traj = []
    W_traj = []

    t = 0.0

    for step in range(MAX_STEPS):
        A = compute_activity(E, E_prev, W, W_prev)
        dt = compute_dt(A)

        times.append(t)
        E_traj.append(E.copy())
        W_traj.append(W.copy())

        E_prev = E.copy()
        W_prev = W.copy()

        positions, velocities, E, W = update_state(positions, velocities, E, W, dt)
        t += dt

        if t >= MAX_AUDIO_DURATION:
            break

    return np.array(times, dtype=np.float64), np.array(E_traj, dtype=np.float64), \
           np.array(W_traj, dtype=np.float64), positions


# -----------------------------
# 7. ENERGY → FREQUENCY & PANNING
# -----------------------------
def energy_to_freq(E):
    E_clip = np.clip(E, -3.0, 3.0)
    x = (E_clip + 3.0) / 6.0
    return FREQ_MIN + x * (FREQ_MAX - FREQ_MIN)


def angle_to_pan(theta):
    return 0.5 + 0.5 * np.sin(theta)


# -----------------------------
# 8. AUDIO SYNTHESIS
# -----------------------------
def generate_eigenzeit_audio(times, E_traj, W_traj, positions):
    total_time = float(times[-1])
    duration = min(total_time, MAX_AUDIO_DURATION)
    n_samples = int(AUDIO_SR * duration)

    print(f"Eigenzeit: {total_time:.2f} s, audio duration: {duration:.2f} s, samples: {n_samples}")

    theta_vals = np.arctan2(positions[:, 1], positions[:, 0])
    pans = angle_to_pan(theta_vals)
    left_gain = 1.0 - pans
    right_gain = pans

    phases = np.zeros(N_SLOTS, dtype=np.float64)
    audio = np.zeros((n_samples, 2), dtype=np.float64)

    times_np = np.asarray(times)

    for s in range(n_samples):
        t = s / AUDIO_SR
        idx = np.searchsorted(times_np, t, side='right') - 1
        if idx < 0:
            idx = 0
        if idx >= E_traj.shape[0]:
            idx = E_traj.shape[0] - 1

        E_step = E_traj[idx]
        W_step = W_traj[idx]

        freqs = energy_to_freq(E_step)
        phase_inc = 2.0 * np.pi * freqs / AUDIO_SR
        phases += phase_inc

        wave = np.sin(phases) * W_step

        audio[s, 0] = np.sum(wave * left_gain)
        audio[s, 1] = np.sum(wave * right_gain)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
        audio *= 0.95

    return audio.astype(np.float32)


# -----------------------------
# 9. MAIN
# -----------------------------
if __name__ == "__main__":
    print("Starting MCM-INFINITY v4 (self-organizing energy universe)…")

    times, E_traj, W_traj, positions = run_universe_v4()

    print("Simulation done.")
    print("Simulated steps:", E_traj.shape[0])
    print("End of eigen-time:", times[-1])

    print("Generating audio…")
    audio = generate_eigenzeit_audio(times, E_traj, W_traj, positions)

    out_dir = "audio_exports"
    os.makedirs(out_dir, exist_ok=True)

    out_name = os.path.join(out_dir, "mcm_infinity_v4.wav")
    write(out_name, AUDIO_SR, (audio * 32767).astype(np.int16))
    print("Audio saved as:", out_name)
    print("DONE.")
