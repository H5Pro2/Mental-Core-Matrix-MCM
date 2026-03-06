import numpy as np
from scipy.io.wavfile import write
import os
# ============================================================
#  MCM-INFINITY: Fully Dynamic Energy Continuum
#  Minimal anchors: -3, 0, +3
#  NO predefined groups, NO archetypes, NO categories
#  Archetypes emerge as attractor clusters
# ============================================================

N_AGENTS = 256
DOMAIN_SIZE = 5.0
COUPLING_RADIUS = 1.5
COUPLING_RADIUS_SQ = COUPLING_RADIUS ** 2
COUPLING_STRENGTH = 0.12
NOISE_LEVEL = 0.15

DT_MAX = 0.12
BETA = 10.0

MAX_STEPS = 4000

AUDIO_SR = 22050
MAX_AUDIO_DURATION = 180.0

FREQ_MIN = 90.0
FREQ_MAX = 1200.0

np.random.seed(42)

# -----------------------------
# INIT: Positions + Energies
# -----------------------------
def init_system():
    positions = np.random.uniform(-DOMAIN_SIZE, DOMAIN_SIZE, (N_AGENTS, 3))
    energies = np.random.uniform(-3.0, 3.0, N_AGENTS)
    return positions, energies

# -----------------------------
# Energy Dynamics
# -----------------------------
def update_energy(pos, E, dt):
    N = len(E)
    new_E = E.copy()

    for i in range(N):
        diff = pos - pos[i]
        dist_sq = np.sum(diff * diff, axis=1)
        neigh = (dist_sq > 0) & (dist_sq < COUPLING_RADIUS_SQ)

        if np.any(neigh):
            mean_neigh = np.mean(E[neigh])
            dE = COUPLING_STRENGTH * (mean_neigh - E[i])
            new_E[i] += dt * dE

    # Add noise scaled with sqrt(dt)
    new_E += np.sqrt(dt) * NOISE_LEVEL * np.random.randn(N)

    return np.clip(new_E, -3.0, 3.0)

# -----------------------------
# Activity / Eigenzeit
# -----------------------------
def compute_activity(E, E_prev):
    return np.mean(np.abs(E - E_prev))

def compute_dt(activity):
    return DT_MAX / (1.0 + BETA * activity)

# -----------------------------
# Run simulation
# -----------------------------
def run_sim():
    pos, E = init_system()
    E_prev = E.copy()

    times = []
    traj = []
    t = 0.0

    for step in range(MAX_STEPS):
        A = compute_activity(E, E_prev)
        dt = compute_dt(A)

        times.append(t)
        traj.append(E.copy())

        E_prev = E.copy()
        E = update_energy(pos, E, dt)
        t += dt

    return np.array(times), np.array(traj), pos

# -----------------------------
# Energy → Frequency
# -----------------------------
def energy_to_freq(E):
    x = (np.clip(E, -3, 3) + 3) / 6
    return FREQ_MIN + x * (FREQ_MAX - FREQ_MIN)

# -----------------------------
# Stereo Panning
# -----------------------------
def angle_to_pan(theta):
    return 0.5 + 0.5 * np.sin(theta)

# -----------------------------
# Audio Synthesis
# -----------------------------
def generate_audio(times, traj, pos):
    total_t = times[-1]
    duration = min(total_t, MAX_AUDIO_DURATION)
    n_samples = int(duration * AUDIO_SR)

    audio = np.zeros((n_samples, 2), dtype=np.float32)
    phases = np.zeros(N_AGENTS)

    theta_vals = np.arctan2(pos[:,1], pos[:,0])
    pans = angle_to_pan(theta_vals)
    left = 1 - pans
    right = pans

    times_np = np.asarray(times)

    for s in range(n_samples):
        t = s / AUDIO_SR
        idx = np.searchsorted(times_np, t, side='right') - 1
        idx = np.clip(idx, 0, traj.shape[0]-1)

        E_step = traj[idx]
        freqs = energy_to_freq(E_step)
        phase_inc = 2 * np.pi * freqs / AUDIO_SR
        phases += phase_inc

        wave = np.sin(phases)
        audio[s,0] = np.sum(wave * left)
        audio[s,1] = np.sum(wave * right)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
        audio *= 0.95

    return audio

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    
    print("Starting MCM-INFINITY v1 (self-organizing energy universe)…")

    times, energy_traj, pos = run_sim()    

    print("Simulation abgeschlossen.")
    print("Simulierte Schritte:", energy_traj.shape[0])
    print("Ende der Eigenzeit:", times[-1])

    audio = generate_audio(times, energy_traj, pos)

    out_dir = "audio_exports"
    os.makedirs(out_dir, exist_ok=True)

    out_name = os.path.join(out_dir, "mcm_infinity_v1.wav")
    write(out_name, AUDIO_SR, (audio * 32767).astype(np.int16))
    print("Audio gespeichert als:", out_name)
    print("FERTIG.")
