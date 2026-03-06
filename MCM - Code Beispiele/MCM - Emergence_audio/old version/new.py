import numpy as np
import soundfile as sf

# ===========================================
# PARAMETERS
# ===========================================
N_GROUPS = 4
N_R_LEVELS = 4
N_PHASE_LEVELS = 4
N_ANGLE_SUB = 4
N_AGENTS = N_GROUPS * N_R_LEVELS * N_PHASE_LEVELS * N_ANGLE_SUB

STEPS = 1500
DT = 0.1

COUPLING_RADIUS = 1.5
COUPLING_STRENGTH = 0.15
NOISE_LEVEL = 0.10

AUDIO_SR = 44100
AUDIO_DURATION = 60.0
SAMPLES = int(AUDIO_SR * AUDIO_DURATION)

# Map energy [-3,3] → Frequenzband
FREQ_MIN = 90
FREQ_MAX = 1200

# Stereo-Panning über θ (0..360°)
def angle_to_pan(theta):
    return 0.5 + 0.5*np.sin(theta)

# Energie → Frequenz
def energy_to_freq(E):
    E_norm = (E + 3.0) / 6.0
    return FREQ_MIN + E_norm * (FREQ_MAX - FREQ_MIN)

# Gruppe → Quadrantenwinkel
def group_angle(g):
    return g * (np.pi / 2.0)

# ===========================================
# INITIALISIERUNG
# ===========================================
def init_system():
    positions = np.zeros((N_AGENTS, 3))
    energies = np.zeros(N_AGENTS)

    r_levels = np.linspace(0.75, 3.0, N_R_LEVELS)
    z_levels = np.linspace(1.0, 4.0, N_PHASE_LEVELS)

    ENERGY_RANGES = {
        0: (-3.0, -1.0),
        1: (-1.0, +1.0),
        2: (+1.0, +2.0),
        3: (+2.0, +3.0),
    }

    idx = 0
    for g in range(N_GROUPS):
        theta_center = group_angle(g)
        E_min, E_max = ENERGY_RANGES[g]
        E_levels = np.linspace(E_min, E_max, N_R_LEVELS)

        for p_idx, z in enumerate(z_levels):
            for r_idx, (r, E) in enumerate(zip(r_levels, E_levels)):
                for a_sub in range(N_ANGLE_SUB):
                    theta_span = (np.pi / 2.0) * 0.8
                    theta_offset = ((a_sub + 0.5) / N_ANGLE_SUB - 0.5) * theta_span
                    theta = theta_center + theta_offset

                    x = r*np.cos(theta)
                    y = r*np.sin(theta)

                    positions[idx] = (x, y, z)
                    energies[idx] = E
                    idx += 1

    return positions, energies


# ===========================================
# DYNAMIK UPDATE
# ===========================================
def update_energies(positions, energies):
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    mask = (dist > 0) & (dist < COUPLING_RADIUS)
    new_E = energies.copy()

    for i in range(N_AGENTS):
        neigh = mask[i]
        if np.any(neigh):
            m = np.mean(energies[neigh])
            new_E[i] += COUPLING_STRENGTH*(m - energies[i])

    new_E += NOISE_LEVEL*np.random.randn(N_AGENTS)
    return np.clip(new_E, -3.0, 3.0)


# ===========================================
# AUDIO GENERATOR
# ===========================================
def generate_audio(energy_traj, theta_vals):
    t = np.linspace(0, AUDIO_DURATION, SAMPLES)
    audio_L = np.zeros(SAMPLES)
    audio_R = np.zeros(SAMPLES)

    # Zeitinterpolation (Mapping Sim-Steps → Samples)
    indices = np.linspace(0, energy_traj.shape[0]-1, SAMPLES).astype(int)

    for i_sample, step_idx in enumerate(indices):
        E_step = energy_traj[step_idx]

        # Frequenzen berechnen
        freqs = energy_to_freq(E_step)

        # Stereo-Panning aus Theta
        pans = angle_to_pan(theta_vals)

        # Summieren aller Agenten
        phases = 2*np.pi*freqs*(i_sample/AUDIO_SR)
        signal = np.sin(phases)

        audio_L[i_sample] = np.sum(signal * (1 - pans))
        audio_R[i_sample] = np.sum(signal * pans)

    # Normalisieren
    max_val = max(np.max(np.abs(audio_L)), np.max(np.abs(audio_R)))
    audio_L /= max_val
    audio_R /= max_val

    return np.stack([audio_L, audio_R], axis=1)


# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    print("MCM-3D-256: Initialisiere System …")
    positions, energies = init_system()

    # Theta (Winkel) aller Agenten für Audio-Panning
    theta_vals = np.arctan2(positions[:,1], positions[:,0])

    print("Simuliere Energie-Dynamik …")
    energy_traj = np.zeros((STEPS, N_AGENTS))
    E = energies.copy()

    for t in range(STEPS):
        energy_traj[t] = E
        E = update_energies(positions, E)

    print("Generiere Audiofeld …")
    audio = generate_audio(energy_traj, theta_vals)

    print("Speichere WAV …")
    sf.write("mcm_3d_256_audio.wav", audio, AUDIO_SR)

    print("Fertig: mcm_3d_256_audio.wav")
