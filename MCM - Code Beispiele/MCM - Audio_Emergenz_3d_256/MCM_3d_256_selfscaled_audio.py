import numpy as np
from scipy.io.wavfile import write

# ============================================================
#  MCM-3D-256 MIT SELBSTSKALIERENDER ZEIT + AUDIOAUSGABE
#  - Emergenz roh
#  - Eigenzeit als Klang
# ============================================================

# -----------------------------
# 1. SYSTEMPARAMETER
# -----------------------------
N_GROUPS = 4            # G1..G4
N_R_LEVELS = 4          # radiale Energiestufen pro Gruppe
N_PHASE_LEVELS = 4      # z-Lagen (Phase)
N_ANGLE_SUB = 4         # feine Unterteilung im Winkel pro Gruppe
N_AGENTS = N_GROUPS * N_R_LEVELS * N_PHASE_LEVELS * N_ANGLE_SUB  # = 256

COUPLING_RADIUS = 1.5
COUPLING_RADIUS_SQ = COUPLING_RADIUS ** 2
COUPLING_STRENGTH = 0.15
NOISE_LEVEL = 0.10

# Selbstskalierende Zeiteinheit
DT_MAX = 0.10           # größter möglicher Zeitschritt
BETA = 12.0             # wie stark Aktivität Zeit komprimiert

MAX_STEPS = 4000        # maximale Simulationsschritte

# Audio-Parameter (Eigenzeit)
AUDIO_SR = 22050        # Abtastrate (bewusst moderat, damit es berechenbar bleibt)
MAX_AUDIO_DURATION = 180.0  # maximale Länge in Sekunden (z.B. 3 Minuten)

# Mapping: Energie [-3,3] -> Frequenzbereich
FREQ_MIN = 90.0
FREQ_MAX = 1200.0

np.random.seed(42)


# -----------------------------
# 2. HILFSFUNKTIONEN (GEOMETRIE & INIT)
# -----------------------------
def group_angle(g: int) -> float:
    """Quadrantenwinkel für Gruppe G1..G4 in Radiant."""
    return g * (np.pi / 2.0)


def init_positions_and_energies():
    """
    Initialisiert:
    - 3D-Positionen (x,y,z)
    - Anfangsenergien E
    """
    ENERGY_RANGES = {
        0: (-3.0, -1.0),   # G1
        1: (-1.0, +1.0),   # G2
        2: (+1.0, +2.0),   # G3
        3: (+2.0, +3.0),   # G4
    }

    r_levels = np.linspace(0.75, 3.0, N_R_LEVELS)
    z_levels = np.linspace(1.0, 4.0, N_PHASE_LEVELS)

    positions = np.zeros((N_AGENTS, 3), dtype=np.float64)
    energies = np.zeros(N_AGENTS, dtype=np.float64)

    idx = 0
    for g in range(N_GROUPS):
        theta_center = group_angle(g)
        E_min, E_max = ENERGY_RANGES[g]
        E_levels = np.linspace(E_min, E_max, N_R_LEVELS)

        for z in z_levels:
            for r, E_val in zip(r_levels, E_levels):
                for a_sub in range(N_ANGLE_SUB):
                    # Angular Subdivision im Quadranten
                    theta_span = (np.pi / 2.0) * 0.8
                    theta_offset = ((a_sub + 0.5) / N_ANGLE_SUB - 0.5) * theta_span
                    theta = theta_center + theta_offset

                    x = r * np.cos(theta)
                    y = r * np.sin(theta)

                    positions[idx] = (x, y, z)
                    energies[idx] = E_val
                    idx += 1

    return positions, energies


# -----------------------------
# 3. ENERGIEDYNAMIK
# -----------------------------
def update_energies(positions, energies, dt):
    """
    Ein Energie-Update mit:
    - lokaler Kopplung (RAM-sparsam, keine NxN-Tensoren)
    - dt-skalierter Dynamik
    - Rauschen ~ sqrt(dt)
    """
    N = energies.shape[0]
    new_E = energies.copy()

    for i in range(N):
        diff = positions - positions[i]           # shape (N,3)
        dist_sq = np.sum(diff * diff, axis=1)     # shape (N,)

        neigh_mask = (dist_sq > 0.0) & (dist_sq < COUPLING_RADIUS_SQ)

        if np.any(neigh_mask):
            neighbor_mean = np.mean(energies[neigh_mask])
            dE = COUPLING_STRENGTH * (neighbor_mean - energies[i])
            new_E[i] += dt * dE

    # Rauschen (roh, ungefiltert)
    new_E += np.sqrt(dt) * NOISE_LEVEL * np.random.randn(N)

    return np.clip(new_E, -3.0, 3.0)


def compute_activity(E, E_prev):
    """
    Feldspannung: mittlere Energiedifferenz.
    """
    return np.mean(np.abs(E - E_prev))


def compute_dt(activity):
    """
    dt = DT_MAX / (1 + BETA * activity)
    Große Aktivität => kleines dt
    Kleine Aktivität => großes dt
    """
    return DT_MAX / (1.0 + BETA * activity)


# -----------------------------
# 4. HAUPTSIMULATION (EIGENZEIT)
# -----------------------------
def run_self_scaled_simulation():
    positions, E = init_positions_and_energies()
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
        E = update_energies(positions, E, dt)
        t += dt

    return np.array(times, dtype=np.float64), np.array(traj, dtype=np.float64), positions


# -----------------------------
# 5. AUDIO-MAPPING: ENERGIE -> KLANG
# -----------------------------
def energy_to_freq(E):
    """
    Mappt Energie E ∈ [-3,3] linear ins Frequenzband [FREQ_MIN, FREQ_MAX].
    Keine Glättung, reine Abbildung.
    """
    E_clip = np.clip(E, -3.0, 3.0)
    x = (E_clip + 3.0) / 6.0  # 0..1
    return FREQ_MIN + x * (FREQ_MAX - FREQ_MIN)


def angle_to_pan(theta):
    """
    Winkel -> Stereo-Panning (0 = links, 1 = rechts).
    Nutzt Sinus für weiche Verteilung.
    """
    return 0.5 + 0.5 * np.sin(theta)


def generate_eigenzeit_audio(times, energy_traj, positions):
    """
    Erzeugt Stereo-Audio in Eigenzeit:
    - Zeitachse des Audios folgt times[-1] (ggf. begrenzt auf MAX_AUDIO_DURATION)
    - Frequenzen kommen direkt aus den Energien
    - 256 Oszillatoren, jeder mit eigener Phase
    """

    total_time = float(times[-1])
    duration = min(total_time, MAX_AUDIO_DURATION)
    n_samples = int(AUDIO_SR * duration)

    print(f"Audio-Eigenzeit: verwende {duration:.2f} s (von {total_time:.2f} s) -> {n_samples} Samples")

    # Winkel für Panning
    theta_vals = np.arctan2(positions[:, 1], positions[:, 0])
    pans = angle_to_pan(theta_vals)   # 0..1
    left_gain = (1.0 - pans)
    right_gain = pans

    # Phasen der einzelnen Oszillatoren
    phases = np.zeros(N_AGENTS, dtype=np.float64)

    audio = np.zeros((n_samples, 2), dtype=np.float64)

    # Für Suche im times-Array: schnelle Binärsuche
    times_np = np.asarray(times)

    for s in range(n_samples):
        t = s / AUDIO_SR

        # passenden Simulationsindex finden (Eigenzeit)
        idx = np.searchsorted(times_np, t, side='right') - 1
        if idx < 0:
            idx = 0
        if idx >= energy_traj.shape[0]:
            idx = energy_traj.shape[0] - 1

        E_step = energy_traj[idx]           # shape (256,)
        freqs = energy_to_freq(E_step)      # shape (256,)

        # Phaseninkrement pro Sample
        phase_inc = 2.0 * np.pi * freqs / AUDIO_SR
        phases += phase_inc

        # rohe Sinuswellen (keine Glättung, reine Oszillation)
        wave = np.sin(phases)               # shape (256,)

        # Summieren mit Panning
        audio[s, 0] = np.sum(wave * left_gain)
        audio[s, 1] = np.sum(wave * right_gain)

    # Normieren
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
        audio *= 0.95

    return audio.astype(np.float32)


# -----------------------------
# 6. AUSFÜHRUNG
# -----------------------------
if __name__ == "__main__":
    print("Starte MCM-3D-256 Selbstskaliert + Audio …")

    # 1) Simulation in Eigenzeit
    times, energy_traj, positions = run_self_scaled_simulation()

    print("Simulation fertig.")
    print("Simulierte Schritte:", energy_traj.shape[0])
    print("Simulierte Eigenzeit (Endzeit):", times[-1], "Sekunden")

    if times.shape[0] > 2:
        print("Erster dt:", times[1] - times[0])
        print("Letzter dt:", times[-1] - times[-2])

    # 2) Audio in Eigenzeit erzeugen
    print("Erzeuge Eigenzeit-Audio …")
    audio = generate_eigenzeit_audio(times, energy_traj, positions)

    # 3) WAV schreiben
    out_name = "mcm_3d_256_selfscaled_audio.wav"
    write(out_name, AUDIO_SR, (audio * 32767).astype(np.int16))
    print("Audio gespeichert als:", out_name)
    print("FERTIG.")
