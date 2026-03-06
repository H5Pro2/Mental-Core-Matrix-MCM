import numpy as np
from scipy.io.wavfile import write
import os
# ============================================================
#  MCM-INFINITY v5
#  - Dynamik wie v4 (selbstorganisierendes Energieuniversum)
#  - ZWEI Audio-Ausgaben:
#       1) Volles Feld (alle Oszillatoren)  -> mcm_infinity_v5_full.wav
#       2) Cluster-Feld (nur stärkste Energiepakete) -> mcm_infinity_v5_clusters.wav
#
#  WICHTIG:
#  - Das Universum selbst wird NICHT verändert.
#  - Nur die Audio-Projektion wird gefiltert (Ausgabe-Ebene).
# ============================================================

# -----------------------------
# 1. GLOBAL PARAMETERS
# -----------------------------
N_SLOTS = 512             # numerische Obergrenze der Feldproben
INIT_ACTIVE_FRAC = 0.5    # anfangs aktiver Anteil

INIT_DOMAIN_SIZE = 30.0   # Startvolumen für Positionen

# Kopplung / Interaktion
COUPLING_RADIUS = 3.0
COUPLING_RADIUS_SQ = COUPLING_RADIUS ** 2
COUPLING_STRENGTH_E = 0.10   # Energiekopplung
COUPLING_STRENGTH_W = 0.05   # Gewichtskopplung

# Zusatz: Wellen-Dynamik (wie v4)
WAVE_THRESHOLD = 2.0         # ab dieser |E| wird eine Welle ausgelöst
WAVE_STRENGTH = 0.12         # Stärke der Welleninjektion
WAVE_DECAY = 0.4             # radiale Abschwächung der Welle

# Rauschen
NOISE_LEVEL_E = 0.10         # Energie-Rauschen
NOISE_LEVEL_V = 0.05         # Geschwindigkeits-Rauschen
NOISE_LEVEL_W = 0.02         # Gewicht-Rauschen

VEL_DAMPING = 0.10           # Dämpfung der Geschwindigkeiten

# Eigenzeit
DT_MAX = 0.15
BETA = 8.0                   # Aktivität → Zeitkompression

MAX_STEPS = 8000             # harte technische Obergrenze

# Gewichtsdynamik
WEIGHT_DECAY = 0.03
WEIGHT_GROWTH_FACTOR = 0.10
WEIGHT_REBIRTH_THRESHOLD = 0.02
WEIGHT_REBIRTH_PROB = 0.01

# Audio
AUDIO_SR = 22050
MAX_AUDIO_DURATION = 180.0   # Sekunden

FREQ_MIN = 90.0
FREQ_MAX = 1200.0

# Cluster-Filter (nur Ausgabe)
CLUSTER_TOP_K = 32           # wie viele "stärkste Pakete" pro Sample hörbar bleiben

np.random.seed(42)


# -----------------------------
# 2. INITIALISIERUNG
# -----------------------------
def init_system():
    """
    Initialisiert das Feld:
    - Positionen zufällig im Volumen
    - Geschwindigkeiten klein zufällig
    - Energien in [-3, +3]
    - Gewichte: Teil aktiv, Rest nahe 0
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
# 3. WEICHE ENERGIE-BEGRÜNZUNG
# -----------------------------
def soft_limit_energy(E, E_max=3.0):
    """
    Weiche Sättigung:
        E_eff = E_max * tanh(E / E_max)
    Verhindert Explosionen ohne hartes Clipping.
    """
    return E_max * np.tanh(E / E_max)


# -----------------------------
# 4. ZUSTANDS-UPDATE (wie v4)
# -----------------------------
def update_state(positions, velocities, energies, weights, dt):
    """
    Aktualisiert:
    - Energien: lokale Kopplung + Wellen + Rauschen
    - Gewichte: Wachstum/Verfall + Rauschen + Rebirth
    - Geschwindigkeiten: Drift + Rauschen + Dämpfung
    - Positionen: freie Bewegung (ohne Wände)
    """
    N = energies.shape[0]
    new_E = energies.copy()
    new_V = velocities.copy()
    new_W = weights.copy()

    # Paarweise Distanzen (für Kopplung & Wellen)
    diff_all = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N,N,3)
    dist_sq_all = np.sum(diff_all * diff_all, axis=2)                     # (N,N)

    for i in range(N):
        if weights[i] <= 0.0:
            # nahezu inaktiver Slot → minimale Dynamik, wird via Rebirth reaktiviert
            continue

        dist_sq = dist_sq_all[i]
        neigh_mask = (dist_sq > 0.0) & (dist_sq < COUPLING_RADIUS_SQ) & (weights > 0.0)

        if np.any(neigh_mask):
            neighbor_E = energies[neigh_mask]
            neighbor_W = weights[neigh_mask]

            # Lokale Energie-Kopplung
            neighbor_mean_E = np.average(neighbor_E, weights=neighbor_W)
            dE = COUPLING_STRENGTH_E * (neighbor_mean_E - energies[i])
            new_E[i] += dt * dE

            # Gewichts-Kopplung
            local_energy_diff = np.abs(neighbor_mean_E - energies[i])
            new_W[i] += dt * (WEIGHT_GROWTH_FACTOR * local_energy_diff)

            # Bewegungsdrift Richtung Nachbarcluster
            neighbor_pos = positions[neigh_mask]
            neighbor_mean_pos = np.average(neighbor_pos, axis=0, weights=neighbor_W)
            direction = neighbor_mean_pos - positions[i]
            accel = 0.03 * direction * local_energy_diff
            new_V[i] += dt * accel

            # Wellenereignisse
            if np.abs(energies[i]) > WAVE_THRESHOLD:
                local_dist_sq = dist_sq[neigh_mask]
                local_dist = np.sqrt(local_dist_sq) + 1e-8
                wave_factor = np.exp(-WAVE_DECAY * local_dist)
                new_E[neigh_mask] += dt * WAVE_STRENGTH * energies[i] * wave_factor

    # Globale Rauschanteile
    new_E += np.sqrt(dt) * NOISE_LEVEL_E * np.random.randn(N)
    new_V += np.sqrt(dt) * NOISE_LEVEL_V * np.random.randn(N, 3)
    new_W += np.sqrt(dt) * NOISE_LEVEL_W * np.random.randn(N)

    # Natürlicher Gewichtsverfall
    new_W -= dt * WEIGHT_DECAY * new_W

    # Rebirth-Mechanik
    dead_mask = new_W < WEIGHT_REBIRTH_THRESHOLD
    dead_indices = np.where(dead_mask)[0]
    if dead_indices.size > 0:
        rebirth_flags = np.random.rand(dead_indices.size) < WEIGHT_REBIRTH_PROB
        for idx, flag in zip(dead_indices, rebirth_flags):
            if flag:
                new_W[idx] = np.random.uniform(0.3, 1.0)
                new_E[idx] = np.random.uniform(-3.0, 3.0)
                new_V[idx] = np.random.normal(0.0, 0.2, size=3)
                positions[idx] = np.random.uniform(-INIT_DOMAIN_SIZE,
                                                   INIT_DOMAIN_SIZE,
                                                   size=3)

    # Gewichte begrenzen
    new_W = np.clip(new_W, 0.0, 1.0)

    # Geschwindigkeiten dämpfen
    new_V *= (1.0 - VEL_DAMPING * dt)

    # Positionen aktualisieren
    new_pos = positions + new_V * dt

    # Energie weich begrenzen
    new_E = soft_limit_energy(new_E, E_max=3.0)

    return new_pos, new_V, new_E, new_W


# -----------------------------
# 5. AKTIVITÄT & EIGENZEIT
# -----------------------------
def compute_activity(E, E_prev, W, W_prev):
    """
    Feldaktivität:
    - gewichtete Energiedifferenz
    - + Gewichtsänderung
    """
    dE = np.mean(np.abs(E - E_prev) * 0.5 * (W + W_prev))
    dW = np.mean(np.abs(W - W_prev))
    return dE + dW


def compute_dt(activity):
    return DT_MAX / (1.0 + BETA * activity)


# -----------------------------
# 6. HAUPTSIMULATION
# -----------------------------
def run_universe_v5():
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
# 7. ENERGY → FREQ & PANNING
# -----------------------------
def energy_to_freq(E):
    E_clip = np.clip(E, -3.0, 3.0)
    x = (E_clip + 3.0) / 6.0
    return FREQ_MIN + x * (FREQ_MAX - FREQ_MIN)


def angle_to_pan(theta):
    return 0.5 + 0.5 * np.sin(theta)


# -----------------------------
# 8a. VOLLES FELD-AUDIO (wie v4)
# -----------------------------
def generate_audio_full(times, E_traj, W_traj, positions):
    total_time = float(times[-1])
    duration = min(total_time, MAX_AUDIO_DURATION)
    n_samples = int(AUDIO_SR * duration)

    print(f"[FULL] Eigenzeit: {total_time:.2f} s, Audio-Dauer: {duration:.2f} s, Samples: {n_samples}")

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
        idx = max(0, min(idx, E_traj.shape[0] - 1))

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
# 8b. CLUSTER-AUDIO (nur stärkste Energiepakete)
# -----------------------------
def generate_audio_clusters(times, E_traj, W_traj, positions, top_k=CLUSTER_TOP_K):
    """
    Filter-Ebene:
    - Universum bleibt unverändert.
    - Pro Sample werden nur die Slots mit höchster "Energielast" hörbar gemacht:
        Score = |E| * Weight
    - Alle anderen Oszillatoren werden stumm geschaltet.
    """
    total_time = float(times[-1])
    duration = min(total_time, MAX_AUDIO_DURATION)
    n_samples = int(AUDIO_SR * duration)

    print(f"[CLUSTERS] Eigenzeit: {total_time:.2f} s, Audio-Dauer: {duration:.2f} s, Samples: {n_samples}")
    print(f"[CLUSTERS] Pro Sample aktiv: top {top_k} Energiepakete")

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
        idx = max(0, min(idx, E_traj.shape[0] - 1))

        E_step = E_traj[idx]
        W_step = W_traj[idx]

        # Score für "Energiepakete": |E| * Gewicht
        score = np.abs(E_step) * W_step
        # Indizes der stärksten top_k
        if top_k < N_SLOTS:
            top_idx = np.argpartition(score, -top_k)[-top_k:]
        else:
            top_idx = np.arange(N_SLOTS)

        # Frequenzen
        freqs = energy_to_freq(E_step)
        phase_inc = 2.0 * np.pi * freqs / AUDIO_SR
        phases += phase_inc

        # Basiswelle
        wave = np.sin(phases)

        # Alles stumm schalten, was NICHT in top_k ist
        mask = np.zeros_like(wave)
        mask[top_idx] = 1.0

        wave_cluster = wave * W_step * mask

        audio[s, 0] = np.sum(wave_cluster * left_gain)
        audio[s, 1] = np.sum(wave_cluster * right_gain)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
        audio *= 0.95

    return audio.astype(np.float32)


# -----------------------------
# 9. MAIN
# -----------------------------
if __name__ == "__main__":
    print("Starte MCM-INFINITY v5 (Universum + Cluster-Filter-Ausgabe)…")

    times, E_traj, W_traj, positions = run_universe_v5()

    print("Simulation abgeschlossen.")
    print("Simulierte Schritte:", E_traj.shape[0])
    print("Ende der Eigenzeit:", times[-1])

    out_dir = "audio_exports"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Volles Feld
    print("Erzeuge FULL-Audio …")
    audio_full = generate_audio_full(times, E_traj, W_traj, positions)
    out_full = os.path.join(out_dir, "mcm_infinity_v5_full.wav")
    write(out_full, AUDIO_SR, (audio_full * 32767).astype(np.int16))
    print("Gespeichert:", out_full)

    # 2) Nur Cluster (stärkste Energiepakete)
    print("Erzeuge CLUSTER-Audio …")
    audio_clusters = generate_audio_clusters(times, E_traj, W_traj, positions)
    out_clusters = os.path.join(out_dir, "mcm_infinity_v5_clusters.wav")
    write(out_clusters, AUDIO_SR, (audio_clusters * 32767).astype(np.int16))
    print("Gespeichert:", out_clusters)

    print("FERTIG.")
