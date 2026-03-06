import numpy as np
from scipy.io.wavfile import write
import os
# ============================================================
#  MCM-INFINITY v3
#  Selbstorganisierendes Energiefeld mit dynamischer "Agentendichte"
#
#  - Nur eine Energiedimension (kontinuierlich, weich begrenzt)
#  - 3D-Position + Geschwindigkeit pro Feldprobe
#  - Jede Probe hat eine Gewichtung (0..1) = "lokale Feldstärke"
#  - Gewichte wachsen/zerfallen dynamisch -> effektive Agentenzahl emergent
#  - Keine Gruppen, keine Archetypen, keine festen Typen
#  - Eigenzeit abhängig von Aktivität
#  - Audio: Frequenz = Energie, Amplitude = Gewicht
# ============================================================

# -----------------------------
# 1. Systemparameter
# -----------------------------
N_SLOTS = 512            # numerische Obergrenze der "Proben" im Feld
INIT_ACTIVE_FRAC = 0.5   # anfangs aktiver Anteil (Rest latent)

INIT_DOMAIN_SIZE = 30.0  # Startvolumen für Positionen, danach keine Wände

COUPLING_RADIUS = 3.0
COUPLING_RADIUS_SQ = COUPLING_RADIUS ** 2
COUPLING_STRENGTH_E = 0.10   # Energiekopplung
COUPLING_STRENGTH_W = 0.05   # Gewichtskopplung

NOISE_LEVEL_E = 0.10         # Energiereauschen
NOISE_LEVEL_V = 0.05         # Bewegungsrauschen
NOISE_LEVEL_W = 0.02         # Rauschen in der Gewichtsdynamik

VEL_DAMPING = 0.10           # sanfte Geschwindigkeitsdämpfung

DT_MAX = 0.15
BETA = 8.0                   # Aktivität -> Eigenzeit-Kompression

MAX_STEPS = 7000             # technische Obergrenze

# Gewichtsdynamik
WEIGHT_DECAY = 0.03          # natürliche Abnahme
WEIGHT_GROWTH_FACTOR = 0.10  # wie stark Energiegradienten Gewicht verstärken
WEIGHT_REBIRTH_THRESHOLD = 0.02  # darunter gilt Slot als "quasi tot"
WEIGHT_REBIRTH_PROB = 0.01       # Wahrscheinlichkeit pro Schritt, wieder zu "spawnen"

# Audio
AUDIO_SR = 22050
MAX_AUDIO_DURATION = 180.0  # Sekunden

FREQ_MIN = 90.0
FREQ_MAX = 1200.0

np.random.seed(42)


# -----------------------------
# 2. Initialisierung
# -----------------------------
def init_system():
    """
    Initialisiert:
    - Positionen zufällig im Volumen
    - Geschwindigkeiten klein zufällig
    - Energien zufällig in [-3, 3]
    - Gewichte: ein Teil aktiv, Rest nahe 0
    """
    positions = np.random.uniform(-INIT_DOMAIN_SIZE,
                                  INIT_DOMAIN_SIZE,
                                  size=(N_SLOTS, 3))
    velocities = np.random.normal(0.0, 0.2, size=(N_SLOTS, 3))
    energies = np.random.uniform(-3.0, 3.0, size=N_SLOTS)

    weights = np.zeros(N_SLOTS, dtype=np.float64)
    active_count = int(INIT_ACTIVE_FRAC * N_SLOTS)
    active_indices = np.random.choice(N_SLOTS, size=active_count, replace=False)
    weights[active_indices] = np.random.uniform(0.3, 1.0, size=active_count)

    return positions, velocities, energies, weights


# -----------------------------
# 3. Weiche Energiebegrenzung
# -----------------------------
def soft_limit_energy(E, E_max=3.0):
    """
    E_eff = E_max * tanh(E / E_max)
    Keine harten Wände, aber keine unendliche Explosion.
    """
    return E_max * np.tanh(E / E_max)


# -----------------------------
# 4. Zustandsupdate (Position, Energie, Gewicht)
# -----------------------------
def update_state(positions, velocities, energies, weights, dt):
    N = energies.shape[0]
    new_E = energies.copy()
    new_V = velocities.copy()
    new_W = weights.copy()

    for i in range(N):
        if weights[i] <= 0.0:
            # fast inaktiver Slot: minimale Dynamik, kann später "wiedergeboren" werden
            continue

        diff = positions - positions[i]           # (N,3)
        dist_sq = np.sum(diff * diff, axis=1)     # (N,)
        neigh_mask = (dist_sq > 0.0) & (dist_sq < COUPLING_RADIUS_SQ) & (weights > 0.0)

        if np.any(neigh_mask):
            # --- Energie-Kopplung (lokaler Ausgleich) ---
            neighbor_mean_E = np.average(energies[neigh_mask], weights=weights[neigh_mask])
            dE = COUPLING_STRENGTH_E * (neighbor_mean_E - energies[i])
            new_E[i] += dt * dE

            # --- Gewichtskopplung: starke lokale Aktivität verstärkt Gewicht ---
            local_energy_diff = np.abs(neighbor_mean_E - energies[i])
            new_W[i] += dt * (WEIGHT_GROWTH_FACTOR * local_energy_diff)

            # --- Bewegungsdrift: sanfte Ausrichtung auf Nachbarcluster ---
            neighbor_mean_pos = np.average(positions[neigh_mask], axis=0, weights=weights[neigh_mask])
            direction = neighbor_mean_pos - positions[i]
            accel = 0.03 * direction * local_energy_diff
            new_V[i] += dt * accel

    # Globale Rauschanteile
    new_E += np.sqrt(dt) * NOISE_LEVEL_E * np.random.randn(N)
    new_V += np.sqrt(dt) * NOISE_LEVEL_V * np.random.randn(N, 3)
    new_W += np.sqrt(dt) * NOISE_LEVEL_W * np.random.randn(N)

    # Gewichtsnatürlichkeit: Verfallsterm
    new_W -= dt * WEIGHT_DECAY * new_W

    # Rebirth: Slots mit sehr geringem Gewicht können zufällig neu "aktiviert" werden
    dead_mask = new_W < WEIGHT_REBIRTH_THRESHOLD
    rebirth_candidates = np.where(dead_mask)[0]
    if rebirth_candidates.size > 0:
        rebirth_flags = np.random.rand(rebirth_candidates.size) < WEIGHT_REBIRTH_PROB
        for idx, flag in zip(rebirth_candidates, rebirth_flags):
            if flag:
                # "Neuer" Energiepunkt an zufälliger Stelle mit frischem Gewicht
                new_W[idx] = np.random.uniform(0.3, 1.0)
                new_E[idx] = np.random.uniform(-3.0, 3.0)
                new_V[idx] = np.random.normal(0.0, 0.2, size=3)
                positions[idx] = np.random.uniform(-INIT_DOMAIN_SIZE,
                                                   INIT_DOMAIN_SIZE,
                                                   size=3)

    # Grenzen für Gewicht: [0, 1]
    new_W = np.clip(new_W, 0.0, 1.0)

    # Geschwindigkeiten dämpfen
    new_V *= (1.0 - VEL_DAMPING * dt)

    # Positionen frei laufen lassen (keine Wände)
    new_pos = positions + new_V * dt

    # Energie weich begrenzen
    new_E = soft_limit_energy(new_E, E_max=3.0)

    return new_pos, new_V, new_E, new_W


# -----------------------------
# 5. Aktivität & Eigenzeit
# -----------------------------
def compute_activity(E, E_prev, W, W_prev):
    """
    Feldaktivität: gewichtete Energieänderung + Gewichtsdynamik.
    """
    dE = np.mean(np.abs(E - E_prev) * (0.5 * (W + W_prev)))
    dW = np.mean(np.abs(W - W_prev))
    return dE + dW


def compute_dt(activity):
    return DT_MAX / (1.0 + BETA * activity)


# -----------------------------
# 6. Hauptsimulation
# -----------------------------
def run_universe_v3():
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
# 7. Energie → Frequenz & Panning
# -----------------------------
def energy_to_freq(E):
    E_clip = np.clip(E, -3.0, 3.0)
    x = (E_clip + 3.0) / 6.0
    return FREQ_MIN + x * (FREQ_MAX - FREQ_MIN)


def angle_to_pan(theta):
    return 0.5 + 0.5 * np.sin(theta)


# -----------------------------
# 8. Audio-Synthese
# -----------------------------
def generate_eigenzeit_audio(times, E_traj, W_traj, positions):
    total_time = float(times[-1])
    duration = min(total_time, MAX_AUDIO_DURATION)
    n_samples = int(AUDIO_SR * duration)

    print(f"Eigenzeit: {total_time:.2f} s, Audio-Dauer: {duration:.2f} s, Samples: {n_samples}")

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

        # Amplitude pro Slot = Gewicht
        wave = np.sin(phases) * W_step

        audio[s, 0] = np.sum(wave * left_gain)
        audio[s, 1] = np.sum(wave * right_gain)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
        audio *= 0.95

    return audio.astype(np.float32)


# -----------------------------
# 9. Ausführung
# -----------------------------
if __name__ == "__main__":
    print("Starte MCM-INFINITY v3 (selbstorganisierendes Feld mit dynamischer Dichte) …")

    times, E_traj, W_traj, positions = run_universe_v3()

    print("Simulation abgeschlossen.")
    print("Simulierte Schritte:", E_traj.shape[0])
    print("Ende der Eigenzeit:", times[-1])

    print("Erzeuge Audio …")
    audio = generate_eigenzeit_audio(times, E_traj, W_traj, positions)

    out_dir = "audio_exports"
    os.makedirs(out_dir, exist_ok=True)

    out_name = os.path.join(out_dir, "mcm_infinity_v3.wav")
    write(out_name, AUDIO_SR, (audio * 32767).astype(np.int16))
    print("Audio gespeichert als:", out_name)
    print("FERTIG.")
