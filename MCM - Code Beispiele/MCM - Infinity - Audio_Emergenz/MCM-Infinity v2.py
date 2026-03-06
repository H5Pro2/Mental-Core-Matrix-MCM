import numpy as np
from scipy.io.wavfile import write
import os
# ============================================================
#  MCM-INFINITY v2
#  Fully dynamic energy continuum
#  - nur eine Energiedimension: [-∞, +∞], weich begrenzt auf ~[-3, +3]
#  - keine Gruppen, keine Archetypen, keine Feldervorgaben
#  - 3D-Raum ohne Wände, Aktoren mit Eigenbewegung
#  - Eigenzeit: dt hängt von Aktivität ab
#  - Audio: 256 Oszillatoren als hörbare Projektion des Feldes
# ============================================================

# -----------------------------
# 1. Systemparameter
# -----------------------------
N_AGENTS = 256

INIT_DOMAIN_SIZE = 20.0   # nur für Startverteilung, danach keine Wände mehr

COUPLING_RADIUS = 2.5
COUPLING_RADIUS_SQ = COUPLING_RADIUS ** 2
COUPLING_STRENGTH = 0.10

NOISE_LEVEL_E = 0.12      # Energiereauschen
NOISE_LEVEL_V = 0.05      # Bewegungsrauschen

VEL_DAMPING = 0.15        # leichte Dämpfung, damit Geschwindigkeiten nicht explodieren

DT_MAX = 0.15             # größter Zeitschritt
BETA = 8.0                # wie stark Aktivität dt komprimiert

MAX_STEPS = 6000          # reine technische Obergrenze

# Audio-Parameter
AUDIO_SR = 22050
MAX_AUDIO_DURATION = 180.0  # Sekunden

# Mapping: Energie -> Frequenz
FREQ_MIN = 90.0
FREQ_MAX = 1200.0

np.random.seed(42)


# -----------------------------
# 2. Initialisierung
# -----------------------------
def init_system():
    """
    Startzustand:
    - Positionen zufällig im großen 3D-Volumen
    - kleine zufällige Startgeschwindigkeiten
    - Energien zufällig im Bereich [-3, 3]
    """
    positions = np.random.uniform(-INIT_DOMAIN_SIZE,
                                  INIT_DOMAIN_SIZE,
                                  size=(N_AGENTS, 3))

    velocities = np.random.normal(0.0, 0.2, size=(N_AGENTS, 3))

    energies = np.random.uniform(-3.0, 3.0, size=N_AGENTS)

    return positions, velocities, energies


# -----------------------------
# 3. Weiche Energiebegrenzung
# -----------------------------
def soft_limit_energy(E, E_max=3.0):
    """
    Weiche Begrenzung der Energie mit tanh:
        E_eff = E_max * tanh(E / E_max)
    Dadurch keine harten Wände, aber keine unendliche Explosion.
    """
    return E_max * np.tanh(E / E_max)


# -----------------------------
# 4. Energiedynamik + Bewegung
# -----------------------------
def update_state(positions, velocities, energies, dt):
    """
    Aktualisiert:
    - Energie (lokale Kopplung + Rauschen)
    - Geschwindigkeit (leichte Kopplung + Rauschen + Dämpfung)
    - Position (freie Bewegung, keine Wände)
    """
    N = energies.shape[0]
    new_E = energies.copy()
    new_V = velocities.copy()

    for i in range(N):
        diff = positions - positions[i]           # (N, 3)
        dist_sq = np.sum(diff * diff, axis=1)     # (N,)

        neigh_mask = (dist_sq > 0.0) & (dist_sq < COUPLING_RADIUS_SQ)

        if np.any(neigh_mask):
            # --- Energie: Angleichung an Nachbarn ---
            neighbor_mean_E = np.mean(energies[neigh_mask])
            dE = COUPLING_STRENGTH * (neighbor_mean_E - energies[i])
            new_E[i] += dt * dE

            # --- Bewegung: sanfter Drift in Richtung lokaler Nachbarn ---
            # Richtungsvektor: Mittel der Nachbarpositionen
            neighbor_mean_pos = np.mean(positions[neigh_mask], axis=0)
            direction = neighbor_mean_pos - positions[i]

            # Energiegewichtung: stärkere Bewegung bei hoher lokaler Differenz
            local_energy_diff = neighbor_mean_E - energies[i]
            accel = 0.05 * direction * local_energy_diff

            new_V[i] += dt * accel

    # Rauschen in Energie und Geschwindigkeit
    new_E += np.sqrt(dt) * NOISE_LEVEL_E * np.random.randn(N)
    new_V += np.sqrt(dt) * NOISE_LEVEL_V * np.random.randn(N, 3)

    # leichte Geschwindigkeitsdämpfung (keine echte Wand, nur Bremsung)
    new_V *= (1.0 - VEL_DAMPING * dt)

    # Positionen frei laufen lassen
    new_pos = positions + new_V * dt

    # Energie weich begrenzen (kein hartes Clipping)
    new_E = soft_limit_energy(new_E, E_max=3.0)

    return new_pos, new_V, new_E


# -----------------------------
# 5. Aktivität und Eigenzeit
# -----------------------------
def compute_activity(E, E_prev):
    """Mittlere Energiedifferenz als Feldaktivität."""
    return np.mean(np.abs(E - E_prev))


def compute_dt(activity):
    """
    Eigenzeit:
        dt = DT_MAX / (1 + BETA * activity)
    """
    return DT_MAX / (1.0 + BETA * activity)


# -----------------------------
# 6. Hauptsimulation
# -----------------------------
def run_self_organizing_universe():
    positions, velocities, E = init_system()
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
        positions, velocities, E = update_state(positions, velocities, E, dt)
        t += dt

        # Optional: früh abbrechen, wenn genügend Eigenzeit für Audio gesammelt ist
        if t >= MAX_AUDIO_DURATION:
            break

    return np.array(times, dtype=np.float64), np.array(traj, dtype=np.float64), positions


# -----------------------------
# 7. Energy → Frequency
# -----------------------------
def energy_to_freq(E):
    """
    Energie E (nach soft_limit) liegt effektiv etwa in [-3, +3].
    Linearer Map auf [FREQ_MIN, FREQ_MAX].
    """
    E_clip = np.clip(E, -3.0, 3.0)
    x = (E_clip + 3.0) / 6.0  # 0..1
    return FREQ_MIN + x * (FREQ_MAX - FREQ_MIN)


def angle_to_pan(theta):
    """Winkel -> Stereo-Panning (0 = links, 1 = rechts)."""
    return 0.5 + 0.5 * np.sin(theta)


# -----------------------------
# 8. Audio-Synthese
# -----------------------------
def generate_eigenzeit_audio(times, energy_traj, positions):
    total_time = float(times[-1])
    duration = min(total_time, MAX_AUDIO_DURATION)
    n_samples = int(AUDIO_SR * duration)

    print(f"Eigenzeit: {total_time:.2f} s, Audio-Dauer: {duration:.2f} s, Samples: {n_samples}")

    # Stereo-Panning aus der xy-Projektion
    theta_vals = np.arctan2(positions[:, 1], positions[:, 0])
    pans = angle_to_pan(theta_vals)
    left_gain = 1.0 - pans
    right_gain = pans

    phases = np.zeros(N_AGENTS, dtype=np.float64)
    audio = np.zeros((n_samples, 2), dtype=np.float64)

    times_np = np.asarray(times)

    for s in range(n_samples):
        t = s / AUDIO_SR

        # passenden Simulationsindex aus Eigenzeit finden
        idx = np.searchsorted(times_np, t, side='right') - 1
        if idx < 0:
            idx = 0
        if idx >= energy_traj.shape[0]:
            idx = energy_traj.shape[0] - 1

        E_step = energy_traj[idx]
        freqs = energy_to_freq(E_step)

        phase_inc = 2.0 * np.pi * freqs / AUDIO_SR
        phases += phase_inc

        wave = np.sin(phases)

        audio[s, 0] = np.sum(wave * left_gain)
        audio[s, 1] = np.sum(wave * right_gain)

    # Normalisieren
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
        audio *= 0.95

    return audio.astype(np.float32)


# -----------------------------
# 9. Ausführung
# -----------------------------
if __name__ == "__main__":
    print("Starte MCM-INFINITY v2 (maximal freies Feld) …")

    times, energy_traj, positions = run_self_organizing_universe()

    print("Simulation abgeschlossen.")
    print("Simulierte Schritte:", energy_traj.shape[0])
    print("Ende der Eigenzeit:", times[-1])

    print("Erzeuge Audio …")
    audio = generate_eigenzeit_audio(times, energy_traj, positions)

    out_dir = "audio_exports"
    os.makedirs(out_dir, exist_ok=True)

    out_name = os.path.join(out_dir, "mcm_infinity_v2.wav")
    write(out_name, AUDIO_SR, (audio * 32767).astype(np.int16))
    print("Audio gespeichert als:", out_name)
    print("FERTIG.")
