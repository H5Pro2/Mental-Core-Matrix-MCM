import numpy as np

# ============================================================
#  MCM-3D-256 MIT SELBSTSKALIERENDER ZEIT (RAM-SPARSAME VERSION)
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

np.random.seed(42)


# -----------------------------
# 2. HILFSFUNKTIONEN
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


def update_energies(positions, energies, dt):
    """
    Ein Energie-Update mit:
    - lokaler Kopplung (RAM-sparsam, keine NxN-Tensoren)
    - dt-skalierter Dynamik
    - Rauschen ~ sqrt(dt)
    """

    N = energies.shape[0]
    new_E = energies.copy()

    # Für jeden Agenten Nachbarn lokal bestimmen
    for i in range(N):
        # Vektor von i zu allen anderen
        diff = positions - positions[i]           # shape (N,3)
        dist_sq = np.sum(diff * diff, axis=1)     # shape (N,)

        # Nachbarn innerhalb des Radius, ohne sich selbst
        neigh_mask = (dist_sq > 0.0) & (dist_sq < COUPLING_RADIUS_SQ)

        if np.any(neigh_mask):
            neighbor_mean = np.mean(energies[neigh_mask])
            dE = COUPLING_STRENGTH * (neighbor_mean - energies[i])
            new_E[i] += dt * dE

    # Rauschen (ungeglättet, emergent roh)
    new_E += np.sqrt(dt) * NOISE_LEVEL * np.random.randn(N)

    # Begrenzen auf MCM-Bereich
    return np.clip(new_E, -3.0, 3.0)


# -----------------------------
# 3. SELBSTSKALIERENDER TAKT
# -----------------------------
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
# 4. HAUPTSIMULATION
# -----------------------------
def run_self_scaled_simulation():
    positions, E = init_positions_and_energies()
    E_prev = E.copy()

    times = []
    traj = []

    t = 0.0

    for step in range(MAX_STEPS):
        # Aktivität messen
        A = compute_activity(E, E_prev)

        # dt aus Aktivität ableiten
        dt = compute_dt(A)

        # Zustände speichern
        times.append(t)
        traj.append(E.copy())

        # Energieupdate
        E_prev = E.copy()
        E = update_energies(positions, E, dt)

        # Zeit fortschreiben
        t += dt

    return np.array(times, dtype=np.float64), np.array(traj, dtype=np.float64), positions


# -----------------------------
# 5. AUSFÜHRUNG
# -----------------------------
if __name__ == "__main__":
    print("Starte MCM-3D-256 Selbstskaliert (RAM-sparsam)…")

    times, energy_traj, positions = run_self_scaled_simulation()

    print("FERTIG.")
    print()
    print("Simulierte Schritte:", energy_traj.shape[0])
    print("Simulierte Eigenzeit (Endzeit):", times[-1], "Sekunden")
    if times.shape[0] > 2:
        print("Erster dt:", times[1] - times[0])
        print("Letzter dt:", times[-1] - times[-2])
    print()
    print("times       shape =", times.shape)
    print("energy_traj shape =", energy_traj.shape)
    print("positions   shape =", positions.shape)
