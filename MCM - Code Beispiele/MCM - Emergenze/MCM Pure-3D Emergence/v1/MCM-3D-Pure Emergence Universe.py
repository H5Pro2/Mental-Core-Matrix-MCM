"""
3D-MCM Pure Emergence Universe
------------------------------

- 3D-Energieraum: (Ex, Ey, Ez) ∈ [-3, 3]^3
- N "Energie-Akteure" bewegen sich durch diesen Raum.
- Kräfte:
    * Zentrumskraft (Rückzug zum Neutralpunkt)
    * Randabstoßung (Vermeidung extremer |E|-Zustände)
    * lokale, gaußförmige Kopplung (selbstorganisierende Cluster)
    * Phasen-Drift (sanfte Expansion)
- Stochastisches Rauschen mit Cooling (anfangs heiß, später kühler)

WICHTIG:
- Keine Clusterregeln, keine Stabilitätslogik, kein "wenn Struktur dann ...".
- Alles, was an Mustern entsteht, ist pure Emergenz.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nur für 3D-Plot nötig

# -------------------------------------------------------------
# 1. Parameter
# -------------------------------------------------------------
N = 200          # Anzahl der "Energiepartikel"
steps = 800      # Zeitschritte
dt = 0.04

# Cooling (Rauschen)
noise_max = 0.8
noise_min = 0.25

# Zentrumskraft λ(t) – wird im Laufe der Zeit stärker
center_min = 0.01
center_max = 0.08

# Lokale Kopplung im 3D-MCM-Raum
coupling = 0.18
sigma = 0.9           # Reichweite der Kopplung im Energieraum

# Randabstoßung (verhindert Festkleben an ±3)
edge_threshold = 2.5
edge_k = 0.6

# Phasen-Drift (sanfte Expansion nach außen, saturiert)
phase_drift = 0.10

seed = 1234
np.random.seed(seed)

# -------------------------------------------------------------
# 2. Initialzustand – frühes "Universum" nahe Zentrum
# -------------------------------------------------------------
Ex = np.random.uniform(-0.4, 0.4, N)
Ey = np.random.uniform(-0.4, 0.4, N)
Ez = np.random.uniform(-0.4, 0.4, N)

Vx = np.zeros(N)
Vy = np.zeros(N)
Vz = np.zeros(N)

# Für Analyse: optional Historie der Energienormen
energy_norm_history = []

# -------------------------------------------------------------
# 3. Kraftberechnung im 3D-MCM-Raum
# -------------------------------------------------------------
def compute_forces_3d(Ex, Ey, Ez, tau):
    """
    Berechnet Kräfte für alle Akteure im 3D-Energieraum.

    tau ∈ [0,1] ist normierte Zeit:
        0 = frühe, heiße Phase
        1 = späte, kühlere, stärker zentrierende Phase
    """
    N = len(Ex)
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Fz = np.zeros(N)

    # ---------- 1. Zentrumskraft (Zug zum Neutralpunkt) ----------
    center_pull = center_min + (center_max - center_min) * tau
    Fx += -center_pull * Ex
    Fy += -center_pull * Ey
    Fz += -center_pull * Ez

    # ---------- 2. Randabstoßung ----------
    # Norm der 3D-Energie
    R = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    edge_mask = R > edge_threshold
    if np.any(edge_mask):
        # Richtung zurück zum Mittelpunkt
        # (negatives Vorzeichen, weil Richtung zum Zentrum)
        Fx[edge_mask] += -edge_k * (Ex[edge_mask] / (R[edge_mask] + 1e-12)) * (R[edge_mask] - edge_threshold)
        Fy[edge_mask] += -edge_k * (Ey[edge_mask] / (R[edge_mask] + 1e-12)) * (R[edge_mask] - edge_threshold)
        Fz[edge_mask] += -edge_k * (Ez[edge_mask] / (R[edge_mask] + 1e-12)) * (R[edge_mask] - edge_threshold)

    # ---------- 3. Lokale Kopplung (Gauß-Kern in 3D) ----------
    # paarweise Differenzen
    dx = Ex[:, None] - Ex[None, :]
    dy = Ey[:, None] - Ey[None, :]
    dz = Ez[:, None] - Ez[None, :]
    dist2 = dx*dx + dy*dy + dz*dz

    weights = np.exp(-dist2 / (2 * sigma**2))
    np.fill_diagonal(weights, 0.0)  # kein Eigen-Einfluss

    # Kopplungskraft ~ gewichtete Differenzen
    Fx += coupling * np.sum(weights * dx, axis=1)
    Fy += coupling * np.sum(weights * dy, axis=1)
    Fz += coupling * np.sum(weights * dz, axis=1)

    # ---------- 4. Phasen-Drift (sanfte Expansion in alle Richtungen) ----------
    Fx += phase_drift * np.tanh(Ex)
    Fy += phase_drift * np.tanh(Ey)
    Fz += phase_drift * np.tanh(Ez)

    return Fx, Fy, Fz

# -------------------------------------------------------------
# 4. Zeitschleife
# -------------------------------------------------------------
for t in range(steps):
    tau = t / (steps - 1)  # normierte Zeit [0,1]

    # Kräfte
    Fx, Fy, Fz = compute_forces_3d(Ex, Ey, Ez, tau)

    # Cooling: Rauschen nimmt mit der Zeit ab
    noise_t = noise_max - (noise_max - noise_min) * tau

    Vx += Fx * dt
    Vy += Fy * dt
    Vz += Fz * dt

    Vx += np.random.normal(scale=noise_t, size=N) * dt
    Vy += np.random.normal(scale=noise_t, size=N) * dt
    Vz += np.random.normal(scale=noise_t, size=N) * dt

    Ex += Vx * dt
    Ey += Vy * dt
    Ez += Vz * dt

    # Begrenzung auf den MCM-Würfel [-3, 3]^3
    Ex = np.clip(Ex, -3, 3)
    Ey = np.clip(Ey, -3, 3)
    Ez = np.clip(Ez, -3, 3)

    # Norm der Energievektoren speichern (zur Grobanalyse)
    energy_norm_history.append(np.sqrt(Ex**2 + Ey**2 + Ez**2).mean())

# -------------------------------------------------------------
# 5. Visualisierung – 3D-Universum im MCM-Raum
# -------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(Ex, Ey, Ez, s=10, alpha=0.7)

ax.set_title("MCM Pure Emergence – 3D-Energie-Universum (finaler Zustand)")
ax.set_xlabel("E_x")
ax.set_ylabel("E_y")
ax.set_zlabel("E_z")

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 6. Optional: Verlauf der mittleren Energienorm
# -------------------------------------------------------------
plt.figure(figsize=(8, 3))
plt.plot(energy_norm_history)
plt.xlabel("Zeit")
plt.ylabel("mittlere |E|")
plt.title("MCM 3D – Entwicklung der mittleren Energienorm")
plt.tight_layout()
plt.show()
