import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D  # 3D-Plot

# =========================================================
# MCM-3D-v4E+
# Full 3D Emergence & Turbulent Clustering
#
# - 3D-MCM-Raum (Ex, Ey, Ez)
# - kompakte Anfangsgaswolke
# - Zentrumskraft + Randabstoßung
# - starke, dichteabhängige Kurzreichweiten-Kopplung
# - voll 3D Wirbelterm (Vorticity) aus Geschwindigkeitsunterschieden
# - Rauschen an Dichte gekoppelt (verstärkt lokale Instabilität)
# - KEINE explizite Vorgabe von Dreh-Richtung
# - DBSCAN-Clustererkennung
# =========================================================

# ---------------------------------------------------------
# Parameter
# ---------------------------------------------------------
N = 230          # Partikelzahl
steps = 900
dt = 0.04

# Cooling
noise_max = 0.55
noise_min = 0.03

# Zentrumskraft
center_min = 0.01
center_max = 0.14

# Kurzreichweiten-Kopplung (stärker als v4E)
coupling_base = 1.4
sigma = 0.20        # noch kürzere Reichweite
alpha_density = 2   # dichteverstärkend

# Randabstoßung
edge_threshold = 2.6
edge_k = 0.90

# leichte Expansion
phase_drift = 0.05

# Wirbel-/Scherstärke (Vorticity)
vorticity_gamma = 1.4   # deutlich stärker als vorher

np.random.seed(42)

# Für Analyse: optional Historie der Energienormen
energy_norm_history = []

# ---------------------------------------------------------
# Initialzustand – kompakte Gaswolke
# ---------------------------------------------------------
Ex = np.random.uniform(-0.4, 0.4, N)
Ey = np.random.uniform(-0.4, 0.4, N)
Ez = np.random.uniform(-0.4, 0.4, N)

Vx = np.zeros(N)
Vy = np.zeros(N)
Vz = np.zeros(N)


# ---------------------------------------------------------
# Kraftfeld mit 3D-Vorticity & Dichtekollaps
# ---------------------------------------------------------
def compute_forces_3d(Ex, Ey, Ez, Vx, Vy, Vz, tau):
    N = len(Ex)
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Fz = np.zeros(N)

    # ---------- 1. Zentrumskraft ----------
    center_pull = center_min + (center_max - center_min) * tau
    Fx += -center_pull * Ex
    Fy += -center_pull * Ey
    Fz += -center_pull * Ez

    # ---------- 2. Randabstoßung ----------
    R = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    edge_mask = R > edge_threshold
    if np.any(edge_mask):
        Fx[edge_mask] += -edge_k * (Ex[edge_mask] / (R[edge_mask] + 1e-12))
        Fy[edge_mask] += -edge_k * (Ey[edge_mask] / (R[edge_mask] + 1e-12))
        Fz[edge_mask] += -edge_k * (Ez[edge_mask] / (R[edge_mask] + 1e-12))

    # ---------- 3. Lokale Kopplung + Dichteverstärkung ----------
    dx = Ex[:, None] - Ex[None, :]
    dy = Ey[:, None] - Ey[None, :]
    dz = Ez[:, None] - Ez[None, :]
    dist2 = dx*dx + dy*dy + dz*dz + 1e-12

    weights = np.exp(-dist2 / (2 * sigma**2))
    np.fill_diagonal(weights, 0.0)

    local_density = np.sum(weights, axis=1)
    strength = coupling_base * (1.0 + alpha_density * local_density)

    Fx += strength * np.sum(weights * dx, axis=1)
    Fy += strength * np.sum(weights * dy, axis=1)
    Fz += strength * np.sum(weights * dz, axis=1)

    # ---------- 4. Voll 3D-Vorticity (Wirbelterm) ----------
    # Geschwindigkeitsdifferenzen
    dvx = Vx[:, None] - Vx[None, :]
    dvy = Vy[:, None] - Vy[None, :]
    dvz = Vz[:, None] - Vz[None, :]

    # Kreuzprodukt dv x dr → Wirbelrichtung (ohne Vorzeichenvorgabe)
    # dr = (dx, dy, dz)
    # (dv x dr)_x = dvy*dz - dvz*dy, etc.
    cx = dvy * dz - dvz * dy
    cy = dvz * dx - dvx * dz
    cz = dvx * dy - dvy * dx

    Fx_vort = vorticity_gamma * np.sum(weights * cx, axis=1)
    Fy_vort = vorticity_gamma * np.sum(weights * cy, axis=1)
    Fz_vort = vorticity_gamma * np.sum(weights * cz, axis=1)

    # Dichteverstärkung → Wirbel wachsen in dichten Regionen stärker
    dens_factor = (1.0 + 0.6 * local_density)
    Fx += Fx_vort * dens_factor
    Fy += Fy_vort * dens_factor
    Fz += Fz_vort * dens_factor

    # ---------- 5. leichte radiale Expansion ----------
    Fx += phase_drift * np.tanh(Ex)
    Fy += phase_drift * np.tanh(Ey)
    Fz += phase_drift * np.tanh(Ez)

    # Norm der Energievektoren speichern (zur Grobanalyse)
    energy_norm_history.append(np.sqrt(Ex**2 + Ey**2 + Ez**2).mean())

    return Fx, Fy, Fz, local_density


# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------
for t in range(steps):
    tau = t / (steps - 1)

    Fx, Fy, Fz, local_density = compute_forces_3d(Ex, Ey, Ez, Vx, Vy, Vz, tau)

    # Cooling
    noise_t = noise_max - (noise_max - noise_min) * tau

    # Rauschen an Dichte gekoppelt:
    # mehr Dichte → mehr lokale chaotische Störung (verstärkt Instabilität)
    Fx += np.random.randn(N) * noise_t * (1.0 + 0.4 * local_density)
    Fy += np.random.randn(N) * noise_t * (1.0 + 0.4 * local_density)
    Fz += np.random.randn(N) * noise_t * 0.8

    # Geschwindigkeiten aktualisieren
    Vx += Fx * dt
    Vy += Fy * dt
    Vz += Fz * dt

    # Positionen aktualisieren
    Ex += Vx * dt
    Ey += Vy * dt
    Ez += Vz * dt

    # auf MCM-Würfel begrenzen
    Ex = np.clip(Ex, -3, 3)
    Ey = np.clip(Ey, -3, 3)
    Ez = np.clip(Ez, -3, 3)


# ---------------------------------------------------------
# DBSCAN-Clustererkennung
# ---------------------------------------------------------
points = np.vstack([Ex, Ey, Ez]).T
db = DBSCAN(eps=0.35, min_samples=6).fit(points)
labels = db.labels_
clusters = np.unique(labels)

# ---------------------------------------------------------
# 3D-Plot: Cluster farbig, Rest grau
# ---------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

for idx, c in enumerate(clusters):
    mask = labels == c
    if c == -1:
        ax.scatter(Ex[mask], Ey[mask], Ez[mask],
                   s=15, c='lightgray', alpha=0.35,
                   label="Noise" if idx == 0 else None)
    else:
        ax.scatter(Ex[mask], Ey[mask], Ez[mask],
                   s=26, color=colors[idx], alpha=0.9,
                   label=f"Cluster {c}")

ax.set_title("MCM-3D-v4E+ – Full 3D Emergence & Turbulent Clustering")
ax.set_xlabel("E_x"); ax.set_ylabel("E_y"); ax.set_zlabel("E_z")
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-3, 3)

if len(clusters) > 1:
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))


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