import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D  # nur für den 3D-Plot

# --------------------------------------------------------
# MCM-3D-Pure-Emergence v3 – mit Rotationsdynamik
# --------------------------------------------------------

# Simulationsparameter
N = 200          # Partikelzahl
steps = 800      # Zeitschritte
dt = 0.04

# Cooling (Rauschen)
noise_max = 0.50
noise_min = 0.02

# Zentrumskraft
center_min = 0.01
center_max = 0.14

# Kurzreichweitige Kopplung
coupling = 0.70
sigma = 0.18

# Dichteverstärkung (density-dependent coupling)
alpha = 0.15

# Randabstoßung
edge_threshold = 2.5
edge_k = 0.85

# leichte Expansion nach außen
phase_drift = 0.06

# Rotationsparameter (Gasrotation im (Ex,Ey)-Plane)
omega_base = 0.18         # globale Grundrotation
omega_jitter = 0.05       # zufällige Abweichung für lokale Turbulenz
rot_scale = 1.0           # Faktor, wie stark sich Rotation auswirkt

np.random.seed(42)

# --------------------------------------------------------
# Initialzustand – kompakte Anfangswolke
# --------------------------------------------------------
Ex = np.random.uniform(-0.4, 0.4, N)
Ey = np.random.uniform(-0.4, 0.4, N)
Ez = np.random.uniform(-0.4, 0.4, N)

Vx = np.zeros(N)
Vy = np.zeros(N)
Vz = np.zeros(N)

# individuelle Rotationsfrequenz pro Partikel
local_omega = omega_base + omega_jitter * np.random.randn(N)


# --------------------------------------------------------
# Kraftberechnung (inkl. dichteabhängiger Kopplung)
# --------------------------------------------------------
def compute_forces_3d(Ex, Ey, Ez, tau):
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Fz = np.zeros(N)

    # ---- Zentrumskraft (Selbstregulation) ----
    center_pull = center_min + (center_max - center_min) * tau
    Fx += -center_pull * Ex
    Fy += -center_pull * Ey
    Fz += -center_pull * Ez

    # ---- Randabstoßung (Vermeidung von |E| > edge_threshold) ----
    R = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    edge_mask = R > edge_threshold
    if np.any(edge_mask):
        Fx[edge_mask] += -edge_k * (Ex[edge_mask] / (R[edge_mask] + 1e-12))
        Fy[edge_mask] += -edge_k * (Ey[edge_mask] / (R[edge_mask] + 1e-12))
        Fz[edge_mask] += -edge_k * (Ez[edge_mask] / (R[edge_mask] + 1e-12))

    # ---- Lokale Kopplung + Dichteverstärkung ----
    dx = Ex[:, None] - Ex[None, :]
    dy = Ey[:, None] - Ey[None, :]
    dz = Ez[:, None] - Ez[None, :]

    dist2 = dx*dx + dy*dy + dz*dz
    weights = np.exp(-dist2 / (2 * sigma**2))
    np.fill_diagonal(weights, 0.0)

    # lokale "Dichte" pro Partikel
    local_density = np.sum(weights, axis=1)

    # verstärkte Kopplung dort, wo es dichter ist
    strength = coupling * (1.0 + alpha * local_density)

    Fx += strength * np.sum(weights * dx, axis=1)
    Fy += strength * np.sum(weights * dy, axis=1)
    Fz += strength * np.sum(weights * dz, axis=1)

    # ---- Expansion (sanft nach außen) ----
    Fx += phase_drift * np.tanh(Ex)
    Fy += phase_drift * np.tanh(Ey)
    Fz += phase_drift * np.tanh(Ez)

    return Fx, Fy, Fz


# --------------------------------------------------------
# Hauptsimulation
# --------------------------------------------------------
for t in range(steps):
    tau = t / (steps - 1)

    Fx, Fy, Fz = compute_forces_3d(Ex, Ey, Ez, tau)

    # Rotationskräfte (Gasrotation im Ex/Ey-Plane)
    # v_rot ~ omega x r  -> in 2D: (-omega * y, omega * x)
    # wir behandeln Rotation als zusätzliche Beschleunigung
    Fx_rot = rot_scale * local_omega * (-Ey)
    Fy_rot = rot_scale * local_omega * ( Ex)
    Fz_rot = 0.0

    Fx += Fx_rot
    Fy += Fy_rot

    # Zusätzlicher Turbulenz-Boost in einer mittleren Phase
    if 200 < t < 320:
        Fx += 0.08 * np.random.randn(N)
        Fy += 0.08 * np.random.randn(N)
        Fz += 0.08 * np.random.randn(N)

    # Cooling: Rauschen nimmt mit der Zeit ab
    noise_t = noise_max - (noise_max - noise_min) * tau

    Vx += Fx * dt + np.random.normal(scale=noise_t, size=N) * dt
    Vy += Fy * dt + np.random.normal(scale=noise_t, size=N) * dt
    Vz += Fz * dt + np.random.normal(scale=noise_t, size=N) * dt

    Ex += Vx * dt
    Ey += Vy * dt
    Ez += Vz * dt

    Ex = np.clip(Ex, -3, 3)
    Ey = np.clip(Ey, -3, 3)
    Ez = np.clip(Ez, -3, 3)


# --------------------------------------------------------
# Clustererkennung (DBSCAN) & farbige Darstellung
# --------------------------------------------------------
points = np.vstack([Ex, Ey, Ez]).T

# eps gut anpassen: grober Richtwert für Clusterabstand
db = DBSCAN(eps=0.35, min_samples=6).fit(points)
labels = db.labels_
clusters = np.unique(labels)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

for idx, c in enumerate(clusters):
    mask = labels == c
    if c == -1:
        # Punkte ohne Clusterzugehörigkeit
        ax.scatter(Ex[mask], Ey[mask], Ez[mask],
                   s=12, c='lightgray', alpha=0.35, label="Noise" if idx == 0 else None)
    else:
        ax.scatter(Ex[mask], Ey[mask], Ez[mask],
                   s=24, color=colors[idx], alpha=0.9,
                   label=f"Cluster {c}")

ax.set_title("MCM Pure Emergence v3 – 3D-Clusterbildung mit Rotationsdynamik")
ax.set_xlabel("E_x")
ax.set_ylabel("E_y")
ax.set_zlabel("E_z")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

if len(clusters) > 1:
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
