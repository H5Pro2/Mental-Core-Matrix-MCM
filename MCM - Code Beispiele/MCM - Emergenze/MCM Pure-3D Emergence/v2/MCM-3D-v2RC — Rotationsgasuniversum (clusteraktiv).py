import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D  # 3D-Plot

# =========================================================
# MCM-3D-v3RC 
# Rotierendes Gasuniversum mit dichteabhängiger Kopplung
# (clusteraktive Version – kritische Fragmentierung)
#
# Eigenschaften:
# - 3D-MCM-Raum (Ex, Ey, Ez)
# - kompakte Gaswolke zu Beginn
# - Rotation ab t = 20 (z-Achsenrotation)
# - dichteabhängige Kopplung verstärkt lokale Überdichten
# - DBSCAN für Clustererkennung
#
# Diese Version zeigt TENDENZ ZU CLUSTERN:
# - In vielen Läufen 1–2 Cluster
# - In manchen Läufen sogar mehrere Fragmente
#
# Unterschied zu v3R:
# - v3R = semikritisch (Cluster selten)
# - v3RC = kritisch (Cluster häufig)
# =========================================================

# ---------------------------------------------------------
# Parameter
# ---------------------------------------------------------
N = 220
steps = 900
dt = 0.04

# Cooling
noise_max = 0.55
noise_min = 0.03

# Zentrumskraft
center_min = 0.01
center_max = 0.14

# Kopplung
coupling_base = 0.65
sigma = 0.22
alpha_density = 0.18

# Randabstoßung
edge_threshold = 2.6
edge_k = 0.85

# Expansion
phase_drift = 0.06

# Rotation
omega_global = 0.12
omega_jitter = 0.05
rotation_start = 20

np.random.seed(42)

# ---------------------------------------------------------
# Initialzustand
# ---------------------------------------------------------
Ex = np.random.uniform(-0.4, 0.4, N)
Ey = np.random.uniform(-0.4, 0.4, N)
Ez = np.random.uniform(-0.4, 0.4, N)

Vx = np.zeros(N)
Vy = np.zeros(N)
Vz = np.zeros(N)

omega_i = omega_global + omega_jitter * np.random.randn(N)

# ---------------------------------------------------------
def compute_forces_3d(Ex, Ey, Ez, tau):
    N = len(Ex)
    Fx = np.zeros(N); Fy = np.zeros(N); Fz = np.zeros(N)

    center_pull = center_min + (center_max - center_min) * tau
    Fx += -center_pull * Ex
    Fy += -center_pull * Ey
    Fz += -center_pull * Ez

    R = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    edge_mask = R > edge_threshold
    if np.any(edge_mask):
        Fx[edge_mask] += -edge_k * (Ex[edge_mask] / (R[edge_mask] + 1e-12))
        Fy[edge_mask] += -edge_k * (Ey[edge_mask] / (R[edge_mask] + 1e-12))
        Fz[edge_mask] += -edge_k * (Ez[edge_mask] / (R[edge_mask] + 1e-12))

    dx = Ex[:, None] - Ex[None, :]
    dy = Ey[:, None] - Ey[None, :]
    dz = Ez[:, None] - Ez[None, :]
    dist2 = dx*dx + dy*dy + dz*dz + 1e-12

    weights = np.exp(-dist2 / (2 * sigma**2))
    np.fill_diagonal(weights, 0.0)

    local_density = np.sum(weights, axis=1)
    strength = coupling_base * (1 + alpha_density * local_density)

    Fx += strength * np.sum(weights * dx, axis=1)
    Fy += strength * np.sum(weights * dy, axis=1)
    Fz += strength * np.sum(weights * dz, axis=1)

    Fx += phase_drift * np.tanh(Ex)
    Fy += phase_drift * np.tanh(Ey)
    Fz += phase_drift * np.tanh(Ez)

    return Fx, Fy, Fz

# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------
for t in range(steps):
    tau = t / (steps - 1)

    Fx, Fy, Fz = compute_forces_3d(Ex, Ey, Ez, tau)
    noise_t = noise_max - (noise_max - noise_min) * tau

    if t >= rotation_start:
        Fx += -omega_i * Ey
        Fy +=  omega_i * Ex

    Vx += Fx * dt + np.random.normal(scale=noise_t, size=N) * dt
    Vy += Fy * dt + np.random.normal(scale=noise_t, size=N) * dt
    Vz += Fz * dt + np.random.normal(scale=noise_t, size=N) * dt

    Ex += Vx * dt
    Ey += Vy * dt
    Ez += Vz * dt

    Ex = np.clip(Ex, -3, 3)
    Ey = np.clip(Ey, -3, 3)
    Ez = np.clip(Ez, -3, 3)

# ---------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------
points = np.vstack([Ex, Ey, Ez]).T
db = DBSCAN(eps=0.40, min_samples=6).fit(points)
labels = db.labels_
clusters = np.unique(labels)

# ---------------------------------------------------------
# Plot
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

ax.set_title("MCM-3D-v3RC – Rotationsgasuniversum (clusteraktiv)")
ax.set_xlabel("E_x"); ax.set_ylabel("E_y"); ax.set_zlabel("E_z")
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-3, 3)

ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
plt.show()
