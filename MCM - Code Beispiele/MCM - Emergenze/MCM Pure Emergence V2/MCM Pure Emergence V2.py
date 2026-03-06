import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. MCM: Energiebereich & Phasen (zur Orientierung)
# -------------------------------------------------------------
# Wir arbeiten auf der kontinuierlichen Achse [-3, +3].
# Die genauen Archetypen (A1..A12) brauchst du hier nicht explizit,
# aber die Grenzen entsprechen deiner MCM:
#   G1: [-3, -1]
#   G2: [-1,  0..+1]
#   Zentrum: 0
#   G3: [+1..+2]
#   G4: [+2..+3]

# -------------------------------------------------------------
# 2. Simulations-Parameter
# -------------------------------------------------------------
N = 120          # Anzahl "mentaler Akteure"
steps = 800
dt = 0.05

# "Temperatur" (Rauschen) – hoch am Anfang, geringer am Ende
noise_max = 0.8
noise_min = 0.25

# Zug zum Zentrum – schwach am Anfang, stärker am Ende
center_min = 0.01
center_max = 0.08

# Kopplungsstärke zwischen nahen Energien
coupling = 0.22

# Abstoßung an den Rändern
edge_threshold = 2.4
edge_k = 0.4

# Phasen-Drift (sanfte Expansion nach außen, begrenzt)
phase_drift = 0.12

seed = 1234
np.random.seed(seed)

# -------------------------------------------------------------
# 3. Initialzustand – Frühphase nahe Zentrum (frühes Universum)
# -------------------------------------------------------------
E = np.random.uniform(-0.3, 0.3, N)  # Energien eng um 0
V = np.zeros(N)                      # "Geschwindigkeit" auf der Achse

# -------------------------------------------------------------
# 4. Hilfsfunktion: Kräfte berechnen
# -------------------------------------------------------------
def mcm_forces(E, tau):
    """
    Berechnet die Kräfte auf alle Akteure.

    tau ∈ [0,1]: normierte Zeit
      - tau = 0   → frühe Phase (heiß, wenig Zentrierung)
      - tau = 1   → späte Phase (kühler, mehr Zentrierung)
    """

    N = len(E)
    F = np.zeros_like(E)

    # 1. Zentrumskraft (Selbstreflexion) – wird im Laufe der Zeit stärker
    center_pull = center_min + (center_max - center_min) * tau
    F += -center_pull * E

    # 2. Abstoßung von den Extremen: verhindert Einfrieren bei ±3
    edge_mask = np.abs(E) > edge_threshold
    F[edge_mask] += -edge_k * (E[edge_mask] - np.sign(E[edge_mask]) * edge_threshold)

    # 3. Lokale Kopplung im Energieraum:
    #    Akteure mit ähnlicher Energie ziehen sich leicht an
    #    -> Clusterbildung entlang der Achse
    #    (Gauß-Kern über Energiedifferenz)
    E_mat = E[None, :] - E[:, None]             # Differenzenmatrix
    dist2 = E_mat**2
    sigma = 0.6
    weights = np.exp(-dist2 / (2 * sigma**2))   # Nachbarschaftsgewicht
    # kein Eigen-Influenz:
    np.fill_diagonal(weights, 0.0)
    # Kopplungskraft ~ gewichtete Differenz
    F += coupling * np.sum(weights * E_mat, axis=1)

    # 4. Phasen-Drift als sanfter Expansionsimpuls
    #    tanh(E) ist symmetrisch & saturiert an den Rändern
    F += phase_drift * np.tanh(E)

    return F

# -------------------------------------------------------------
# 5. Zeitentwicklung
# -------------------------------------------------------------
history = []

for t in range(steps):
    tau = t / (steps - 1)  # normierte Zeit [0,1]

    # Kräfte
    F = mcm_forces(E, tau)

    # Zeitabhängiges Rauschen (Cooling)
    noise_t = noise_max - (noise_max - noise_min) * tau

    # Euler-Schritt
    V += F * dt
    V += np.random.normal(scale=noise_t, size=N) * dt
    E += V * dt

    # Begrenzung auf MCM-Bereich [-3, +3]
    E = np.clip(E, -3, 3)

    history.append(E.copy())

history = np.array(history)

# -------------------------------------------------------------
# 6. Visualisierung
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.title("MCM Pure Emergence v2 – Energieverlauf über Zeit (Cooling & symmetrisch)")
plt.imshow(history.T, aspect='auto', cmap='coolwarm',
           extent=[0, steps, -3, 3])
plt.colorbar(label='Energie')
plt.xlabel("Zeit")
plt.ylabel("Energieposition")
plt.show()
