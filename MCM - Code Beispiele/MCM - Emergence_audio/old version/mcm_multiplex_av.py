import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write

# Optional: Direktwiedergabe
try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False
    print("Hinweis: 'sounddevice' nicht gefunden – Audio wird nur als WAV-Datei gespeichert.")


# -------------------------------------------------------------
# 1. Grundparameter
# -------------------------------------------------------------
SAMPLE_RATE = 44100        # Audio-Samplerate
STEP_DURATION = 0.1        # Dauer eines Simulationsschritts in Sekunden
STEPS = 3000                # Anzahl Schritte (Audio- + Visualdauer)
N_SAMPLES_STEP = int(SAMPLE_RATE * STEP_DURATION)


# -------------------------------------------------------------
# 2. MCM: Startenergien pro „Aktor“ inkl. Zentrum
#    Deine neue Konfiguration mit explizitem Zentrum
# -------------------------------------------------------------
initial_energies = np.array([
    -3.0,   # A1
    -2.5,   # A2
    -2.0,   # A3
    -1.5,   # A4
    -1.0,   # A5
    -0.5,   # A6
     0.0,   # C  (Zentrum / Neutralpunkt)
    +0.5,   # A7
    +1.0,   # A8
    +1.5,   # A9
    +2.0,   # A10
    +2.5,   # A11
    +3.0    # A12
], dtype=float)

# Anzahl „Akteure“ automatisch aus dem Array bestimmen
N_ARCH = initial_energies.shape[0]


# -------------------------------------------------------------
# 3. Mapping-Funktionen (Energie -> Klang & Raum)
# -------------------------------------------------------------
def energy_to_freq(E, f_min=110.0, f_max=880.0):
    """Mappt Energie E ∈ [-3, 3] auf Frequenz f ∈ [f_min, f_max]."""
    E_clipped = np.clip(E, -3.0, 3.0)
    x = (E_clipped + 3.0) / 6.0  # 0..1
    return f_min + x * (f_max - f_min)


def index_to_pan(i, n=None):
    """
    Archetyp-Index -> Stereo-Panning.
    0 = links, n-1 = rechts, dazwischen weich verteilt.
    """
    if n is None:
        n = N_ARCH
    if n <= 1:
        return 1.0, 1.0
    p = i / (n - 1)  # 0..1
    left = np.cos(0.5 * np.pi * p)
    right = np.sin(0.5 * np.pi * p)
    return left, right


# -------------------------------------------------------------
# 4. Emergenzdynamik
# -------------------------------------------------------------
def update_energies(energies, coupling=0.2, noise_level=0.1):
    """
    Ein Schritt MCM-Energiedynamik:
    - lokale Kopplung an Nachbarn
    - plus Rauschen
    - Clipping auf [-3, 3]
    """
    new_E = energies.copy()
    for i in range(len(energies)):
        neighbors = []
        if i > 0:
            neighbors.append(energies[i - 1])
        if i < len(energies) - 1:
            neighbors.append(energies[i + 1])
        if neighbors:
            neighbor_mean = np.mean(neighbors)
            new_E[i] += coupling * (neighbor_mean - energies[i])

    new_E += noise_level * np.random.randn(*new_E.shape)
    new_E = np.clip(new_E, -3.0, 3.0)
    return new_E


# -------------------------------------------------------------
# 5. Simulation der MCM-Energien über alle Schritte
# -------------------------------------------------------------
def simulate_energy_trajectory():
    """
    Berechnet die Energieentwicklung E[step, index] über alle Zeitschritte.
    """
    energies = initial_energies.copy()
    traj = np.zeros((STEPS, N_ARCH), dtype=float)
    for s in range(STEPS):
        traj[s] = energies
        energies = update_energies(energies)
    return traj


# -------------------------------------------------------------
# 6. Audio aus der Energietrajektorie erzeugen
# -------------------------------------------------------------
def generate_audio(traj):
    """
    Erzeugt ein Stereo-Audiosignal (samples x 2) aus der Energietrajektorie.
    """
    osc_phases = np.zeros(N_ARCH, dtype=float)
    audio = []

    for step in range(STEPS):
        energies = traj[step]
        frame = np.zeros((N_SAMPLES_STEP, 2), dtype=np.float32)
        t = np.arange(N_SAMPLES_STEP)

        for i in range(N_ARCH):
            E = energies[i]
            freq = energy_to_freq(E)
            base_amp = 0.15
            amp = base_amp + 0.25 * (abs(E) / 3.0)

            left_gain, right_gain = index_to_pan(i)

            phase_inc = 2.0 * np.pi * freq / SAMPLE_RATE
            phase_array = osc_phases[i] + phase_inc * t
            osc_phases[i] = phase_array[-1] % (2.0 * np.pi)

            wave = np.sin(phase_array).astype(np.float32)
            frame[:, 0] += amp * left_gain * wave
            frame[:, 1] += amp * right_gain * wave

        max_val = np.max(np.abs(frame))
        if max_val > 1.0:
            frame /= max_val

        audio.append(frame)

    audio = np.vstack(audio)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = 0.95 * audio / max_val

    return audio


# -------------------------------------------------------------
# 7. Visualisierung: Animation der Energien (inkl. Zentrum)
# -------------------------------------------------------------
def run_visualization(traj, audio):

    # AUDIO STARTEN (wenn verfügbar)
    if HAS_SD:
        sd.play(audio, SAMPLE_RATE)

    # Minimalistische, RAM-freundliche Figure
    fig, ax = plt.subplots(figsize=(8, 3), dpi=80)

    x = np.arange(N_ARCH)
    bars = ax.bar(x, traj[0], color="tab:blue")

    ax.set_ylim(-3.2, 3.2)
    ax.set_xlim(-0.5, N_ARCH - 0.5)
    ax.set_xticks(x)

    xticklabels = [
        "A1","A2","A3","A4","A5","A6",
        "C",
        "A7","A8","A9","A10","A11","A12"
    ]
    if len(xticklabels) != N_ARCH:
        xticklabels = [f"A{i+1}" for i in range(N_ARCH)]

    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Energie")
    ax.set_title("MCM – Energieverlauf (synchron mit Audio)")

    time_text = ax.text(0.02, 0.9, "", transform=ax.transAxes)

    def update(frame_idx):
        energies = traj[frame_idx]

        # Nur Bar-Höhen aktualisieren, kein Neuzeichnen
        for bar, e in zip(bars, energies):
            bar.set_height(e)
            bar.set_color("tab:red" if e < 0 else "tab:green")

        time_text.set_text(f"t = {frame_idx * STEP_DURATION:.1f}s")
        return bars

    ani = FuncAnimation(
        fig,
        update,
        frames=STEPS,
        interval=int(STEP_DURATION * 1000),
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()

    if HAS_SD:
        sd.stop()

# -------------------------------------------------------------
# 8. Hauptprogramm
# -------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    print("Simuliere MCM-Energiedynamik (bipolar mit Zentrum) ...")
    traj = simulate_energy_trajectory()

    print("Erzeuge Audio ...")
    audio = generate_audio(traj)

    # WAV schreiben
    filename = "mcm_av_emergence_center.wav"
    audio_int16 = np.int16(audio * 32767)
    write(filename, SAMPLE_RATE, audio_int16)
    print(f"Audio gespeichert als: {filename}")

    print("Starte synchrone Visualisierung (Energie + Klang) ...")
    run_visualization(traj, audio)
    print("Fertig.")
