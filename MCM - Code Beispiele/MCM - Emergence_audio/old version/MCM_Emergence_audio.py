import numpy as np

# Optional: zum Speichern als WAV
from scipy.io.wavfile import write

# Optional: für Direktwiedergabe (pip install sounddevice)
try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False
    print("Hinweis: sounddevice nicht gefunden. Audio wird nur als WAV-Datei gespeichert.")


# -------------------------------------------------------------
# 1. Grundeinstellungen für Audio
# -------------------------------------------------------------
SAMPLE_RATE = 44100      # Abtastrate
DT_AUDIO = 0.1           # Dauer eines Simulationsschritts in Sekunden
STEPS = 1200              # Anzahl Zeitschritte (gesamt ~ 20 Sekunden Audio)
N_SAMPLES_STEP = int(SAMPLE_RATE * DT_AUDIO)

N_ARCH = 12              # A1 ... A12


# -------------------------------------------------------------
# 2. MCM: Archetypen & Energieachse
#    (hier nur als Referenz; Du kannst später pro Archetyp
#     noch Phasen / Eigenschaften ergänzen)
# -------------------------------------------------------------
# Energiebelegung nach deiner Beschreibung (vereinfachte Form):
# G1: -3, -2, -1    (A1, A2, A3)
# G2: -1, 0, +1     (A4, A5, A6)
# G3: +1, +2, +2    (A7, A8, A9)
# G4: +2, +3, +3    (A10, A11, A12)

initial_energies = np.array([
    -3.0,   # A1
    -2.0,   # A2
    -1.0,   # A3
    -1.0,   # A4
     0.0,   # A5
    +1.0,   # A6
    +1.0,   # A7
    +2.0,   # A8
    +2.0,   # A9
    +2.0,   # A10
    +3.0,   # A11
    +3.0    # A12
], dtype=float)

energies = initial_energies.copy()


# -------------------------------------------------------------
# 3. Mapping: Energie → Frequenz, Position → Stereo-Panning
# -------------------------------------------------------------
def energy_to_freq(E, f_min=110.0, f_max=880.0):
    """
    Mappt Energie E ∈ [-3, 3] auf Frequenz im Bereich [f_min, f_max].
    """
    E_clipped = np.clip(E, -3.0, 3.0)
    # Normierung auf [0, 1]
    x = (E_clipped + 3.0) / 6.0
    return f_min + x * (f_max - f_min)


def index_to_pan(i, n=N_ARCH):
    """
    Mappt den Archetyp-Index i (0 ... n-1) auf Stereo-Panning.
    p = 0.0 -> komplett links, p = 1.0 -> komplett rechts.
    Wir nutzen eine sin/cos-Verteilung für weiche Übergänge.
    """
    if n <= 1:
        return 1.0, 1.0

    p = i / (n - 1)  # 0.0 ... 1.0
    left = np.cos(0.5 * np.pi * p)
    right = np.sin(0.5 * np.pi * p)
    return left, right


# -------------------------------------------------------------
# 4. Emergenzdynamik auf der MCM-Achse
# -------------------------------------------------------------
def update_energies(energies, coupling=0.2, noise_level=0.1):
    """
    Ein einfacher emergenter Update-Schritt:
    - jedes A_i koppelt an seine Nachbarn (lokale Mittelwertanpassung)
    - zusätzlich Rauschen
    - Begrenzung auf [-3, 3]
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
            # lokale Kopplung: Richtung Nachbarn
            new_E[i] += coupling * (neighbor_mean - energies[i])

    # Stochastische Fluktuation
    new_E += noise_level * np.random.randn(*new_E.shape)

    # Grenzen der MCM-Achse
    new_E = np.clip(new_E, -3.0, 3.0)
    return new_E


# -------------------------------------------------------------
# 5. Audio-Synthese: pro Schritt ein Klangframe aus allen Archetypen
# -------------------------------------------------------------
def generate_mcm_audio():
    """
    Erzeugt ein Stereo-Audio-Signal (N x 2) aus der MCM-Dynamik.
    """
    global energies

    # Phasen der Oszillatoren pro Archetyp (für kontinuierliche Wellen)
    osc_phases = np.zeros(N_ARCH, dtype=float)

    audio = []

    for step in range(STEPS):
        # 1) Emergenzdynamik auf der Energieachse
        energies = update_energies(energies)

        # 2) Audio-Frame vorbereiten
        frame = np.zeros((N_SAMPLES_STEP, 2), dtype=np.float32)

        # Zeitvektor für diesen Frame (relativ, 0 .. DT_AUDIO)
        # Hier nur für die Phase; der eigentliche Phase-Lauf kommt über osc_phases
        t = np.arange(N_SAMPLES_STEP)

        for i in range(N_ARCH):
            E = energies[i]

            # Frequenz aus Energie
            freq = energy_to_freq(E)

            # Amplitude: hängt von |E| ab, plus kleine Grundlautstärke
            base_amp = 0.15
            amp = base_amp + 0.25 * (abs(E) / 3.0)  # ~0.15 .. 0.4

            # Stereo-Panning
            left_gain, right_gain = index_to_pan(i, N_ARCH)

            # Oszillator-Phase updaten
            phase_inc = 2.0 * np.pi * freq / SAMPLE_RATE
            # lineare Phase über das Frame
            phase_array = osc_phases[i] + phase_inc * t
            osc_phases[i] = phase_array[-1] % (2.0 * np.pi)

            # Grundwelle (Sinus); später kann man hier z.B. Noise / FM / Filter einbauen
            wave = np.sin(phase_array).astype(np.float32)

            # Lautstärke & Panning anwenden
            frame[:, 0] += amp * left_gain * wave   # Left
            frame[:, 1] += amp * right_gain * wave  # Right

        # Softes Clipping im Frame
        max_val = np.max(np.abs(frame))
        if max_val > 1.0:
            frame /= max_val

        audio.append(frame)

    # Alle Frames hintereinander
    audio = np.vstack(audio)

    # Gesamtnormalisierung (Headroom)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = 0.95 * audio / max_val

    return audio


# -------------------------------------------------------------
# 6. Hauptlauf: Audio erzeugen, speichern, optional abspielen
# -------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)  # Reproduzierbarkeit (Emergenz bleibt stochastisch, aber deterministisch)

    print("Erzeuge MCM-Emergenz-Audio ...")
    audio = generate_mcm_audio()

    # Als WAV speichern
    filename = "mcm_emergence.wav"
    # Konvertiere nach int16 für WAV
    audio_int16 = np.int16(audio * 32767)
    write(filename, SAMPLE_RATE, audio_int16)
    print(f"Fertig. Datei gespeichert als: {filename}")

    # Optional direkt abspielen
    if HAS_SD:
        print("Spiele Audio ab ...")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()
        print("Wiedergabe beendet.")
    else:
        print("sounddevice nicht verfügbar – nur Datei wurde erzeugt.")
