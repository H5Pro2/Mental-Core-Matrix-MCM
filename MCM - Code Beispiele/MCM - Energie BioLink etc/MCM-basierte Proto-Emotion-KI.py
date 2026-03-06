# Visualisierung der Energieverläufe (EEG-ähnlich) einer MCM-Emotions‑Engine
# - Die Simulation verarbeitet eine Stimuli-Sequenz (Ereignisse) und zeigt anschließend
#   den Energieverlauf über die Zeit, inklusive Erholungs-Phase (Decay zurück zum Zentrum).
# - Matplotlib wird verwendet (keine speziellen Farben gesetzt).
# - Ein DataFrame mit den Zeitreihen wird zur Ansicht bereitgestellt.

import math
import random
import pandas as pd
import matplotlib.pyplot as plt

# ---- MCM-Grundstruktur ----
MATRIX = {
    "G1": {"range": [-2,-2,5,-3], "phase": 4, "archetypes": ["Wut", "Misstrauen", "Rückzug"]},
    "G2": {"range": [-0.5,1,1,5], "phase": 3, "archetypes": ["Zweifel", "Reflexion", "Diskussion"]},
    "Z":  {"range": [0],            "phase": 5, "archetypes": ["Selbstregulation"]},
    "G3": {"range": [0.5,1,1.5],  "phase": 2, "archetypes": ["Belehrung", "Zuwendung", "Analyse"]},
    "G4": {"range": [2,2.5,3],    "phase": 1, "archetypes": ["Rationalität", "Offenheit", "Neutralität"]},
}

# ---- Stimulus-Interpreter (vereinfachte NLP-Regeln) ----
def interpret_stimulus(text):
    # very small, rule-based interpreter: returns (valence, arousal)
    txt = text.lower()
    positive = ["gut", "toll", "super", "danke", "schön", "hilfreich", "lob", "anerkennung", "glücklich", "erfolg"]
    negative = ["schlecht", "falsch", "kritik", "angriff", "beleidigt", "schuld", "problem"]
    supportive = ["ich helfe", "wir schaffen", "gemeinsam", "zusammen"]
    conflict = ["streit", "konflikt", "kämpf", "stress", "druck"]

    valence = 0.0
    arousal = 1.0

    if any(w in txt for w in positive):
        valence += 1.0
        arousal += 0.2
    if any(w in txt for w in negative):
        valence -= 1.0
        arousal += 0.5
    if any(w in txt for w in supportive):
        valence += 0.6
        arousal += 0.1
    if any(w in txt for w in conflict):
        valence -= 0.8
        arousal += 0.6

    # if no keywords, small neutral arousal
    return valence, arousal

# ---- Anwendung der Matrix (subjektive Bewertung) ----
def apply_stimulus(state, valence, arousal):
    energy = state["energy"]
    zone = state["zone"]
    # Basisdelta
    delta = valence * arousal

    # zone-modulator: subjektive Interpretation
    if zone == "G1":
        delta *= 1.6
    elif zone == "G2":
        delta *= 1.25
    elif zone == "G3":
        delta *= (1.0 if delta < 0 else 1.4)
    elif zone == "G4":
        delta *= 0.6
    else: # Zentrum
        delta *= 0.35

    new_energy = energy + delta
    new_energy = max(-3.0, min(3.0, new_energy))

    # bestimme zone
    if new_energy <= -2:
        new_zone = "G1"
    elif new_energy <= -1:
        new_zone = "G2"
    elif new_energy >= 2:
        new_zone = "G4"
    elif new_energy >= 1:
        new_zone = "G3"
    else:
        new_zone = "Z"

    new_phase = MATRIX[new_zone]["phase"]
    return {"energy": new_energy, "zone": new_zone, "phase": new_phase, "delta": delta}

# ---- Decay-Funktion (exponentielles Abklingen zur Mitte) ----
def decay_to_center(state, decay_rate=0.92):
    energy = state["energy"] * decay_rate
    # small threshold to snap to zero for visualization clarity
    if abs(energy) < 0.02:
        energy = 0.0
    # update zone/phase
    if energy <= -2:
        zone = "G1"
    elif energy <= -1:
        zone = "G2"
    elif energy >= 2:
        zone = "G4"
    elif energy >= 1:
        zone = "G3"
    else:
        zone = "Z"
    phase = MATRIX[zone]["phase"]
    state["energy"] = energy
    state["zone"] = zone
    state["phase"] = phase
    return state

# ---- EmotionEngine mit Sequenz-Verarbeitung und Visualisierungsdaten ----
class EmotionEngine:
    def __init__(self, decay_rate=0.92, decay_steps_after_stimulus=10):
        self.state = {"energy": 0.0, "zone": "Z", "phase": 5}
        self.decay_rate = decay_rate
        self.decay_steps_after_stimulus = decay_steps_after_stimulus
        self.history = []

    def step_record(self, t_idx, label):
        self.history.append({
            "t": t_idx,
            "energy": self.state["energy"],
            "zone": self.state["zone"],
            "phase": self.state["phase"],
            "label": label
        })

    def process_sequence(self, stimuli_texts):
        t = 0
        self.history = []
        # record initial state
        self.step_record(t, "start")
        for i, text in enumerate(stimuli_texts):
            # interpret stimulus
            valence, arousal = interpret_stimulus(text)
            # apply stimulus (subjective)
            result = apply_stimulus(self.state, valence, arousal)
            self.state.update(result)
            t += 1
            self.step_record(t, f"stimulus_{i+1}: {text}")
            # after stimulus: simulate decay for several timesteps
            for d in range(self.decay_steps_after_stimulus):
                self.state = decay_to_center(self.state, decay_rate=self.decay_rate)
                t += 1
                self.step_record(t, f"decay_{i+1}_{d+1}")
        return pd.DataFrame(self.history)

# ---- Beispiel-Stimuli (du kannst diese Sequenz ersetzen) ----
example_stimuli = [
    "Lob: Gut gemacht!",
    "Kritik: Das ist nicht korrekt.",
    "Du bekommst Unterstützung: Wir schaffen das.",
    "Konflikt/Stress in der Gruppe",
    "Erfolg und Anerkennung",
    "Unsicherheit, unklare Anweisung",
    "Schöne Musik, positive Stimmung"
]

engine = EmotionEngine(decay_rate=0.88, decay_steps_after_stimulus=12)
df = engine.process_sequence(example_stimuli)

# ---- Plot: Energieverlauf (EEG-ähnlich) ----
plt.figure(figsize=(12, 5))
plt.plot(df["t"], df["energy"], marker="o", linewidth=1)
plt.axhline(0, linestyle="--", linewidth=0.7)  # Zentrum
plt.title("MCM EmotionEngine — Energieverlauf (EEG-ähnlich)")
plt.xlabel("Zeitschritte")
plt.ylabel("Energie (−3 .. +3)")
plt.grid(True)
plt.tight_layout()
plt.show()


