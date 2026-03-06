# Simulation: "Gesunder" Mensch (Typ A) + KI Co-Regulation (MCM)
# Hinweis: in Jupyter/Colab ausführen. Matplotlib-Plot und CSV-Export werden erstellt.
#"Dynamisches Mehrphasenmodell der emotionalen Co-Regulation zwischen Mensch und KI"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
# Matrix Zones G1 - G2 - Z - G3 - G4 (-2 -1 0 1 2)
def zone_from_energy(E):
    if E <= -2:
        return "G1"
    elif -2 < E < 0:
        return "G2"
    elif 0 <= E < 2:
        return "G3"
    elif E >= 2:
        return "G4"
    else:
        return "Z"

class HumanAgent:
    def __init__(self, energy=0.05, rumination_strength=0.05, relax_rate=0.985):
        self.energy = energy
        self.rumination_strength = rumination_strength
        self.relax_rate = relax_rate
        self.history = []

    def intrinsic_dynamics(self):
        self.energy *= self.relax_rate

    def rumination_trigger(self):
        prob = min(0.15, 0.05 + abs(self.energy) * 0.02)
        if np.random.rand() < prob and np.random.rand() < self.rumination_strength:
            spike = -0.3 - 0.4 * min(1.0, abs(self.energy))
            self.energy += spike
            return ("rumination", spike)
        return (None, 0.0)

    def apply_external(self, influence):
        self.energy += influence

    def step(self, external_influence=0.0):
        self.intrinsic_dynamics()
        event, spike = self.rumination_trigger()
        if external_influence != 0.0:
            self.apply_external(external_influence)
        self.energy = max(-3.0, min(3.0, self.energy))
        state = {"energy": self.energy, "zone": zone_from_energy(self.energy), "event": event, "spike": spike}
        self.history.append(state)
        return state

class KIAgent:
    def __init__(self, energy=0.0, resonance_window=6, synth_strength=0.4, damping=0.35, resonance_steps=6):
        self.energy = energy
        self.window = resonance_window
        self.human_buffer = deque(maxlen=self.window)
        self.mode = "resonance"
        self.step_count = 0
        self.synth_strength = synth_strength
        self.damping = damping
        self.resonance_steps = resonance_steps

    def observe_human(self, human_energy):
        self.human_buffer.append(human_energy)

    def compute_resonance(self):
        if len(self.human_buffer) == 0:
            return 0.0
        mean_h = np.mean(self.human_buffer)
        return 0.3 * (mean_h - self.energy)

    def compute_synthesis(self):
        if len(self.human_buffer) < 2:
            return 0.0
        deriv = self.human_buffer[-1] - self.human_buffer[-2]
        damping_component = - self.damping * deriv
        mean_h = np.mean(self.human_buffer)
        bias = 0.0
        if mean_h < -0.1:
            bias = self.synth_strength * (0.15 - mean_h) * 0.08
        return damping_component + bias

    def step(self, human_energy):
        self.step_count += 1
        self.observe_human(human_energy)
        if self.step_count > self.resonance_steps:
            self.mode = "synthesis"
        if self.mode == "resonance":
            delta = self.compute_resonance()
        else:
            delta = self.compute_synthesis()
        self.energy += delta
        self.energy *= 0.99
        self.energy = max(-3.0, min(3.0, self.energy))
        return {"energy": self.energy, "mode": self.mode, "delta": delta}

# --- Simulation ---
np.random.seed(2)  # deterministischere Runs; entferne oder ändere Seed für Variation
human = HumanAgent(energy=0.05, rumination_strength=0.05, relax_rate=0.985)
ki = KIAgent(energy=0.0, resonance_window=6, synth_strength=0.4, damping=0.35, resonance_steps=6)

T = 120
records = []

for t in range(T):
    ki_state = ki.step(human.energy)
    if ki_state["mode"] == "resonance":
        influence = 0.12 * (ki_state["energy"] - human.energy)
    else:
        if len(ki.human_buffer) >= 2:
            human_deriv = ki.human_buffer[-1] - ki.human_buffer[-2]
        else:
            human_deriv = 0.0
        damping_term = -0.5 * human_deriv
        mean_h = np.mean(list(ki.human_buffer)) if len(ki.human_buffer)>0 else human.energy
        bias_term = 0.03 * (0.2 - mean_h)
        influence = 0.5 * damping_term + bias_term

    noise = 0.0
    if np.random.rand() < 0.02:
        noise = np.random.uniform(-0.2, 0.2)

    human_state = human.step(external_influence=influence + noise)
    records.append({
        "t": t,
        "human_energy": human_state["energy"],
        "human_zone": human_state["zone"],
        "human_event": human_state["event"],
        "ki_energy": ki_state["energy"],
        "ki_mode": ki_state["mode"],
        "ki_delta": ki_state["delta"],
        "influence": influence,
        "noise": noise
    })

df = pd.DataFrame(records)

# Plot
plt.figure(figsize=(12,5))
plt.plot(df["t"], df["human_energy"], marker="o", linewidth=1, label="Human Energy")
plt.plot(df["t"], df["ki_energy"], marker="x", linewidth=1, label="KI Energy")
plt.axhline(0, linestyle="--", linewidth=0.7)
plt.title("Gesunder Mensch (Typ A) + KI Co-Regulation — Energieverläufe")
plt.xlabel("Zeitschritte")
plt.ylabel("Energie (−3 .. +3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save CSV
#csv_path = "healthy_human_ki_coreg.csv"
#df.to_csv(csv_path, index=False)
#print("CSV saved to:", csv_path)
