"Multiagenten-System mit Energiezustandsdynamik und sozialer Kopplung"

import numpy as np
import random

# ---------------------------------------
# MCM Grundlagen 
# ---------------------------------------

def zone_from_energy(E):
    """Ermittelt die Zone basierend auf Energie."""
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

def relax(E, rate=0.1):
    """Rückkehr zum Zentrum."""
    return E - rate * np.sign(E)


# ---------------------------------------
# Agentenklasse
# ---------------------------------------

class Agent:
    def __init__(self, name, energy=0.0, sensitivity=1.0):
        self.name = name
        self.energy = energy
        self.sensitivity = sensitivity
    
    def apply_stimulus(self, value):
        """Externer Stimulus."""
        self.energy += value * self.sensitivity
    
    def social_influence(self, others, coupling=0.05):
        """Soziale Kopplung: Agenten nähern sich emotional an."""
        for other in others:
            if other is not self:
                self.energy += coupling * (other.energy - self.energy)
    
    def update(self):
        """Erholung -> Zentrum"""
        self.energy = relax(self.energy)
    
    def state(self):
        return {
            "name": self.name,
            "energy": round(self.energy, 3),
            "zone": zone_from_energy(self.energy)
        }


# ---------------------------------------
# Simulation
# ---------------------------------------

def simulate(agents, stimuli_sequence, steps=30):
    history = []

    for t in range(steps):

        # 1. Stimuli anwenden (optional)
        if t < len(stimuli_sequence):
            target, value = stimuli_sequence[t]
            target.apply_stimulus(value)

        # 2. Soziale Kopplung (alle beeinflussen sich gegenseitig)
        for agent in agents:
            agent.social_influence(agents)

        # 3. Erholung jedes Agenten
        for agent in agents:
            agent.update()

        # 4. Zustand speichern
        history.append([a.state() for a in agents])

    return history


# ---------------------------------------
# Beispiel: 3 Agenten
# ---------------------------------------

A = Agent("Alice", energy= 0.5, sensitivity=1.2)
B = Agent("Bob",   energy=-1.0, sensitivity=1.0)
C = Agent("Cara",  energy= 2.0, sensitivity=0.8)

agents = [A, B, C]

stimuli = [
    (A, +1.5),   # Alice bekommt ein Kompliment
    (B, -2.0),   # Bob bekommt Kritik
    (C, +1.0),   # Cara erlebt Erfolg
    (A, -1.0),   # Alice wird verunsichert
]

history = simulate(agents, stimuli, steps=20)

# Ausgabe der Entwicklung
for t, snapshot in enumerate(history):
    print(f"\n--- Zeit t={t} ---")
    for s in snapshot:
        print(s)
