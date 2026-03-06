import random
import matplotlib.pyplot as plt

# --- Definition der Zonen mit klaren, nicht überlappenden Energiebereichen ---
MCM_ZONES = {
    'G1': {'energy': [-3, -2], 'labels': ['Wut', 'Misstrauen', 'Rückzug'], 'phase': 4, 'color':'#FF4C4C'},
    'G2': {'energy': [-2, -1], 'labels': ['Zweifel', 'Reflexion', 'Diskussion'], 'phase': 3, 'color':'#FFAA4C'},
    'Zentrum': {'energy': [-1, 1], 'labels': ['Selbstreflexion'], 'phase': 5, 'color':'#FFFF4C'},
    'G3': {'energy': [1, 2], 'labels': ['Belehrung', 'Zuwendung', 'Analyse'], 'phase': 2, 'color':'#4CFFAA'},
    'G4': {'energy': [2, 3], 'labels': ['Rationalität', 'Offenheit', 'Neutralität'], 'phase': 1, 'color':'#4C4CFF'}
}

# --- KI-Agent ---
class EmotionAgent:
    def __init__(self):
        self.energy = 0
        self.zone = 'Zentrum'
        self.phase = 5
        self.label = 'Selbstreflexion'
        self.energy_history = []

    def self_regulate(self):
        """Selbstregulation: Extremwerte werden Richtung Zentrum ausgeglichen"""
        if self.energy < -2:
            self.energy += 0.5
        elif self.energy > 2:
            self.energy -= 0.5

    def update(self, stimulus):
        """Aktualisiert den emotionalen Zustand basierend auf Stimulus + Selbstregulation"""
        self.energy += stimulus
        self.energy = max(-3, min(3, self.energy))

        # Selbstregulation
        self.self_regulate()

        # Zone und Label bestimmen
        for zone, data in MCM_ZONES.items():
            if data['energy'][0] <= self.energy <= data['energy'][-1]:
                self.zone = zone
                self.phase = data['phase']
                weights = [1.0 / self.phase] * len(data['labels'])
                self.label = random.choices(data['labels'], weights=weights)[0]
                break

        self.energy_history.append(self.energy)

# --- Simulation ---
agent = EmotionAgent()
stimuli_sequence = [0, -1, -0.5, 1, 0.7, -1, 1.5, -0.8, 0.3, -0.7, 1]  # externe Inputs !! Demo sequenz

for stimulus in stimuli_sequence:
    agent.update(stimulus)

# --- Visualisierung ---
plt.figure(figsize=(12,6))

# Zonen als farbige Hintergrundbereiche
for zone, data in MCM_ZONES.items():
    plt.axhspan(data['energy'][0], data['energy'][-1], color=data['color'], alpha=0.3, label=zone)

# Energieverlauf des Agenten als Linie mit Markern
plt.plot(agent.energy_history, marker='o', color='black', label='Agent Energie')

# Optional: Labels über jedem Punkt anzeigen
for i, label in enumerate(agent.energy_history):
    plt.text(i, agent.energy_history[i]+0.05, f'{agent.energy_history[i]:.1f}', fontsize=8, ha='center')

plt.title('Adaptive emotionale Simulation mit Selbstregulation (MCM)')
plt.xlabel('Zeit / Simulationsschritt')
plt.ylabel('Energielevel')
plt.ylim(-3.2, 3.2)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
