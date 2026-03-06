import numpy as np
import random

# ---------------------------------------
# MCM Energie-Update
# ---------------------------------------

def relax(E, rate=0.05):
    return E - rate * np.sign(E)

def apply_reaction(E, action):
    """Reaktionen ändern die Energie unterschiedlich."""
    effects = {
        0:  0.0,   # neutral bleiben
        1: -0.6,   # beruhigen
        2: +0.4,   # ermutigen
        3: -1.0,   # klarer Einspruch
        4: +1.0,   # starke Zustimmung
    }
    return E + effects[action]

def reward(E):
    """Belohnung für Nähe zur Mitte."""
    return -abs(E)  # max bei E=0


# ---------------------------------------
# Reinforcement Learning Agent (Q-Learning)
# ---------------------------------------

class EmotionRL:
    def __init__(self):
        self.actions = [0, 1, 2, 3, 4]
        self.q = {}  # Q-Table

    def get_Qs(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(len(self.actions))
        return self.q[state]

    def choose_action(self, state, eps=0.2):
        if random.random() < eps:
            return random.choice(self.actions)
        return np.argmax(self.get_Qs(state))

    def learn(self, state, action, new_state, r, alpha=0.1, gamma=0.9):
        Qs = self.get_Qs(state)
        target = r + gamma * max(self.get_Qs(new_state))
        Qs[action] += alpha * (target - Qs[action])


# ---------------------------------------
# Simulation einer Episode
# ---------------------------------------

def run_episode(agent):
    E = random.uniform(-3, 3)  # zufälliger Start in G1–G4 MCM Bereiche
    trajectory = []

    for t in range(20):
        state = round(E, 1)

        action = agent.choose_action(state)
        E2 = apply_reaction(E, action)
        E2 = relax(E2)

        r = reward(E2)

        new_state = round(E2, 1)
        agent.learn(state, action, new_state, r)

        trajectory.append((E, action, E2, r))

        E = E2

    return trajectory


# ---------------------------------------
# Training
# ---------------------------------------

agent = EmotionRL()

for episode in range(2000):
    run_episode(agent)

print("Training abgeschlossen.")

# Testen: die KI reagiert optimal
test = run_episode(agent)
for step in test[:10]:
    print(step)
