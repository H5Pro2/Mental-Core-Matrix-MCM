import numpy as np
from sklearn.cluster import DBSCAN
import random

# --------------------------------------------------
# Wahrnehmung
# --------------------------------------------------

class Perception:

    def encode(self, stimulus):
        """Stimulus → Energieimpuls"""

        mapping = {
            "positive": +1.45,
            "negative": -0.65,
            "threat": -1.75,
            "reward": +1.55,
            "neutral": 0.0
        }

        return mapping.get(stimulus, 0.0)
    
# --------------------------------------------------
# MCM SelfModel
# --------------------------------------------------
class SelfModel:

    def evaluate(self, energy):

        mean_e = float(np.mean(energy))

        if mean_e >= 1.2:
            return "excited"

        if mean_e <= -1.5:
            return "stressed"

        return "stable"
    
# --------------------------------------------------
# MCM Feld
# --------------------------------------------------

class MCMField:

    def __init__(self, n_agents=80):

        self.N = n_agents

        self.energy = np.random.uniform(-0.3,0.3,self.N)
        self.velocity = np.zeros(self.N)

        self.k_center = 0.08
        self.coupling = 0.2
        self.noise = 0.05


    def step(self, impulse):

        # Energieimpuls
        self.energy += impulse

        # Zentrumskraft
        force = -self.k_center * self.energy

        # lokale Kopplung
        diff = self.energy[:,None] - self.energy[None,:]
        weights = np.exp(-(diff**2)/0.5)

        coupling_force = self.coupling * np.sum(weights * diff, axis=1)

        # Dynamik
        self.velocity += force + coupling_force
        self.velocity += np.random.randn(self.N) * self.noise

        self.energy += self.velocity

        self.energy = np.clip(self.energy,-3,3)


# --------------------------------------------------
# Clusterbildung
# --------------------------------------------------

class ClusterDetector:

    def detect(self, energy):

        points = energy.reshape(-1,1)

        db = DBSCAN(eps=0.4,min_samples=4).fit(points)

        labels = db.labels_

        clusters = []

        for c in set(labels):
            if c == -1:
                continue
            clusters.append(points[labels==c])

        return clusters


# --------------------------------------------------
# Gedächtnis
# --------------------------------------------------

class Memory:

    def __init__(self):
        self.memory = []

    def store(self, clusters):

        self.memory = []

        for c in clusters:

            center = np.mean(c)

            self.memory.append({
                "center": float(center),
                "strength": int(len(c))
            })

    def strongest(self):

        if not self.memory:
            return None

        return max(self.memory, key=lambda x: x["strength"])

    def replay_impulse(self, replay_scale=0.25):

        if not self.memory:
            return 0.0

        item = random.choice(self.memory)

        return replay_scale * float(item["center"])


# --------------------------------------------------
# Attraktoren
# --------------------------------------------------

class AttractorSystem:

    def choose(self, memory):

        if memory is None:
            return "neutral"

        e = memory["center"]

        if e < -1.5:
            return "defense"

        if -1.5 <= e < -0.3:
            return "analysis"

        if -0.3 <= e < 1.2:
            return "cooperate"

        if e >= 1.2:
            return "explore"


# --------------------------------------------------
# Handlungssystem
# --------------------------------------------------

class ActionSystem:

    def act(self, attractor):

        actions = {
            "defense": "block / withdraw",
            "analysis": "observe / process",
            "cooperate": "engage socially",
            "explore": "seek novelty",
            "neutral": "idle"
        }

        return actions[attractor]


# --------------------------------------------------
# KI Agent
# --------------------------------------------------

class MCM_AI:

    def __init__(self):

        self.perception = Perception()
        self.field = MCMField()
        self.cluster = ClusterDetector()
        self.memory = Memory()
        self.attractor = AttractorSystem()
        self.action = ActionSystem()


    def step(self, stimulus):

        # Wahrnehmung
        impulse = self.perception.encode(stimulus)

        # Feld Dynamik
        self.field.step(impulse)

        # Clusterbildung
        clusters = self.cluster.detect(self.field.energy)

        # Gedächtnis
        self.memory.store(clusters)

        # Attraktor
        attractor = self.attractor.choose(self.memory.strongest())

        # Handlung
        action = self.action.act(attractor)

        return action


# --------------------------------------------------
# Beispiel
# --------------------------------------------------

ai = MCM_AI()

stimuli = ["neutral","positive","negative","reward","threat"]

for s in stimuli:

    # Feld entspannen
    ai.field.energy *= 0.5
    ai.field.velocity *= 0.3

    action = ai.step(s)

    print(s,"→",action)