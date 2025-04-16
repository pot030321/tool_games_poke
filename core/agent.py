import numpy as np

class ClusteredAgent:
    def __init__(self, encoder, vectors, labels, action_map, confidence_threshold=0.90):
        self.encoder = encoder
        self.vectors = vectors
        self.labels = labels
        self.action_map = action_map
        self.threshold = confidence_threshold

    def predict_cluster(self, vec):
        sims = self.vectors @ vec / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(vec))
        best_idx = int(np.argmax(sims))
        confidence = np.max(sims)
        cluster = str(self.labels[best_idx])
        return cluster, confidence

    def get_action(self, cluster):
        return self.action_map.get(cluster, None)

    def should_review(self, cluster, confidence):
        return confidence < self.threshold or cluster not in self.action_map