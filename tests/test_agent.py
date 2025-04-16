import numpy as np
from core.agent import ClusteredAgent


def test_predict_cluster():
    dummy_vecs = np.array([[1, 0], [0, 1]])
    dummy_labels = np.array([0, 1])
    dummy_action_map = {"0": "tap:100,100", "1": "wait"}

    agent = ClusteredAgent(
        encoder=None,
        vectors=dummy_vecs,
        labels=dummy_labels,
        action_map=dummy_action_map,
        confidence_threshold=0.9
    )

    test_vec = np.array([1, 0])
    cluster, confidence = agent.predict_cluster(test_vec)

    assert cluster == "0"
    assert confidence > 0.99
    print("✅ test_predict_cluster passed")


def test_should_review():
    agent = ClusteredAgent(
        encoder=None,
        vectors=np.array([[1, 0]]),
        labels=np.array([0]),
        action_map={"0": "tap:123,456"},
        confidence_threshold=0.95
    )

    # Case 1: good confidence, known cluster
    assert agent.should_review("0", 0.96) is False

    # Case 2: low confidence
    assert agent.should_review("0", 0.5) is True

    # Case 3: unknown cluster
    assert agent.should_review("999", 0.99) is True

    print("✅ test_should_review passed")


if __name__ == "__main__":
    test_predict_cluster()
    test_should_review()
