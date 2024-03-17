import uuid

import pytest
import torch

from streamvigil.core import AnomalyDetector, ModelPool
from streamvigil.detectors import BasicDetector
from tests.mock import MockAutoEncoder

# Mock
auto_encoder = MockAutoEncoder(input_dim=10, hidden_dim=8, latent_dim=5)
detector = BasicDetector(auto_encoder)


def test_add_model():
    pool = ModelPool(detector, max_model_num=3)
    model_id = pool.add_model()
    assert model_id is not None
    assert isinstance(model_id, uuid.UUID)

    pool.add_model()
    pool.add_model()
    assert len(pool._pool) == 3

    # Exceed max number of models
    with pytest.raises(ValueError):
        pool.add_model()


def test_is_drift():
    pool = ModelPool(detector, reliability_threshold=0.8)
    assert pool.is_drift() is True

    pool._reliability = 0.9
    assert pool.is_drift() is False


def test_get_models():
    pool = ModelPool(detector)
    pool.add_model()
    pool.add_model()
    pool.add_model()
    models = pool.get_models()
    assert len(models) == 3
    assert isinstance(models[0], AnomalyDetector)


def test_similarity():
    pool = ModelPool(detector)
    x = torch.randn(5, 10)
    model_id1 = pool.add_model()
    model_id2 = pool.add_model()
    similarity = pool.similarity(x, model_id1, model_id2)
    assert similarity > 0.0


def test_merge_models():
    pool = ModelPool(detector)
    src_id = pool.add_model()
    dst_id = pool.add_model()
    pool.add_model()

    assert len(pool.get_models()) == 3
    pool._merge_models(src_id, dst_id)
    assert len(pool.get_models()) == 2


def test_find_most_similar_model():
    pool = ModelPool(detector, max_model_num=10)

    x = torch.randn(5, 10)

    target_id = pool.add_model()

    # Error when there is no model other than target
    with pytest.raises(ValueError):
        pool.find_most_similar_model(x, target_id)

    # Add models
    pool.add_model()
    pool.add_model()
    pool.add_model()
    pool.add_model()
    pool.add_model()

    id, sim = pool.find_most_similar_model(x, target_id)
    assert id is not None
    assert id != target_id
    assert sim > 0.0


def test_compress():
    pool = ModelPool(detector, max_model_num=10)

    x = torch.randn(5, 10)

    dst_id = pool.add_model()

    pool.add_model()
    pool.add_model()

    assert pool.compress(x, dst_id)


def test_predict():
    pool = ModelPool(detector)
    pool.add_model()
    pool.add_model()
    pool.add_model()

    x = torch.randn(5, 10)

    scores = pool.predict(x)
    assert scores.dim() == 1
    assert scores.numel() == 5
