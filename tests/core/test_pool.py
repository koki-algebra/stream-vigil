import uuid

import pytest

from streamvigil.core import Model, ModelPool
from tests.mock import MockAutoEncoder

# Mock
auto_encoder = MockAutoEncoder(input_dim=10, hidden_dim=8, latent_dim=5)


def test_add_model():
    pool = ModelPool(auto_encoder, max_model_num=3)
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
    pool = ModelPool(auto_encoder, reliability_threshold=0.8)
    assert pool.is_drift() is True

    pool._reliability = 0.9
    assert pool.is_drift() is False


def test_get_models():
    pool = ModelPool(auto_encoder)
    pool.add_model()
    pool.add_model()
    pool.add_model()
    models = pool.get_models()
    assert len(models) == 3
    assert isinstance(models[0], Model)
