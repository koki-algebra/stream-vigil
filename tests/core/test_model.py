import uuid

from streamvigil.core import Model
from tests.mock import MockAutoEncoder

# Mock
auto_encoder = MockAutoEncoder(input_dim=10, hidden_dim=8, latent_dim=5)


def test_model_id():
    model = Model(auto_encoder)

    assert model.model_id is not None
    assert isinstance(model.model_id, uuid.UUID)
