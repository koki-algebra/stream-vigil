import uuid

import pytest

from streamvigil.core import AutoEncoder, Model


@pytest.fixture
def mock_auto_encoder(mocker) -> AutoEncoder:
    return mocker.MagicMock(spec=AutoEncoder)


@pytest.fixture
def model(mock_auto_encoder) -> Model:
    return Model(mock_auto_encoder)


def test_model_id(model: Model):
    assert model.model_id is not None
    assert isinstance(model.model_id, uuid.UUID)
