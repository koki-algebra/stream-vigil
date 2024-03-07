import uuid

import pytest
import torch

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


def test_reliability_setter_valid(model: Model):
    model.reliability = 0.8
    assert model.reliability == 0.8


def test_reliability_setter_invalid(model: Model):
    with pytest.raises(ValueError):
        model.reliability = -0.5
    with pytest.raises(ValueError):
        model.reliability = 1.5


def test_predict(model: Model, mock_auto_encoder):
    # Mock AutoEncoder.forward() method
    input_data = torch.randn(10, 5)
    expected_output = torch.randn(10, 5)
    mock_auto_encoder.forward.return_value = expected_output

    # Test predict method
    output = model.predict(input_data)
    assert torch.allclose(output, expected_output)
    mock_auto_encoder.forward.assert_called_once_with(input_data)
