import torch

from ao_lab.models import PredictorMLP


def test_model_shape_and_finite():
    model = PredictorMLP(history=6, n_modes=12, hidden=16)
    x = torch.randn(4, 6, 12)
    y = model(x)
    assert y.shape == (4, 12)
    assert torch.isfinite(y).all()
