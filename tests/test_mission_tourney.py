import pytest

torch = pytest.importorskip("torch")

from tools.nn.general.mission_tourney import variant_advantage  # noqa: E402


def test_advantage_is_value_less_mean_over_variants():
    v = torch.tensor([[0.0, 0.3, -0.1, 0.2], [0.0, 0.0, 0.0, 0.0]])
    a = variant_advantage(v)
    assert torch.allclose(a[0], torch.tensor([-0.1, 0.2, -0.2, 0.1]))
    assert torch.allclose(a.sum(1), torch.zeros(2), atol=1e-6)   # the baseline is the row's mean: they sum to 0
    assert torch.equal(a[1], torch.zeros(4))                     # equal variants: no signal
