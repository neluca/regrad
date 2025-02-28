import torch
from regrad import Var


def test_sanity_check():
    x = Var(-4.0, req_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    x_var, y_var = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    x_torch, y_torch = x, y

    assert abs(y_var.val - y_torch.data.item()) < 1e-6
    assert abs(x_var.grad - x_torch.grad.item()) < 1e-6
