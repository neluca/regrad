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


def test_more_ops():
    a = Var(-4.0, req_grad=True)
    b = Var(2.0, req_grad=True)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    arg, brg, grg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(grg.val - gpt.data.item()) < tol
    # backward pass went well
    assert abs(arg.grad - apt.grad.item()) < tol
    assert abs(brg.grad - bpt.grad.item()) < tol


def test_more_sin_cos():
    a = Var(-4.0, req_grad=True)
    b = Var(2.0, req_grad=True)
    g = a.sin() + b.cos()
    g.backward()
    arg, brg, grg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    g = a.sin() + b.cos()
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(grg.val - gpt.data.item()) < tol
    # backward pass went well
    assert abs(arg.grad - apt.grad.item()) < tol
    assert abs(brg.grad - bpt.grad.item()) < tol


def test_exp_pow():
    a = Var(-4.0, req_grad=True)
    b = Var(2.0, req_grad=True)
    g = a ** 2 + b.exp()
    g.backward()
    arg, brg, grg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    g = a ** 2 + b.exp()
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(grg.val - gpt.data.item()) < tol
    # backward pass went well
    assert abs(arg.grad - apt.grad.item()) < tol
    assert abs(brg.grad - bpt.grad.item()) < tol


def test_tanh():
    a = Var(-0.5, req_grad=True)
    b = Var(2.0, req_grad=True)
    g = a.tanh() + b.tanh()
    g.backward()
    arg, brg, grg = a, b, g

    a = torch.Tensor([-0.5]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    g = a.tanh() + b.tanh()
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(grg.val - gpt.data.item()) < tol
    # backward pass went well
    assert abs(arg.grad - apt.grad.item()) < tol
    assert abs(brg.grad - bpt.grad.item()) < tol


def test_log_sqrt():
    a = Var(3.0, req_grad=True)
    b = Var(2.0, req_grad=True)
    g = a.log() + b.sqrt()
    g.backward()
    arg, brg, grg = a, b, g

    a = torch.Tensor([3.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    g = a.log() + b.sqrt()
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(grg.val - gpt.data.item()) < tol
    # backward pass went well
    assert abs(arg.grad - apt.grad.item()) < tol
    assert abs(brg.grad - bpt.grad.item()) < tol
