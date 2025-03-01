# regrad
**regrad** is an educational implementation of reverse mode automatic differentiation that is **<u>distinct</u>** from Karpathy's [micrograd ](https://github.com/karpathy/micrograd).

### Example usage

```python
from regrad import Var
from tools import draw_to_html

a = Var(-4.0, req_grad=True)
b = Var(2.0, req_grad=True)
c_5 = Var(5)  # const
c = a + b
d = a * b + b ** 3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += c_5 * d + (b - a).relu()
e = c - d
f = e ** 2
g = f / 2.0
g += 10.0 / f
print(f'{g.val:.4f}')   # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}')  # prints 222.1341, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}')  # prints 978.7784, i.e. the numerical value of dg/db

y = a + a ** 2
draw_to_html(y, "computed_graph_pow")
```

![sigmoid](./doc/computed_graph_pow.png)

### Basic sigmoid

```python
from regrad import Var
from tools import draw_to_html


def sigmoid(x: Var) -> Var:
    return 1 / (1 + (-x).exp())


y = sigmoid(Var(0.5, req_grad=True))
draw_to_html(y, "sigmoid")
```

![sigmoid](./doc/computed_graph_sigmoid.png)

### Basic  MLP

```python
from regrad import Var
from tools.nn import MLP
from tools import draw_to_html

model = MLP(2, [3, 1])  # 3-neurons, 1-layer
print("number of parameters", len(model.parameters()))
x = Var(0.5)
y = model([x, x ** 2])
draw_to_html(y, "computed_graph_mlp", "BT")
```

![mlp](./doc/computed_graph_mlp.png)

### Training a neural net

![moons_mlp](./doc/moons_mlp.png)

### Tracing / visualization

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

