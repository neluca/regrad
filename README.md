# regrad
**regrad** is an educational implementation of reverse mode automatic differentiation.

Here's a basic example:

```python
from regrad import Var

x = Var(-4.0, req_grad=True)
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
print(y, x.grad)
```

Outputs

```
-20.0 46.0
```

