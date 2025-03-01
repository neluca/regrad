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
print(f'{g.val:.4f}')  # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}')  # prints 222.1341, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}')  # prints 978.7784, i.e. the numerical value of dg/db

y = a + a ** 2
draw_to_html(y, "computed_graph_pow")
