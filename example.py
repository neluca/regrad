from regrad import Var

a = Var(-5, req_grad=True)
b = Var(-3, req_grad=True)
c = Var(-2, req_grad=True)

x = (a * b * c).relu()

x.backward()
print(x)
print(a.grad, b.grad, c.grad)

x = a * b * c
x.backward()
print(x)
print(a.grad, b.grad, c.grad)

a.grad = None
b.grad = None
c.grad = None

y = a ** 2 + 4 * b * a + c + 10
y = y.tanh()
y.backward()
print(y.grad, a.grad, b.grad, c.grad)
