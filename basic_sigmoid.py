from regrad import Var
from tools import draw_to_html


def sigmoid(x: Var) -> Var:
    return 1 / (1 + (-x).exp())


y = sigmoid(Var(0.5, req_grad=True))
draw_to_html(y, "sigmoid")

