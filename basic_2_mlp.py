from regrad import Var
from tools.nn import MLP
from tools import draw_to_html

model = MLP(1, [3, 1])  # 3-neurons, 1-layer
print("number of parameters", len(model.parameters()))
y = model([Var(1)])
draw_to_html(y, "computed_graph_mlp", "BT")
