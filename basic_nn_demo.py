import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_moons
from regrad import Var
from tools.nn import MLP

data, label = make_moons(n_samples=130, noise=0.15)
label = label * 2 - 1  # make y be -1 or 1
model = MLP(2, [16, 16, 1])  # 2-layer neural network

print(model)
print("number of parameters", len(model.parameters()))


def loss(batch_size=None):
    if batch_size is None:
        x, y = data, label
    else:
        shuffle_index = np.random.permutation(data.shape[0])[:batch_size]
        x, y = data[shuffle_index], label[shuffle_index]
    x_input = [list(map(Var, x_row)) for x_row in x]
    output_y = list(map(model, x_input))
    loss_value = [(1 + -yi * output_yi).relu() for yi, output_yi in zip(y, output_y)]
    data_loss = sum(loss_value) * (1.0 / len(loss_value))

    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    accuracy = [(yi > 0) == (output_yi.val > 0) for yi, output_yi in zip(y, output_y)]
    return total_loss, sum(accuracy) / len(accuracy)


total_loss, acc = loss()
print(total_loss, acc)

# optimization
for epoch in range(50):

    # forward
    total_loss, acc = loss()

    # backward
    model.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.9 * epoch / 100
    for p in model.parameters():
        p.val -= learning_rate * p.grad

    if epoch % 1 == 0:
        print(f"step {epoch} loss {total_loss.val:.4f}, accuracy {acc * 100:.4f}%")

# visualize decision boundary

h = 0.25
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
x_mesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Var, x_row)) for x_row in x_mesh]
output_y = list(map(model, inputs))
z = np.array([s.val > 0 for s in output_y])
z = z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, z, cmap=cm.Spectral, alpha=0.6)
plt.scatter(data[:, 0], data[:, 1], c=label, s=40, cmap=cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.savefig("doc/moons_mlp.png")
plt.show()
plt.close()
