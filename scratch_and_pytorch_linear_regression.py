# manual
import numpy as np

# f = w * x
# f = 2 * x ; ground truth

X = np.arange(start=1, stop=5 + 1, step=1, dtype=np.float32)
Y = (lambda X: 2 * X)(X)

print(X)
print(Y)

w = 0.0

# model prediction


def forward(x):
    return w * x

# loss = MSE


def loss(y, y_hat):
    return ((y_hat - y)**2).mean()

# gradient for MSE = 1/N * (w*x - y)**2 -> d(MSE)/dw = 1/N * 2*x * (w*x - y)


def gradient(x, y, y_hat):
    return np.dot(2 * x, y_hat - y).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
epochs = 20

for epoch in range(epochs):
    y_hat = forward(X)
    l = loss(Y, y_hat)

    # error
    dw = gradient(X, Y, y_hat)

    # update the weights
    w -= learning_rate * dw

    if not (epoch + 1) % 2:
        print(f'Epoch {epoch + 1}: w = {w:.4f}, loss = {l:.4f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')


# using pytrorch
import torch  # noqa

X = torch.arange(start=1, end=5 + 1, step=1, dtype=torch.float32)
Y = (lambda x: 2 * x)(X)

print(X)
print(Y)

w = torch.tensor(0, dtype=torch.float32, requires_grad=True)

# model prediction


def forward(x: torch.tensor) -> torch.tensor:
    return w * x

# loss = MSE


def loss(y: torch.tensor, y_hat: torch.tensor) -> torch.tensor:
    return ((y_hat - y)**2).mean()

# gradient for MSE = 1/N * (w*x - y)**2 -> d(MSE)/dw = 1/N * 2*x * (w*x - y)


def gradient(x: torch.tensor, y: torch.tensor, y_hat: torch.tensor) -> torch.tensor:
    return np.dot(2 * x, y_hat - y).mean()


print(f'Prediction before training in pytorch: f(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    y_hat = forward(X)
    l = loss(Y, y_hat)

    # make the gradient available
    l.backward()  # dl/dw

    # gradient
    dw = w.grad

    # we don't want this to be computed in the backward prop later, so
    with torch.no_grad():  # we can also use torch.inference_mode()
        w -= learning_rate * dw

    w.grad.zero_()

    if not (epoch + 1) % 10:
        print(f'Epoch {epoch + 1}: w = {w:.4f}, loss = {l:.4f}')

print(f'Prediction after training in pytorch: f(5) = {forward(5):.3f}')
