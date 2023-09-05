# # manual
# import numpy as np
#
# # f = w * x
# # f = 2 * x ; ground truth
#
# X = np.arange(start=1, stop=5 + 1, step=1, dtype=np.float32)
# Y = (lambda X: 2 * X)(X)
#
# w = 0.0
#
# # model prediction
#
#
# def forward(x):
#     return w * x
#
# # loss = MSE
#
#
# def loss(y, y_hat):
#     return ((y_hat - y)**2).mean()
#
# # gradient for MSE = 1/N * (w*x - y)**2 -> d(MSE)/dw = 1/N * 2*x * (w*x - y)
#
#
# def gradient(x, y, y_hat):
#     return np.dot(2 * x, y_hat - y).mean()
#
#
# print(f'Prediction before training: f(5) = {forward(5):.3f}')
#
# # training
# learning_rate = 0.01
# epochs = 10
#
# for epoch in range(epochs):
#     y_hat = forward(X)
#     l = loss(Y, y_hat)
#
#     # error
#     dw = gradient(X, Y, y_hat)
#
#     # update the weights
#     w -= learning_rate * dw
#
#     if not (epoch + 1) % 2:
#         print(f'Epoch {epoch + 1}: w = {w:.4f}, loss = {l:.4f}')
#
# print(f'Prediction after training: f(5) = {forward(5):.3f}')
#
# print('\n')

# using pytrorch
import torch  # noqa
import numpy as np
from torch import tensor

X = torch.arange(start=1, end=5 + 1, step=1, dtype=torch.float32)
Y = (lambda x: 2 * x)(X)

w = torch.tensor(0, dtype=torch.float32, requires_grad=True)

# model prediction


def forward(x: tensor) -> tensor:
    return w * x

# loss = MSE


def loss(y: tensor, y_hat: tensor) -> tensor:
    return ((y_hat - y)**2).mean()

# gradient for MSE = 1/N * (w*x - y)**2 -> d(MSE)/dw = 1/N * 2*x * (w*x - y)


def gradient(x: tensor, y: tensor, y_hat: tensor) -> tensor:
    return np.dot(2 * x, y_hat - y).mean()


print(f'Prediction before training in pytorch: f(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    y_hat = forward(X)
    l = loss(Y, y_hat)

    # make the gradient available
    l.backward()  # dl/dw

    # gradient
    dw = w.grad

    # we don't want this to be computed in the backward prop later, so
    with torch.no_grad():  # we can also use torch.inference_mode()
        # update the weights
        w -= learning_rate * dw

    w.grad.zero_()

    if not (epoch + 1) % 10:
        print(f'Epoch {epoch + 1}: w = {w:.4f}, loss = {l:.4f}')

print(f'Prediction after training in pytorch: f(5) = {forward(5):.3f}')

print('\n')

# # using pytorch loss and optimizer
# import torch  # noqa
#
# X = torch.arange(start=1, end=5 + 1, step=1, dtype=torch.float32)
# Y = (lambda x: 2 * x)(X)
#
# w = torch.tensor(0, dtype=torch.float32, requires_grad=True)
#
# # model prediction
#
#
# def forward(x: torch.tensor) -> torch.tensor:
#     return w * x
#
#
# # training
# learning_rate = 0.01
# epochs = 10
#
# # loss = MSE
# loss = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(params=[w], lr=learning_rate, momentum=0.3)
#
# print(
#     f'Prediction before training in pytorch with loss and optimizer: f(5) = {forward(5):.3f}')
#
#
# for epoch in range(epochs):
#     y_hat = forward(X)
#     l = loss(Y, y_hat)
#
#     # make the gradient available
#     l.backward()  # dl/dw
#
#     # update the weights
#     optimizer.step()
#
#     optimizer.zero_grad()
#
#     if not (epoch + 1) % 10:
#         print(f'Epoch {epoch + 1}: w = {w:.4f}, loss = {l:.4f}')
#
# print(
#     f'Prediction after training in pytorch with loss and optimizer: f(5) = {forward(5):.3f}')
#
# print('\n')
#
# # using pytorch model
# import torch  # noqa
#
# X = torch.arange(start=1, end=5 + 1, step=1, dtype=torch.float32)
# Y = (lambda x: 2 * x)(X)
#
# # going from [1, 2, 3, 4, 5] -> [[1], [2], [3], [4], [5]]
# X = X.view(X.shape[0], 1)
# Y = Y.view(Y.shape[0], 1)
#
# X_test = torch.tensor([5], dtype=torch.float32)
#
# n_samples, n_features = X.shape
#
# # model = torch.nn.Linear(in_features=1, out_features=1, device=device)
#
#
# class LinearRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim) -> None:
#         super(LinearRegression, self).__init__()
#         self.lin = torch.nn.Linear(
#             in_features=input_dim, out_features=output_dim)
#
#     def forward(self, x):
#         return self.lin(x)
#
#
# model = LinearRegression(input_dim=n_features,
#                          output_dim=n_features)
#
# # model prediction
#
# # training
# learning_rate = 0.01
# epochs = 10
#
# # loss = MSE
# loss = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(
#     params=model.parameters(), lr=learning_rate)
#
# print(
#     f'Prediction before training in pytorch with model: f(5) = {model(X_test).item():.3f}')
#
#
# for epoch in range(epochs):
#     y_hat = model(X)
#     l = loss(Y, y_hat)
#
#     # make the gradient available
#     l.backward()  # dl/dw
#
#     # update the weights
#     optimizer.step()
#
#     optimizer.zero_grad()
#
#     if not (epoch + 1) % 10:
#         [w, b] = model.parameters()
#
#         print(
#             f'Epoch {epoch + 1}: w = {w.item():.4f}, loss = {l:.4f}')
#         # print(
#         #     f'Epoch {epoch + 1}: w = {model.weight.item():.4f}, loss = {l:.4f}')
#
# print(
#     f'Prediction after training in pytorch with model: f(5) = {model(X_test).item():.3f}')
