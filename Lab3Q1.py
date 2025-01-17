import torch
import matplotlib.pyplot as plt
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2], dtype=torch.float32).view(-1, 1)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6], dtype=torch.float32).view(-1, 1)
w = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)
learning_rate = 0.001
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()
epochs = 25
losses = []
for epoch in range(epochs):
    y_pred = x * w + b
    loss = mse_loss(y_pred, y)
    losses.append(loss.item())
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()
