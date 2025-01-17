import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.w * x + self.b
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).view(-1, 1)
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]).view(-1, 1)
dataset = RegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
epochs = 100
losses = []
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs for Linear Regression')
plt.show()
print(f'Final weight (w): {model.w.item()}')
print(f'Final bias (b): {model.b.item()}')
