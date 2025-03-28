import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Load and preprocess the data
df = pd.read_csv("daily_csv.csv")
df = df.dropna()
y = df['Price'].values
x = np.arange(1, len(y), 1)
minm = y.min()
maxm = y.max()
y = (y - minm) / (maxm - minm)

# Sequence generation
Sequence_Length = 10
X = []
Y = []
for i in range(0, len(y) - Sequence_Length):
    list1 = y[i:i + Sequence_Length]
    X.append(list1)
    Y.append(y[i + Sequence_Length])

X = np.array(X)
Y = np.array(Y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=False)

# Define custom Dataset
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

# DataLoader for training
train_dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1, :]  # Only get the last time step
        output = self.fc1(output)  # Apply the fully connected layer
        return output

model = LSTMModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 1500
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        inputs = inputs.view(-1, Sequence_Length, 1)  # Reshape inputs for LSTM (batch, seq_len, input_size)
        y_pred = model(inputs).view(-1)  # Predict
        loss = criterion(y_pred, targets)
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        running_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")

# Testing
model.eval()  # Set model to evaluation mode
test_dataset = NGTimeSeries(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

y_pred = []
y_true = []

with torch.no_grad():  # Disable gradient calculation during testing
    for inputs, targets in test_loader:
        inputs = inputs.view(-1, Sequence_Length, 1)
        outputs = model(inputs).view(-1)
        y_pred.append(outputs.numpy())
        y_true.append(targets.numpy())

y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)

# Inverse normalization to get actual price values
y_pred = y_pred * (maxm - minm) + minm
y_true = y_true * (maxm - minm) + minm

# Plot the results
plt.plot(y_pred, label='Predicted')
plt.plot(y_true, label='Original')
plt.legend()
plt.show()

# Plot the full data and predicted values
plt.plot(np.concatenate([y, y_pred[-len(y):]]))  # Assuming the predicted values match the length of y
plt.show()
