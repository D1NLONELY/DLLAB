import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Load the data
df = pd.read_csv("daily_csv.csv")

# Preprocess the data - Drop NA values in the dataset
df = df.dropna()
y = df['Price'].values
x = np.arange(1, len(y) + 1, 1)

# Normalize the input range between 0 and 1
minm = y.min()
maxm = y.max()
y = (y - minm) / (maxm - minm)

# Prepare the sequences for training
Sequence_Length = 10
X = []
Y = []
for i in range(0, len(y) - Sequence_Length):
    list1 = []
    for j in range(i, i + Sequence_Length):
        list1.append(y[j])
    X.append(list1)
    Y.append(y[i + Sequence_Length])

# Convert from list to array
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=False)

# Define the custom Dataset class
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

# Create DataLoader for training data
dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)

# Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _status = self.rnn(x)
        output = output[:, -1, :]  # Take the output of the last time step
        output = self.fc1(torch.relu(output))  # Apply ReLU and then the fully connected layer
        return output

# Instantiate the model
model = RNNModel()

# Define optimizer and loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 1500
for epoch in range(epochs):
    model.train()
    for j, data in enumerate(train_loader):
        optimizer.zero_grad()
        x_batch, y_batch = data
        y_pred = model(x_batch.view(-1, Sequence_Length, 1)).reshape(-1)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    # Print the loss at regular intervals
    if epoch % 50 == 0:
        print(f"{epoch}th iteration: Loss = {loss.item()}")

# Prepare the test set
test_set = NGTimeSeries(x_test, y_test)
test_pred = model(test_set[:][0].view(-1, Sequence_Length, 1)).view(-1)

# Plot the actual vs predicted values for the test set
plt.plot(test_pred.detach().numpy(), label='Predicted')
plt.plot(test_set[:][1].view(-1), label='Actual')
plt.legend()
plt.show()

# Undo normalization
y_actual = y_test * (maxm - minm) + minm
y_pred = test_pred.detach().numpy() * (maxm - minm) + minm

# Plot original data vs predicted values
plt.plot(y_actual)
plt.plot(range(len(y_actual) - len(y_pred), len(y_actual)), y_pred)
plt.show()
