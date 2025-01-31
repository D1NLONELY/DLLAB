import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Define reduced CNN model architecture
class ReducedCNN(nn.Module):
    def __init__(self):
        super(ReducedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Reduced to 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Reduced to 32 channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # Reduced from 128 to 64 neurons
        self.fc2 = nn.Linear(64, 10)  # 10 output classes for MNIST digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply conv1 + relu + maxpool
        x = self.pool(F.relu(self.conv2(x)))  # Apply conv2 + relu + maxpool
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = F.relu(self.fc1(x))  # Fully connected layer 1 + relu
        x = self.fc2(x)  # Output layer (no activation)
        return x


# Define a function to count learnable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Define a function to train and evaluate the model
def train_and_evaluate(model, train_loader, test_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the original and reduced models
original_model = q3().to(device)
reduced_model = ReducedCNN().to(device)

# Calculate learnable parameters for original and reduced models
original_params = count_parameters(original_model)
reduced_params = count_parameters(reduced_model)

# Train and evaluate both models, and record the results
accuracies = []
param_drops = []

# List of models to evaluate
models = [original_model, reduced_model]
for model in models:
    print(f"Training model: {model.__class__.__name__}")
    accuracy = train_and_evaluate(model, train_loader, test_loader, device)
    param_drop = ((original_params - count_parameters(model)) / original_params) * 100
    accuracies.append(accuracy)
    param_drops.append(param_drop)
    print(f"Accuracy: {accuracy}%")
    print(f"Parameter Drop: {param_drop}%")

# Step 4: Plot the percentage drop in parameters vs accuracy
plt.figure(figsize=(8, 6))
plt.plot(param_drops, accuracies, marker='o')
plt.xlabel('Percentage Drop in Parameters (%)')
plt.ylabel('Accuracy (%)')
plt.title('Parameter Drop vs Accuracy')
plt.grid(True)
plt.show()

