import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
import unicodedata
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import zipfile
import requests

# Step 1: Download and unzip the data (if not already done)
# Download the data if you haven't done so already
url = "https://download.pytorch.org/tutorial/data.zip"
response = requests.get(url)
with open("data.zip", "wb") as file:
    file.write(response.content)

# Unzip the data
with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Step 2: Prepare the dataset
# Reading the data from the extracted files
language_names = {}
data_folder = "/home/student/Desktop/220962123/lab 8/data/data/names"
print(f"Full path to the data folder: {os.path.abspath(data_folder)}")

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        language = filename.split('.')[0]
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as file:
            names = file.read().splitlines()
            language_names[language] = names

# Step 3: Preprocess the data (convert to lowercase, remove non-ASCII characters, etc.)
all_languages = list(language_names.keys())
language_to_idx = {language: idx for idx, language in enumerate(all_languages)}
idx_to_language = {idx: language for language, idx in language_to_idx.items()}

# Define the alphabet (all possible characters in the names)
alphabet = string.ascii_lowercase + " '"
n_letters = len(alphabet)


# Helper function to convert a name to a tensor of letter indices
def name_to_tensor(name):
    name = unicodedata.normalize("NFD", name)  # Normalize characters
    name = name.lower()
    tensor = torch.zeros(len(name), 1, n_letters)
    for li, letter in enumerate(name):
        tensor[li][0][alphabet.find(letter)] = 1
    return tensor


# Prepare the dataset
class NameDataset(Dataset):
    def __init__(self, language_names, language_to_idx):
        self.data = []
        for language, names in language_names.items():
            for name in names:
                self.data.append((name_to_tensor(name), language_to_idx[language]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name_tensor, label = self.data[idx]
        return name_tensor, label


# Create the dataset and data loader
dataset = NameDataset(language_names, language_to_idx)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# Step 4: Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[-1])  # Only use the last output of the RNN
        return out


# Step 5: Initialize the model, criterion, and optimizer
hidden_size = 128
output_size = len(language_names)
model = RNNModel(n_letters, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the Model
epochs = 500
for epoch in range(epochs):
    total_loss = 0
    for i, (name_tensor, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(name_tensor)
        loss = criterion(output, torch.tensor([label]))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')


# Step 7: Evaluate the Model
# Function to predict the language of a name
def predict_language(name):
    name_tensor = name_to_tensor(name)
    with torch.no_grad():
        output = model(name_tensor)
        _, predicted = torch.max(output, 1)
    return idx_to_language[predicted.item()]


# Test the model with a few examples
print("Predicted language for 'jackson':", predict_language("jackson"))
print("Predicted language for 'duarte':", predict_language("duarte"))
print("Predicted language for 'kimura':", predict_language("kimura"))
