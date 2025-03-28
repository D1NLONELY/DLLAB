import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Load and preprocess data
text = open("data.txt", "r").read().lower()  # Load text file and convert to lowercase
chars = sorted(list(set(text)))  # Get unique characters
char_to_int = {c: i for i, c in enumerate(chars)}  # Map characters to integers
int_to_char = {i: c for i, c in enumerate(chars)}  # Map integers to characters

# Prepare dataset
seq_length = 100  # Length of input sequence
dataX = []
dataY = []
for i in range(0, len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
n_vocab = len(chars)

# Convert data to PyTorch tensors
X = torch.tensor(dataX, dtype=torch.long)
y = torch.tensor(dataY, dtype=torch.long)

# Define Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        output, hidden = self.lstm(x)  # Shape: (batch_size, seq_length, hidden_dim)
        output = self.fc(output[:, -1, :])  # Use the last time step's output
        return output, hidden

# Initialize model and hyperparameters
embedding_dim = 128
hidden_dim = 256
model = LSTMModel(n_vocab, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 20
for epoch in range(epochs):
    for inputs, targets in dataloader:
        batch_size = inputs.size(0)  # Get current batch size dynamically

        # Initialize hidden state with current batch size
        hidden_state = (torch.zeros(2, batch_size, hidden_dim),
                        torch.zeros(2, batch_size, hidden_dim))

        optimizer.zero_grad()
        outputs, hidden_state = model(inputs.to(torch.int64), hidden_state)

        loss = criterion(outputs.view(-1, n_vocab), targets.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Predict next characters with temperature scaling and top-k sampling
def predict_next_chars(seed_text, n_chars, temperature=1.0, top_k=5):
    model.eval()
    pattern = [char_to_int[char] for char in seed_text]
    generated_text = seed_text

    # Initialize hidden state for prediction with batch size of 1
    hidden_state = (torch.zeros(2, 1, hidden_dim), torch.zeros(2, 1, hidden_dim))

    with torch.no_grad():
        for _ in range(n_chars):
            inputs = torch.tensor(pattern[-seq_length:], dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_length)
            outputs, hidden_state = model(inputs.to(torch.int64), hidden_state)
            outputs = outputs.squeeze(0) / temperature
            probs = torch.softmax(outputs, dim=-1).numpy()
            top_k_indices = np.argsort(probs)[-top_k:]
            top_k_probs = probs[top_k_indices]
            top_k_probs = top_k_probs / top_k_probs.sum()
            predicted_index = np.random.choice(top_k_indices, p=top_k_probs)
            generated_text += int_to_char[predicted_index]
            pattern.append(predicted_index)

    print(generated_text)

# Example usage
seed_text = "this is a sample text"
predict_next_chars(seed_text[:seq_length], 100, temperature=0.7, top_k=5)
