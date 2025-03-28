import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Prepare the dataset
text = "hello there, this is a simple character predictor model using an RNN."  # Example text data
chars = list(set(text))  # Get unique characters
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}  # Map characters to indices
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}  # Map indices back to characters

# Hyperparameters
seq_length = 10  # Length of the input sequence
hidden_size = 128  # Size of the hidden layer
batch_size = 1  # We'll predict one character at a time
epochs = 500  # Number of training epochs
learning_rate = 0.001
temperature = 0.7  # Temperature for sampling

# Step 2: Convert the text into sequences of indices
X_data = []
Y_data = []

for i in range(len(text) - seq_length):
    X_data.append([char_to_idx[ch] for ch in text[i:i + seq_length]])
    Y_data.append(char_to_idx[text[i + seq_length]])

X_data = torch.tensor(X_data, dtype=torch.long)
Y_data = torch.tensor(Y_data, dtype=torch.long)


# Step 3: Define the LSTM Model (instead of basic RNN)
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only use the last output of the LSTM
        return out


# Initialize the model
model = CharLSTM(input_size=len(chars), hidden_size=hidden_size, output_size=len(chars))

# Step 4: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 5: Training Loop
for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    # Reshape the input for LSTM
    X_batch = X_data
    X_batch = torch.nn.functional.one_hot(X_batch, num_classes=len(chars)).float()  # One-hot encoding

    # Forward pass
    output = model(X_batch)

    # Compute the loss
    loss = criterion(output, Y_data)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Step 6: Text Generation (Using the trained model with Temperature Sampling)
model.eval()


# Function for sampling with temperature
def sample_with_temperature(logits, temperature=1.0):
    # Apply temperature scaling
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


# Generate text using the trained model with temperature sampling
def predict_next_char(model, start_str, char_to_idx, idx_to_char, seq_length=10, temperature=0.7):
    model.eval()
    input_seq = [char_to_idx[ch] for ch in start_str]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    input_seq = torch.nn.functional.one_hot(input_seq, num_classes=len(char_to_idx)).float()

    predicted_char = ''
    for _ in range(100):  # Predict next 100 characters
        output = model(input_seq)
        predicted_idx = sample_with_temperature(output, temperature)
        predicted_char = idx_to_char[predicted_idx]

        print(predicted_char, end='')

        input_seq = torch.tensor([predicted_idx], dtype=torch.long).unsqueeze(0)
        input_seq = torch.nn.functional.one_hot(input_seq, num_classes=len(char_to_idx)).float()


# Test the model by generating text
predict_next_char(model, "hello", char_to_idx, idx_to_char, temperature=0.7)
