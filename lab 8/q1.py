import torch
import torch.nn as nn
import torch.optim as optim


# Step 1: Prepare Fibonacci Data
def generate_fibonacci_sequence(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence


# Function to create input-output pairs for training
def prepare_data(fib_sequence, seq_length):
    inputs = []
    targets = []
    for i in range(len(fib_sequence) - seq_length):
        inputs.append(fib_sequence[i:i + seq_length])
        targets.append(fib_sequence[i + seq_length])
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


# Generate a Fibonacci sequence
fib_sequence = generate_fibonacci_sequence(50)  # 50 Fibonacci numbers for example
seq_length = 5  # Number of previous Fibonacci numbers to use for prediction

# Prepare input and output data
inputs, targets = prepare_data(fib_sequence, seq_length)


# Step 2: Define the RNN Model
class FibonacciRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(FibonacciRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Only use the last output of the RNN
        return out


# Initialize the model
model = FibonacciRNN(input_size=1, hidden_size=32, output_size=1)

# Step 3: Train the Model
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reshape the inputs for RNN [batch_size, seq_length, input_size]
inputs = inputs.view(-1, seq_length, 1)
targets = targets.view(-1, 1)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(inputs)

    # Compute the loss
    loss = criterion(output, targets)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# Step 4: Test the Model with Manual Input

def get_manual_input():
    # Ask the user to input the Fibonacci sequence
    user_input = input(f"Enter the last {seq_length} Fibonacci numbers separated by spaces: ")
    user_input = list(map(int, user_input.split()))

    # Check if the input length matches the expected sequence length
    if len(user_input) != seq_length:
        print(f"Error: Please enter exactly {seq_length} Fibonacci numbers.")
        return None

    # Convert to tensor and reshape for RNN input
    return torch.tensor(user_input, dtype=torch.float32).view(1, seq_length, 1)


# Allow the user to manually input the sequence for prediction
test_input = get_manual_input()

if test_input is not None:
    # Predict the next Fibonacci number
    model.eval()
    with torch.no_grad():
        predicted = model(test_input)
        print(f"Predicted next Fibonacci number: {predicted.item():.0f}")
