import torch
import math
num1 = float(input("Enter 1st Number : "))
x = torch.tensor(num1, requires_grad=True)
f = torch.exp(-x**2 - 2*x - torch.sin(x))
f.backward()
grad_pytorch = x.grad.item()
grad_analytical = - (2*x.item() + 2 + math.cos(x.item())) * math.exp(-x.item()**2 - 2*x.item() - math.sin(x.item()))
print(f"Gradient computed by PyTorch: {grad_pytorch}")
print(f"Analytical gradient: {grad_analytical}")