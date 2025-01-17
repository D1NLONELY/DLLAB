import torch
num1 = float(input("Enter Value of X : "))
x = torch.tensor(num1, requires_grad=True)
y = 8*x**4 + 3*x**3 + 7*x**2 + 6*x + 3
y.backward()
grad_pytorch = x.grad.item()
grad_analytical = 32*x.item()**3 + 9*x.item()**2 + 14*x.item() + 6
print(f"Gradient computed by PyTorch: {grad_pytorch}")
print(f"Analytical gradient: {grad_analytical}")