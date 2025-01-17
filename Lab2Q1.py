import torch
num1 = float(input("Enter 1st Number : "))
num2 = float(input("Enter 2nd Number : "))
a = torch.tensor(num1, requires_grad=True)
b = torch.tensor(num2, requires_grad=True)
x = 2 * a + 3 * b
y = 5 * a**2 + 3 * b**3
z = 2 * x + 3 * y
dz_da = 2 * 2 + 3 * (10 * a)
print(f"Manually computed dz/da: {dz_da.item()}")
z.backward()
print(f"PyTorch computed dz/da: {a.grad.item()}")