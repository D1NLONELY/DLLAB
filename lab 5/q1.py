import torch
import torch.nn.functional as F
image = torch.rand(6, 6)
print("image =", image)
image = image.unsqueeze(dim=0).unsqueeze(dim=0)
print("image.shape =", image.shape)
kernel = torch.ones(3, 3)
kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
print("kernel =", kernel)
outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("outimage.shape (stride=1, padding=0):", outimage.shape)
outimage_stride2 = F.conv2d(image, kernel, stride=2, padding=0)
print("outimage.shape (stride=2, padding=0):", outimage_stride2.shape)
outimage_padding1 = F.conv2d(image, kernel, stride=1, padding=1)
print("outimage.shape (stride=1, padding=1):", outimage_padding1.shape)
total_params = kernel.numel() + 1
print("Total parameters in the network:", total_params)
