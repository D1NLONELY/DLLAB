import torch
import torch.nn as nn
import torch.nn.functional as F
image = torch.rand(1, 1, 6, 6)
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
output_conv2d = conv_layer(image)
print("Output of Conv2d with out_channels=3:")
print(output_conv2d.shape)
kernel = torch.rand(3, 1, 3, 3)
output_func_conv2d = F.conv2d(image, kernel, stride=1, padding=0)
print("Output of functional conv2d with out_channels=3:")
print(output_func_conv2d.shape)
