#include libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")