import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the training and testing datasets
train_data = [(torch.randn(100), torch.randint(0, 10, (1,))) for _ in range(1000)]
test_data = [(torch.randn(100), torch.randint(0, 10, (1,))) for _ in range(100)]

# Define the data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Train the non-quantized model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.squeeze())
        loss.backward()
        optimizer.step()

# Test the non-quantized model
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target.squeeze()).sum().item()

print("Accuracy of the non-quantized model: %d %%" % (100 * correct / total))

# Quantize the model

torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

# Test the quantized model
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target.squeeze()).sum().item()

print("Accuracy of the quantized model: %d %%" % (100 * correct / total))
