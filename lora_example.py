import torch
import torch.nn as nn
import torch.optim as optim

# 1. Full Model Training Process
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(100):
    inputs = torch.randn(32, 10)
    targets = torch.randint(2, (32, 1)).float()
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{100}, Loss: {loss.item()}')

# 2. LoRA method applied to random subspaces of the model weights
optimizer_lora_random = optim.SGD([model.fc1.weight], lr=0.01)

for epoch in range(50):
    inputs = torch.randn(32, 10)
    targets = torch.randint(2, (32, 1)).float()
    
    optimizer_lora_random.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_lora_random.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Fine-tuning (random) Epoch {epoch+1}/{50}, Loss: {loss.item()}')

# 3. LoRA method applied to strongly activated regions of the model weights
# Define a mask for the first layer weights based on the weight magnitude
with torch.no_grad():
    mask = (model.fc1.weight > model.fc1.weight.mean()).float()

for epoch in range(50):
    inputs = torch.randn(32, 10)
    targets = torch.randint(2, (32, 1)).float()
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    with torch.no_grad():
        model.fc1.weight.grad *= mask
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Fine-tuning (strong activation) Epoch {epoch+1}/{50}, Loss: {loss.item()}')

