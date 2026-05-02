import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run this once to load the train and test data into a dataloader class
# that will provide the batches.

batch_size_train = 64
batch_size_test = 1000

# Image transformations
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])

# Get datasets and convert to dataloaders
train_loader = torch.utils.data.DataLoader(
  datasets.MNIST(root='data', train=True, download=True, transform=transform), 
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST(root='data', train=False, download=True, transform=transform), 
  batch_size=batch_size_test, shuffle=True)


# PART ONE
# TODO Change this class to implement
# 1. A valid convolution with kernel size 5, 1 input channel and 10 output channels
# 2. A max pooling operation over a 2x2 area
# 3. A Relu
# 4. A valid convolution with kernel size 5, 10 input channels and 20 output channels
# 5. A 2D Dropout layer
# 6. A max pooling operation over a 2x2 area
# 7. A relu
# 8. A flattening operation
# 9. A fully connected layer mapping from (whatever dimensions we are at-- find out using .shape) to 50
# 10. A ReLU
# 11. A fully connected layer mapping from 50 to 10 dimensions
# 12. A softmax function.

# Replace this class which implements a minimal network (which still does okay)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)   # 28 -> conv5 -> 24 -> pool2 -> 12 -> conv5 -> 8 -> pool2 -> 4, so 20*4*4=320
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)              # 1.  valid conv 5x5, 1->10
        x = F.max_pool2d(x, 2)         # 2.  max pool 2x2
        x = F.relu(x)                  # 3.  relu
        x = self.conv2(x)              # 4.  valid conv 5x5, 10->20
        x = self.drop(x)               # 5.  dropout2d
        x = F.max_pool2d(x, 2)         # 6.  max pool 2x2
        x = F.relu(x)                  # 7.  relu
        x = x.flatten(1)               # 8.  flatten
        x = F.relu(self.fc1(x))        # 9-10.  fc 320->50 + relu
        x = self.fc2(x)                # 11. fc 50->10
        x = F.log_softmax(x, dim=1)    # 12. softmax (log for NLLLoss)
        return x


# PART TWO

# TODO Implement the main training routine
# Return a list containing loss at each training iteration
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.squeeze() == target).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
  model.eval()
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for data, target in loader:
      data = data.to(device)
      target = target.to(device)
      output = model(data)

      test_loss += criterion(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += (pred.squeeze() == target).sum().item()

  return test_loss / len(loader.dataset), correct / len(loader.dataset)


# PART THREE

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = F.nll_loss

n_epochs = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, n_epochs + 1):
    tr_loss, tr_acc = train(model, train_loader, optimizer, criterion)
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    test_losses.append(te_loss)
    test_accs.append(te_acc)
    print(f"Epoch {epoch}: train loss={tr_loss:.4f} acc={tr_acc:.4f} | test loss={te_loss:.4f} acc={te_acc:.4f}")


# PART FOUR

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
epochs = range(1, n_epochs + 1)

axes[0].plot(epochs, train_losses, 'o-', label='Train')
axes[0].plot(epochs, test_losses, 'o-', label='Test')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss vs Epoch")
axes[0].legend()

axes[1].plot(epochs, train_accs, 'o-', label='Train')
axes[1].plot(epochs, test_accs, 'o-', label='Test')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy vs Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("hw6_p4_plots.png", dpi=150)
plt.show()

# Both training and test losses decrease over epochs, and both accuracies
# increase -- the model is learning and generalizing well.
#
# We may observe that test loss is slightly LOWER and test accuracy slightly
# HIGHER than training. This is counterintuitive (typically expect train to
# outperform test), but it happens here because Dropout2d is active during
# training (randomly zeroing feature maps, making training harder) while it
# is disabled during evaluation. So the model at test time is strictly more
# capable than during any individual training forward pass, which can push
# test metrics slightly ahead of train metrics.