import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import random
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
BATCH_SIZE = 128
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1

# Image transformations
transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                ])

# Get datasets and convert to dataloaders
train_loader = DataLoader(
    datasets.CIFAR10(root='data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(
    datasets.CIFAR10(root='data', train=False, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False)


# Let's draw some of the training data

def plot_grid(dataset, classes, grid_size=3):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = random.randint(0, len(dataset) - 1)
            img = dataset[idx]
            input_tensor = img.unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                img = img * 0.5 + 0.5 # unnormalize image
                npimg = img.cpu().numpy()
                axes[i, j].imshow(np.transpose(npimg, (1, 2, 0)))
                axes[i, j].set_title(f"Class: {cifar_classes[classes[idx]]}", fontsize=10)
                axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

examples = enumerate(test_loader)
cifar_classes = test_loader.dataset.classes
batch_idx, (example_data, example_targets) = next(examples)
plot_grid(example_data, example_targets)


# PART ONE

# TODO Implement a multi-layer perceptron.
# In __init__(), define:
# 1. A linear layer (fc1) mapping in_features to hidden_features.
# 2. A linear layer (fc2) mapping hidden_features to in_features.
# 3. A dropout layer with drop_rate.

# In forward(), process the input x through:
# 1. fc1, followed by ReLU activation and dropout.
# 2. fc2, followed by dropout.

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

# TODO Implement the TransformerEncoderLayer.
# In __init__(), define:
# 1. Two LayerNorm layers (norm1, norm2), each with embed_dim.
# 2. A MultiHeadAttention layer (attn), given embed_dim, num_heads, and drop_rate (batch_first=True).
# 3. An MLP layer (mlp), given embed_dim, mlp_dim, and drop_rate.

# In forward(), process x as follows, ensuring residual connections are added after each sub-layer:
# 1. Normalize x using norm1.
# 2. Apply attn to the normalized x (using it as query, key, and value). Remember to extract the attention output.
# 3. Add the attention output to the original x (residual connection).
# 4. Normalize the result using norm2.
# 5. Apply mlp to the normalized result.
# 6. Add the MLP output to the input of the mlp (residual connection).

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x

class PatchEmbedding(nn.Module):
  def __init__(self, img_size, patch_size, in_channels, embed_dim):
      super().__init__()
      self.patch_size = patch_size
      self.proj = nn.Conv2d(in_channels=in_channels,
                            out_channels=embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size)
      num_patches = (img_size // patch_size) ** 2
      self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
      self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))


  def forward(self, x):
      B = x.size(0)
      x = self.proj(x) # (B, E, H/P, W/P)
      x = x.flatten(2).transpose(1, 2) # (B, N, E)
      cls_token = self.cls_token.expand(B, -1 , -1)
      x = torch.cat((cls_token, x), dim=1)
      x = x + self.pos_embed
      return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

  def forward(self, x):
      x = self.patch_embed(x)
      x = self.encoder(x)
      x = self.norm(x)
      cls_token = x[:, 0]
      return self.head(cls_token)


# PART TWO

# TODO Implement the main training routine
# Return the average loss and classification accuracy

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

            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.squeeze() == target).sum().item()

    return test_loss / len(loader.dataset), correct / len(loader.dataset)


# PART THREE

model = VisionTransformer(
    img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=CHANNELS,
    num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, depth=DEPTH,
    num_heads=NUM_HEADS, mlp_dim=MLP_DIM, drop_rate=DROP_RATE
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()

n_epochs = 20
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
plt.show()
