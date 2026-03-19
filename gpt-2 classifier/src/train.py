"""
Training script for GPT2ForSequenceClassification on 20 Newsgroups dataset.

This script trains a GPT-2 based classifier without using any HuggingFace libraries.
"""

import argparse
import json
import os
import time
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gpt2 import GPT2Config, GPT2ForSequenceClassification
from torch.utils.tensorboard import SummaryWriter


class NewsgroupsDataset(Dataset):
    def __init__(self, filepath, max_length=1024):
        self.samples = []
        self.max_length = max_length
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        token_ids = sample["token_ids"][:self.max_length]  # truncate
        label = sample["label"]

        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        token_ids = token_ids + [0] * padding_length  # 0 = padding token

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            output = model(input_ids)
            preds = output.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    # TODO: implement the training loop for GPT2ForSequenceClassification on the 20 Newsgroups dataset.
    # You can use any techniques or implementations you like.

    # setup tensorboard for logging
    writer = SummaryWriter(log_dir="../runs/classifier")
    
    # set up the model, optimizer, and loss functions
    config = GPT2Config()
    config.num_labels = 20

    model = GPT2ForSequenceClassification(
        config=config,
        lm_bin_path="../checkpoints/gpt2_model.pth"  # pre-trained LM weights
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01) # use adam optimizer (highly reliable, industry standard)
    criterion = nn.CrossEntropyLoss() # loss function


    # make dataloaders
    train_dataset = NewsgroupsDataset("../data/20_newsgroups_train.jsonl", max_length=512)
    val_dataset = NewsgroupsDataset("../data/20_newsgroups_val.jsonl", max_length=512)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) # shuffling helps prevent the model from learning patterns simply based on order of the samples
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


    # lets train for 3 epochs
    num_epochs = 3  # 2-5 epochs is typical for fine-tuning
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} average training loss: {avg_loss:.4f}")

        # validation / running evals, once per epoch loop
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

        # log results on tensorboard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../checkpoints/classifier_model.pth")
    print("Saved classifier checkpoint to ../checkpoints/classifier_model.pth")

    writer.close()