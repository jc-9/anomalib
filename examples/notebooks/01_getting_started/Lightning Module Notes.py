"""
Original Code from the video : https://lightning.ai/docs/pytorch/stable/data/datamodule.html

"""

# Original Code from the video : https://lightning.ai/docs/pytorch/stable/data/datamodule.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import os

# PyTorch (Implied)
class MNISTClassifier(nn.Module):
    # Typo in method name: _init_
    def _init_(self):
        # Typo: torch.nn.Linear is implicitly used but imports are missing
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # Spacing/case issue: x. View
        x = x.View(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        # Typo/incomplete line: reLunx
        return x


# download datal
if global_rank == 0:
    # Typo: os,getcwd()
    mnist_train = MNIST(os.getcwd(), train=True, download=True)
    mnist_test = MNIST(os.getcwd(), train=False, download=True)

dist.barrier()

# transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# Typo: nnist_train
nnist_train = MNIST(os.getcwd(),train=True, transform=transform)
mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

# split data
# Typo: nnist_train
nnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
# Typo/redundant download: nnist_test
nnist_test = MNIST(os.getcwd(), train=False, download=True)

# build dataloaders
pytorch_model = MNISTClassifier()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=1e-3)


# Function definition for loss
def cross_entropy_loss(logits, labels): return F.nll_loss(logits, labels)


num_epochs = 1
for epoch in range(num_epochs):
    # Indentation issue on next line
    for train_batch in mnist_train:
        x, y = train_batch
        logits = pytorch_model(x)
        # Typo: oss
        loss = cross_entropy_loss(logits, y)
        # Typo: orint
        print('train loss: ', loss.item())
        # Indentation issues
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

with torch.no_grad():
    val_loss = []
    # Indentation issue
    for val_batch in mnist_val:
        x, y = val_batch
        logits = pytorch_model(x)
        val_loss.append(cross_entropy_loss(logits, y).item())


""" GEMINI UPDATED CODE https://gemini.google.com/app/5ae6e1d4e0106751
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import os

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3  # Increased from 1 for a more meaningful run
# Set dummy variables for distributed environment (defaulting to single-process)
global_rank = 0
dist = None  # Replace with torch.distributed if running distributedly


# ----------------------------------------------------------------------

## MNIST Classifier Model

class MNISTClassifier(nn.Module):
    # Corrected method name: __init__
    def __init__(self):
        super().__init__()
        # Input: 28*28 = 784 features (flattened image)
        self.layer_1 = nn.Linear(28 * 28, 128)
        # Output: 10 classes (digits 0-9)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten the image tensor: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)

        # Linear layer 1 -> ReLU activation
        x = F.relu(self.layer_1(x))

        # Linear layer 2 (logits)
        x = self.layer_2(x)

        # Return log-probabilities (required for F.nll_loss)
        return F.log_softmax(x, dim=1)


# ----------------------------------------------------------------------

if __name__ == '__main__':

    ## 1. Data Preparation and Preprocessing

    # Only download if necessary and handle distributed synchronization
    if global_rank == 0:
        # Download the data
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)
        # if dist: dist.barrier()

    # Define standard MNIST data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize with standard MNIST mean and std
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the full training data with transforms
    mnist_train_full = MNIST(os.getcwd(), train=True, transform=transform)

    # Split the full training set into training and validation sets
    train_size = 55000
    val_size = 5000
    mnist_train, mnist_val = random_split(mnist_train_full, [train_size, val_size])

    # Load the test set (can be used for final evaluation)
    mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    # Build DataLoaders for batching and shuffling
    train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------------------------

    ## 2. Training Setup

    # Instantiate model
    pytorch_model = MNISTClassifier()

    # Define optimizer (Adam)
    optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=LEARNING_RATE)


    # Define loss function (Negative Log-Likelihood Loss)
    def cross_entropy_loss(logits, labels):
        return F.nll_loss(logits, labels)


    # ----------------------------------------------------------------------

    ## 3. Training and Validation Loop

    print("\n--- Starting Training ---")

    for epoch in range(NUM_EPOCHS):

        # --- Training Phase ---
        pytorch_model.train()  # Set model to training mode
        train_running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):

            # Forward pass: get logits (log-probs)
            logits = pytorch_model(x)

            # Calculate loss
            loss = cross_entropy_loss(logits, y)
            train_running_loss += loss.item()

            # Zero gradients, backward pass, and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss periodically (adjusted from the original snippet)
            if (batch_idx + 1) % 100 == 0:
                print(
                    f'  Epoch {epoch + 1}/{NUM_EPOCHS} | Batch {batch_idx + 1}/{len(train_loader)} | Train Loss: {loss.item():.4f}')

        avg_train_loss = train_running_loss / len(train_loader)

        # --- Validation Phase ---
        pytorch_model.eval()  # Set model to evaluation mode
        val_loss_list = []

        with torch.no_grad():  # Disable gradient calculation
            for x, y in val_loader:
                # Forward pass
                logits = pytorch_model(x)

                # Calculate validation loss
                val_loss = cross_entropy_loss(logits, y).item()
                val_loss_list.append(val_loss)

        avg_val_loss = sum(val_loss_list) / len(val_loss_list)

        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} Summary ---")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}\n")
        """