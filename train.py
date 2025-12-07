"""
3. train.py - Trains MLP to predict expert actions from game states

Supervised learning pipeline:
- Train/test split (80/20) for generalization evaluation [5]
- Cross-entropy loss for multi-class classification [5]
- Adam optimizer for efficient convergence [4]
- Batch processing with DataLoader

    References: see full version in Bibliography.txt
    [4] Kingma & Ba (2015) - Adam Optimizer
    [5] LeCun et al. (1998) - MNIST: Train/test split, loss function principles
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import trange

from architecture import PacmanNetwork
from data import PacmanDataset
from plotting import setup_training_plots, update_training_plots


def evaluate_model(model, loader, device):
    """Evaluate model accuracy on dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, actions in loader:
            features = features.to(device)
            actions = actions.to(device, dtype=torch.long)
            outputs = model(features)
            _, predicted = torch.max(outputs, dim=1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()

    model.train()
    return correct / total


if __name__ == "__main__":
    # follows the same supervised-learning as [5]
    #   Dataset -> DataLoader -> model -> Adam -> loss.backward()
    #   -> optim.step() -> loss tracking.
    #
    # Changes : 
    #   1) Custom 80/20 split via random_split (MNIST already provides
    #      train/test sets).
    #   2) Per-epoch evaluation on the test set.
    #   3) Best-model checkpointing based on test accuracy.

    # Hyperparameters
    batch_size = 256  # Samples per batch
    epochs = 32  # Training iterations over full dataset
    show_every = 8  # Plot frequency
    learning_rate = 1e-3  # Adam optimizer learning rate

    # Load expert demonstrations
    dataset = PacmanDataset(path="datasets/pacman_dataset.pkl")
    print(f"Dataset: {len(dataset)} samples")

    # 80/20 train/test split
    dataset_size = len(dataset)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    print(f"Train: {len(train_set)} | Test: {len(test_set)}")

    # Dataloaders for batching
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=batch_size)

    # Initialize network
    model = PacmanNetwork()

    device = "cpu"

    model = model.to(device)

    # Optimizer and loss tracking
    losses = []
    optim = Adam(model.parameters(), lr=learning_rate)

    # Best model tracking
    best_accuracy = 0.0
    save_file = "pacman_model.pth"

    # Test accuracy tracking
    test_accuracies = []
    epoch_numbers = []

    # Setup real-time plotting (loss & accuracy in one window)
    num_plots = (epochs // show_every) + 1
    fig, axes, ax_acc = setup_training_plots(num_plots)
    plot_idx = 0

    # Training loop
    model.train()  # Enable dropout/batchnorm training mode

    for i in (pbar := trange(epochs)):
        for features, actions in loader_train:
            # Move tensors to the selected device [5]
            features = features.to(device)
            actions = actions.to(device, dtype=torch.long)

            # Forward pass + loss
            loss = model.loss(features, actions)

            # Backpropagation [5]
            optim.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optim.step()  # Update weights

            # Track progress [5]
            losses.append(loss.item())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Evaluate on test set every epoch
        current_accuracy = evaluate_model(model, loader_test, device)

        # Track accuracy for plotting
        test_accuracies.append(current_accuracy * 100)  # Convert to percentage
        epoch_numbers.append(i)

        # Save best model
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), save_file)
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Best Acc": f"{best_accuracy:.2%}"
            })

        model.train()  # Back to training mode

        # Periodic visualization
        # (same idea as MNIST template: lightweight live loss plots
        # to verify convergence)
        if i % show_every == 0:
            update_training_plots(
                axes, ax_acc, plot_idx, losses, epoch_numbers,
                test_accuracies, i, fig
            )
            plot_idx += 1

    # Final plots
    update_training_plots(
        axes, ax_acc, -1, losses, epoch_numbers,
        test_accuracies, "Final", fig, is_final=True
    )

    # Final evaluation with best model
    print(f"\nBest model saved -> accuracy: {best_accuracy:.2%}")

    # Keep graphs open
    plt.show()