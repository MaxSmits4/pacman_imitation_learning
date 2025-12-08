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

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import trange

from architecture import PacmanNetwork
from data import PacmanDataset


def visualization(num_plots):
    """
    Initialize and manage matplotlib plots for training monitoring.

    Returns:
        setup_plots: Function to initialize plots
        update_plots: Function to update plots during training
    """
    def setup_plots(num_plots):
        """Initialize matplotlib figures for loss and accuracy plots."""
        fig = plt.figure(figsize=(5 * num_plots + 8, 4))

        # Create grid: loss plots on left, accuracy on right
        gs = fig.add_gridspec(
            1, num_plots + 1, width_ratios=[5] * num_plots + [8]
        )

        # Loss subplot axes
        axes = [fig.add_subplot(gs[0, i]) for i in range(num_plots)]

        # Accuracy subplot axis
        ax_acc = fig.add_subplot(gs[0, num_plots])
        ax_acc.set_title("Test Accuracy vs Epoch")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.grid()

        # Bring matplotlib window to front
        fig.canvas.manager.show()
        fig.canvas.flush_events()

        return fig, axes, ax_acc

    def update_plots(
        axes, ax_acc, plot_idx, losses, epoch_numbers,
        test_accuracies, epoch, fig, is_final=False
    ):
        """Update loss and accuracy plots."""
        # Update loss plot
        axes[plot_idx].plot(losses)
        axes[plot_idx].set_title(f"Epoch {epoch}" if not is_final else "Final")
        axes[plot_idx].set_ylabel("Loss")
        axes[plot_idx].set_xlabel("Steps")
        axes[plot_idx].set_xscale("log")
        axes[plot_idx].grid()

        # Update accuracy plot
        ax_acc.clear()
        ax_acc.plot(
            epoch_numbers, test_accuracies, 'b-',
            linewidth=2, marker='o'
        )
        title_suffix = " (Final)" if is_final else ""
        ax_acc.set_title(f"Test Accuracy vs Epoch{title_suffix}")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.grid()
        ax_acc.set_ylim([0, 100])

        # Refresh display
        plt.figure(fig.number)
        plt.tight_layout()
        if is_final:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.pause(0.01)

    return setup_plots, update_plots


def evaluate_acc_model(model, loader, device):
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
    #   1) 80/20 split via random_split
    #   2) Per-epoch evaluation on the test set.
    #   3) Best-model checkpointing based on test accuracy.

    # Hyperparameters 
    batch_size = 256
    epochs = 150  
    show_every = 30  # Graph every 30 epochs
    learning_rate = 8e-4  # LR slightly higher in order to converge faster 

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

    # Optimizer 
    losses = []
    optim = Adam(model.parameters(), lr=learning_rate)

    # Best model tracking
    best_accuracy = 0.0
    best_epoch = 0
    save_file = "pacman_model.pth"

    # Test accuracy tracking
    test_accuracies = []
    epoch_numbers = []

    # Setup real-time plotting (loss & accuracy in one window)
    num_plots = (epochs // show_every) + 1
    setup_plots, update_plots = visualization(num_plots)
    fig, axes, ax_acc = setup_plots(num_plots)
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
        current_accuracy = evaluate_acc_model(model, loader_test, device)

        # Track accuracy for plotting
        test_accuracies.append(current_accuracy * 100)  # Convert to percentage
        epoch_numbers.append(i)

        # Save best model
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_epoch = i
            torch.save(model.state_dict(), save_file)
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Best Acc": f"{best_accuracy:.2%}",
                "Epoch": best_epoch
            })

        model.train()  # Back to training mode

        # Periodic visualization
        # like MNIST template: lightweight live loss plots
        # to verify convergence
        if i % show_every == 0:
            update_plots(
                axes, ax_acc, plot_idx, losses, epoch_numbers,
                test_accuracies, i, fig
            )
            plot_idx += 1

    # Final plots
    update_plots(
        axes, ax_acc, -1, losses, epoch_numbers,
        test_accuracies, "Final", fig, is_final=True
    )

    # Final evaluation with best model
    print(f"\nBest model saved at epoch {best_epoch} -> accuracy: {best_accuracy:.2%}")

    # Keep graphs open
    plt.show()