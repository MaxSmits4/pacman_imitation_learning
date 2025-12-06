"""
3.train.py - Training Pipeline for Pacman
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from architecture import PacmanNetwork
from data import PacmanDataset


class Pipeline(nn.Module):
    """
    Complete training pipeline.
    """

    def __init__(self, path: str):
        """
        Initialize the pipeline.

        Args:
            path: Path to pacman_dataset.pkl
        """
        super().__init__()

            # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load dataset
        self.dataset = PacmanDataset(path)

            # Initialize model
        self.model = PacmanNetwork().to(self.device)

            # Loss: CrossEntropyLoss is standard for multi-class classification
            # Recommandé dans le cours pour les problèmes de catégories (5 actions possibles)
            # Formule: L(θ) = -1/N Σ Σ y_ij log f_i(x_j; θ)
        self.criterion = nn.CrossEntropyLoss()

            # [Ref 8] Kingma & Ba (2015) - Adam optimizer
            # [Ref 9] keon/deep-q-learning - lr=1e-3 validated for Atari games
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3  
        )

    def train(self, batch_size: int = 128, epochs: int = 50, val_ratio: float = 0.2):
        """
        Run the complete training.

        Args:
            batch_size: Batch size (default 128) - optimal for hardware and stability
            epochs: Number of epochs (default 50) - empirically validated
            val_ratio: Validation ratio (0.2 = 20%) - standard 80/20 split

        Returns:
            None (saves model to pacman_model.pth)
        """
        print("Training the neural network:") 

            # Split train/validation
        dataset_size = len(self.dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size

        train_set, val_set = random_split(self.dataset, [train_size, val_size])

            # DataLoaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            # Track best model
        best_val_acc = 0.0
        best_state_dict = None
        best_epoch = 0

            # Training loop
        for epoch in range(1, epochs + 1):

                # Training phase
            self.model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total if val_total > 0 else 0.0

                # Display progress
            print(f"Epoch {epoch}/{epochs} - accuracy: {val_acc:.2%}")

                # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = self.model.state_dict()
                best_epoch = epoch

            # Load and save best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        torch.save(self.model.state_dict(), "pacman_model.pth")
        print(f"Best model is saved! ->  {best_val_acc:.4f} (converged at epoch {best_epoch}/{epochs})")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
