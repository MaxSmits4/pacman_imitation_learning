"""
2.architecture.py - Optimized Neural Network for Pacman Imitation Learning
"""

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Optimized MLP to imitate the expert Pacman player.

        - Input: 23 features (restored critical features)
        - Output: 5 logits (one per action)
        - Architecture: 23 → 128 → 64 → 32 → 5
        - Activation: GELU (smooth, modern, used in GPT/BERT)
        - Regularization: Dropout(0.15) + BatchNorm

    Design rationale:
        - 23 features includes CRITICAL danger signals (ghost_adjacent, danger_level)
        - 3 hidden layers: sufficient capacity without overfitting
        - 128 neurons first layer (~5.5x input, good ratio)
        - GELU activation: smoother gradients than ReLU, better performance
        - Dropout 0.15: regularization without being too aggressive
        - BatchNorm: training stability
        - ~13,000 parameters: sweet spot for 15k training examples (VERIFIED: larger models overfit!)
  """

    def __init__(self):
        super().__init__()

        # Architecture: Linear → BatchNorm → GELU → Dropout 
        # Layer ordering inspired by Ioffe & Szegedy (2015) [1]
        self.net = nn.Sequential(
            # First hidden layer: 23 → 128
            nn.Linear(23, 128),
            nn.BatchNorm1d(128),  # [1] BatchNorm
            nn.GELU(),            # [2] GELU
            nn.Dropout(0.15),     # [3] Dropout

            # Second hidden layer: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),   
            nn.GELU(),            
            nn.Dropout(0.15),     

            # Third hidden layer: 64 → 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  
            nn.GELU(),           
            nn.Dropout(0.15),    

            # Output layer: 32 → 5 (no activation - CrossEntropyLoss expects raw logits)
            nn.Linear(32, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Tensor of shape (batch_size, 23) - normalized features

        Returns:
            Tensor of shape (batch_size, 5) - logits for each action
        """
        return self.net(x)
