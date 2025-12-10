"""
2. architecture.py - Neural Network Architecture for Pacman Imitation Learning
"""

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    MLP for predicting Pacman actions from game state.
    """

    def __init__(
        self,
        input_features=23, 
        num_actions=5,
        hidden_dims=[256, 128, 64], s
        activation=nn.GELU(),
        dropout=0.3  
    ):
        super().__init__()

        layers = []

        # Input layer: 23 → 256
        layers.append(nn.Linear(input_features, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))  # Normalize activations
        layers.append(activation)
        layers.append(nn.Dropout(dropout))  # Prevent overfitting

        # Hidden layers: 256 → 128 → 64
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))

        # Output layer: 64 → 5 (no activation, raw logits)
        layers.append(nn.Linear(hidden_dims[-1], num_actions))

        self.net = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features → logits"""
        return self.net(x)

    def loss(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for training"""
        pred_actions = self.forward(features)
        return self.criterion(pred_actions, actions)
