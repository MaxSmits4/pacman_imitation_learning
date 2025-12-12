"""
2. architecture.py - Neural Network Architecture for Pacman Imitation Learning
"""

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    MLP for predicting Pacman actions from game state.

    Full architecture: 23 → [128 → 64 → 32] → 5
    Each hidden layer uses: Linear → BatchNorm → GELU → Dropout
    """

    def __init__(
        self,
        input_features=23,
        num_actions=5,
        hidden_dims=[128, 64, 32],
        activation=nn.GELU(),
        dropout=0.3
    ):
        super().__init__()

        layers = []

        # Linear: z = x·W^T + b
        layers.append(nn.Linear(input_features, hidden_dims[0]))

        layers.append(nn.BatchNorm1d(hidden_dims[0]))

        # GELU
        layers.append(activation)

        # Dropout: Randomly zeros 30% of activations (only in TRAIN mode)
        layers.append(nn.Dropout(dropout))

        # HIDDEN LAYERS: 128 → 64 → 32
        for i in range(len(hidden_dims) - 1):

            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

            layers.append(activation)

            layers.append(nn.Dropout(dropout))


        layers.append(nn.Linear(hidden_dims[-1], num_actions))

        self.net = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for training

        Arguments:
            features: [batch_size, 23] game states
            actions: [batch_size] ground truth actions (integers 0-4)

        Returns:
            Scalar loss value
        """
        pred_actions = self.forward(features)
        return self.criterion(pred_actions, actions)
