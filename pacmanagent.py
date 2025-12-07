"""
4. pacmanagent.py - Use trained MLP to predict expert actions during gameplay
"""

import torch
from pacman_module.game import Agent, Directions
from data import state_to_tensor, INDEX_TO_ACTION


class PacmanAgent(Agent):
    """
    Neural network agent for Pacman.
    Predicts actions by passing game state through trained MLP.
    """

    def __init__(self, model):
        """
        Initialize agent with trained model.

        Arguments:
            model: Trained PacmanNetwork instance
        """
        super().__init__()
        self.model = model
        self.model.eval()  # Disable dropout/batchnorm

    def get_action(self, state):
        """
        Predict best legal action for current game state.

        Process:
        1. Convert GameState to 23 features
        2. Forward pass through network
        3. Return highest-scoring legal action

        Arguments:
            state: GameState object

        Returns:
            Direction (NORTH/SOUTH/EAST/WEST/STOP)
        """
        # Get legal moves (walls block some directions)
        legal_actions = state.getLegalPacmanActions()

        # Convert state to tensor (23 features)
        x = state_to_tensor(state).unsqueeze(0)  # Add batch dimension

        # Predict action probabilities
        with torch.no_grad():  # No gradient tracking for inference
            logits = self.model(x)[0]  # Remove batch dimension
            probs = torch.softmax(logits, dim=0)  # Convert to probabilities
            sorted_indices = torch.argsort(probs, descending=True).tolist()  # Sort by confidence

        # Return best legal action
        for i in sorted_indices:
            action = INDEX_TO_ACTION[i]
            if action in legal_actions:
                return action

        return Directions.STOP  # Fallback

