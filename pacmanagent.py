"""
4. pacmanagent.py - Use trained MLP to predict expert actions during gameplay
"""

import torch

from pacman_module.game import Agent, Directions

from data import INDEX_TO_ACTION, state_to_tensor


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
        self.action_history = []

    def registerInitialState(self, state):
        """
        Called at the start of each game by the engine.
        Reset the action history so memory does not leak between games.
        """
        self.action_history = []

    def get_action(self, state):
        """
        Predict best legal action for current game state.

        Process:
        1. Convert GameState to 32 features
        2. Forward pass through network
        3. Return highest-scoring legal action

        Arguments:
            state: GameState object

        Returns:
            Direction (NORTH/SOUTH/EAST/WEST/STOP)
        """
        # Get legal moves (walls block some directions)
        legal_actions = state.getLegalPacmanActions()

        # Convert state to tensor (32 features)
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
                # Update memory with chosen action for next state
                self.action_history.append(action)
                self.action_history = self.action_history[-5:]
                return action

        return Directions.STOP  # Fallback
