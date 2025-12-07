"""
1. data.py - Feature Engineering

Feature extraction and normalization for Pacman imitation learning.
Following best practices from classic supervised learning cf MNIST ref[5] :
- Normalize all features to similar scale ([0,1] range)
- Convert game state to fixed-size feature vector
- Map actions to integer indices for classification

"""

import pickle
import torch
from torch.utils.data import Dataset
from pacman_module.game import Directions


# Action to index mapping (NORTH=0, SOUTH=1, EAST=2, WEST=3, STOP=4)
ACTIONS = [
    Directions.NORTH,
    Directions.SOUTH,
    Directions.EAST,
    Directions.WEST,
    Directions.STOP,
]

ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = {index: action for action, index in ACTION_TO_INDEX.items()}


def state_to_tensor(state: object) -> torch.Tensor:
    """
    Convert GameState to 23 normalized features:
    - Position (2): pac_pos_x, pac_pos_y
    - Ghost (4): direction_to_ghost_x, direction_to_ghost_y,
                 ghost_mantt_dist, ghost_adjacent
    - Food (4): n_food, direction_to_food_x, direction_to_food_y,
                closest_food_dist
    - Maze (5): dist_north, dist_south, dist_east, dist_west, is_corner
    - Danger (3): danger_level, ghost_blocks_food, escape_options
    - Legal (5): legal_north, legal_south, legal_east, legal_west, legal_stop

    Returns: Tensor of shape (23) 1D
    """

    # Position (2)
    pac_pos_x, pac_pos_y = state.getPacmanPosition()

    # Ghost info (4)
    ghost_positions = state.getGhostPositions()
    ghost_x, ghost_y = ghost_positions[0]
    ghost_x, ghost_y = ghost_positions[0]
    ghost_mantt_dist = float(
        abs(pac_pos_x - ghost_x) + abs(pac_pos_y - ghost_y)
    )  # 1
    direction_to_ghost_x = float(ghost_x - pac_pos_x)  # 2
    direction_to_ghost_y = float(ghost_y - pac_pos_y)  # 3
    ghost_adjacent = 1.0 if ghost_mantt_dist == 1.0 else 0.0  # 4

    # Food info (4)
    food = state.getFood()
    food_positions = food.asList()
    n_food = float(len(food_positions))  # 1

    if food_positions:
        # Find closest food
        distances_to_all_foods = [
            abs(pac_pos_x - fx) + abs(pac_pos_y - fy)
            for fx, fy in food_positions
        ]
        closest_food_index = int(
            torch.argmin(torch.tensor(distances_to_all_foods))
        )
        closest_food_x, closest_food_y = food_positions[closest_food_index]
        closest_food_dist = float(distances_to_all_foods[closest_food_index])
        direction_to_food_x = float(closest_food_x - pac_pos_x)  # 2
        direction_to_food_y = float(closest_food_y - pac_pos_y)  # 3
    else:
        closest_food_dist = 0.0  # 4
        direction_to_food_x = 0.0
        direction_to_food_y = 0.0

    # Maze geometry (5)
    W = state.getWalls()
    maze_W, maze_H = W.width, W.height

    def dist_until_wall(x, y, dx, dy):
        """Count free cells in direction (dx, dy) before hitting wall"""
        distance = 0
        while True:
            x += dx
            y += dy
            if not (0 <= x < maze_W and 0 <= y < maze_H) or W[x][y]:
                break
            distance += 1
        return float(distance)

    dist_north = dist_until_wall(pac_pos_x, pac_pos_y, 0, 1)  # 1
    dist_south = dist_until_wall(pac_pos_x, pac_pos_y, 0, -1)  # 2
    dist_east = dist_until_wall(pac_pos_x, pac_pos_y, 1, 0)  # 3
    dist_west = dist_until_wall(pac_pos_x, pac_pos_y, -1, 0)  # 4
    is_corner = (
        1.0 if sum(
            1 for d in [dist_north, dist_south, dist_east, dist_west]
            if d > 0
        ) <= 2 else 0.0
    )  # 5

    # Danger features (3)
    legal_actions = state.getLegalPacmanActions()

    danger_level = 1.0 / ghost_mantt_dist  # 1

    ghost_blocks_food = 0.0  # 2
    if ghost_positions and food_positions:
        same_direction = (
            direction_to_ghost_x * direction_to_food_x > 0
            or direction_to_ghost_y * direction_to_food_y > 0
        )
        if same_direction and ghost_mantt_dist < closest_food_dist:
            ghost_blocks_food = 1.0

    escape_options = sum(
        1.0 for a in legal_actions if a != Directions.STOP
    ) / 4.0  # 3

    # Legal actions (5)
    legal_flags = [
        1.0 if action in legal_actions else 0.0 for action in ACTIONS
    ]

    # In order to Normalize
    MAX_DIST = 20.0
    MAX_COORD = 20.0

    features = [
        # Position (2)
        float(pac_pos_x) / MAX_COORD,
        float(pac_pos_y) / MAX_COORD,
        # Ghost (4)
        direction_to_ghost_x / MAX_DIST,
        direction_to_ghost_y / MAX_DIST,
        ghost_mantt_dist / MAX_DIST,
        ghost_adjacent,
        # Food (4)
        n_food / 50.0,
        direction_to_food_x / MAX_DIST,
        direction_to_food_y / MAX_DIST,
        closest_food_dist / MAX_DIST,
        # Maze (5)
        dist_north / 10.0,
        dist_south / 10.0,
        dist_east / 10.0,
        dist_west / 10.0,
        is_corner,
        # Danger (3)
        danger_level,
        ghost_blocks_food,
        escape_options,
        # Legal actions (5)
        *legal_flags,
    ]

    return torch.tensor(features, dtype=torch.float32)


class PacmanDataset(Dataset):
    """
    Custom PyTorch Dataset for Pacman expert demonstrations.

    Load (state, action) pairs and converts them to tensors for training.
    """

    def __init__(self, path: str):
        super().__init__()

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.labels = []

        for state, action in data:
            feature_tensor = state_to_tensor(state)
            action_index = ACTION_TO_INDEX[action]

            self.inputs.append(feature_tensor)
            self.labels.append(action_index)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.labels[index]
