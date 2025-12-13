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
    Convert GameState to 32 normalized features (descriptive, not prescriptive):
    - Position (2): pac_pos_x, pac_pos_y
    - Ghost (7): direction_to_ghost_x, direction_to_ghost_y,
                 ghost_mantt_dist,
                 ghost_in_north, ghost_in_south, ghost_in_east, ghost_in_west
    - Food (9): n_food, avg_food_dist, direction_to_food_x, direction_to_food_y,
                closest_food_dist, food_in_north, food_in_south,
                food_in_east, food_in_west
    - Maze (9): dist_north, dist_south, dist_east, dist_west,
                is_corner, is_corridor, is_open_space,
                dist_to_center_x, dist_to_center_y
    - Legal (5): legal_north, legal_south, legal_east, legal_west, legal_stop

    Returns: Tensor of shape (32) 1D
    """

    # Position (2)
    pac_pos_x, pac_pos_y = state.getPacmanPosition()

    # Ghost info (5)
    ghost_positions = state.getGhostPositions()
    ghost_x, ghost_y = ghost_positions[0]
    ghost_mantt_dist = float(
        abs(pac_pos_x - ghost_x) + abs(pac_pos_y - ghost_y)
    )  # 1
    dist_to_ghost_x = float(ghost_x - pac_pos_x)  # 2
    dist_to_ghost_y = float(ghost_y - pac_pos_y)  # 3

    # Directional ghost presence (4 features: N, S, E, W)
    ghost_in_north = 1.0 if dist_to_ghost_y > 0 else 0.0  # 4
    ghost_in_south = 1.0 if dist_to_ghost_y < 0 else 0.0  # 5
    ghost_in_east = 1.0 if dist_to_ghost_x > 0 else 0.0  # 6
    ghost_in_west = 1.0 if dist_to_ghost_x < 0 else 0.0  # 7

    # Food info (9)
    food = state.getFood()
    food_positions = food.asList()
    n_food = float(len(food_positions))  # 1

    if food_positions:
        # Average distance to all food
        avg_food_dist = sum(abs(pac_pos_x - fx) + abs(pac_pos_y - fy)
                           for fx, fy in food_positions) / len(food_positions)  # 2
        # Find closest food
        distances_to_all_foods = []
        for fx, fy in food_positions:
            distances_to_all_foods.append(abs(pac_pos_x - fx) + abs(pac_pos_y - fy))

        closest_food_index = distances_to_all_foods.index(min(distances_to_all_foods))
        closest_food_x, closest_food_y = food_positions[closest_food_index]
        closest_food_dist = float(distances_to_all_foods[closest_food_index])  # 5
        dist_to_food_x = float(closest_food_x - pac_pos_x)  # 3
        dist_to_food_y = float(closest_food_y - pac_pos_y)  # 4

        # Directional food presence (4 features: N, S, E, W)
        food_in_north = 1.0 if dist_to_food_y > 0 else 0.0  # 6
        food_in_south = 1.0 if dist_to_food_y < 0 else 0.0  # 7
        food_in_east = 1.0 if dist_to_food_x > 0 else 0.0  # 8
        food_in_west = 1.0 if dist_to_food_x < 0 else 0.0  # 9
    else:
        avg_food_dist = 0.0
        closest_food_dist = 0.0
        dist_to_food_x = 0.0
        dist_to_food_y = 0.0
        food_in_north = 0.0
        food_in_south = 0.0
        food_in_east = 0.0
        food_in_west = 0.0

    # Maze geometry (9)
    W = state.getWalls()
    maze_W, maze_H = W.width, W.height

    # Center of maze (middle of the two sides)
    center_x = float(maze_W - 1) / 2.0
    center_y = float(maze_H - 1) / 2.0

    # Distance from center
    dist_to_center_x = float(pac_pos_x - center_x)  # 1 (negative = west of center)
    dist_to_center_y = float(pac_pos_y - center_y)  # 2 (negative = south of center)

    #Dist from walls
    dist_north = dist_until_wall(pac_pos_x, pac_pos_y, 0, 1, W)  # 3
    dist_south = dist_until_wall(pac_pos_x, pac_pos_y, 0, -1, W)  # 4
    dist_east = dist_until_wall(pac_pos_x, pac_pos_y, 1, 0, W)  # 5
    dist_west = dist_until_wall(pac_pos_x, pac_pos_y, -1, 0, W)  # 6

    
    legal_actions = state.getLegalPacmanActions()
    num_legal_moves = len([a for a in legal_actions if a != Directions.STOP])

    # Corner when  1-2 legal moves (excluding STOP)
    is_corner = 1.0 if num_legal_moves <= 2 else 0.0  # 7
    # Lane when exactly 2 opposite directions 
    is_lane = 0.0
    if num_legal_moves == 2:
        has_north_south = (Directions.NORTH in legal_actions and
                          Directions.SOUTH in legal_actions)
        has_east_west = (Directions.EAST in legal_actions and
                        Directions.WEST in legal_actions)
        is_lane = 1.0 if (has_north_south or has_east_west) else 0.0  # 8
    # Open space when 3-4 legal moves
    is_open_space = 1.0 if num_legal_moves >= 3 else 0.0  # 9

    # Legal actions (5)
    legal_flags = []
    for action in ACTIONS:
        if action in legal_actions:
            legal_flags.append(1.0)
        else:
            legal_flags.append(0.0)
            
    # Normalize based on actual maze dimensions
    # Maximum Manhattan distance is from one corner to the opposite corner
    max_manhattan_dist = (maze_W - 1) + (maze_H - 1)

    features = [
        # Position (2) - normalized by maze dimensions
        float(pac_pos_x) / float(maze_W),
        float(pac_pos_y) / float(maze_H),
        # Ghost (7) - descriptive features only
        dist_to_ghost_x / float(maze_W),  # X-direction
        dist_to_ghost_y / float(maze_H),  # Y-direction
        ghost_mantt_dist / float(max_manhattan_dist),  # Manhattan distance
        ghost_in_north,
        ghost_in_south,
        ghost_in_east,
        ghost_in_west,
        # Food (9)
        n_food,  # Number of food pellets remaining (raw count, not normalized)
        avg_food_dist / float(max_manhattan_dist),  # Average distance to all food
        dist_to_food_x / float(maze_W),  # X-direction to closest food
        dist_to_food_y / float(maze_H),  # Y-direction to closest food
        closest_food_dist / float(max_manhattan_dist),  # Distance to closest food
        food_in_north,
        food_in_south,
        food_in_east,
        food_in_west,
        # Maze (9) - wall distances + space type + position relative to center
        dist_to_center_x / (float(maze_W) / 2.0),  # Normalized by half-width
        dist_to_center_y / (float(maze_H) / 2.0),  # Normalized by half-height
        dist_north / float(maze_H),
        dist_south / float(maze_H),
        dist_east / float(maze_W),
        dist_west / float(maze_W),
        is_corner,
        is_lane,
        is_open_space,
        # Legal actions (5)
        *legal_flags,
    ]

    return torch.tensor(features, dtype=torch.float32)

def dist_until_wall(x, y, dx, dy, W):
        distance = 0
        while True:
            x += dx
            y += dy
            if not (0 <= x < W.width and 0 <= y < W.height) or W[x][y]:
                break
            distance += 1
        return float(distance)

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
        return self.inputs[index], torch.tensor(self.labels[index], dtype=torch.long)
