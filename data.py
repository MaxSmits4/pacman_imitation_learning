"""
1.data.py - Dataset and Feature Engineering for Pacman
"""

import pickle
import torch
from torch.utils.data import Dataset
from pacman_module.game import Directions


# 1) ACTION <=> INDEX MAPPINGS
ACTIONS = [
    Directions.NORTH,   # Index 0
    Directions.SOUTH,   # Index 1
    Directions.EAST,    # Index 2
    Directions.WEST,    # Index 3
    Directions.STOP,    # Index 4
]

ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = {index: action for action, index in ACTION_TO_INDEX.items()}


def state_to_tensor(state: object) -> torch.Tensor:
    """
    Extract 23 normalized ~[0,1] features from a GameState.

    - Pacman position (2): pac_pos_x, pac_pos_y
    - Ghost info (4): direction_to_ghost_x, direction_to_ghost_y, closest_ghost_dist, ghost_adjacent
    - Food info (4): n_food, direction_to_food_x, direction_to_food_y, closest_food_dist
    - Maze geometry (5): dist_north, dist_south, dist_east, dist_west, is_corner
    - Advanced danger (3): danger_level, ghost_blocks_food, escape_options
    - Legal actions (5): legal_flags (one-hot encoding)

    Args:
        state: GameState object from Pacman engine

    Returns:
        1D Tensor of 23 normalized float32 values
    """

    # 1) PACMAN POSITION (2)
    pac_pos_x, pac_pos_y = state.getPacmanPosition()

    # 2) GHOST INFO (4) 
    ghost_positions = state.getGhostPositions()

    ghost_x, ghost_y = ghost_positions[0]

        # Manhattan dist between pac & ghost
    ghost_mantt_dist = float(abs(pac_pos_x - ghost_x) + abs(pac_pos_y - ghost_y))

    direction_to_ghost_x = float(ghost_x - pac_pos_x)
    direction_to_ghost_y = float(ghost_y - pac_pos_y)


        # If FLAG Adjacent ghost = 1 = immediate danger
    ghost_adjacent = 1.0 if ghost_mantt_dist == 1.0 else 0.0

    # 3) FOOD INFO : Guide Pacman toward the closest food
    food = state.getFood()  # Grid containing all positions with food
    food_positions = food.asList()  # Convert to list of coordinates [(x1,y1), (x2,y2), etc]
    n_food = float(len(food_positions))  # Total number of remaining food

    if food_positions:
            # list of Manhattan dist from each food
        distances_to_all_foods = [abs(pac_pos_x - food_x) + abs(pac_pos_y - food_y)
        for food_x, food_y in food_positions]

            # Find which food is closest
        closest_food_index = int(torch.argmin(torch.tensor(distances_to_all_foods)))
        closest_food_x, closest_food_y = food_positions[closest_food_index]

            # Distance to that closest food
        closest_food_dist = float(distances_to_all_foods[closest_food_index])

            # Direction to food (same logic as for ghost) -  Positive = food to the right/up
        direction_to_food_x = float(closest_food_x - pac_pos_x)
        direction_to_food_y = float(closest_food_y - pac_pos_y)
    else:
            # No more food thus game is won
        closest_food_dist = 0.0
        direction_to_food_x = 0.0
        direction_to_food_y = 0.0

    # 4) MAZE GEOMETRY
        # Goal: Understand the maze structure around Pacman
        # In order to avoid dead ends and detect dangerous corners
    W = state.getWalls()  # Wall grid (True = wall, False = free passage)
    maze_W, maze_H = W.width, W.height  

    def dist_until_wall(pac_start_x, pac_start_y, d_x, d_y):
        """
        Count how many cells Pacman can move in a direction before hitting a wall.

        Args:
            start_x, start_y: Pacman's starting position
            d_x, d_y: Direction to explore 
                for Nord : dx = 0, dy = +1

        Returns:
            Number of free cells before hitting a wall
        """
        distance = 0
        pac_current_x, pac_current_y = pac_start_x, pac_start_y

        while True:
                # Move one cell in the direction
            pac_current_x += d_x 
            pac_current_y += d_y

                # Check if we're outside the maze
            if not (0 <= pac_current_x < maze_W and 0 <= pac_current_y < maze_H):
                break

                # Check if we hit a wall
            if W[pac_current_x][pac_current_y]:
                break

            # Otherwise, increment distance
            distance += 1

        return float(distance)

        # distance to walls in all 4 directions : 4 features here
    dist_north = dist_until_wall(pac_pos_x, pac_pos_y, 0, 1)   # How many cells upward?
    dist_south = dist_until_wall(pac_pos_x, pac_pos_y, 0, -1)
    dist_east = dist_until_wall(pac_pos_x, pac_pos_y, 1, 0)
    dist_west = dist_until_wall(pac_pos_x, pac_pos_y, -1, 0)

        # 5th feature - Corner detection : Count how many directions are free around Pacman
    free_directions = 0
    for dist in [dist_north, dist_south, dist_east, dist_west]:
        if dist > 0:
            free_directions += 1

        # If ≤2 free directions then Pacman is in a corner or narrow corridor 
    is_corner = 1.0 if free_directions <= 2 else 0.0

    # 5) DANGER FEATURES - 3 features
    legal_actions = state.getLegalPacmanActions() 

        #Feature 1
        # if dist=1: danger = 1 high danger & if dist=10: danger = 0.1 low danger
    danger_level = 1.0 / ghost_mantt_dist

        #Feature 2 -  1.0 if true, 0.0 otherwise
    ghost_blocks_food = 0.0
    if ghost_positions and food_positions:
        # signs of direction_to_ghost and direction_to_food are identical
        same_direction = (direction_to_ghost_x * direction_to_food_x > 0 or
                         direction_to_ghost_y * direction_to_food_y > 0)
        # AND ghost is closer than food
        if same_direction and ghost_mantt_dist < closest_food_dist:
            ghost_blocks_food = 1.0

        #Feature 3 - Escape options : How many directions can Pacman take to escape? (excluding STOP)
        # Divided by 4.0 to normalize between 0 and 1
    escape_options = sum(1.0 for action in legal_actions if action != Directions.STOP) / 4.0

    # 6) Convert legal_actions (list of Action objects) to a numerical vector
    legal_flags = [1.0 if action in legal_actions else 0.0 for action in ACTIONS]

    # 7) NORMALIZATION AND VECTOR CONSTRUCTION
        # Neural networks converge better when all features are normalized
        # Without normalization, a feature with value 100
        # would dominate a feature with value 0.5 even if both are important.
    MAX_DIST = 20.0
    MAX_COORD = 20.0

    # Final X vector: 23 normalized features
    features = [
        # 1) PACMAN POSITION (2 features)
        float(pac_pos_x) / MAX_COORD,  
        float(pac_pos_y) / MAX_COORD,  

        # 2) GHOST INFORMATION (4 features)
        direction_to_ghost_x / MAX_DIST,  
        direction_to_ghost_y / MAX_DIST,  
        ghost_mantt_dist / MAX_DIST,      
        ghost_adjacent,                   

        # 3) FOOD INFORMATION (4 features)
        n_food / 50.0,                    
        direction_to_food_x / MAX_DIST,   
        direction_to_food_y / MAX_DIST,   
        closest_food_dist / MAX_DIST,     

        # 4) MAZE GEOMETRY (5 features)
        dist_north / 10.0, 
        dist_south / 10.0,  
        dist_east / 10.0,  
        dist_west / 10.0,   
        is_corner,          

        # 5) DANGER FEATURES (3 features)
        danger_level,       
        ghost_blocks_food,  
        escape_options,    

        # 6) LEGAL ACTIONS (5 features)
        *legal_flags,   
    ]

    # Convert Python list to PyTorch tensor (format expected by the network)
    return torch.tensor(features, dtype=torch.float32)


class PacmanDataset(Dataset):
    """
    PyTorch Dataset for loading Pacman expert data.

    Args:
        path: Path to pacman_dataset.pkl

    Attributes:
        inputs: List of feature tensors
        labels: List of action indices
    """

    def __init__(self, path: str):
        """
        Load dataset and convert all states to tensors.

        Args:
            path: Path to pickle file
        """
        with open(path, "rb") as f: 
            data = pickle.load(f)#Load dataset expert witch pair (state, action)

        self.inputs = []
        self.labels = []

        for state, action in data:
            # For state : Convert GameState to feature tensor 
            feature_tensor = state_to_tensor(state) 
            # For action : Convert action to 0/1/2/3/4)
            action_index = ACTION_TO_INDEX[action]

            self.inputs.append(feature_tensor)
            self.labels.append(action_index)

    def __len__(self) -> int:
        """Returns number of samples."""
        return len(self.inputs)

    def __getitem__(self, index: int):
        """
        Get a sample.s

        Args:
            index: Sample index

        Returns:
            Tuple (feature_tensor, action_index)
        """
        return self.inputs[index], self.labels[index]
