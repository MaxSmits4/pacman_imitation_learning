import numpy as np
import random
import torch

torch.cuda.empty_cache()

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import SmartyGhost

from architecture import PacmanNetwork
from pacmanagent import PacmanAgent


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

path_to_saved_model = "pacman_model.pth"
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PacmanNetwork().to(device)
state_dict = torch.load(path_to_saved_model, map_location=device)
model.load_state_dict(state_dict)
model.eval()

pacman_agent = PacmanAgent(model)

score, elapsed_time, nodes = runGame(
    layout_name="test_layout",
    pacman=pacman_agent,
    ghosts=[SmartyGhost(1)],
    beliefstateagent=None,
    displayGraphics=True,
    expout=0.0,
    hiddenGhosts=False,
)

print(f"Score: {score}")
print(f"Computation time: {elapsed_time}")
