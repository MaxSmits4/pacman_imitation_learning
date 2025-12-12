"""
3. train.py - Trains MLP to predict expert actions from game states

    References:
    [4] Kingma & Ba (2015) - Adam Optimizer
    [5] LeCun et al. (1998) - MNIST: Train/test split, loss function principles
"""

import random
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from architecture import PacmanNetwork
from data import PacmanDataset

# for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def model_accuracy(model, loader):
    """
    Evaluate model accuracy on dataset.
    """
    correct_pred = 0
    nb_action_batch = 0

    # START: MNIST inspired §Evaluate the model
    model.eval()  

    with torch.no_grad():
        for features, actions in loader: #action vector 256 expert actions
            outputs = model.forward(features)   # FORWARD - Output=logits
            _, predicted = torch.max(outputs, dim=1)  # max in logits vector = direction predicted
            nb_action_batch += actions.size(0) # size(0) = nbr d'elements in the batch
            correct_pred += (predicted == actions).sum().item()  # correct_pred = our model prediction vs expert choices
    # END: MNIST

    model.train()  # Retour en training mode
    return correct_pred / nb_action_batch  # = accuracy

 
if __name__ == "__main__":
    # Training pipeline inspired by MNIST supervised learning [5]:
    #   Dataset -> DataLoader -> model -> Adam optimizer
    #   -> loss.backward() -> optim.step()

    #  80/20 -> 2 DataLoader test & eval
    #   Test: forward -> loss -> backward ->uptade
    #   Eval: forward 


    # Hyperparameters
    batch_size = 128
    epochs = 150
    learning_rate = 1e-3  # Slightly higher LR for faster convergence
    # (empirical: large enough to converge quickly, but small enough not to oscillate)

    # Load expert demonstrations (Pacman-specific)
    dataset = PacmanDataset(path="datasets/pacman_dataset.pkl")
    print(f"Dataset: {len(dataset)} samples")

    # 80/20 train/test split (MNIST best practice)
    dataset_size = len(dataset)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    print(f"Train: {len(train_set)} | Test: {len(test_set)}\n")

    # START: MNIST inspired §Training loop
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=batch_size)
    # END: MNIST

    # Initialize model
    model = PacmanNetwork()

   
    # Adam optimizer = smart version of gradient descent
    # Adam adapts the learning rate for each weight individually.
    # training is faster and more stable.
    # The parameters of the model are linked to the optimizer. 
    # This allows the optimizer to keep track of parameters 
    # that should be updated during training.
    optim = Adam(model.parameters(), lr=learning_rate)


    # Best model tracking
    best_accuracy = 0.0
    best_epoch = 0
    save_file = "pacman_model.pth"

    # Training loop
    print("Starting training...\n")

    # START: MNIST inspired §Training loop
    model.train()  # Training mode

    for epoch in range(epochs):
        # dataloader: outil qui decompose le dataset en batch
        # et mélanger -> pour que chaque batch soit représentatif de tout le data set
        for features, actions in loader_train:
            
            loss = model.loss(features, actions) #forward & loss

            optim.zero_grad() # Gradients ACCUMULATE by default in PyTorch

            loss.backward()# gradients (∂loss/∂θ) for each θ & retroactivly modify every θ 

            optim.step() #θnew ← θold - LR ∇_θ L̂(θ) = uptade = adam
        # END: MNIST


        test_accuracy = model_accuracy(model, loader_test)


    #Vizualisation in terminal:
        print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {test_accuracy:.2%}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            # START: MNIST inspired §Save and load module
            torch.save(model.state_dict(), save_file) #dict Python containing every parametres
            # END: MNIST

    # print Final results in terminal
    print(f"\nBest model: epoch {best_epoch + 1} "
          f"with {best_accuracy:.2%} accuracy")
    print(f"Saved to: {save_file}")
