"""
3. train.py - Trains MLP to predict expert actions from game states

Supervised learning pipeline (inspired by MNIST training [5]):
- Train/test split (80/20) for generalization evaluation
- Cross-entropy loss for multi-class classification
- Adam optimizer for efficient convergence [4]
- Batch processing with DataLoader

    References: see full version in Bibliography.txt
    [4] Kingma & Ba (2015) - Adam Optimizer
    [5] LeCun et al. (1998) - MNIST: Train/test split, loss function principles
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from architecture import PacmanNetwork
from data import PacmanDataset


def model_accuracy(model, loader, device):
    """
    Evaluate model accuracy on dataset.
    """
    correct_pred = 0
    Batch_size = 0

    # START: MNIST inspired §Evaluate the model
    with torch.no_grad():
        for features, actions in loader:
            features = features.to(device)
            actions = actions.to(device, dtype=torch.long) # action = expert direction
            outputs = model(features) # model output predictions: [batch_size, 5] = logits
            _, predicted = torch.max(outputs, dim=1) # predicted = max in logits vector = direction predicted
            Batch_size += actions.size(0)  # size(0) = nbr d'elements in the sbatch
            correct_pred += (predicted == actions).sum().item() #correct_pred = our model prediction vs expert choises
    # END: MNIST

    model.train() # trainnig mode
    return correct_pred / Batch_size # = accuracy

 
if __name__ == "__main__":
    # Training pipeline inspired by MNIST supervised learning [5]:
    #   Dataset -> DataLoader -> model -> Adam optimizer
    #   -> loss.backward() -> optim.step()

    # Lors du trainning les batch peuvent être non représentatif (par exemple un batch d'une data bizarre)
    # du coup maintenir des meta data inter batch devient éroner -> par conséquen lors du training on se base sur les
    # meta data final etablie lors de la phase de training: running_mean, running_var
    
    model.eval()  #evaluation mode (Dropout OFF, BatchNorm μ&σ off)

    # No need to track gradients during evaluation:
    # - Gradients are only needed for training (to update weights via backprop)
    # - Saves memory and speeds up computation
  

    # Hyperparameters
    batch_size = 256
    epochs = 150
    learning_rate = 8e-4  # Slightly higher LR for faster convergence
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
    device = "cpu"
    model = PacmanNetwork().to(device)

   
    # Adam optimizer = smart version of gradient descent
    # Adam adapts the learning rate for each weight individually.
    # training is faster and more stable.
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
            
            loss = model.loss(features, actions)

            # Gradients ACCUMULATE by default in PyTorch.
            # We need to reset them to zero before computing new gradients.
            optim.zero_grad()

    
            loss.backward()

            optim.step() #θ ← θ - η · (adapted_gradient) 
        # END: MNIST

    
        test_accuracy = model_accuracy(model, loader_test, device)


    #Vizualisation in terminal:
        print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {test_accuracy:.2%}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            # START: MNIST inspired §Save and load module
            torch.save(model.state_dict(), save_file) # utile ?
            # END: MNIST

    # print Final results in terminal
    print(f"\nBest model: epoch {best_epoch + 1} "
          f"with {best_accuracy:.2%} accuracy")
    print(f"Saved to: {save_file}")
