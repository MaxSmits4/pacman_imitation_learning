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


def evaluate_accuracy(model, loader, device):
    """
    Evaluate model accuracy on dataset.
    """
    correct = 0
    total = 0

    # START: MNIST inspired §Evaluate the model
    # Evaluation = measure performance WITHOUT learning.
    # We only do forward passes to compute metrics, no weight updates.

    # Dropout OFF -> pdt la phase de training certain neurone sont etaint et ce délibérement
    # -> permet d'eviter l'overfitting: en gros un réseau au cours de l'utilisation peux ne finir par utiliser que
    # de neurone critique pour l'output -> si ceux là ce trompe, le réseau ce trompe
    # BatchNorm inference mode: permet en mode training de normalise les batch -> sur base d'une moyenne et d'une variante
    # glissante qui evolue au cours de batch rencontré


    # Lors du trainning les batch peuvent être non représentatif (par exemple un batch d'une data bizarre)
    # du coup maintenir des meta data inter batch devient éroner -> par conséquen lors du training on se base sur les
    # meta data final etablie lors de la phase de training: running_mean, running_var
    model.eval()  # disable training specific behaviors (e.g., Dropout OFF, BatchNorm inference mode)

    # No need to track gradients during evaluation:
    # - Gradients are only needed for training (to update weights via backprop)
    # - Here we just want predictions, not to learn
    # - Saves memory and speeds up computation
    #
    # Link to gradient descent theory:
    # During evaluation we DO NOT compute:
    #   ∇_θ L̂(θ) (loss.backward)
    # and we DO NOT apply:
    #   θ ← θ - η ∇_θ L̂(θ) (optim.stem)
    # because evaluation is not an optimization step.
    with torch.no_grad():
        for features, actions in loader:
            features = features.to(device)
            actions = actions.to(device, dtype=torch.long) # type long car ce sont des indice de class
            outputs = model(features) # model output prediction: [batch_size, 5] -> chaque output est link a un tenseur 1D de 5 schore
            _, predicted = torch.max(outputs, dim=1) # selectionne le max qui est la classe prédie
            total += actions.size() # ici pourquoi 0 ?????????????????
            correct += (predicted == actions).sum().item()
    # END: MNIST

    model.train() # returun in train mod
    return correct / total # accuration

 
if __name__ == "__main__":
    # Training pipeline inspired by MNIST supervised learning [5]:
    #   Dataset -> DataLoader -> model -> Adam optimizer
    #   -> loss.backward() -> optim.step()
    #
    # Adaptations for Pacman:
    #   1) 80/20 train/test split (from MNIST best practices)
    #   2) Per-epoch evaluation on test set -> on complet session of train and test
    #   3) Save best model based on test accuracy

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

    # START: MNIST inspired §Training loop
    # Adam optimizer = smart version of gradient descent
    # Instead of using the same learning rate for all weights,
    # Adam adapts the learning rate for each weight individually.
    # This makes training faster and more stable.
    optim = Adam(model.parameters(), lr=learning_rate)
    # END: MNIST

    # Best model tracking
    best_accuracy = 0.0
    best_epoch = 0
    save_file = "pacman_model.pth"

    # Training loop
    print("Starting training...\n")

    # START: MNIST inspired §Training loop
    # Training mode = enable learning-specific behaviors.
    model.train()

    for epoch in range(epochs):
        # dataloader: outil qui decompose le dataset en batch (parti de par exemple 256 item du dataset
        # et mélanger -> pour que chaque batch soit représentatif de tout le data set)
        for features, actions in loader_train:
            # By convention, model.loss(...) computes an empirical loss:
            #   L̂(θ) = (1/N) Σ_i ℓ(f_θ(x_i), y_i)
            # where ℓ is the cross-entropy since this is multi-class classification.
            #
            # This is the quantity we actually minimize in practice.
            loss = model.loss(features, actions)

            # Empty previous gradient values of tracked parameters
            optim.zero_grad()

            # Backprop computes gradients of the empirical loss:
            #   ∇_θ L̂(θ) -> it's here that we compute the gradient
            #   on dérive chaque parametre un a un θ (chaque poids entre les neurone), pour chaque un regarde ça derivée
            #   du coup ci c'est positif ça signifie que ce parametre augmente la loss -> il faudra le diminuer
            #   si la derrivée est negative, ce poid diminue la loss -> il faudra l'augmenter
            #   Forward → Loss → Backward → Update weights
            loss.backward()

            # Parameter update step (optimization):
            # Canonical SGD-style formula:
            #   θ ← θ - η ∇_θ L̂(θ)
            #   -> tweaking des poids sur base du backward
            # With Adam, the update is still driven by ∇_θ L̂(θ),
            # but the effective step is adaptively scaled.
            optim.step()
        # END: MNIST

        # Per-epoch evaluation:
        # This measures performance on the held-out split without learning.
        test_accuracy = evaluate_accuracy(model, loader_test, device)

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
