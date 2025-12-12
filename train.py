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
            total += actions.size(0)  # size(0) = nombre d'éléments dans le batch
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
            # ═══════════════════════════════════════════════════════════════════
            # TRAINING STEP: Forward → Loss → Backward → Update
            # ═══════════════════════════════════════════════════════════════════
            # This is ONE optimization step on ONE batch.
            # Each epoch has ~47 batches (12015 samples / 256 batch_size ≈ 47)
            #
            # INPUT:
            #   features: [batch_size, 23] game states
            #   actions: [batch_size] expert actions (0-4)
            #
            # Example batch (batch_size=256):
            #   features shape: [256, 23]
            #   actions shape: [256]
            #
            # ───────────────────────────────────────────────────────────────────
            # STEP 1: FORWARD PASS + COMPUTE LOSS
            # ───────────────────────────────────────────────────────────────────
            # Computes empirical loss on this batch:
            #   L̂(θ) = (1/N) Σᵢ₌₁ᴺ ℓ(f_θ(xᵢ), yᵢ)
            #
            # Where:
            #   N = batch_size (256)
            #   f_θ(xᵢ) = model prediction for example i
            #   yᵢ = true action for example i
            #   ℓ = cross-entropy loss (for multi-class classification)
            #
            # CONCRETE:
            # For each of 256 examples:
            #   1. Forward pass: xᵢ → [23→128→64→32→5] → logits
            #   2. Compute loss: -log(softmax(logits)[yᵢ])
            #   3. Average over 256 examples → single scalar loss
            #
            # Example output: loss = 1.234 (scalar tensor)
            loss = model.loss(features, actions)

            # ───────────────────────────────────────────────────────────────────
            # STEP 2: ZERO GRADIENTS
            # ───────────────────────────────────────────────────────────────────
            # Gradients ACCUMULATE by default in PyTorch.
            # We need to reset them to zero before computing new gradients.
            #
            # Sets ALL .grad attributes to zero:
            #   model.net[0].weight.grad = zeros([128, 23])
            #   model.net[0].bias.grad = zeros([128])
            #   ... (for all layers)
            optim.zero_grad()

            # ═══════════════════════════════════════════════════════════════════
            # BACKPROPAGATION: Compute gradients ∇_θ L̂(θ)
            # ═══════════════════════════════════════════════════════════════════
            # This is where PyTorch computes ALL gradients using chain rule.
            #
            # GRADIENT = Derivative of loss with respect to each parameter
            # For each weight w and bias b in the network:
            #   gradient = ∂L/∂w  or  ∂L/∂b
            #
            # WHAT IT MEANS:
            # "If I increase this parameter by a tiny amount, how much does the loss change?"
            #
            # - If ∂L/∂w > 0 (positive): increasing w increases loss
            #   → We should DECREASE w to reduce loss
            #
            # - If ∂L/∂w < 0 (negative): increasing w decreases loss
            #   → We should INCREASE w to reduce loss
            #
            # CONCRETE EXAMPLE - Neurone 1 of first hidden layer (23→128):
            # This neuron has 24 parameters: 23 weights + 1 bias
            #
            # After backward(), these gradients are stored in:
            #   model.net[0].weight.grad[0]  →  [23] gradient values for weights
            #   model.net[0].bias.grad[0]    →  1 gradient value for bias
            #
            # Example gradient values:
            #   ∂L/∂w₁,₁ = 0.05   → weight w₁,₁ should decrease slightly
            #   ∂L/∂w₁,₂ = -0.12  → weight w₁,₂ should increase
            #   ∂L/∂w₁,₃ = 0.03   → weight w₁,₃ should decrease slightly
            #   ...
            #   ∂L/∂b₁ = -0.08    → bias b₁ should increase
            #
            # HOW PYTORCH COMPUTES THIS (Chain Rule):
            # Starting from loss L, going backward through layers:
            #
            # ∂L/∂w = ∂L/∂output · ∂output/∂activation · ∂activation/∂z · ∂z/∂w
            #         ↑            ↑                      ↑               ↑
            #    (from next    (Dropout)            (GELU)        (= input)
            #      layer)      (BatchNorm)
            #
            # For a weight in layer L:
            # ∂L/∂wᴸ = (gradient from layer L+1) · (local gradient at layer L)
            #
            # This propagates backward from output (layer 5) to input (layer 1).
            #
            # VISUALIZATION OF GRADIENT FLOW:
            # Loss (scalar)
            #   ↓ ∂L/∂logits
            # Linear5 (32→5)  ← gradients: ∂L/∂W⁵, ∂L/∂b⁵
            #   ↓ ∂L/∂a⁴
            # Dropout4
            #   ↓
            # GELU4
            #   ↓
            # BatchNorm4  ← gradients: ∂L/∂γ⁴, ∂L/∂β⁴
            #   ↓ ∂L/∂z⁴
            # Linear4 (64→32)  ← gradients: ∂L/∂W⁴, ∂L/∂b⁴
            #   ↓
            # ... (continues backward through all layers)
            #   ↓
            # Linear1 (23→128)  ← gradients: ∂L/∂W¹, ∂L/∂b¹
            #
            # After backward(), ALL parameters have their .grad filled with
            # the derivative of the loss with respect to that parameter.
            loss.backward()

            # ═══════════════════════════════════════════════════════════════════
            # OPTIMIZER STEP: Update parameters using gradients
            # ═══════════════════════════════════════════════════════════════════
            # Now that we have gradients (from backward()), we update the weights.
            #
            # VANILLA GRADIENT DESCENT FORMULA:
            #   θ_new = θ_old - η · ∇L(θ)
            #
            # Where:
            #   θ = parameter (weight or bias)
            #   η (eta) = learning rate (8e-4 in our case)
            #   ∇L(θ) = gradient (computed by backward())
            #
            # CONCRETE EXAMPLE with one weight:
            #   Current value: w₁,₁ = 0.5
            #   Gradient: ∂L/∂w₁,₁ = 0.2 (positive → should decrease)
            #   Learning rate: η = 0.001
            #
            #   Update:
            #   w₁,₁ ← 0.5 - 0.001 × 0.2 = 0.5 - 0.0002 = 0.4998
            #
            # If gradient was negative (-0.2):
            #   w₁,₁ ← 0.5 - 0.001 × (-0.2) = 0.5 + 0.0002 = 0.5002
            #
            # ADAM OPTIMIZER (what we actually use):
            # Adam is smarter than vanilla gradient descent. Instead of:
            #   θ ← θ - η · gradient
            #
            # Adam uses:
            #   θ ← θ - η · (adapted_gradient)
            #
            # Where adapted_gradient takes into account:
            #   1) Momentum: exponential moving average of past gradients
            #      → Helps overcome local minima, smooths updates
            #
            #   2) Adaptive learning rate: different effective η for each parameter
            #      → Parameters with large gradients get smaller steps
            #      → Parameters with small gradients get larger steps
            #
            # Adam formulas (simplified):
            #   m ← β₁·m + (1-β₁)·gradient        (momentum)
            #   v ← β₂·v + (1-β₂)·gradient²       (variance)
            #   θ ← θ - η · m/√(v + ε)             (update)
            #
            # Where β₁≈0.9, β₂≈0.999 (default Adam parameters)
            #
            # WHAT HAPPENS IN CODE:
            # For EVERY parameter in the model:
            #   1. Read current value: θ_old
            #   2. Read gradient: ∂L/∂θ (from .grad)
            #   3. Compute adaptive step using Adam formula
            #   4. Update: θ_new = θ_old - step
            #
            # After this line, all weights and biases have been updated to
            # (hopefully) reduce the loss on future predictions.
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
