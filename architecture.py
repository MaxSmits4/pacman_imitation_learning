"""
2. architecture.py - Neural Network Architecture for Pacman Imitation Learning

═══════════════════════════════════════════════════════════════════════════════
COMPLETE NETWORK OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

INPUT: Game state [batch_size, 23 features]
  ↓
LAYER 1: Linear(23→128) → BatchNorm → GELU → Dropout
  [batch, 23] → [batch, 128] → [batch, 128] → [batch, 128] → [batch, 128]
  ↓
LAYER 2: Linear(128→64) → BatchNorm → GELU → Dropout
  [batch, 128] → [batch, 64] → [batch, 64] → [batch, 64] → [batch, 64]
  ↓
LAYER 3: Linear(64→32) → BatchNorm → GELU → Dropout
  [batch, 64] → [batch, 32] → [batch, 32] → [batch, 32] → [batch, 32]
  ↓
OUTPUT: Linear(32→5) [no activation, raw logits for CrossEntropyLoss]
  [batch, 32] → [batch, 5]

TOTAL PARAMETERS:
  Layer 1 Linear: 128×(23+1) = 3,072
  Layer 1 BatchNorm: 128×2 = 256 (γ and β)
  Layer 2 Linear: 64×(128+1) = 8,256
  Layer 2 BatchNorm: 64×2 = 128
  Layer 3 Linear: 32×(64+1) = 2,080
  Layer 3 BatchNorm: 32×2 = 64
  Output Linear: 5×(32+1) = 165
  ──────────────────────────────
  TOTAL ≈ 14,021 learnable parameters

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE DETAILS (what happens inside each block):

1) LINEAR (Fully-connected layer):
   Transformation affine: z = x·W^T + b
   - Input: features vector [batch, input_dim]
   - Weights: matrix W [output_dim, input_dim]
   - Bias: vector b [output_dim]
   - Output: z [batch, output_dim]

   Each neuron j receives ALL input features with its own weights:
   z_j = w_{j,1}·x_1 + w_{j,2}·x_2 + ... + w_{j,n}·x_n + b_j

   Example: Linear(23, 128) creates 128 neurons, each with 23 weights + 1 bias
   → Total parameters: 128 × (23 + 1) = 3,072 params

2) BATCHNORM1D:
   Goal: Stabilize activation scale during training

   IN TRAIN mode (per neuron, across batch):
   ẑ = (z - μ_batch) / √(σ²_batch + ε)
   Then re-scale with learned params:
   y = γ·ẑ + β

   Where:
   - μ_batch, σ²_batch: mean/variance computed on current batch
   - γ (gamma/weight): scale parameter [num_features]
   - β (beta/bias): shift parameter [num_features]
   - Initially: γ=1, β=0

   IN EVAL mode:
   Uses running_mean and running_var (computed during training)
   instead of batch statistics → ensures consistent behavior

   CONCRETE EXAMPLE:
   Input z (3 examples, 2 neurons):
     [[10, 100],
      [20, 200],
      [30, 300]]

   For neuron 0: μ=20, σ≈8.16
     ẑ[:, 0] = [(10-20)/8.16, (20-20)/8.16, (30-20)/8.16]
             = [-1.22, 0.00, 1.22]

   Then y[:, 0] = γ₀·ẑ[:, 0] + β₀

3) GELU (Activation):
   Smooth non-linearity: a = GELU(z)
   - Positive values: pass through almost unchanged
   - Negative values: progressively attenuated (not hard cut like ReLU)
   - More "soft" than ReLU, better gradient flow

4) DROPOUT:
   IN TRAIN: Randomly sets activations to 0 with probability p
   a' = a ⊙ m, where m ~ Bernoulli(1-p)

   IN EVAL: Does nothing (identity function)
   a' = a

   Why: Forces network to not rely on specific neurons → prevents overfitting

DATAFLOW THROUGH ONE LAYER:
x → Linear(z) → BatchNorm(y) → GELU(a) → Dropout(a') → next layer

For one neuron j:
x → z_j = Σ w_i·x_i + b → y_j = γ(ẑ_j) + β → a_j = GELU(y_j) → a'_j (maybe 0)
"""

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    MLP for predicting Pacman actions from game state.

    Full architecture: 23 → [128 → 64 → 32] → 5
    Each hidden layer uses: Linear → BatchNorm → GELU → Dropout
    """

    def __init__(
        self,
        input_features=23,
        num_actions=5,
        hidden_dims=[128, 64, 32],
        activation=nn.GELU(),
        dropout=0.3
    ):
        super().__init__()

        layers = []

        # ═══════════════════════════════════════════════════════════════════
        # INPUT LAYER: 23 → 128
        # ═══════════════════════════════════════════════════════════════════
        # Linear: z = x·W^T + b, where W is [128, 23], b is [128]
        # Creates 128 neurons, each processing all 23 input features
        layers.append(nn.Linear(input_features, hidden_dims[0]))
    
        # BatchNorm: Normalizes the 128 outputs (neuron by neuron over batch)
        # Has 128 γ (weight) and 128 β (bias) learnable parameters
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

        # GELU: Smooth activation, applied element-wise
        layers.append(activation)

        # Dropout: Randomly zeros 30% of activations (only in TRAIN mode)
        layers.append(nn.Dropout(dropout))

        # ═══════════════════════════════════════════════════════════════════
        # HIDDEN LAYERS: 128 → 64 → 32
        # ═══════════════════════════════════════════════════════════════════
        for i in range(len(hidden_dims) - 1):
            # Linear(input_dim, output_dim):
            # - input_dim: number of weights per neuron
            # - output_dim: number of neurons in this layer
            # Iteration 0: 128 → 64 (64 neurons, each with 128 weights)
            # Iteration 1: 64 → 32 (32 neurons, each with 64 weights)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

            # BatchNorm: Normalizes each of the output_dim neurons
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

            # GELU activation
            layers.append(activation)

            # Dropout
            layers.append(nn.Dropout(dropout))

        # ═══════════════════════════════════════════════════════════════════
        # OUTPUT LAYER: 32 → 5
        # ═══════════════════════════════════════════════════════════════════
        # No BatchNorm, no activation, no Dropout
        # Outputs raw logits for CrossEntropyLoss
        # CrossEntropyLoss applies softmax internally
        layers.append(nn.Linear(hidden_dims[-1], num_actions))

        self.net = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features → logits

        CONCRETE EXAMPLE (batch_size=3):
        Input x: [batch=3, features=23]
          [[0.5, 0.2, ..., 0.8],   ← Game state 1
           [0.1, 0.9, ..., 0.3],   ← Game state 2
           [0.7, 0.4, ..., 0.6]]   ← Game state 3

        After Linear(23→128): [3, 128]
          Each of 3 examples now represented by 128 numbers (neuron activations)

        After BatchNorm: [3, 128]
          For each of 128 neurons (columns):
            - Compute mean/std across 3 examples (rows)
            - Normalize: ẑ = (z - μ) / σ
            - Re-scale: y = γ·ẑ + β

        After GELU: [3, 128]
          Applied element-wise, smooth non-linearity

        After Dropout (TRAIN mode): [3, 128]
          Randomly zeros ~30% of values

        ... (through 128→64→32 layers similarly) ...

        Final output: [3, 5]
          [[2.1, -0.5, 1.3, 0.2, -1.1],   ← Logits for example 1
           [1.5,  0.8, -0.3, 2.0, 0.1],   ← Logits for example 2
           [-0.2, 1.7, 0.5, -1.0, 3.2]]   ← Logits for example 3

        These are RAW LOGITS (not probabilities yet).
        CrossEntropyLoss will apply softmax internally.
        """
        return self.net(x)

    def loss(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for training

        Args:
            features: [batch_size, 23] game states
            actions: [batch_size] ground truth actions (integers 0-4)

        Returns:
            Scalar loss value

        FORMULA:
        For multi-class classification with C classes:
        L = - (1/N) Σᵢ log(softmax(logitsᵢ)[yᵢ])

        Where:
        - N = batch size
        - logitsᵢ = predicted logits for example i
        - yᵢ = true class label for example i
        - softmax(z)ⱼ = exp(zⱼ) / Σₖ exp(zₖ)

        CONCRETE EXAMPLE:
        Predicted logits for example 1: [2.1, -0.5, 1.3, 0.2, -1.1]
        True action: 0 (go North)

        Softmax:
        exp([2.1, -0.5, 1.3, 0.2, -1.1]) / sum(exp(.C..))
        ≈ [0.65, 0.05, 0.29, 0.10, 0.03]

        Loss for this example:
        -log(0.65) ≈ 0.43

        If prediction was perfect (prob=1.0 for correct class):
        -log(1.0) = 0 (minimum loss)

        If prediction was terrible (prob=0.01 for correct class):
        -log(0.01) ≈ 4.6 (high loss)

        The loss measures: "How confident was the model in the CORRECT answer?"
        """
        pred_actions = self.forward(features)
        return self.criterion(pred_actions, actions)
