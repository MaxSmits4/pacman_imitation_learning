# Pacman - Imitation Learning

Project for **INFO8006 - Introduction to Artificial Intelligence** (ULiège).

## Description

Train a neural network to imitate an expert Pacman player. The model learns from (game state, expert action) pairs and predicts which action to take in each situation.

## Installation

```bash
pip install torch pandas
```

## Usage

**1. Train the model:**
```bash
python train.py
```
This generates `pacman_model.pth`.

**2. Run the game:**
```bash
python run.py
```

## Structure & ordre de lecture:

- `1.data.py` - Feature extraction and dataset
- `2.architecture.py` - Neural network (MLP)
- `3.train.py` - Training pipeline
- `4.pacmanagent.py` - Agent using the trained model
