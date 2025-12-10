# Pacman - Imitation Learning

Project for **INFO8006 - Introduction to Artificial Intelligence** (ULiège).

## Description

Train a neural network to imitate an expert Pacman player. The model learns from (game state, expert action) pairs and predicts which action to take in each situation.

## Installation

### Prérequis
- Anaconda ou Miniconda installé
- Python 3.8+

### Créer et configurer l'environnement conda

**1. Créer l'environnement conda :**
```bash
conda create -n pacman python=3.10 -y
```

**2. Activer l'environnement :**
```bash
conda activate pacman
```

**3. Installer PyTorch (avec support GPU) :**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**OU pour CPU uniquement :**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**4. Installer les autres dépendances :**

**Méthode simple (recommandée) - Utiliser requirements.txt :**
```bash
pip install -r requirements.txt
```
Le fichier `requirements.txt` contient : torch, torchvision, einops

**OU manuellement :**
```bash
pip install einops
```

### Vérifier l'installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
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
