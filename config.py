# config.py
import os
import torch

DATA_DIR = os.path.join(os.getcwd(), 'data')
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 45
# DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# DATA_DIR = '/Users/srirammandalika/Downloads/MedMNIST'
# BATCH_SIZE = 128
# NUM_EPOCHS = 45
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


