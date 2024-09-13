"""
HERE I AM ANSWERING THE FOLLOWING QUESTIONS:

IS THE RECONSTRUCTION OF RANDOM MUTATIONS WORSE THAN A REALISTIC EXAMPLE?
The answer is:
"""

import collections
import os
import sys

import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt


from signaturesnet import DATA, TRAINED_MODELS
from signaturesnet.utilities.io import save_model
from signaturesnet.models.vae_classifier import VaeClassifier
from signaturesnet.loggers.generator_logger import GeneratorLogger
from signaturesnet.utilities.plotting import plot_matrix
from signaturesnet.utilities.io import read_model
from signaturesnet.utilities.io import read_data_classifier
from signaturesnet.utilities.io import sort_signatures
from sklearn.metrics import roc_curve, auc

def get_dist_vec(reals, reconstructions):
    N = len(reals)
    D = torch.zeros(N)
    for i in range(N):
        D[i] = nn.MSELoss()(reals[i], reconstructions[i])
    return D

def plot_hist(fake_d_np, real_d_np):
    plt.figure(figsize=(10, 6))

    plt.hist(fake_d_np, bins=200, alpha=0.5, label='Fake Distances', color='blue')
    plt.hist(real_d_np, bins=200, alpha=0.5, label='Real Distances', color='orange')

    plt.title('Histogram of Fake and Real Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()
    
def plot_roc(binary_labels, probabilities):
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(binary_labels, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    dev = "cuda"
    model_dir = os.path.join(TRAINED_MODELS, "vae_classifier/ae_nummut_sigmoid_2")
    model = read_model(model_dir, device=dev).eval()

    train_data, val_data = read_data_classifier(
        device=dev,
        experiment_id="datasets/detector",
    )

    dataloader = DataLoader(
        dataset=val_data,
        batch_size=1000,
        shuffle=True,
    )
    
    with torch.no_grad():
        for inputs, labels, _, nummut, _ in tqdm(dataloader):

            pred, mean, std = model(inputs, nummut, noise=False)

            # Plot histogram of distances
            distances = torch.clip(get_dist_vec(inputs, pred).to(dev), 0, 0.001)
            
            fake_d = distances[(labels == 0).squeeze(-1)]
            real_d = distances[(labels == 1).squeeze(-1)]
            
            plot_hist(fake_d.cpu().numpy(), real_d.cpu().numpy())
            probs = 1 - (distances / 0.001)  # very improvable
            plot_roc(labels.cpu().numpy(), probs.cpu().numpy())
            
            print("here")