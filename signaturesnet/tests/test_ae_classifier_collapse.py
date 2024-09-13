"""
HERE I AM ANSWERING THE FOLLOWING QUESTIONS:

IS THERE A POSTERIOR COLLAPSE? IS THE NETWORK RECONSTRUCTING SOMETHING CLOSE TO THE ORIGINAL INPUT?
The answers are: No posterior collapse, Yes good reconstruction.
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

from signaturesnet import DATA, TRAINED_MODELS
from signaturesnet.utilities.io import save_model
from signaturesnet.models.vae_classifier import VaeClassifier
from signaturesnet.loggers.generator_logger import GeneratorLogger
from signaturesnet.utilities.plotting import plot_matrix
from signaturesnet.utilities.io import read_model
from signaturesnet.utilities.io import read_data_classifier
from signaturesnet.utilities.io import sort_signatures

def get_dist_mat(m):
    N = len(m)
    D = torch.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i,j] = nn.MSELoss()(m[i], m[j])
    return D

if __name__ == "__main__":
    dev = "cuda"
    model_dir = os.path.join(TRAINED_MODELS, "vae_classifier/ae_nummut_sigmoid")
    model = read_model(model_dir, device=dev).eval()

    train_data, val_data = read_data_classifier(
        device=dev,
        experiment_id="datasets/detector",
    )

    dataloader = DataLoader(
        dataset=val_data,
        batch_size=100,
        shuffle=True,
    )
    
    with torch.no_grad():
        for inputs, labels, _, nummut, _ in tqdm(dataloader):
            inputs = inputs[(labels == 1).squeeze(-1)]
            nummut = nummut[(labels == 1).squeeze(-1)]
            
            pred, mean, std = model(inputs, nummut, noise=False)


            real_distnces = get_dist_mat(inputs)
            rec_distances = get_dist_mat(pred)
            latent_distances = get_dist_mat(mean)
            
            plot_matrix(
                matrix=real_distnces / 0.005,
                filename="/home/oleguer/software/SigNet/real_dist",
                title="Real pairwise distances")
            plot_matrix(
                matrix=rec_distances / 0.005,
                filename="/home/oleguer/software/SigNet/reconstructed_dist",
                title="Reconstructed pairwise distances")
            plot_matrix(
                matrix=latent_distances,
                filename="/home/oleguer/software/SigNet/latent_dist",
                title="Latent pairwise distances")
            
            print("here")