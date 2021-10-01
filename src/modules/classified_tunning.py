import os
import sys

import numpy as np
import torch

from utilities.io import read_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.finetuner import FineTuner
from models.classifier import Classifier

class ClassifiedFinetuner:
    # TODO: At some point we'll be using the CombinedFInetuner
    # instead of a single finetuner

    def __init__(self,
                 classifier_file,
                 realistic_model_file,
                 random_model_file,
                 classification_cutoff=0.5,
                 device="cpu"):
        self.classification_cutoff = classification_cutoff
        self.device = device

        self.classifier = read_model(classifier_file)
        self.realistic_finetuner = read_model(realistic_model_file)
        self.random_finetuner = read_model(random_model_file)

    def __call__(self,
                 mutation_dist,
                 weights,
                 num_mut):
        # Classify inputs depending on whether they come from a real distribution or not
        classification = self.classifier(mutation_dist=mutation_dist,
                                         num_mut=num_mut).view(-1)

        # Remember input indexes
        ind = np.array(range(mutation_dist.size()[0]))
        ind_real = ind[classification >= self.classification_cutoff]
        ind_rand = ind[classification < self.classification_cutoff]
        ind_order = np.concatenate((ind_real, ind_rand))

        # Select and finetune mutations classified as real
        mut_dist_real = mutation_dist[ind_real, ...]
        weights_real = weights[ind_real, ...]
        num_mut_real = num_mut[ind_real]
        real_guess = self.realistic_finetuner(mutation_dist=mut_dist_real,
                                              weights=weights_real,
                                              num_mut=num_mut_real)

        # Select and finetune mutations classified as random
        mut_dist_rand = mutation_dist[ind_rand, ...]
        weights_rand = weights[ind_rand, ...]
        num_mut_rand = num_mut[ind_rand]
        rand_guess = self.random_finetuner(mutation_dist=mut_dist_rand,
                                           weights=weights_rand,
                                           num_mut=num_mut_rand)

        # Join predictions and re-order them as originally
        joined_guess = torch.cat((real_guess, rand_guess), dim=0)
        joined_guess = torch.cat(
            (joined_guess, torch.tensor(ind_order).reshape(-1, 1)), dim=1)
        joined_guess = joined_guess[joined_guess[:, -1].sort()[1]]
        joined_guess = joined_guess[:, :-1]
        return joined_guess