import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from signaturesnet.loggers import ClassifierLogger
from signaturesnet.models.one_class_svm_classifier import OneClassSVM
from signaturesnet.utilities.data_partitions import DataPartitions
from signaturesnet.utilities.io import save_model

def get_random_subset(tensor, subset_size):
    num_rows = tensor.size(0)
    if subset_size > num_rows:
        raise ValueError("subset_size cannot be greater than the number of rows in the tensor")
    indices = torch.randperm(num_rows)[:subset_size]
    subset = tensor[indices]
    return subset

class OCSvmClassifierTrainer:
    def __init__(
            self,
            iterations,
            train_data,
            val_data,
            loging_path="../runs",
            log_freq=100,
            model_path=None,  # File where to save model learned weights None to not save
            device=torch.device("cuda:0")
        ):
        
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.device = device
        self.log_freq = log_freq
        self.model_path = model_path
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.logger = ClassifierLogger()
        self.cutoff = 0

    def objective(
            self,
            lr,
            run=None
        ):

        x_train = self.train_dataset.inputs[(self.train_dataset.labels == 1).squeeze()].to(self.device)
        x_train = get_random_subset(x_train, 100)
        
        model = OneClassSVM(
            X=x_train,
        )
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        l_vals = collections.deque(maxlen=50)
        max_found = -np.inf
        step = 0
        for iteration in tqdm(range(self.iterations)):
            model.train()
            optimizer.zero_grad()
            result = model.forward(lagrange_multiplier=1.0)
            train_loss = result["loss"]

            train_loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_prediction = model.predict(self.val_dataset.inputs.to(self.device))

                l_vals.append(train_loss.item())
                max_found = max(max_found, -np.nanmean(l_vals))

            if run and step % self.log_freq == 0:
                self.logger.log(
                    train_loss=train_loss,
                    train_prediction=torch.ones(x_train.size(0)).type(torch.int64),  # Does not apply
                    train_label=torch.ones(x_train.size(0)).type(torch.int64),  # Does not apply
                    val_loss=torch.tensor(0.0),  # Does not apply
                    val_prediction=(val_prediction > self.cutoff).type(torch.int64),
                    val_label=self.val_dataset.labels.type(torch.int64).squeeze(-1),
                    step=step
                )

            if self.model_path is not None and step % 500 == 0:
                save_model(model=model, directory=self.model_path)
            step += 1
        if self.model_path is not None:
            save_model(model=model, directory=self.model_path)
        
        if run is not None:
            run.finish()
        return max_found


def train_ocsvm_classifier(config) -> float:
    """Train a classification model and get the validation score

    Args:
        config (dict): Including all the needed args
        to load data, and train the model 
    """
    from signaturesnet.utilities.io import read_data_classifier

    dev = "cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    print("Using device:", dev)

    run = None
    if config["enable_logging"]:
        run = wandb.init(
            project=config["wandb_project_id"],
            entity='sig-net',
            config=config,
            name=config["model_id"]
        )

    train_data, val_data = read_data_classifier(
        device=dev,
        experiment_id=config["data_id"]
    )

    trainer = OCSvmClassifierTrainer(
        iterations=config["iterations"],  # Passes through all dataset
        train_data=train_data,
        val_data=val_data,
        device=torch.device(dev),
        model_path=os.path.join(config["models_dir"], config["model_id"])
    )

    min_val = trainer.objective(
        lr=config["lr"],
        run=run,
    )

    return min_val