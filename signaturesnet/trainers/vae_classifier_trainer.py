
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
from signaturesnet.utilities.metrics import get_jensen_shannon, get_kl_divergence
from signaturesnet.utilities.io import save_model
from signaturesnet.utilities.generator_data import GeneratorData
from signaturesnet.utilities.oversampler import OverSampler, CancerTypeOverSampler
from signaturesnet.models import Generator
from signaturesnet.loggers.generator_logger import GeneratorLogger

class VaeClassifierTrainer:
    def __init__(
            self,
            iterations,
            train_data,
            val_data,
            signatures,
            lagrange_param=1.0,
            loging_path="../runs",
            num_classes=94,
            log_freq=1000,
            model_path=None,  # File where to save model learned weights None to not save
            device=torch.device("cuda:0"),
        ):
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.device = device
        self.log_freq = log_freq
        self.lagrange_param = lagrange_param
        self.model_path = os.path.join(TRAINED_MODELS, model_path)
        self.train_dataset = train_data
        self.val_dataset = val_data

        self.logger = GeneratorLogger(
            train_inputs=train_data.inputs,
            val_inputs=val_data.inputs,
            signatures=signatures,
            device=device)


    def __loss(self, ins, pred, z_mu, z_std):
        # kl_div = (0.5*(z_std.pow(2) + z_mu.pow(2) - 2*torch.log(z_std) - 1).sum(dim=1)).mean(dim=0)
        reconstruction = nn.MSELoss()(ins, pred)
        # reconstruction = get_jensen_shannon(predicted_label=pred, true_label=input)
        # return reconstruction + self.adapted_lagrange_param*kl_div
        return reconstruction

    def objective(self,
                  batch_size,
                  lr_encoder,
                  lr_decoder,
                  num_hidden_layers,
                  latent_dim,
                  plot=False):

        print(batch_size, lr_encoder, lr_decoder,
              num_hidden_layers, latent_dim)

        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=int(batch_size),
            shuffle=True,
        )
        model = Generator(
            input_size=int(self.num_classes),
            num_hidden_layers=int(num_hidden_layers),
            latent_dim=int(latent_dim),
            device=self.device.type
        )
        model.to(self.device)
        model.train()

        # wandb.watch(model, log_freq=100)

        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        optimizer = optim.Adam([
            {'params': model.encoder_layers.parameters(), 'lr': lr_encoder},
            {'params': model.decoder_layers.parameters(), 'lr': lr_decoder}
        ])

        # l_vals = collections.deque(maxlen=50)
        # max_found = -np.inf
        step = 0
        # total_steps = 1000*len(self.train_dataset)
        total_steps = self.iterations*len(self.train_dataset)
        # self.batch_size_factor = batch_size/len(self.train_dataset)
        self.batch_size_factor = 1.
        train_DQ99R = None
        for iteration in range(self.iterations):
            for train_input, train_labels, _, _, _ in tqdm(dataloader):
                train_input = train_input[(train_labels == 1).squeeze(-1)]
                optimizer.zero_grad()
                train_pred, train_mean, train_std = model(train_input, noise=False)
                # self.adapted_lagrange_param = self.lagrange_param
                # if step < total_steps*0.8:
                #     self.adapted_lagrange_param = self.lagrange_param * \
                #         float(total_steps - step)/float(total_steps)
                # else:
                #     self.adapted_lagrange_param = self.lagrange_param * \
                #         float(total_steps - total_steps*0.8)/float(total_steps)
                train_loss = self.__loss(
                    ins=train_input,
                    pred=train_pred,
                    z_mu=train_mean,
                    z_std=train_std,
                )

                train_loss.backward()
                optimizer.step()

                
                    # l_vals.append(val_loss.item())
                    # max_found = max(max_found, -np.nanmean(l_vals))

                if plot and step % self.log_freq == 0 and step > 0:
                    print("Validating...")
                    model.eval()
                    with torch.no_grad():
                        val_inputs = self.val_dataset.inputs[(self.val_dataset.labels == 1).squeeze(-1)][:100]
                        val_pred, val_mean, val_std = model(
                            val_inputs, noise=False
                        )
                        val_loss = self.__loss(
                            ins=val_inputs,
                            pred=val_pred,
                            z_mu=val_mean,
                            z_std=val_std
                        )
                    print("Validating [DONE]")

                    print("Logging...")
                    
                    current_train_DQ99R = self.logger.log(
                        train_loss=train_loss,
                        train_prediction=train_pred,
                        train_label=train_input,
                        val_loss=val_loss,
                        val_prediction=val_pred,
                        val_label=val_inputs,
                        train_mu=train_mean,
                        train_sigma=train_std,
                        val_mu=val_mean,
                        val_sigma=val_std,
                        step=step,
                        model=model,
                    )
                    # TODO: Add posterior collapse metric
                    
                    train_DQ99R = current_train_DQ99R if current_train_DQ99R is not None else train_DQ99R
                    model.train()
                    print("Logging [DONE]")

                if self.model_path is not None and step % 500 == 0:
                    print("Saving model...")
                    save_model(model=model, directory=self.model_path)
                    print("Saving model [DONE]")
                step += 1
        if self.model_path is not None:
            save_model(model=model, directory=self.model_path)
        
        # Return last mse and KL obtained in validation
        return train_DQ99R

def log_results(config, train_DQ99R, out_csv_path):
    model_results = pd.DataFrame({"batch_size": [config["batch_size"]],
                                  "lr_encoder": [config["lr_encoder"]],
                                  "lr_decoder": [config["lr_decoder"]],
                                  "num_hidden_layers": [config["num_hidden_layers"]],
                                  "latent_dim": [config["latent_dim"]],
                                  "lagrange_param": [config["lagrange_param"]],
                                  "train_DQ99R": [train_DQ99R]})
    model_results.to_csv(out_csv_path,
                         header=False, index=False, mode="a")

def train_vae_classifier(config, data_folder=DATA + "/") -> float:
    """Train a classification model and get the validation score

    Args:
        config (dict): Including all the needed args
        to load data, and train the model 
    """
    from signaturesnet.utilities.io import read_data_classifier
    from signaturesnet.utilities.io import sort_signatures

    dev = "cuda" if config["device"] == "cuda" and torch.cuda.is_available(
    ) else "cpu"
    print("Using device:", dev)

    if config["enable_logging"]:
        run = wandb.init(project=config["wandb_project_id"],
                   entity='sig-net',
                   config=config,
                   name=config["model_id"])

    train_data, val_data = read_data_classifier(
        device=dev,
        experiment_id=config["data_id"],
    )

    signatures = sort_signatures(
        file=data_folder + "data.xlsx",
        mutation_type_order=data_folder + "mutation_type_order.xlsx")

    trainer = VaeClassifierTrainer(
        iterations=config["iterations"],  # Passes through all dataset
        train_data=train_data,
        val_data=val_data,
        signatures=signatures,
        lagrange_param=config["lagrange_param"],
        num_classes=config["num_classes"],
        device=torch.device(dev),
        model_path=os.path.join(config["models_dir"], config["model_id"]),
    )

    train_DQ99R = trainer.objective(
        batch_size=config["batch_size"],
        lr_encoder=config["lr_encoder"],
        lr_decoder=config["lr_decoder"],
        num_hidden_layers=config["num_hidden_layers"],
        latent_dim=config["latent_dim"],
        plot=config["enable_logging"],
    )

    wandb.log({"train_DQ99R_score": train_DQ99R})

    if config["enable_logging"]:
        run.finish()
    return train_DQ99R


if __name__ == "__main__":
    from signaturesnet import TRAINING_CONFIGS
    from signaturesnet.utilities.io import read_config

    config = read_config(path=os.path.join(TRAINING_CONFIGS, "vae_classifier/vc_config.yaml"))
    
    DQ99R = train_vae_classifier(config=config,)
    print("DQ99R:", DQ99R)