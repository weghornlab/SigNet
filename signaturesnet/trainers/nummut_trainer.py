import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchmetrics import Accuracy
from tqdm import tqdm
from signaturesnet.utilities.plotting import plot_confusion_matrix
import wandb

from signaturesnet.models import NumMutNet
from signaturesnet.loggers.nummut_logger import NumMutLogger
from signaturesnet.utilities.io import save_model
from signaturesnet import DATA, TRAINED_MODELS
from signaturesnet.utilities.temporal_io import read_data_nummutnet


class NumMutTrainer:
    def __init__(self,
                 train_data,
                 val_data,
                 iterations,
                 log_freq=100,
                 log_path="../runs",
                 model_path=None,
                 device=torch.device("cuda:0"),):
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.iterations = iterations
        self.log_freq = log_freq
        self.log_path = log_path
        self.model_path = model_path
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = NumMutLogger()

    def objective(self,
                  batch_size,
                  lr,
                  num_hidden_layers,
                  num_units,
                  plot=False):

        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        model = NumMutNet(n_layers=int(num_hidden_layers),
                          hidden_dim=int(num_units))
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        

        step = 0
        l_vals, max_found = [], -np.inf
        for _ in range(self.iterations):
            for train_input, train_label, _, _, _ in tqdm(dataloader):
                model.train()  # NOTE: Very important! Otherwise we zero the gradient
                optimizer.zero_grad()   
                train_prediction = model(train_input)

                train_loss = self.criterion(input=train_prediction,
                                            target=train_label.squeeze(),)

                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    one_hot_label = torch.nn.functional.one_hot(train_label, num_classes=train_prediction.shape[-1]).squeeze()
                    accuracy_train = Accuracy().to(self.device)(preds=train_prediction,
                                                target=one_hot_label)
                    val_prediction = model(self.val_dataset.inputs)
                    val_loss = self.criterion(input=val_prediction,
                                              target=self.val_dataset.labels.squeeze(),)
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                if plot and step % self.log_freq == 0:
                    one_hot_label = torch.nn.functional.one_hot(self.val_dataset.labels.squeeze(),
                                                                num_classes=train_prediction.shape[-1]).squeeze()
                    accuracy_val = Accuracy().to(self.device)(preds=val_prediction,
                                              target=one_hot_label)

                    self.logger.log(train_loss=train_loss,
                                    train_classification_metrics={"accuracy": accuracy_train},
                                    val_loss=val_loss,
                                    val_classification_metrics={"accuracy": accuracy_val},
                                    step=step)

                # if self.model_path is not None and step % 1000 == 0:
                #     save_model(model=model, directory=self.model_path + '_it' + str(step))
                step += 1
        save_model(model=model, directory=self.model_path)

        with torch.no_grad():
            val_prediction = model(self.val_dataset.inputs)
            val_prediction = torch.argmax(val_prediction, dim=1)
            label_val = self.val_dataset.labels.squeeze()
            labels = torch.unique(torch.cat((val_prediction, label_val))).tolist()
            conf_mat = plot_confusion_matrix(label_val, val_prediction, labels)
            wandb.log({"Confusion Matrix": wandb.Image(conf_mat)})

        return max_found

def train_nummutnet(config,
                    models_folder=TRAINED_MODELS,
                    data_folder=DATA):
    train_data, val_data = read_data_nummutnet(path=os.path.join(data_folder, "datasets/num_muts/"),
                                               device=config["device"],)
    
    if config["enable_logging"]:
        run = wandb.init(project=config["wandb_project_id"],
                         entity='signet_2',
                         config=config,
                         name=config["model_name"])


    trainer = NumMutTrainer(train_data=train_data,
                            val_data=val_data,
                            iterations=config["iterations"],
                            model_path=os.path.join(models_folder, config["model_name"]),
                            device=config["device"],)

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_hidden_layers=config["num_hidden_layers"],
                                num_units=config["num_neurons"],
                                plot=True)

    if config["enable_logging"]:
        wandb.log({"validation_score": min_val})
        run.finish()

    return min_val