import collections
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.svm import OneClassSVM

from signaturesnet import DATA, TRAINED_MODELS

def plot_box(labels, values):
    # Convert tensors to numpy arrays
    labels_np = labels.cpu().numpy()
    values_np = values.cpu().numpy()
    
    # Separate values based on labels
    values_class_0 = values_np[labels_np == 0]
    values_class_1 = values_np[labels_np == 1]
    
    # Prepare data for the box plot
    data = [values_class_0, values_class_1]
    
    # Create box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=['Class 0', 'Class 1'])
    plt.title('Box Plot of Values by Class')
    plt.ylabel('Values')
    plt.ylim(-0, 10.0)
    plt.show()


def binary_classification_metrics(predictions, labels):
    # Ensure predictions and labels are binary (0 or 1)
    predictions = predictions.int()
    labels = labels.int()
    
    # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
    TP = ((predictions == 1) & (labels == 1)).sum().item()
    FP = ((predictions == 1) & (labels == 0)).sum().item()
    TN = ((predictions == 0) & (labels == 0)).sum().item()
    FN = ((predictions == 0) & (labels == 1)).sum().item()

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Confusion matrix
    confusion_matrix = torch.tensor([[TN, FP],
                                     [FN, TP]])

    # Return results as a dictionary
    return {
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1_score, 2),
        'confusion_matrix': confusion_matrix,
    }


class DistanceBasedClassifier(nn.Module):
    def __init__(self, train_data):
        super(DistanceBasedClassifier, self).__init__()
        self.train_data = train_data
        
    @torch.no_grad()
    def forward(self, mutation_dist, num_mut=None):
        """nummut is not used in this model, but it is included for compatibility with other models
        """
        distances = torch.cdist(mutation_dist, self.train_data)
        # IDEA: Improvable doing -mean(top-k of -dist) to avoid outliers
        min_dist = torch.min(distances, dim=1).values * 10
        return min_dist


class LatentDistanceBasedClassifier(nn.Module):
    def __init__(self, train_data, train_numuts, ae_classifier):
        super(LatentDistanceBasedClassifier, self).__init__()
        self.ae_classifier = ae_classifier
        self.train_projection, _ = ae_classifier.encode(train_data, train_numuts)
        
    @torch.no_grad()
    def forward(self, mutation_dist, num_mut=None):
        projection, _ = self.ae_classifier.encode(mutation_dist, num_mut)
        distances = torch.cdist(projection, self.train_projection)
        # IDEA: Improvable doing -mean(top-k of -dist) to avoid outliers
        min_dist = torch.min(distances, dim=1).values / 10
        return min_dist


class OCSVMClassifier():
    def __init__(self, train_data):
        self.ocsvm = OneClassSVM(kernel='rbf', gamma='auto').fit(train_data)
        
    def forward(self, mutation_dist, num_mut=None):
        """nummut is not used in this model, but it is included for compatibility with other models
        """
        results = self.ocsvm.predict(mutation_dist)
        results[results == -1] = 0
        return results


if __name__ == "__main__":
    from signaturesnet.utilities.io import read_data_classifier
    from signaturesnet.utilities.io import read_model
    
    
    # DATA  -------------------------------
    dev = "cuda"
    train_data, val_data = read_data_classifier(
        device=dev,
        experiment_id="datasets/detector",
    )

    train_input = train_data.inputs
    train_input = train_input[(train_data.labels == 1).squeeze(-1)]
    train_nummuts = train_data.num_mut[(train_data.labels == 1).squeeze(-1)]

    val_inputs, val_nummuts = val_data.inputs, val_data.num_mut
    val_labels = val_data.labels.squeeze(-1)


    # Euclidean distance classifier  -------------------------------
    classifier = DistanceBasedClassifier(train_input)
    classifier_logits = classifier.forward(val_inputs)
    # plot_box(labels=val_labels, values=classifier_logits)
    
    threshold = 1.0
    classifier_guess = (classifier_logits <= threshold).to(torch.int64)
    metrics = binary_classification_metrics(
        predictions=classifier_guess,
        labels=val_labels
    )
    print("Euclidean distance classifier metrics:")
    pprint(metrics)


    # COMPARING DISTANCE IN LATENT SPACE -------------------------------
    model_dir = os.path.join(TRAINED_MODELS, "vae_classifier/ae_nummut_sigmoid")
    ae_classifier = read_model(model_dir, device="cuda").eval()
    classifier = LatentDistanceBasedClassifier(
        train_data=train_input,
        train_numuts=train_nummuts,
        ae_classifier=ae_classifier,
    )
    classifier_logits = classifier.forward(val_inputs, val_data.num_mut)
    # plot_box(labels=val_labels, values=classifier_logits)
    
    threshold = 1.0
    classifier_guess = (classifier_logits <= threshold).to(torch.int64)
    metrics = binary_classification_metrics(
        predictions=classifier_guess,
        labels=val_labels
    )
    print("Latent distance classifier metrics:")
    pprint(metrics)
    
    
    # OCSV Classifier  -------------------------------
    ocsvm = OCSVMClassifier(train_input.cpu().numpy())
    ocsvm_guess = torch.tensor(ocsvm.forward(val_inputs.cpu().numpy()))
    # plot_box(labels=val_labels, values=ocsvm_guess)
    metrics = binary_classification_metrics(
        predictions=ocsvm_guess.cpu(),
        labels=val_labels.cpu()
    )
    print("OCSVM classifier metrics:")
    pprint(metrics)
    
    print("here")