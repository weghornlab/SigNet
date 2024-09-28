"""
This is the script used to find the thresholds for the distance-based classifiers. It is used to find
the best threshold for each class of mutations based on the F1 score.
The thresholds are found by using the ROC curve and finding the threshold
that maximizes the F1 score.
"""

from pprint import pprint

import torch
from sklearn.metrics import roc_curve, roc_auc_score

from signaturesnet.models.distance_classifier import DistanceBasedClassifier
from signaturesnet.utilities.plotting import (
    plot_box,
    plot_roc,
    plot_histogram,
)
from signaturesnet.utilities.metrics import binary_classification_metrics


def find_threshold(labels, guesses):
    fpr, tpr, thresh = roc_curve(labels.detach().numpy(), guesses.detach().numpy(), pos_label=1)
    best_threshold = 0
    best_f1 = 0
    for i in range(len(fpr)):
        precision = tpr[i] / (tpr[i] + fpr[i]) if (tpr[i] + fpr[i]) > 0 else 0
        recall = tpr[i]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh[i]
    return best_threshold, best_f1


if __name__ == "__main__":
    from signaturesnet.utilities.io import read_data_classifier

    # Load reference and validation data
    reference_data, val_data = read_data_classifier(
        device="cuda",  # put "cpu" here if you don't have a GPU
        experiment_id="datasets/detector",
    )

    # Instantiate the classifier
    reference_mutations = reference_data.inputs[(reference_data.labels == 1).squeeze(-1)]
    classifier = DistanceBasedClassifier(reference_mutations)


    # Validation data
    val_inputs, val_nummuts = val_data.inputs, val_data.num_mut
    val_labels = val_data.labels.squeeze(-1)
    distances, _ = classifier.forward(val_inputs, num_mut=val_nummuts)
    classifier_logits = 1 - distances

    n_mut_class = torch.log(val_nummuts).to(torch.int)
    classes = torch.unique(n_mut_class)
    
    thresholds = {}
    for c in classes:
        print(f"Class {c}")
        c_pos = (n_mut_class == c).squeeze(-1)
        c_labels = val_labels[c_pos].flatten()
        c_logits = classifier_logits[c_pos].flatten()
        
        # plot_box(labels=c_labels, values=c_logits)
        plot_roc(labels=c_labels.cpu(), guesses=c_logits.cpu(), name="n~=" + str(int(torch.exp(c).item())))

        thresh, f1 = find_threshold(labels=c_labels.cpu(), guesses=c_logits.cpu())
        print(f"Threshold: {thresh}, F1: {f1}")
        thresholds[c.item()] = thresh
    
    pprint(thresholds)
    print("DONE")
