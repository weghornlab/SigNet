from pprint import pprint

from signaturesnet.models.distance_classifier import DistanceBasedClassifier
from signaturesnet.utilities.plotting import (
    plot_box,
    plot_roc,
    plot_histogram,
)
from signaturesnet.utilities.metrics import binary_classification_metrics


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

    classifier_distances, classifier_guess = classifier.forward(val_inputs, num_mut=val_nummuts)
    
    metrics = binary_classification_metrics(
        predictions=classifier_guess,
        labels=val_labels
    )
    pprint(metrics)
    print("DONE")
