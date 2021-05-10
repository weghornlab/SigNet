import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

def plot_signature(signature, labels):
    plt.bar(range(96), signature, tick_label=labels)
    plt.xticks(rotation=90)
    plt.show()


def plot_confusion_matrix(label_list, predicted_list, class_names):
    conf_mat = confusion_matrix(label_list.numpy(), predicted_list.numpy())
    plt.figure(figsize=(15, 10))

    df_cm = pd.DataFrame(conf_mat, index=class_names,
                         columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return conf_mat

def plot_weights(guessed_labels, guessed_error_pos, guessed_error_neg, sigs_names):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    ax.bar(range(num_classes),guessed_labels, yerr=[abs(guessed_error_neg), abs(guessed_error_pos)], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.tight_layout()
    plt.show()

def plot_weights_comparison(true_labels, guessed_labels, guessed_error_pos, guessed_error_neg, sigs_names):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    ax.bar(range(num_classes),guessed_labels, yerr=[abs(guessed_error_neg), abs(guessed_error_pos)], align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    ax.bar(np.array(range(num_classes))+0.2, true_labels, width=0.2, align='center')
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.tight_layout()
    plt.show()

def plot_weights_comparison_deconstructSigs(true_labels, deconstructSigs_labels, guessed_labels, guessed_error_pos, guessed_error_neg, sigs_names):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    ax.bar(range(num_classes),guessed_labels, yerr=[abs(guessed_error_neg), abs(guessed_error_pos)], align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    ax.bar(np.array(range(num_classes))+0.2, true_labels, width=0.2, align='center')
    ax.bar(np.array(range(num_classes))-0.2,deconstructSigs_labels, width=0.2, align='center')
    ax.axhline(0.05, 0, num_classes, linestyle='--', c='red')
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.tight_layout()
    plt.show()
     
def plot_interval_performance(label_batch, train_weight_guess, prediction_pos, prediction_neg, sigs_names):
    lower_bound = train_weight_guess - abs(prediction_neg)
    upper_bound = train_weight_guess + abs(prediction_pos)
    lower = label_batch - lower_bound
    upper = prediction_pos - label_batch
    num_error = torch.sum(lower<0, dim=0)
    num_error += torch.sum(upper<0, dim=0)
    num_error = num_error / label_batch.shape[0]
    num_classes = 72
    plt.bar(range(num_classes), 100*num_error, align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel("Percentage of error (%)")
    plt.xticks(range(num_classes), sigs_names, rotation='vertical')
    plt.title('Confidence intervals performance')
    plt.show()
        
if __name__ == "__main__":
    deconstructSigs_labels = [0.1, 0.7, 0.2]
    real_labels = [0.2, 0.5, 0.3]
    guessed_labels = [0.25, 0.6, 0.2]
    guessed_error = [0.01, 0.04, 0.001]
    sigs_names = ["SBS1", "SBS2", "SBS3"]

    plot_weights_comparison_deconstructSigs(real_labels, deconstructSigs_labels, guessed_labels, guessed_error, sigs_names)