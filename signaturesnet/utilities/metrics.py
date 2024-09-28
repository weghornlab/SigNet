import copy

import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn


# USED IN CLASSIFIER
def accuracy(prediction, label):
    assert(prediction.shape == label.shape)
    assert(prediction.dtype == torch.int64)
    assert(label.dtype == torch.int64)
    return (torch.sum(prediction == label).float()/torch.numel(prediction))*100.

def false_realistic(prediction, label):
    assert(prediction.shape == label.shape)
    assert(prediction.dtype == torch.int64)
    # assert(label.dtype == torch.int64)
    # print(label[prediction == 1])
    # print(torch.sum(label[prediction == 1] == 0))
    return torch.true_divide(torch.sum(label[prediction == 1] == 0), (torch.numel(label)))*100 

def false_random(prediction, label):
    assert(prediction.shape == label.shape)
    assert(prediction.dtype == torch.int64)
    # assert(label.dtype == torch.int64)
    return torch.true_divide(torch.sum(label[prediction == 0] == 1), (torch.numel(label)))*100 

# USED IN FINE TUNER
def get_MSE(predicted_label, true_label):
    return torch.nn.MSELoss()(predicted_label, true_label)

def get_cosine_similarity(predicted_label, true_label, dim=None):
    predicted_label = torch.nn.functional.normalize(
        predicted_label, dim=1, p=2)
    true_label = torch.nn.functional.normalize(true_label, dim=1, p=2)
    if dim is not None:
        return torch.abs(torch.einsum("ij,ij->i", (predicted_label, true_label)))
    return torch.mean(torch.abs(torch.einsum("ij,ij->i", (predicted_label, true_label))))

def get_negative_cosine_similarity(predicted_label, true_label):
    return -get_cosine_similarity(predicted_label, true_label)

def get_entropy(predicted_label):
    entropy = torch.mean(torch.distributions.categorical.Categorical(
        probs=predicted_label.cpu().detach()).entropy())
    return entropy

def get_cross_entropy2(predicted_label, true_label):
    #predicted_label_local = copy.deepcopy(predicted_label)
    #predicted_label_local += 1e-6
    return torch.mean(-torch.einsum("ij,ij->i", (true_label, torch.log(predicted_label))))

def get_kl_divergence(predicted_label, true_label):
    _EPS = 1e-9
    predicted_label = predicted_label + _EPS
    return get_cross_entropy2(predicted_label, true_label) - get_entropy(true_label)

def get_jensen_shannon(predicted_label, true_label):
    _EPS = 1e-6
    r = (predicted_label + true_label)/2 + _EPS
    term1 = torch.mean(torch.einsum("ij,ij->i",(predicted_label, torch.log(torch.div(predicted_label + _EPS, r)))))
    term2 = torch.mean(torch.einsum("ij,ij->i",(true_label, torch.log(torch.div(true_label + _EPS, r)))))
    return 0.5 * (term1 + term2) 

def get_wasserstein_distance(predicted_label, true_label):
    dist = 0
    for i in range(predicted_label.shape[0]):
        dist += wasserstein_distance(
            predicted_label[i, :].cpu().detach().numpy(), true_label[i, :].cpu().detach().numpy())
    return torch.from_numpy(np.array(dist/predicted_label.shape[0]))

def get_classification_metrics(label_batch, prediction_batch, cutoff=0.01):
    batch_size = float(label_batch.shape[0])
    kl = get_kl_divergence(prediction_batch, label_batch)
    label_mask = (label_batch > cutoff).type(torch.int).float()
    prediction_mask = (prediction_batch > cutoff).type(torch.int).float()
    fp = torch.sum(label_mask - prediction_mask < -0.1)
    fn = torch.sum(label_mask - prediction_mask > 0.1)
    tp = torch.sum(torch.einsum("bi,bi->b", prediction_mask, label_mask))
    tn = torch.sum(torch.einsum("bi,bi->b", 1 - prediction_mask, 1 - label_mask))
    fpr = fp/(tn + fp)
    fnr = fn/(tp + fn)
    sensitivity = tp/torch.sum(label_mask)
    specificity = tn/torch.sum(1 - label_mask)
    accuracy = (tp + tn)/(batch_size*label_batch.shape[1])
    precision = tp / (tp + fp)
    mae = torch.abs(label_batch - prediction_batch)
    mae_by_sign = torch.mean(torch.abs(label_batch - prediction_batch), dim=0)
    MAE_p = torch.sum(torch.einsum("bi,bi->b", mae, label_mask))/(tp + fn)
    MAE_n = torch.sum(torch.einsum("bi,bi->b", mae, 1 - label_mask))/(tn + fp)
    # Q95_p = torch.quantile(mae[label])
    return {"fpr": fpr*100, "fnr": fnr*100,
            "sens: tp/p %": sensitivity*100., "spec: tn/n %": specificity*100., "accuracy %": accuracy*100., "precision %": precision*100.,
            "MAE_p": MAE_p, "MAE_n": MAE_n, "MAE":mae.mean(), "KL":kl, "MAE_sign":mae_by_sign.detach()}

def get_fp_fn_soft(label_batch, prediction_batch, cutoff=0.05, softness=100):
    label_mask = nn.Sigmoid()((label_batch - cutoff)*softness)  # ~0 if under cutoff, ~1 if over cutoff (element-wise)
    prediction_mask = nn.Sigmoid()((prediction_batch - cutoff)*softness)  # ~0 if under cutoff, ~1 if over cutoff (element-wise)
    fp = torch.sum(nn.ReLU()(prediction_mask - label_mask))  # only count when pred ~= 1 and label ~= 0
    fn = torch.sum(nn.ReLU()(label_mask - prediction_mask))  # only count when label ~= 1 and pred ~= 0
    return fp, fn

def probs_batch_to_sigs(label_batch, prediction_batch, cutoff=0.05, num_classes=72):
    label_sigs_list = torch.zeros(0, dtype=torch.long)
    predicted_sigs_list = torch.zeros(0, dtype=torch.long)
    for i in range(len(label_batch)):
        for j in range(len(label_batch[i])):
            if label_batch[i][j] > cutoff and prediction_batch[i][j] > cutoff:
                label_sigs_list = torch.cat(
                    [label_sigs_list, torch.from_numpy(np.array([j]))])
                predicted_sigs_list = torch.cat(
                    [predicted_sigs_list, torch.from_numpy(np.array([j]))])
            if label_batch[i][j] > cutoff and prediction_batch[i][j] < cutoff:
                label_sigs_list = torch.cat(
                    [label_sigs_list, torch.from_numpy(np.array([j]))])
                predicted_sigs_list = torch.cat(
                    [predicted_sigs_list, torch.from_numpy(np.array([num_classes]))])
            if label_batch[i][j] < cutoff and prediction_batch[i][j] > cutoff:
                label_sigs_list = torch.cat(
                    [label_sigs_list, torch.from_numpy(np.array([num_classes]))])
                predicted_sigs_list = torch.cat(
                    [predicted_sigs_list, torch.from_numpy(np.array([j]))])
    return label_sigs_list, predicted_sigs_list

# USED IN ERROR FINDER 
def distance_to_interval(label, pred_lower, pred_upper):
    batch_size = float(pred_lower.shape[0])
    lower = label - pred_lower
    upper = pred_upper - label
    lower = nn.ReLU()(-lower)
    upper = nn.ReLU()(-upper)
    dist = lower + upper
    return dist

def interval_width(pred_lower, pred_upper):
    return abs(pred_upper-pred_lower)

def get_pi_metrics_by_sig(label, pred_lower, pred_upper, dim=0):
    """Used to compare prediction interval guesses.
    
    Returns:
        in_prop [float]: Proportion of labels in (pred_lower, pred_upper)
        mean_interval_width [float]: Mean width of the intervals
    """
    EPS = 1e-6
    result = {}
    k_hu = (label <= pred_upper).type(torch.float)  # 1 if label < upper; else 0
    k_hl = (pred_lower <= label).type(torch.float)  # 1 if lower < label; else 0
    k_h = torch.einsum("be,be->be", k_hl, k_hu)  # 1 if label in (lower, upper) else 0
    result["in_prop"] = torch.mean(k_h, dim)  # Hard Prediction Interval Coverage Probability
    result["in_prop_present"] = torch.masked_select(k_h, label > 0.5).mean()  # This flattens all values into a 1-d tensor
    result["in_prop_absent"] = torch.masked_select(k_h, label <= 0.5).mean()

    pred_upper[pred_upper!=pred_upper] = 0
    pred_lower[pred_lower!=pred_lower] = 0
    interval_width = torch.max(torch.zeros_like(label), pred_upper - pred_lower)
    result["mean_interval_width"] = torch.mean(interval_width, dim)
    interval_width_present = torch.masked_select(interval_width, label > EPS)
    interval_width_absent = torch.masked_select(interval_width, label <= EPS)
    result["mean_interval_width_present"] = interval_width_present.mean()
    result["mean_interval_width_absent"] = interval_width_absent.mean()
    return result

def get_pi_metrics(label, pred_lower, pred_upper, collapse=True, dim=0):
    metrics_by_signature = get_pi_metrics_by_sig(label, pred_lower, pred_upper, dim=dim)
    if collapse:
        for key in metrics_by_signature:
            metrics_by_signature[key] = metrics_by_signature[key].mean()
    return metrics_by_signature

def get_soft_qd_loss(label, pred_lower, pred_upper, conf=0.01, lagrange_mult=1e-4, softening_factor=500.0):
    """Used to optimize:
    
    min pred_upper - pred_lower
    s.t. p(label in (pred_lower, pred_upper)) >= 1 - conf

    Algorithm 1 of https://arxiv.org/pdf/1802.07167.pdf

    conf [float]: Interval confidence, proportion of missplaced points which is "ok" usually ~0.05
    lagrange_mult [float]: "How much do we care about missclassifications"
                            Larger lagrange_mult means more correctly guessed points at the expense of larger intervals
    softening_factor [float]: The bigger the closer it is to the real function
                              This means better guesses but harder optimization (less differentiable)
    """
    EPS_ = 1e-9
    EPS2_ = 1e-3

    # Hard in-between constrain
    with torch.no_grad():
        k_hu = (label <= pred_upper).type(torch.float)  # 1 if label <= upper; else 0
        k_hl = (pred_lower <= label).type(torch.float)  # 1 if lower <= label; else 0
        k_h = torch.einsum("be,be->be", k_hl, k_hu)  # 1 if label in (lower, upper) else 0
        PICP_h = torch.mean(k_h)  # Prediction Interval Coverage Probability

    # Softened in-between constrain (same as before but differentiable)
    k_su = nn.Sigmoid()((pred_upper - label + EPS2_)*softening_factor)
    k_sl = nn.Sigmoid()((label - pred_lower + EPS2_)*softening_factor)
    k_s = torch.einsum("be,be->be", k_sl, k_su)
    PICP_s = torch.mean(k_s)  # Soft Prediction Interval Coverage Probability

    # Compute Mean Prediction Interval Width (MPIW)
    MPIW = torch.sum(torch.einsum("be,be->be", (pred_upper - pred_lower), k_h))/(torch.sum(k_h) + EPS_)

    # Compute constrain
    n = float(torch.numel(label))  # number elements in the input
    constrain = (n/(conf*(1.0 - conf)))*torch.max(torch.tensor(0).to(PICP_s), (1.0 - conf) - PICP_s)**2
    
    # Compute and return loss
    loss = MPIW + lagrange_mult*constrain
    return loss, PICP_h, MPIW


def get_entropy(data):
    """Return average entropy across 
    """
    _EPS = 1e-9
    return -torch.sum(data * torch.log(data + _EPS))/data.shape[0]

def get_std(data):
    """Return average std across 
    """
    return torch.mean(torch.std(data, axis=1))

def get_present_sigs(data):
    """Return average std across 
    """
    return torch.mean(torch.count(data, axis=1))

def get_reconstruction_error(mutation_dist, guess, signatures):
    reconstruction = torch.einsum("ij,bj->bi", (signatures, guess))
    errors = torch.sum(torch.nn.MSELoss(reduction='none')(reconstruction, mutation_dist), dim=1)
    return errors

def sets_distances(real, fake):
    """Distances between points of two sets
    """
    def min_dist(point_set, point):
        """Minimum distance between a set and a point
        """
        return torch.sqrt(((point_set - point).pow(2)).mean(dim=1).min())
    with torch.no_grad():
        real_distances = torch.tensor([min_dist(fake, p) for p in real])
        fake_distances = torch.tensor([min_dist(real, p) for p in fake])
    return real_distances, fake_distances

def prop_distances(real, fake):
    """Distances between the proportion of signatures in the real and fake samples
    """
    real_changed = real.detach().clone()
    fake_changed = fake.detach().clone()

    real_changed[real_changed<0.01] = 0
    real_changed[real_changed>=0.01] = 1
    prop_tumors_real = torch.sum(real_changed, dim=0)/real_changed.shape[0]

    fake_changed[fake_changed<0.01] = 0
    fake_changed[fake_changed>=0.01] = 1
    prop_tumors_fake = torch.sum(fake_changed, dim=0)/fake_changed.shape[0]
        
    se_prop = ((prop_tumors_fake-prop_tumors_real)**2).detach().numpy()
    mse_prop = np.mean(se_prop)
    return se_prop, mse_prop

def get_distances_metrics(distances, quantiles=[0.99]):
    metrics = {}
    metrics["mean"] = np.format_float_scientific(torch.mean(distances).item(), precision = 2, exp_digits=1)
    metrics["median"] = np.format_float_scientific(torch.median(distances).item(), precision = 2, exp_digits=1)
    metrics["max"] = np.format_float_scientific(torch.max(distances).item(), precision = 2, exp_digits=1)
    quantiles = torch.quantile(distances, torch.tensor(quantiles), keepdim=True)
    metrics["quantiles"] = np.round(quantiles.detach().cpu().numpy(), 3)
    return metrics

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

METRICS_DICT = {
    "mse" : get_MSE,
    "cos" : get_negative_cosine_similarity,
    "cross_ent" : get_cross_entropy2,
    "KL" : get_kl_divergence,
    "JS" : get_jensen_shannon,
    "W" : get_wasserstein_distance,
}

if __name__ == "__main__":
    torch.seed = 0

    losses = []
    mpiws = []
    picps = []
    lower = torch.tensor([[0.1]])
    label = torch.tensor([[0.25]])
    u_s = [u/100. for u in range(100)]
    for u in u_s:
        upper = torch.tensor([[u]])

        # in_prop, mean_interval_width = get_pi_metrics(label, lower, upper)
        # print("in_prop", in_prop)
        # print("mean_interval_width", mean_interval_width)

        # loss, picp, mpiw = get_soft_qd_loss(label, lower, upper, conf=0.05, lagrange_mult=0.5, softening_factor=10)
        loss, picp, mpiw = distance_to_interval(label, lower, upper, lagrange_mult=1.0)
        # print("loss", loss)
        losses.append(loss)
        picps.append(picp)
        mpiws.append(mpiw)

    import matplotlib.pyplot as plt
    plt.plot(u_s, losses, label="loss")
    plt.plot(u_s, picps, label="in_prop")
    plt.plot(u_s, mpiws, label="width")
    plt.legend()
    plt.show()