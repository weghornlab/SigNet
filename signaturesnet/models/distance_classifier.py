import torch

class DistanceBasedClassifier(torch.nn.Module):
    def __init__(self, reference_mutations):
        super(DistanceBasedClassifier, self).__init__()
        self.reference_mutations = reference_mutations


    @torch.no_grad()
    def forward(self, mutation_dist, num_mut=None, p=5, k=5):
        """1 means in-distribution, 0 means out-of-distribution
        """
        distances = torch.cdist(mutation_dist, self.reference_mutations, p=p)
        
        values, _ = torch.topk(distances, k=5, largest=False, dim=1)
        dist = values.mean(dim=1) * 10

        # min_dist = torch.min(distances, dim=1).values * 10
        classes = self._classify(dist, num_mut)
        return dist, classes

    def _classify(self, dists, num_mut):
        # These were found empirically through a validation set
        thresholds = {
            2: -0.23375274240970612,
            3: 0.09172570705413818,
            4: 0.4452713131904602,
            5: 0.6428919434547424,
            6: 0.7619860172271729,
            7: 0.8058968186378479,
            8: 0.8330588936805725,
            9: 0.8443573117256165,
            10: 0.8582318425178528,
            11: 0.8613357543945312
        }
        default_threshold = 0.82
        n_mut_class = torch.log(num_mut).to(torch.int)
        
        logits = 1 - dists
        
        is_in_distro = torch.zeros_like(n_mut_class)

        acc_pos_sum, acc_indist_sum = 0, 0

        for c in torch.unique(n_mut_class):
            threshold = thresholds.get(c.item(), default_threshold)
            
            positions = (n_mut_class == c).squeeze(-1)
            class_logits = logits[positions]
            class_indistro = (class_logits > threshold).to(torch.int)
            is_in_distro[positions] = class_indistro.unsqueeze(-1)
            
            acc_pos_sum += torch.sum(positions)
            acc_indist_sum += torch.sum(class_indistro)

        return is_in_distro.flatten()