"""
Implements a One-Class SVM classifier using PyTorch.

Original paper: https://papers.nips.cc/paper_files/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf
I implement the dual formulation.
"""

import torch
import torch.optim as optim

# RBF Kernel Function
def rbf_kernel(X1, X2, gamma=0.5):
    dist_matrix = torch.cdist(X1, X2, p=2)
    return torch.exp(-gamma * dist_matrix ** 2)

class OneClassSVM(torch.nn.Module):
    def __init__(self, X, kernel_func=rbf_kernel, nu=0.5):
        """
        Initialize the One-Class SVM model.

        Args:
        - X: The training data (tensor of shape [num_samples, num_features]).
        - kernel_func: The kernel function to use (e.g., RBF kernel).
        - nu: The parameter nu controlling the number of support vectors (0 < nu <= 1).
        """
        # x = X.clone().detach().cpu().numpy()  # TODO: Think of a way to make this serializable
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("X")
        self.init_args.pop("kernel_func")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "OneClassSVM"
        
        super(OneClassSVM, self).__init__()
        self.X = X  # TODO: We probably need some kind of projection here
        self.kernel_func = kernel_func
        self.nu = nu
        self.num_samples = X.shape[0]
        self.alpha = torch.nn.Parameter(torch.rand(self.num_samples))  # Initialize alpha
        self.rho = None  # To be computed after training

    def forward(self, lagrange_multiplier=1.0):
        # Calculate the kernel matrix
        K = self.kernel_func(self.X, self.X)
        # Dual objective function
        objective = 0.5 * torch.sum(self.alpha @ K @ self.alpha)

        # Constraints
        alpha_sum_constraint = (torch.sum(self.alpha) - 1) ** 2
        alpha_positive_constrain = torch.sum(torch.relu(-self.alpha))
        alpha_upper_limit = 1/(self.nu * self.num_samples)
        alpha_small_constrain = torch.sum(torch.relu(self.alpha - alpha_upper_limit))

        # Combine constraints into a penalty
        constraint_loss = alpha_sum_constraint + alpha_positive_constrain + alpha_small_constrain

        loss = objective + lagrange_multiplier * constraint_loss
        return {
            "loss": loss,
            "alpha_sum_constraint": alpha_sum_constraint,
            "alpha_positive_constrain": alpha_positive_constrain,
            "alpha_small_constrain": alpha_small_constrain,
        }

    @torch.no_grad()
    def compute_rho(self):
        # Compute the kernel matrix
        K = self.kernel_func(self.X, self.X)

        # Identify support vectors (where alpha is non-zero within a small tolerance)
        support_vector_indices = (self.alpha > 1e-5).nonzero(as_tuple=True)[0]
        support_alphas = self.alpha[support_vector_indices]
        support_vectors = self.X[support_vector_indices]

        # Compute rho using all support vectors
        rho_values = []
        for i, sv in enumerate(support_vectors):
            rho_value = torch.sum(support_alphas * K[support_vector_indices, i])
            rho_values.append(rho_value.item())

        # Average rho values
        self.rho = sum(rho_values) / len(rho_values)

    @torch.no_grad()
    def predict(self, X_new):
        """
        Return the decision values for new data points X_new.
        
        If the datapoint is positive, it is in-distribution.
        """
        
        # Ensure rho is computed
        if self.rho is None:
            self.compute_rho()

        # Compute the kernel between the training data (support vectors) and the new data point(s)
        K = self.kernel_func(self.X, X_new)

        # Compute decision values
        decision_values = torch.matmul(K.T, self.alpha) - self.rho

        return decision_values