import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid


class FineTuner(nn.Module):
    def __init__(self,
                 num_classes=72,
                 num_hidden_layers=2,
                 num_units=400,
                 cutoff=0.01,
                 sigmoid_params=[[500, 1000], [5000, 2000], [10000, 5000]]):
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self._cutoff = cutoff
        self._EPS = 1e-6
        self.sigmoid_params = sigmoid_params
        self.num_classes = num_classes

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _apply_cutoff(self, comb):
        mask = (comb > self._cutoff).type(torch.int).float()
        comb = comb*mask
        comb = torch.div(comb, torch.sum(
        comb, axis=1).reshape((-1, 1)) + self._EPS)
        return comb

class FineTunerLowNumMut(FineTuner):

    def __init__(self,
                 num_classes=72,
                 num_hidden_layers=2,
                 num_units=400,
                 cutoff=0.001,
                 sigmoid_params=[5000, 2000]):
        super(FineTunerLowNumMut, self).__init__(num_classes=num_classes,
                                                 num_hidden_layers=num_hidden_layers,
                                                 num_units=num_units,
                                                 cutoff=cutoff,
                                                 sigmoid_params=sigmoid_params)
        self.init_args["model_type"] = "FineTunerLowNumMut"
        print(self.init_args)

        # Num units of the mutations path
        num_units_branch_mut = 3
        num_units_joined_path = num_units + 3*num_units_branch_mut

        # Mutvec path
        self.layer_mutvec_1 = nn.Linear(96, num_units)
        self.layer_mutvec_2 = nn.Linear(num_units, num_units)

        # Nummut path
        self.layer_numut_low = nn.Linear(1, num_units_branch_mut)
        self.layer_numut_mid = nn.Linear(1, num_units_branch_mut)
        self.layer_numut_large = nn.Linear(1, num_units_branch_mut)
        self.layer_numut_joint = nn.Linear(
            3*num_units_branch_mut, 3*num_units_branch_mut)

        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units_joined_path, num_units_joined_path)
                     for _ in range(num_hidden_layers)])

        self.output_layer = nn.Linear(num_units_joined_path, num_classes)
        self.activation = nn.LeakyReLU(0.1)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self,
                mutation_dist,
                num_mut):
        # Input head
        mutation_dist = self.activation(self.layer_mutvec_1(mutation_dist))
        mutation_dist = self.activation(self.layer_mutvec_2(mutation_dist))

        # Number of mutations head
        num_mut_low = nn.Sigmoid()((num_mut - self.sigmoid_params[0][0]) / self.sigmoid_params[0][1])
        num_mut_mid = nn.Sigmoid()((num_mut - self.sigmoid_params[1][0]) / self.sigmoid_params[1][1])
        num_mut_large = nn.Sigmoid()((num_mut - self.sigmoid_params[2][0]) / self.sigmoid_params[2][1])
        num_mut_low = self.activation(self.layer_numut_low(num_mut_low))
        num_mut_mid = self.activation(self.layer_numut_mid(num_mut_mid))
        num_mut_large = self.activation(self.layer_numut_large(num_mut_large))
        num_mut = torch.cat([num_mut_low, num_mut_mid, num_mut_large], dim=1)
        num_mut = self.activation(self.layer_numut_joint(num_mut))

        # Concatenate
        comb = torch.cat([mutation_dist, num_mut], dim=1)
        assert(not torch.isnan(comb).any())

        # Apply shared layers
        for layer in self.hidden_layers:
            comb = self.activation(layer(self.dropout(comb)))

        # Apply output layer
        comb = self.output_layer(comb)
        # comb += baseline_guess
        comb = self.softmax(comb)

        

        # If in eval mode, send small values to 0
        if not self.training:
            comb = self._apply_cutoff(comb)
        return comb


class FineTunerLargeNumMut(FineTuner, nn.Module):

    def __init__(self,
                 num_classes=72,
                 num_hidden_layers=2,
                 num_units=400,
                 cutoff=0.001,
                 sigmoid_params=[5000, 2000]):
        super(FineTunerLargeNumMut, self).__init__(num_classes=num_classes,
                                                 num_hidden_layers=num_hidden_layers,
                                                 num_units=num_units,
                                                 cutoff=cutoff,
                                                 sigmoid_params=sigmoid_params)
        self.init_args["model_type"] = "FineTunerLargeNumMut"

        self.input_layer = nn.Linear(96+num_classes+3, num_units)  # Baseline guess path
        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units, num_units)
                     for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(num_units, num_classes)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self,
                mutation_dist,
                baseline_guess,
                num_mut):
        # Number of mutations head
        num_mut_1 = torch.log10(num_mut)/6
        num_mut_2 = nn.Sigmoid()(
            (num_mut-self.sigmoid_params[0])/self.sigmoid_params[1])
        num_mut_3 = nn.Sigmoid()((num_mut-10000)/200)

        # Concatenate
        comb = torch.cat([mutation_dist, baseline_guess,
                          num_mut_1, num_mut_2, num_mut_3], dim=1)
        comb = self.activation(self.input_layer(comb))

        # Apply shared layers
        for layer in self.hidden_layers:
            comb = self.activation(layer(comb))

        # Apply output layer
        comb = self.output_layer(comb)
        comb = comb + baseline_guess
        comb = nn.ReLU()(comb)
        comb = comb/torch.sum(comb, dim=1).reshape(mutation_dist.shape[0], 1)

        # If in eval mode, send small values to 0
        if not self.training:
            comb = self._apply_cutoff(comb)
        return comb


def baseline_guess_to_finetuner_guess(trained_finetuner_dir, data):
    # Load finetuner and compute guess_1
    import gc
    from utilities.io import read_model

    finetuner = read_model(directory=trained_finetuner_dir)
    finetuner.to("cpu")

    with torch.no_grad():
        data.prev_guess = finetuner(mutation_dist=data.inputs,
                                    baseline_guess=data.prev_guess,
                                    num_mut=data.num_mut)
    del finetuner
    gc.collect()
    torch.cuda.empty_cache()
    return data
