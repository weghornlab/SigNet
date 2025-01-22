import os
import logging
import pathlib

import pandas as pd
import numpy as np
import torch

from signaturesnet import DATA, TRAINED_MODELS
from signaturesnet.utilities.normalize_data import normalize_data
from signaturesnet.utilities.io import read_model, read_signatures
from signaturesnet.models import Baseline
from signaturesnet.modules import CombinedFinetuner, ClassifiedFinetunerErrorfinder
from signaturesnet.utilities.plotting import plot_weights
from signaturesnet.modules.euc_classifier import euc_classifier

class SigNet:

    def __init__(self,
                 # classifier=os.path.join(TRAINED_MODELS, "detector"),
                 finetuner_realistic_low=os.path.join(TRAINED_MODELS, "finetuner_low"),
                 finetuner_realistic_large=os.path.join(TRAINED_MODELS, "finetuner_large"),
                 errorfinder=os.path.join(TRAINED_MODELS, "errorfinder"),
                 opportunities_name_or_path=None,
                 signatures_path=os.path.join(DATA, "data.xlsx"),
                 mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx")):

        signatures = read_signatures(file=signatures_path,
                                     mutation_type_order=mutation_type_order)
        self.signatures = signatures # TODO(oleguer): Remove, this is only for debugging
        self.baseline = Baseline(signatures)
        finetuner = CombinedFinetuner(low_mum_mut_dir=finetuner_realistic_low,
                                      large_mum_mut_dir=finetuner_realistic_large)

        classifier = euc_classifier
        errorfinder = read_model(errorfinder)

        self.finetuner_errorfinder = ClassifiedFinetunerErrorfinder(classifier=classifier,
                                                                    finetuner=finetuner,
                                                                    errorfinder=errorfinder)
        self.opportunities_name_or_path = opportunities_name_or_path\
            if opportunities_name_or_path != 'None' else None
        logging.info("SigNet loaded!")

    def __call__(self,
                 mutation_dataset,
                 numpy=True,
                 only_NNLS=False,
                 nworkers=1,
                 cutoff = 0.01):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_dataset (pd.DataFrame(batch_size, 96)): Batch of mutation vectors to decompose
            numpy (bool): Whether to convert the outputs into numpy arrays (otherwise it'd be torch.Tensor). Default: True
            nworkers (int): Num of threads to run the

        Returns:
            results (dict)
        """
        with torch.no_grad():
            # Sort input data columns
            mutation_order = pd.read_excel(os.path.join(DATA, "mutation_type_order.xlsx"))
            mutation_dataset = mutation_dataset[list(mutation_order['Type'])]
            sample_names = mutation_dataset.index

            mutation_vec = torch.tensor(mutation_dataset.values.astype(np.float32), dtype=torch.float, device='cpu')
            num_mutations = torch.sum(mutation_vec, dim=1)

            # Normalize input data
            if self.opportunities_name_or_path is not None:
                mutation_vec = normalize_data(mutation_vec, self.opportunities_name_or_path)

            sums = torch.sum(mutation_vec, dim=1).reshape(-1, 1)
            normalized_mutation_vec = mutation_vec / sums
  
            # Run NNLS
            logging.info("Obtaining NNLS guesses...")
            self.baseline_guess = self.baseline.get_weights_batch(input_batch=normalized_mutation_vec, 
                                                                  n_workers=nworkers)
            logging.info("Obtaining NNLS guesses... DONE")

            if only_NNLS:
                result = SigNetResult(mutation_dataset,
                                  weights=self.baseline_guess,
                                  lower=torch.full((mutation_dataset.shape[0],72), float('nan')),
                                  upper=torch.full((mutation_dataset.shape[0],72), float('nan')),
                                  classification=torch.full((mutation_dataset.shape[0],1), float('nan')),
                                  normalized_input=normalized_mutation_vec)
                return result

            # Finetune guess and aproximate errors
            signet_res = self.finetuner_errorfinder(mutation_dist=normalized_mutation_vec,
                                                    baseline_guess=self.baseline_guess,
                                                    num_mut=num_mutations.reshape(-1, 1),
                                                    cutoff=cutoff)
            result = SigNetResult(mutation_dataset,
                                  weights=signet_res["finetuner_guess"],
                                  lower=signet_res["error_lower"],
                                  upper=signet_res["error_upper"],
                                  classification=signet_res["classification"],
                                  normalized_input=normalized_mutation_vec)
            
            logging.info("Success: SigNet result obtained!")
        return result

class SigNetResult:

    def __init__(self,
                 mutation_dataset,
                 weights,
                 lower,
                 upper,
                 classification,
                 normalized_input):
        self.mutation_dataset = mutation_dataset
        self.weights = weights
        self.lower = lower
        self.upper = upper
        self.classification = classification
        self.normalized_input = normalized_input
        self.sig_names = list(pd.read_excel(os.path.join(DATA, "data.xlsx")).columns)[1:]

    def get_output(self, format="numpy"):
        """ 
        Obtain the predicted outputs in one of these formats: ["numpy", "pandas", "tensor"]
        Args:
            format: one of: "numpy", "pandas", "tensor"
        Returns:
            weights, lower bound, upper bound, classification, normalized_input in the given format.
        """
        assert format in ["numpy", "pandas", "tensor"]
        if format == "numpy":
            weights = self.weights.detach().numpy()
            lower = self.lower.detach().numpy()
            upper = self.upper.detach().numpy()
            classification = self.classification.detach().numpy()
            normalized_input = self.normalized_input.detach().numpy()
        if format == "pandas":
            weights = pd.DataFrame(self.weights.detach().numpy())
            lower = pd.DataFrame(self.lower.detach().numpy())
            upper = pd.DataFrame(self.upper.detach().numpy())
            classification = pd.DataFrame(self.classification.detach().numpy())
            normalized_input = pd.DataFrame(self.normalized_input.detach().numpy())
        if format == "tensor":
            return self.weights, self.lower, self.upper, self.classification, self.normalized_input
        return weights, lower, upper, classification, normalized_input

    def save(self, 
             path='Output', 
             name='signet'):
        """ 
        Save outputs into a file.
        Args:
            path (str): path to the directory where the files will be written.
        """
        logging.info("Writting results: %s..."%path)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        weights, lower, upper, classification, _ = self.get_output(format="pandas")

        try:
            weights.columns = self.sig_names + ['Unknown']
            weights.index = self.mutation_dataset.index
            weights.to_csv(path + "/weight_guesses-"+name+".csv", header=True, index=True)
        except:
            weights.columns = self.sig_names
            weights.index = self.mutation_dataset.index
            weights.to_csv(path + "/weight_guesses-"+name+".csv", header=True, index=True)

        lower.columns = self.sig_names
        lower.index = self.mutation_dataset.index
        lower.to_csv(path + "/lower_bound_guesses-"+name+".csv", header=True, index=True)
    
        upper.columns = self.sig_names
        upper.index = self.mutation_dataset.index
        upper.to_csv(path + "/upper_bound_guesses-"+name+".csv", header=True, index=True)
    
        classification.columns = ['Classification']
        classification.index = self.mutation_dataset.index
        classification.to_csv(path + "/classification_guesses-"+name+".csv", header=True, index=True)

        self.mutation_dataset.to_csv(path + "/mutation_counts_input-"+name+".csv", header=True, index=True)
        logging.info("Writting results: %s... DONE"%path)

    def plot_results(self, 
                     compute = 'True',
                     save = True,
                     path = 'Output/plots'):
        """ 
        Shows and saves (if applicable) the plots of the signature decompositions.
        Args:
            path (str): path to the directory where the plots will be saved.
            save (bool): whether to save the plot into a file or not.
        """
        if compute == 'True':
            logging.info("Plotting results: %s..."%path)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            samples = list(self.mutation_dataset.index)
            for i in range(self.weights.shape[0]):
                plot_weights(guessed_labels=self.weights[i,:72], 
                            pred_upper=self.upper[i,:], 
                            pred_lower=self.lower[i,:], 
                            sigs_names=self.sig_names, 
                            save=save, 
                            plot_path=path + "/plot_%s.png"%samples[i])
            logging.info("Plotting results: %s... DONE"%path)