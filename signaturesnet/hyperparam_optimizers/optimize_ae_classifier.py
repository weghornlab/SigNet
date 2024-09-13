import os
import sys

from skopt.space import Real, Integer
import torch
import wandb

from signaturesnet.utilities.io import read_data_classifier, sort_signatures
from signaturesnet.trainers import VaeClassifierTrainer
from signaturesnet.HyperParameterOptimizer.gaussian_process import GaussianProcessSearch

experiment_id = "aeclassifier"
iterations = 15

batch_sizes = Integer(name='batch_size', low=10, high=5000)
learning_rates_encoder = Real(name='lr_encoder', low=0.00001, high=0.001)
learning_rates_decoder = Real(name='lr_decoder', low=0.00001, high=0.001)
latent_dim = Integer(name='latent_dim', low=10, high=80)
num_units = Integer(name='num_units', low=150, high=350)
num_units_branch_mut = Integer(name='num_units_branch_mut', low=1, high=20)
num_hidden_layers = Integer(name='num_hidden_layers', low=1, high=8)

input_file = None  # Use None to start from zero
# NOTE(claudia): Is this a problem if running it on multiple nodes of the cluster?
output_file = "search_results_" + experiment_id + ".csv"

if __name__ == "__main__":
    # Select training device
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = "cpu"
    print("Using device:", dev)

    # Read data
    train_data, val_data = read_data_classifier(
        device=dev,
        experiment_id='../data/datasets/detector')

    signatures = sort_signatures(
        file="../data/data.xlsx",
        mutation_type_order="../data/mutation_type_order.xlsx")

    # Instantiate Classifier trainer
    trainer = VaeClassifierTrainer(
        iterations=iterations,  # Passes through all dataset
        train_data=train_data,
        val_data=val_data,
        signatures=signatures,
        device=torch.device(dev),
        model_path=None)
        # model_path=os.path.join("vae_classifier/", "hypeparam_search"))

    # Define hyperparameters to train
    search_space = [batch_sizes, learning_rates_encoder, learning_rates_decoder, latent_dim, num_units, num_units_branch_mut, num_hidden_layers]
    fixed_space = {"plot": True}

    def objective(batch_size,
                  lr_encoder,
                  lr_decoder,
                  num_hidden_layers,
                  latent_dim,
                  num_units,
                  num_units_branch_mut,
                  plot=False):
        
        config = {"batch_size": batch_size,
                  "lr_encoder": lr_encoder,
                  "lr_decoder": lr_decoder,
                  "latent_dim": latent_dim,
                  "num_hidden_layers": num_hidden_layers,
                  "num_units":num_units,
                  "num_units_branch_mut":num_units_branch_mut}
        
        run = wandb.init(project='bayesian-' + experiment_id,
                         entity='sig-net',
                         config=config,
                         name=str(config))

        val = trainer.objective(batch_size=batch_size,
                                lr_encoder=lr_encoder,
                                lr_decoder=lr_decoder,
                                num_hidden_layers=num_hidden_layers,
                                latent_dim=latent_dim,
                                num_units=num_units,
                                num_units_branch_mut=num_units_branch_mut,
                                plot=plot,
                                run=run)
        print(val)
        
        wandb.log({"validation_score": val})
        run.finish()
        return val

    # Start optimization
    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=objective,
                                      input_file=input_file,
                                      output_file=output_file)  # Store tested points
    gp_search.init_session()
    best_metaparams, best_val = gp_search.get_maximum(
        n_calls=500,
        n_random_starts=75,
        noise=0.01,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)
