import os

import torch

from signaturesnet import DATA, TRAINED_MODELS
from signaturesnet.utilities.io import read_model

def parse_data(data_dir):
    def _parse_file(f):
        with open(f, "r") as file:
            return torch.tensor([
                    float(l.split("\t")[1].replace("\n", ""))
                        for l in file.readlines()
                ])
    
    data = {
        f.replace("mutational_profile_", "").replace("_TSS_100_UKBB", "").replace(".txt", ""):
            _parse_file(os.path.join(data_dir, f))
                for f in os.listdir(data_dir)
    }
    return data        


def plot(x, y, z):
    fig, ax1 = plt.subplots()
    
    # Plotting the first data set
    color = 'tab:red'
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Reconstruction Distance', color=color)
    ax1.plot(x, y, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating a second y-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Z_mu', color=color)  
    ax2.plot(x, z, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding a legend and showing the plot
    fig.tight_layout()  # Optional to fit well into the plot
    plt.show()

if __name__ == "__main__":
    real_data_test = True
    synthetic_data_test = True

    if real_data_test:
        data_dir = os.path.join(DATA, "datasets", "vae_classification_test")
        model_dir = os.path.join(TRAINED_MODELS, "vae_classifier/classifier_nummut_sigmoid")

        data = parse_data(data_dir)

        downstream = sorted([int(k.split("_")[1]) for k in data.keys() if "downstream" in k])
        upstream = sorted([int(k.split("_")[1]) for k in data.keys() if "upstream" in k])

        vae_classifier = read_model(model_dir, device="cpu").eval()
        
        with torch.no_grad():
            metrics = {}
            for e in downstream:
                inp = data[f"downstream_{e}"].unsqueeze(0)
                # normalize
                num_mut = inp.sum().unsqueeze(0).unsqueeze(0)
                inp = inp / inp.sum()
                out, z_mu, z_var = vae_classifier.forward(inp, num_mut, noise=False)
                reconstruction_dist = torch.nn.MSELoss()(out, inp)
                if e == 1:
                    print(inp)
                    print(num_mut)
                    print(out)
                metrics[f"downstream_{e}"] = {
                    "rec_dist": reconstruction_dist.item(),
                    "z_mu": z_mu.abs().mean().item(),
                    "z_var": z_var.abs().mean().item(),
                }
                
            for e in upstream:
                inp = data[f"upstream_{e}"].unsqueeze(0)
                # normalize
                num_mut = inp.sum().unsqueeze(0).unsqueeze(0)
                inp = inp / inp.sum()
                out, z_mu, z_var = vae_classifier.forward(inp, num_mut, noise=False)
                reconstruction_dist = torch.nn.MSELoss()(out, inp)
                
                metrics[f"upstream_{e}"] = {
                    "rec_dist": reconstruction_dist.item(),
                    "z_mu": z_mu.abs().mean().item(),
                    "z_var": z_var.abs().mean().item(),
                }
        
        
        # Plot metrics
        import matplotlib.pyplot as plt
        x = [e for e in downstream]
        y = torch.tensor([metrics[f"downstream_{e}"]["rec_dist"] for e in downstream])
        z = torch.tensor([metrics[f"downstream_{e}"]["z_mu"] for e in downstream])

        #normalize
        y = (y - y.mean()) / y.std()
        z = (z - z.mean()) / z.std()

        # same for upstream, yes I'm being very lazy here
        xx = [e for e in upstream]
        yy = torch.tensor([metrics[f"upstream_{e}"]["rec_dist"] for e in upstream])
        zz = torch.tensor([metrics[f"upstream_{e}"]["z_mu"] for e in upstream])
        
        yy = (yy - yy.mean()) / yy.std()
        zz = (zz - zz.mean()) / zz.std()

        plt.plot(x, 0.6*y + 0.4*z, label="Downstream Dissimilarity")
        plt.plot(xx, 0.6*yy + 0.4*zz, label="Upstream Dissimilarity")
        plt.legend()
        plt.show()

    if synthetic_data_test:
        import pandas as pd
        import numpy as np

        data_dir = os.path.join(DATA, "datasets", "detector")
        test_input = pd.read_csv(data_dir + "/test_input.csv", header=None)
        test_num_mut = pd.read_csv(data_dir + "/test_num_mut.csv", header=None)
        test_label = pd.read_csv(data_dir + "/test_label.csv", header=None)

        model_dir = os.path.join(TRAINED_MODELS, "vae_classifier/classifier_nummut_sigmoid")
        vae_classifier = read_model(model_dir, device="cpu").eval()

        with torch.no_grad():
            metrics = {}
            test_input_realistic = torch.tensor(test_input.loc[test_label[0]==1].values.astype(np.float32))
            test_nummut_realistic = torch.tensor(test_num_mut.loc[test_label[0]==1].values.astype(np.float32))
            print(test_input_realistic)
            print(test_nummut_realistic)
            out, z_mu, z_var = vae_classifier.forward(test_input_realistic, test_nummut_realistic, noise=False)
            print(out)
            reconstruction_dist = torch.nn.MSELoss(reduction='none')(out, test_input_realistic)
            reconstruction_dist_mean = torch.nn.MSELoss()(out, test_input_realistic)
            metrics[f"realistic"] = {
                "rec_dist_mean": reconstruction_dist_mean.item(),
                "z_mu_mean": z_mu.abs().mean().item(),
                "z_var_mean": z_var.abs().mean().item(),
                "rec_dist": reconstruction_dist.sum(axis=1).sqrt().tolist(),
                "z_mu": z_mu.abs().mean(axis=1).tolist(),
                "z_var": z_var.abs().mean(axis=1).tolist(),
            }
            test_input_random = torch.tensor(test_input.loc[test_label[0]==0].values.astype(np.float32))
            test_nummut_random = torch.tensor(test_num_mut.loc[test_label[0]==0].values.astype(np.float32))
            out, z_mu, z_var = vae_classifier.forward(test_input_random, test_nummut_random, noise=False)
            reconstruction_dist = torch.nn.MSELoss(reduction='none')(out, test_input_random)
            reconstruction_dist_mean = torch.nn.MSELoss()(out, test_input_random)
            metrics[f"random"] = {
                "rec_dist_mean": reconstruction_dist_mean.item(),
                "z_mu_mean": z_mu.abs().mean().item(),
                "z_var_mean": z_var.abs().mean().item(),
                "rec_dist": reconstruction_dist.sum(axis=1).sqrt().tolist(),
                "z_mu": z_mu.abs().mean(axis=1).tolist(),
                "z_var": z_var.abs().mean(axis=1).tolist(),
            }   

        # Plot metrics
        import matplotlib.pyplot as plt
        y = torch.tensor(metrics["realistic"]["rec_dist"])
        z = torch.tensor(metrics["realistic"]["z_mu"])
        x = [1 for _ in y]
        ymean = torch.tensor([metrics["realistic"]["rec_dist_mean"]])
        zmean = torch.tensor([metrics["realistic"]["z_mu_mean"]])

        #normalize
        y = (y - y.mean()) / y.std()
        z = (z - z.mean()) / z.std()

        # same for upstream, yes I'm being very lazy here
        yy = torch.tensor(metrics["random"]["rec_dist"])
        zz = torch.tensor(metrics["random"]["z_mu"])
        xx = [0 for _ in yy]
        yymean = torch.tensor([metrics["random"]["rec_dist_mean"]])
        zzmean = torch.tensor([metrics["random"]["z_mu_mean"]])

        yy = (yy - yy.mean()) / yy.std()
        zz = (zz - zz.mean()) / zz.std()

        plt.scatter(x, 0.6*y + 0.4*z, label="Realistic Dissimilarity")
        plt.scatter(xx, 0.6*yy + 0.4*zz, label="Random Dissimilarity")
        plt.scatter([1], 0.6*ymean + 0.4*zmean, label="Realistic Average Dissimilarity")
        plt.scatter([0], 0.6*yymean + 0.4*zzmean, label="Random Average Dissimilarity")
        plt.legend()
        plt.show()