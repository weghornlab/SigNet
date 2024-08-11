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
    data_dir = os.path.join(DATA, "datasets", "vae_classification_test")
    model_dir = os.path.join(TRAINED_MODELS, "vae_classifier/classifier_3")
    
    data = parse_data(data_dir)
    
    downstream = sorted([int(k.split("_")[1]) for k in data.keys() if "downstream" in k])
    upstream = sorted([int(k.split("_")[1]) for k in data.keys() if "upstream" in k])
    
    vae_classifier = read_model(model_dir, device="cpu").eval()
    
    with torch.no_grad():
        metrics = {}
        for e in downstream:
            inp = data[f"downstream_{e}"].unsqueeze(0)
            # normalize
            inp = inp / inp.sum()
            out, z_mu, z_var = vae_classifier.forward(inp, noise=False)
            reconstruction_dist = torch.nn.MSELoss()(out, inp)
            
            metrics[f"downstream_{e}"] = {
                "rec_dist": reconstruction_dist.item(),
                "z_mu": z_mu.abs().mean().item(),
                "z_var": z_var.abs().mean().item(),
            }
            
        for e in upstream:
            inp = data[f"upstream_{e}"].unsqueeze(0)
            # normalize
            inp = inp / inp.sum()
            out, z_mu, z_var = vae_classifier.forward(inp, noise=False)
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
    # z = torch.tensor([metrics[f"downstream_{e}"]["z_mu"] for e in downstream])

    #normalize
    y = (y - y.mean()) / y.std()
    # z = (z - z.mean()) / z.std()

    # same for upstream, yes I'm being very lazy here
    xx = [e for e in upstream]
    yy = torch.tensor([metrics[f"upstream_{e}"]["rec_dist"] for e in upstream])
    # zz = torch.tensor([metrics[f"upstream_{e}"]["z_mu"] for e in upstream])
    
    yy = (yy - yy.mean()) / yy.std()
    # zz = (zz - zz.mean()) / zz.std()

    plt.plot(x, y, label="Downstream Dissimilarity")
    plt.plot(xx, yy, label="Upstream Dissimilarity")
    plt.legend()
    plt.show()

    print(here)