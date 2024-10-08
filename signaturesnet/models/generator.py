import torch
from torch import nn

from signaturesnet.utilities.data_generator import DataGenerator

class Generator(nn.Module):
    
    def __init__(
            self,
            input_size=72,
            num_hidden_layers=2,
            latent_dim=100,
            device="cuda"
            ) -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "Generator"
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        encoder_layers = []
        for i in range(num_hidden_layers):
            in_features = int(input_size - (input_size-latent_dim)*i/num_hidden_layers)
            out_features = int(input_size - (input_size-latent_dim)*(i+1)/num_hidden_layers)
            # print("in_features:", in_features, "out_features:", out_features)
            layer = nn.Linear(in_features, out_features)
            # layernorm = torch.nn.LayerNorm(in_features)
            # encoder_layers.append(layernorm)
            encoder_layers.append(layer)
        self.encoder_layers = nn.ModuleList(modules=encoder_layers)
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.var_layer = nn.Linear(latent_dim, latent_dim)

        decoder_layers = []
        for i in reversed(range(num_hidden_layers+1)):
            in_features = int(input_size - (input_size-latent_dim)*(i+1)/(num_hidden_layers + 1))
            out_features = int(input_size - (input_size-latent_dim)*i/(num_hidden_layers + 1))
            layer = nn.Linear(in_features, out_features)            
            # layernorm = torch.nn.LayerNorm(in_features)
            # decoder_layers.append(layernorm)
            decoder_layers.append(layer)

        self.decoder_layers = nn.ModuleList(modules=decoder_layers)
        self.activation = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        

        self.Normal = torch.distributions.Normal(0, 1)
        if device == "cuda":
            print("Sending generator to cuda")
            self.Normal.loc = self.Normal.loc.cuda()
            self.Normal.scale = self.Normal.scale.cuda()
        self.device = device


    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        z_mu = self.mean_layer(x)
        z_var = torch.exp(self.var_layer(x))
        return z_mu, z_var

    def decode(self, x):
        for layer in self.decoder_layers[:-1]:
            x = self.activation(layer(x))
        x = self.relu(self.decoder_layers[-1](x))
        x = x/x.sum(dim=1).reshape(-1,1)
        return x

    def forward(self, x, noise=True):
        z_mu, z_var = self.encode(x)
        if noise:
            # z = z_mu + z_var*self.Normal.sample(z_mu.shape)
            z = torch.randn(size = (z_mu.size(0), z_mu.size(1)), device=self.device)
            z = z_mu + z_var*z
        else:
            z = z_mu
        x = self.decode(z)
        return x, z_mu, z_var

    def generate_weights(self, n_samples:int, std = 1.0, cutoff = 0.01):
        shape = tuple((n_samples, self.latent_dim))
        z = self.Normal.sample(shape)*std
        labels = self.decode(z)
        if cutoff > 0:
            labels[labels < cutoff] = 0
            labels = labels/labels.sum(dim=1).reshape(-1,1)
        return labels

    def generate_weights_and_samples(self,
                                     n_samples,
                                     signatures,
                                     realistic_number_of_mutations=False,
                                     nummutnet=None,
                                     std = 1.0,
                                     cutoff = 0.01):
        """Generate a dataset of realistic signature weights and their sampled mutation vectors.

        Args:
            n_samples (int): Number of samples to generate.
            signatures (torch.tensor): Signature matrix.
            realistic_number_of_mutations (bool, optional): Wether to use the nummut model to generate
                realistic number of mutations for each sample. Defaults to False.
            nummutnet (NumMutNet, optional): Model to estimate the number of mutations for a given signature weight combination. Defaults to None.
            std (float, optional): Standard deviation of latent variables. Defaults to 1.0.
            cutoff (float, optional): Minimum weight assigned t signature. Defaults to 0.01.
        
        Returns:
            inputs (mutational vector)
            labels (including an appended column with the number of mutations used)
        """
        labels = self.generate_weights(n_samples, std, cutoff)
        data_generator = DataGenerator(signatures)
        
        if realistic_number_of_mutations:
            assert nummutnet is not None  # Must provide a mutnet model!
            num_mutations = nummutnet.get_nummuts(labels)
        else:
            num_mutations = np.random.randint(1, 5e3, size=n_samples)

        return data_generator.make_input(labels,
                                        split=None,
                                        large_low=None,
                                        augmentation=1,
                                        nummuts=num_mutations)
    


