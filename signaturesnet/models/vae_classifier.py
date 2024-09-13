import torch
from torch import nn

class VaeClassifier(nn.Module):
    
    def __init__(self,
                 input_size=96,
                 num_hidden_layers=2,
                 latent_dim=100,
                 num_units=200,
                 num_units_branch_mut=10,
                 sigmoid_params=[5000,1000],
                 device="cuda") -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "VaeClassifier"

        super(VaeClassifier, self).__init__()
        self.sigmoid_params = sigmoid_params
        self.latent_dim = latent_dim

        # Input path
        # 96 = total number of possible muts
        self.layer1_1 = nn.Linear(input_size, num_units)
        # Number of mutations path
        self.layer1_2 = nn.Linear(1, num_units_branch_mut)

        self.layer2_1 = nn.Linear(num_units, num_units)
        self.layer2_2 = nn.Linear(num_units_branch_mut, num_units_branch_mut)

        merged_input_size = num_units + num_units_branch_mut
        encoder_layers = []
        for i in range(num_hidden_layers):
            in_features = int(merged_input_size - (merged_input_size-latent_dim)*i/num_hidden_layers)
            out_features = int(merged_input_size - (merged_input_size-latent_dim)*(i+1)/num_hidden_layers)
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
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.Normal = torch.distributions.Normal(0, 1)
        if device == "cuda":
            print("Sending generator to cuda")
            self.Normal.loc = self.Normal.loc.cuda()
            self.Normal.scale = self.Normal.scale.cuda()
        self.device = device

    def encode(self, mutation_dist, num_mut):
        # Input head
        mutation_dist = self.activation(self.layer1_1(mutation_dist))
        mutation_dist = self.activation(self.layer2_1(mutation_dist))
        # Number of mutations head
        num_mut = nn.Sigmoid()(
            (num_mut-self.sigmoid_params[0])/self.sigmoid_params[1])
        num_mut = self.activation(self.layer1_2(num_mut))
        num_mut = self.activation(self.layer2_2(num_mut))
        # Concatenate
        x = torch.cat([mutation_dist, num_mut], dim=1)

        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        z_mu = self.mean_layer(x)
        z_var = torch.exp(self.var_layer(x))
        # z_var = torch.ones_like(z_mu)
        # z_var = torch.ones_like(z_mu)*0.7
        # z_var = torch.ones_like(z_mu)*0.0001
        return z_mu, z_var

    def decode(self, x):
        for layer in self.decoder_layers[:-1]:
            x = self.activation(layer(x))
        x = self.relu(self.decoder_layers[-1](x))
        # x = self.sigmoid(self.decoder_layers[-1](x))
        x = x/x.sum(dim=1).reshape(-1,1)
        return x

    def forward(self, mutation_dist, num_mut, noise=False):
        z_mu, z_var = self.encode(mutation_dist, num_mut)
        if noise:
            # z = z_mu + z_var*self.Normal.sample(z_mu.shape)
            z = torch.randn(size = (z_mu.size(0), z_mu.size(1)), device=self.device)
            z = z_mu + z_var*z
        else:
            z = z_mu
        x = self.decode(z)
        return x, z_mu, z_var
