import numpy as np
import pandas as pd
import torch

def euc_classifier(mutation_dist, num_mut, train_file_path="signaturesnet/data/train_all_input.csv", measures_path = 'signaturesnet/data/measure_2.0_1.4e-02_interpolated.txt'):
	""" 
	Classify a set of mutations as realistic or not based on the Euclidean distance to the training data.
	Args:
		mutation_dist(torch.Tensor): Tensor of shape (n, 96) with the mutational distribution for each sample
		num_mut(torch.Tensor): Tensor of shape (n,) with the number of mutations for each sample
		train_file_path(str): Path to the training data file
	Returns:
		classification(torch.Tensor): Tensor of shape (n,) with the classification of each sample
	"""
	# Load training data
	train_data = np.loadtxt(train_file_path, delimiter=",")
	train_tensor = torch.tensor(train_data, dtype=torch.float32)

	# Load the measures
	measures_file = pd.read_csv(measures_path, sep='\t', header=None)
	measures_dict = pd.Series(measures_file[1].values, index=measures_file[0]).to_dict() # Convert it into a dictionary

	classification = []
	for i in range(mutation_dist.shape[0]):
		# Get the mutation distribution and number of mutations for the current sample
		mutation_instance = mutation_dist[i]
		num_mut_instance = int(num_mut[i].item())

		# If the number of mutations is less than 10, classify as random
		if num_mut_instance < 10:
			classification.append(0)
		else:
			# Compute Euclidean distances in a vectorized way
			distances = torch.sqrt(torch.sum((train_tensor - mutation_instance) ** 2, dim=1))
			# Find the minimum distance
			min_distance = torch.min(distances)

			if num_mut_instance > 200000:
				measure = measures_dict[200000]
			else:
				measure = measures_dict[num_mut_instance]
			
			# Compare the minimum distance with the measure: classify as realistic if the distance is less than the measure
			classification.append(1 if min_distance <= measure else 0)

	# Return the classification as a tensor
	return torch.tensor(classification, dtype=torch.float32)
