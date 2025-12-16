import numpy as np
import pandas as pd
import torch
import gzip
import os
from signaturesnet import DATA

def euclidean_classifier(mutation_dist, num_mut):
	""" 
	Classify a set of mutations as realistic or not based on the Euclidean distance to the training data.
	Args:
		mutation_dist(torch.Tensor): Tensor of shape (n, 96) with the mutational distribution for each sample
		num_mut(torch.Tensor): Tensor of shape (n,) with the number of mutations for each sample
	Returns:
		classification(torch.Tensor): Tensor of shape (n,) with the classification of each sample
	"""
	realistic_path = os.path.join(DATA,"realistic_profiles.csv.gz")
	thresholds_path = os.path.join(DATA,'measure_6_4.0e-02_7_interpolated.txt')
	# Load realistic profiles
	with gzip.open(realistic_path, 'rt') as f:
		realistic_data = np.loadtxt(f, delimiter=",")
	realistic_tensor = torch.tensor(realistic_data, dtype=torch.float32) # Convert it into a tensor

	# Load the thresholds
	thresholds_file = pd.read_csv(thresholds_path, sep='\t', header=None)
	thresholds_dict = pd.Series(thresholds_file[1].values, index=thresholds_file[0]).to_dict() # Convert it into a dictionary

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
			distances = torch.sqrt(torch.sum((realistic_tensor - mutation_instance)**2, dim=1))
			# Find the minimum distance
			min_distance = torch.min(distances)

			if num_mut_instance > 200000:
				threshold = thresholds_dict[200000]
			else:
				threshold = thresholds_dict[num_mut_instance]
			
			# Compare the minimum distance with the threshold: classify as realistic if the distance is less than the threshold
			classification.append(1 if min_distance <= threshold else 0)

	# Return the classification as a tensor
	return torch.tensor(classification, dtype=torch.float32)
