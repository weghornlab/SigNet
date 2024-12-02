import numpy as np
import torch

def get_measure(n):
	""" 
	Obtain the measure for a given number of mutations given a set of intervals.
	Args:
		n(int): Number of mutations
	Returns:
		measure: float
	"""
	data = [
		(1, 5, 0.363756),
		(5, 10, 0.322294),
		(10, 15, 0.250683), 
		(15, 20, 0.222612), 
		(20, 30, 0.192311), 
		(30, 50, 0.154168),
		(50, 75, 0.128083), 
		(75, 100, 0.110249), 
		(100, 200, 0.089018), 
		(200, 300, 0.072420),
		(300, 400, 0.062052), 
		(400, 500, 0.056765), 
		(500, 700, 0.051645), 
		(700, 1000, 0.045896),
		(1000, 2000, 0.038491), 
		(2000, 5000, 0.030448), 
		(5000, 7000, 0.026473),
		(7000, 10000, 0.024518), 
		(10000, 20000, 0.022149), 
		(20000, 50000, 0.019657),
		(50000, 100000, 0.018322), 
		(100000, float('inf'), 0.016398)
    ]
    # Find the interval that contains the number of mutations
	for start, end, measure in data:
		if start <= n < end:
			# Return the measure
			return measure

def euc_classifier(mutation_dist, num_mut, train_file_path="signaturesnet/data/train_all_input.csv"):
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

	classification = []
	for i in range(mutation_dist.shape[0]):
		# Get the mutation distribution and number of mutations for the current sample
		mutation_instance = mutation_dist[i]
		num_mut_instance = num_mut[i].item()

		# If the number of mutations is less than 10, classify as random
		if num_mut_instance < 10:
			classification.append(0)
		else:
			# Compute Euclidean distances in a vectorized way
			distances = torch.sqrt(torch.sum((train_tensor - mutation_instance) ** 2, dim=1))
			# Find the minimum distance
			min_distance = torch.min(distances)
			
			print(distances)
			measure = get_measure(num_mut_instance)
			print(f"Num mutations: {num_mut_instance}")
			print(f"Min distance: {min_distance}, measure: {measure}")

			# Compare the minimum distance with the measure: classify as realistic if the distance is less than the measure
			classification.append(1 if min_distance <= measure else 0)

	# Return the classification as a tensor
	return torch.tensor(classification, dtype=torch.float32)
