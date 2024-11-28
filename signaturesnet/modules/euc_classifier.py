import numpy as np

# Function to find the measure for a number inside a range
def get_measure(n):
    # Interval (start, end) and its associated value
	data = [
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
	
	for start, end, measure in data:
		if start <= n < end:
			return measure

def euc_classifier(mutation_dist, num_mut, train_file_path = "../data/train_all_input.csv"):

	# If there are less than 10 mutations, discard
	if num_mut < 10:
		return 0
	
	else: 
		deltas = []
		# Calculate the euclidean distance between the mutation profile (normalised) of the sample and each training instance
		with open(train_file_path, "r") as file:
			for line in file:
				train_instance = np.array(line.strip().split(","), dtype=float)
				eucl_dist = np.power(np.sum(np.power(np.absolute(np.array(mutation_dist) - train_instance), 2)), 1./2)
				deltas.append(eucl_dist)

		# Take the minimum distance
		min_delta = np.min(deltas)
		# Get the value for the number of mutations
		measure = get_measure(num_mut)

		# If the minimum distance is less than the measure, classify as realistic
		if min_delta <= measure:
			return 1
		else:
			return 0
