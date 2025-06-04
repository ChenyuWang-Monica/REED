import numpy as np
from collections import defaultdict

# File path
file_path = "./mad_pred_mean_label_pairs_epoch_0.txt"

save_path = "./mad_pred_mean_label_pairs_with_error_epoch_0.txt"

# Load data from file
with open(file_path, "r") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

# # Parse the predicted and ground-truth value pairs
pairs = []
for i in range(1, len(lines)):
    pairs.append([round(float(num), 3) for num in lines[i].split(", ")])

trips = []
for pair in pairs:
    trips.append((pair[0], pair[1], round(abs(pair[0] - pair[1]), 3)))
trips.sort(key=lambda trip: trip[2])
    
with open(save_path, "w") as f:
    for trip in trips:
        f.write(str(trip) + "\n")

# # Define interval size
# interval_size = 5

# # Dictionary to hold errors grouped by ground-truth value intervals
# error_dict = defaultdict(list)

# # Process each pair and compute errors
# for predicted_value, ground_truth_value in pairs:
#     # Find the interval in which the ground-truth value falls
#     interval_start = (int(ground_truth_value) // interval_size) * interval_size
#     interval_end = interval_start + interval_size
#     interval_key = f"{interval_start}-{interval_end}"
    
#     # Calculate prediction error
#     error = predicted_value - ground_truth_value
    
#     # Store the error in the appropriate interval
#     error_dict[interval_key].append(error)

# # Calculate mean error and sample count for each interval
# interval_stats = [(interval, np.mean(errors), len(errors)) for interval, errors in error_dict.items()]

# # Sort intervals by their numeric start value
# interval_stats_sorted = sorted(interval_stats, key=lambda x: int(x[0].split('-')[0]))

# # Display sorted mean errors and sample counts per interval
# for interval, mean_error, sample_count in interval_stats_sorted:
#     print(f"Interval {interval}: Mean Error = {mean_error:.2f}, Sample Count = {sample_count}")