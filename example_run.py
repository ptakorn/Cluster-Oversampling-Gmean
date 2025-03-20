
from COG_v1_5 import apply_algorithm

# Example CSV file path (replace with your dataset CSV)
dataset_path = 'your_dataset.csv'

# Example parameters
n_clusters = 5
target_ir = 0.8
minority_class_label = 1

# Run the algorithm
final_gmean, total_synthetic = apply_algorithm(
    file_path=dataset_path,
    n_clusters=n_clusters,
    target_ir=target_ir,
    minority_class=minority_class_label
)

print(f"Final G-Mean on Test Set: {final_gmean:.4f}")
print(f"Total Synthetic Instances Generated: {total_synthetic}")
