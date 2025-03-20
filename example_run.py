
from COG_v1_5 import apply_algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Example 1 - Basic usage with default Decision Tree
print("===== Example 1: Basic Usage =====")
gmean1, synthetic1 = apply_algorithm(
    file_path='dataset1.csv',
    n_clusters=5,
    target_ir=0.8,
    minority_class=1
)
print(f"G-Mean: {gmean1:.4f} | Synthetic Samples: {synthetic1}\n")

# Example 2 - Different number of clusters and higher target IR
print("===== Example 2: More Clusters and Higher Target IR =====")
gmean2, synthetic2 = apply_algorithm(
    file_path='dataset2.csv',
    n_clusters=8,
    target_ir=1.2,
    minority_class=1
)
print(f"G-Mean: {gmean2:.4f} | Synthetic Samples: {synthetic2}\n")

# Example 3 - Target minority class is labeled as '2'
print("===== Example 3: Different Minority Class Label =====")
gmean3, synthetic3 = apply_algorithm(
    file_path='dataset3.csv',
    n_clusters=6,
    target_ir=0.9,
    minority_class=2
)
print(f"G-Mean: {gmean3:.4f} | Synthetic Samples: {synthetic3}\n")

# Example 4 - Custom classifier: Random Forest
print("===== Example 4: Using RandomForestClassifier =====")
gmean4, synthetic4 = apply_algorithm(
    file_path='dataset4.csv',
    n_clusters=7,
    target_ir=1.0,
    minority_class=1,
    base_classifier=RandomForestClassifier(n_estimators=100, random_state=42)
)
print(f"G-Mean: {gmean4:.4f} | Synthetic Samples: {synthetic4}\n")

# Example 5 - Experiment with high patience for optimization
print("===== Example 5: High Patience Setting =====")
gmean5, synthetic5 = apply_algorithm(
    file_path='dataset5.csv',
    n_clusters=4,
    target_ir=1.0,
    minority_class=1,
    patience=5
)
print(f"G-Mean: {gmean5:.4f} | Synthetic Samples: {synthetic5}\n")

