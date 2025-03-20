
# Cluster Oversampling G-mean (COG v1.5) Algorithm

## 📌 Overview
This repository contains the Python implementation of the **Cluster Oversampling G-mean (COG v1.5)** algorithm, designed for handling imbalanced classification problems by optimizing the G-mean metric.

The implementation is aligned with the paper:
Prexawanprasut, T., & Banditwattanawong, T. (2024). *Improving Minority Class Recall through a Novel Cluster-Based Oversampling Technique*. Informatics, 11(2), 35. https://doi.org/10.3390/informatics11020035

## ⚙️ Files
- `COG_v1_5.py`: Full implementation of the COG v1.5 algorithm
- `example_run.py`: Example script to run the algorithm
- `requirements.txt`: Required Python libraries
- `LICENSE`: MIT License for use and distribution

## 🚀 Installation
```
git clone https://github.com/YourUsername/COG-Oversampling.git
cd COG-Oversampling
pip install -r requirements.txt
```

## 🧠 Example Usage
```python
from COG_v1_5 import apply_algorithm

final_gmean, total_synthetic = apply_algorithm(
    file_path='your_dataset.csv',
    n_clusters=5,
    target_ir=0.8,
    minority_class=1
)
print(f"Final G-Mean: {final_gmean:.4f}")
print(f"Total Synthetic Samples: {total_synthetic}")
```

## 📜 License
MIT License - feel free to use, modify, and share with attribution.
