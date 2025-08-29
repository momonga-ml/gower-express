import time
import numpy as np
import pandas as pd
import gower_exp

# Create a larger test dataset
np.random.seed(42)
n_samples = 500
n_features = 10

# Mix of numerical and categorical features
data = {}
for i in range(n_features // 2):
    data[f'num_{i}'] = np.random.normal(0, 1, n_samples)

categories = ['A', 'B', 'C', 'D', 'E']
for i in range(n_features // 2):
    data[f'cat_{i}'] = np.random.choice(categories, n_samples)

df = pd.DataFrame(data)

print("Benchmarking Gower distance computation...")
print(f"Dataset shape: {df.shape}")

# Test with smaller subset for timing
test_df = df.iloc[:100].copy()

start_time = time.time()
result = gower_exp.gower_matrix(test_df)
end_time = time.time()

print(f"Computation time: {end_time - start_time:.4f} seconds")
print(f"Result shape: {result.shape}")
print(f"Sample distance: {result[0, 1]:.6f}")

# Check if numba is being used
from gower_exp.gower_dist import NUMBA_AVAILABLE
print(f"Numba available: {NUMBA_AVAILABLE}")

# Test gower_topn
print("\nTesting gower_topn...")
start_time = time.time()
topn_result = gower_exp.gower_topn(test_df.iloc[:1], test_df, n=5)
end_time = time.time()

print(f"Top-N computation time: {end_time - start_time:.4f} seconds")
print(f"Top-5 indices: {topn_result['index']}")
print(f"Top-5 distances: {topn_result['values']}")
