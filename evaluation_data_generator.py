import numpy as np

num_samples = 1000
temperature = np.round(np.random.uniform(26.7, 40, num_samples), 2)
humidity    = np.round(np.random.uniform(40, 100, num_samples), 2)

data = np.column_stack((temperature, humidity))

np.savetxt("input.txt", data, fmt="%.2f")

print("✅ input.txt created")