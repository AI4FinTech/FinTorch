from fintorch.datasets import elliptic

# Load the elliptic dataset
dataset = elliptic.EllipticDataset()

# Display the first records
for i in range(1):
    print(dataset[i])
