from fintorch.datasets import AmazonFraudDataset

# Usage example
data_path = "/path/to/your/data"
dataset = AmazonFraudDataset(data_path)
dataloader = dataset.get_dataloader(batch_size=32, shuffle=True, num_workers=4)
for batch in dataloader:
    # Process each batch of data
    # ...
    # Print example records
    for record in batch:
        print(record)