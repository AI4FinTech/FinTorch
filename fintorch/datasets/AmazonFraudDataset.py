# Amazon-Fraud is a multi-relational graph dataset built upon the Amazon review dataset, 
# which can be used in evaluating graph-based node classification, fraud detection, 
# and anomaly detection models.

import torch
from torch.utils.data import Dataset

class AmazonFraudDataset(Dataset):
    def __init__(self, data_path):
        # Initialize dataset here
        self.data = self.load_data(data_path)
    
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve a single sample from the dataset
        sample = self.data[idx]
        # Preprocess the sample if needed
        # ...
        # Return the preprocessed sample
        return sample
    
    def load_data(self, data_path):
        # Load the dataset from the specified path
        # ...
        # Return the loaded dataset
        raise NotImplementedError
    
    def collate_fn(self, batch):
        # Define how to collate multiple samples into a batch
        # ...
        # Return the collated batch
        raise NotImplementedError
    
    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        # Create a PyTorch DataLoader for this dataset
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
        return dataloader

