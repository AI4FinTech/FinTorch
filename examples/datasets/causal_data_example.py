#!/usr/bin/env python3
"""
Example script demonstrating how to use the CausalDataModule for loading
causal relationship datasets from the CausalFormer repository.

This script shows how to:
1. Load different types of causal datasets (diamond, fork, mediator, v)
2. Access the data and groundtruth causal relationships
3. Visualize the data and causal structure (adjacency matrix + directed graph)
4. Train a simple model using the data loader

The visualization includes:
- Adjacency matrix heatmap showing causal strengths
- Directed graph representation with nodes and weighted edges
- Time series plots showing past and future data windows
"""

import os
import matplotlib.pyplot as plt
import polars as pl
import torch
import numpy as np
from torch import nn
import lightning as L
import networkx as nx

from fintorch.datasets.causal_data import (
    CausalDataModule, 
    create_causal_datamodule,
    get_causal_data_dir,
    clear_causal_data,
    list_available_datasets,
    get_dataset_info,
    get_clean_adjacency_matrix
)


def explore_dataset(dataset_type: str, local_dir: str = None):
    """
    Explore a causal dataset by loading it and showing basic statistics.
    
    Args:
        dataset_type (str): Type of dataset ('diamond', 'fork', 'mediator', 'v')
        local_dir (str): Directory to store downloaded data. If None, uses ~/.fintorch_data/causal
    """
    print(f"\n{'='*60}")
    print(f"Exploring {dataset_type.upper()} Dataset")
    print(f"{'='*60}")
    
    # Create data module
    data_module = create_causal_datamodule(
        dataset_type=dataset_type,
        local_dir=local_dir,
        time_step=20,
        output_window=10,
        batch_size=32,
        train_split=0.7,
        val_split=0.15
    )
    
    # Setup the data module
    data_module.setup()
    
    # Get basic information about the dataset
    dataset = data_module.dataset
    print(f"Dataset type: {dataset.dataset_type}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of nodes (time series): {dataset.series_dim}")
    print(f"Time steps per sample: {dataset.time_steps}")
    print(f"Future steps to predict: {dataset.future_steps}")
    print(f"Feature dimension per node: {dataset.features_dim}")
    
    # Show data splits
    print(f"\nData splits:")
    print(f"  Training samples: {len(data_module.train_dataset)}")
    print(f"  Validation samples: {len(data_module.val_dataset)}")
    print(f"  Test samples: {len(data_module.test_dataset)}")
    
    # Get and display groundtruth causal relationships
    groundtruth = data_module.get_groundtruth()
    if groundtruth is not None:
        print(f"\nGroundtruth causal relationships:")
        print(groundtruth)
        
        # Get clean adjacency matrix (removes index columns)
        adjacency_matrix = get_clean_adjacency_matrix(groundtruth)
        print(f"\nClean adjacency matrix (shape: {adjacency_matrix.shape}):")
        print(adjacency_matrix)
        
        # If the groundtruth is an adjacency matrix, show it
        if groundtruth.shape[0] == groundtruth.shape[1]:
            print(f"\nCausal adjacency matrix (shape: {groundtruth.shape}):")
            print(groundtruth.to_numpy())
        else:
            print(f"\nGroundtruth matrix (shape: {groundtruth.shape}):")
            print(groundtruth.to_numpy())
    else:
        print("\nNo groundtruth file found or could not be loaded.")
    
    # Show a sample from the dataset
    sample = dataset[0]
    print(f"\nSample data shapes:")
    print(f"  Past data: {sample['past_data'].shape}")
    print(f"  Future data: {sample['future_data'].shape}")
    print(f"  Static data: {sample['static_data'].shape}")
    print(f"  Target: {sample['target'].shape}")
    
    # Load a batch to demonstrate data loading
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    return data_module


def visualize_causal_structure(data_module: CausalDataModule, save_plot: bool = False):
    """
    Visualize the causal structure as both an adjacency matrix and a directed graph.
    
    This function creates a side-by-side visualization showing:
    1. Adjacency matrix heatmap with causal strengths
    2. Directed graph with nodes and weighted edges representing causal relationships
    
    Args:
        data_module (CausalDataModule): The data module containing the dataset
        save_plot (bool): Whether to save the plot to a file
    """
    groundtruth = data_module.get_groundtruth()
    
    if groundtruth is None:
        print("No groundtruth available for visualization.")
        return
    
    # Get clean adjacency matrix (automatically handles index columns)
    adjacency_matrix = get_clean_adjacency_matrix(groundtruth)
    
    # Create subplots for both matrix and graph visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot adjacency matrix
    im = ax1.imshow(adjacency_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax1, label='Causal strength')
    ax1.set_title(f'Causal Adjacency Matrix - {data_module.dataset_type.upper()} Dataset')
    ax1.set_xlabel('Target Node')
    ax1.set_ylabel('Source Node')
    
    # Add grid and node labels
    ax1.grid(True, alpha=0.3)
    
    # Add node labels if matrix is reasonably small
    if adjacency_matrix.shape[0] <= 10:
        y_labels = [f'Node {i}' for i in range(adjacency_matrix.shape[0])]
        x_labels = [f'Node {i}' for i in range(adjacency_matrix.shape[1])]
        ax1.set_xticks(range(adjacency_matrix.shape[1]))
        ax1.set_xticklabels(x_labels, rotation=45)
        ax1.set_yticks(range(adjacency_matrix.shape[0]))
        ax1.set_yticklabels(y_labels)
    
    # Plot causal graph
    visualize_causal_graph(groundtruth, ax2, data_module.dataset_type)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'{data_module.dataset_type}_causal_structure.png', dpi=300, bbox_inches='tight')
        print(f"Saved causal structure plot to {data_module.dataset_type}_causal_structure.png")
    
    plt.show()


def visualize_causal_graph(groundtruth, ax, dataset_type: str):
    """
    Visualize the causal relationships as a directed graph.
    
    Args:
        groundtruth: Polars DataFrame containing causal relationships
        ax: Matplotlib axis to plot on
        dataset_type: Type of dataset for the title
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Parse the groundtruth data to extract edges
    # The groundtruth appears to have columns: [source, target, weight]
    groundtruth_np = groundtruth.to_numpy()
    
    # Determine unique nodes
    if groundtruth_np.shape[1] >= 3:
        sources = groundtruth_np[:, 0]
        targets = groundtruth_np[:, 1]
        weights = groundtruth_np[:, 2]
    else:
        # If only 2 columns, assume binary relationships
        sources = groundtruth_np[:, 0]
        targets = groundtruth_np[:, 1]
        weights = np.ones(len(sources))
    
    # Add all unique nodes
    all_nodes = np.unique(np.concatenate([sources, targets]))
    G.add_nodes_from(all_nodes)
    
    # Add edges with weights
    for source, target, weight in zip(sources, targets, weights):
        if weight > 0:  # Only add edges with positive weights
            G.add_edge(source, target, weight=weight)
    
    # Create layout
    try:
        # Try hierarchical layout for causal graphs
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        # Fallback to circular layout
        pos = nx.circular_layout(G)
    
    # Draw nodes
    node_colors = ['lightblue' for _ in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=1000, alpha=0.8)
    
    # Draw edges with varying thickness based on weight
    edges = G.edges(data=True)
    if edges:
        edge_weights = [d['weight'] for _, _, d in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [3 * w / max_weight for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                              alpha=0.6, edge_color='gray', 
                              arrowsize=20, arrowstyle='->', 
                              connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    labels = {node: f'Node {node}' for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')
    
    # Add edge labels for weights if there aren't too many edges
    if len(G.edges()) <= 10:
        edge_labels = {(u, v): f'{d["weight"]:.1f}' for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)
    
    ax.set_title(f'Causal Graph - {dataset_type.upper()} Dataset')
    ax.axis('off')
    
    # Add legend
    ax.text(0.02, 0.98, 'Arrow direction: causal influence\nEdge thickness: causal strength', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def visualize_time_series_sample(data_module: CausalDataModule, sample_idx: int = 0, save_plot: bool = False):
    """
    Visualize a sample time series from the dataset.
    
    Args:
        data_module (CausalDataModule): The data module containing the dataset
        sample_idx (int): Index of the sample to visualize
        save_plot (bool): Whether to save the plot to a file
    """
    dataset = data_module.dataset
    sample = dataset[sample_idx]
    
    past_data = sample['past_data'].squeeze(-1)  # Remove feature dimension
    future_data = sample['future_data'].squeeze(-1)
    
    # Create time indices
    past_time = range(-past_data.shape[0], 0)
    future_time = range(0, future_data.shape[0])
    
    plt.figure(figsize=(12, 8))
    
    # Plot each node's time series
    num_nodes = past_data.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_nodes))
    
    for node_idx in range(num_nodes):
        plt.plot(past_time, past_data[:, node_idx], 
                color=colors[node_idx], linestyle='-', 
                label=f'Node {node_idx} (past)', alpha=0.8)
        plt.plot(future_time, future_data[:, node_idx], 
                color=colors[node_idx], linestyle='--', 
                label=f'Node {node_idx} (future)', alpha=0.8)
    
    plt.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='Present')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Value')
    plt.title(f'Time Series Sample - {data_module.dataset_type.upper()} Dataset')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig(f'{data_module.dataset_type}_time_series_sample.png', dpi=300, bbox_inches='tight')
        print(f"Saved time series plot to {data_module.dataset_type}_time_series_sample.png")
    
    plt.show()


class SimpleLSTMPredictor(L.LightningModule):
    """
    Simple LSTM model for time series prediction to demonstrate training with causal data.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = None, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size or input_size)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        output = self.fc(lstm_out[:, -1, :])
        return output
    
    def training_step(self, batch, batch_idx):
        # Extract past data and target
        past_data = batch['past_data']  # (batch_size, time_steps, num_nodes, 1)
        target = batch['target']        # (batch_size, future_steps, num_nodes, 1)
        
        # Reshape for LSTM: combine node and feature dimensions
        batch_size, time_steps, num_nodes, features = past_data.shape
        past_data = past_data.view(batch_size, time_steps, num_nodes * features)
        
        # For simplicity, predict the first future step
        target_first_step = target[:, 0, :, :].view(batch_size, num_nodes * features)
        
        # Forward pass
        predictions = self(past_data)
        loss = self.criterion(predictions, target_first_step)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        past_data = batch['past_data']
        target = batch['target']
        
        batch_size, time_steps, num_nodes, features = past_data.shape
        past_data = past_data.view(batch_size, time_steps, num_nodes * features)
        target_first_step = target[:, 0, :, :].view(batch_size, num_nodes * features)
        
        predictions = self(past_data)
        loss = self.criterion(predictions, target_first_step)
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def train_simple_model(data_module: CausalDataModule, max_epochs: int = 5):
    """
    Train a simple LSTM model on the causal dataset.
    
    Args:
        data_module (CausalDataModule): The data module containing the dataset
        max_epochs (int): Maximum number of training epochs
    """
    print(f"\nTraining simple LSTM model on {data_module.dataset_type} dataset...")
    
    # Get dataset properties
    dataset = data_module.dataset
    input_size = dataset.series_dim * dataset.features_dim
    
    # Create model
    model = SimpleLSTMPredictor(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        output_size=input_size,
        learning_rate=0.001
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        logger=False,  # Disable logging for this example
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    print(f"Training completed!")
    
    return model, trainer


def demonstrate_utility_functions():
    """Demonstrate the utility functions for managing causal datasets."""
    print(f"\n{'='*60}")
    print("Demonstrating Utility Functions")
    print(f"{'='*60}")
    
    # Show data directory location
    data_dir = get_causal_data_dir()
    print(f"Causal data directory: {data_dir}")
    
    # List available datasets (before downloading)
    available = list_available_datasets()
    print(f"Currently available datasets: {available}")
    
    # Show dataset-specific directory
    diamond_dir = get_causal_data_dir('diamond')
    print(f"Diamond dataset directory: {diamond_dir}")
    
    return data_dir


def main():
    """Main function to run all examples."""
    print("CausalDataModule Example Script")
    print("===============================")
    
    # Demonstrate utility functions first
    data_dir = demonstrate_utility_functions()
    
    # Define the dataset types to explore
    dataset_types = ['diamond', 'fork', 'mediator', 'v']
    local_dir = None  # Use default ~/.fintorch_data/causal
    
    # Explore each dataset type
    data_modules = {}
    for dataset_type in dataset_types:
        try:
            data_module = explore_dataset(dataset_type, local_dir)
            data_modules[dataset_type] = data_module
            
            # Show dataset info after loading
            info = get_dataset_info(dataset_type)
            print(f"\nDataset info for {dataset_type}:")
            print(f"  Data files: {info['data_files']}")
            print(f"  Has groundtruth: {info['has_groundtruth']}")
            print(f"  Total size: {info['total_size_mb']} MB")
            
            # Visualize the causal structure (adjacency matrix and directed graph)
            visualize_causal_structure(data_module, save_plot=True)
            
            # Visualize a time series sample
            visualize_time_series_sample(data_module, save_plot=True)
            
        except Exception as e:
            print(f"Error processing {dataset_type} dataset: {e}")
            continue
    
    # Train a simple model on one of the datasets (if available)
    if data_modules:
        dataset_to_train = list(data_modules.keys())[0]
        print(f"\n{'='*60}")
        print(f"Training Example on {dataset_to_train.upper()} Dataset")
        print(f"{'='*60}")
        
        try:
            model, trainer = train_simple_model(data_modules[dataset_to_train], max_epochs=3)
            print("Model training example completed successfully!")
        except Exception as e:
            print(f"Error during model training: {e}")
    
    # Show final utility function examples
    print(f"\n{'='*60}")
    print("Final Utility Function Examples")
    print(f"{'='*60}")
    
    # List all available datasets after downloading
    available = list_available_datasets()
    print(f"Available datasets after downloading: {available}")
    
    # Show how to clear a specific dataset (commented out to avoid actually clearing)
    print(f"\nTo clear a specific dataset, use:")
    print(f"clear_causal_data('diamond')  # Clears only diamond dataset")
    print(f"clear_causal_data()  # Clears all causal datasets")
    
    print(f"\n{'='*60}")
    print("Example script completed!")
    print(f"Data stored in: {get_causal_data_dir()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()