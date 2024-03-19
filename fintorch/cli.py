"""Console script for fintorch."""

import click

from fintorch.datasets import elliptic as e


@click.group()
def fintorch():
    """FinTorch CLI - Your financial AI toolkit"""
    pass


@fintorch.command()
@click.argument("dataset")
def datasets(dataset):
    """Download financial datasets"""
    # Implement your dataset download logic here
    click.echo(f"Downloading dataset: {dataset}")
    if dataset == "elliptic":
        e.EllipticDataset("~/.fintorch_data", force_reload=True)


@fintorch.command()
@click.argument("model")
def train(model):
    """Train financial models"""
    # Implement your model training logic here
    click.echo(f"Training model: {model}")


if __name__ == "__main__":
    fintorch()
