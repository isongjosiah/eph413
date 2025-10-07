"""
Flower server for federated learning.
"""

import flwr as fl
from scripts.models.federated_client import FlowerClient
from scripts.models.central_model import PolygenicNeuralNetwork
from scripts.data.synthetic.genomic import partition_data, prepare_feature_matrix
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.preprocessing import StandardScaler

from scripts.models.strategy_factory import get_strategy
from flwr.common import ndarrays_to_parameters


def client_fn(cid: str, partitions):
    """Create a Flower client."""
    # Load data for this client
    client_data = partitions[int(cid)]

    X = client_data.iloc[:, :-1].values
    y_str = client_data.iloc[:, -1].values

    # Convert y to numeric
    y = np.array([0 if val == "Short" else 1 for val in y_str])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)
    )

    # Create model
    n_variants = X_train.shape[1]
    model = PolygenicNeuralNetwork(n_variants=n_variants, n_loci=100)

    return FlowerClient(model, train_dataset, val_dataset).to_client()


def run_federated_simulation(num_clients: int, strategy_name: str = "FedAvg"):
    """Run federated learning simulation."""

    feature_matrix = prepare_feature_matrix()
    partitions = partition_data(feature_matrix, num_partitions=num_clients)

    # Create a temporary model to get initial parameters
    n_variants = feature_matrix.shape[1] - 1
    temp_model = PolygenicNeuralNetwork(n_variants=n_variants, n_loci=100)
    initial_parameters_ndarrays = [
        val.cpu().numpy() for _, val in temp_model.state_dict().items()
    ]
    initial_parameters = ndarrays_to_parameters(initial_parameters_ndarrays)

    strategy = get_strategy(strategy_name, initial_parameters)

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, partitions),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    return history


if __name__ == "__main__":
    history = run_federated_simulation(num_clients=3)
    print(history)
