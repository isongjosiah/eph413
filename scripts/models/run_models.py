"""
This script orchestrates the execution of different models, including federated and centralized approaches.
It loads the prepared data, runs the selected models, and generates comparison outputs and graphs
for analysis.
"""

import pandas as pd
from scripts.models.central_model import PolygenicNeuralNetwork
from sklearn.model_selection import train_test_split
from scripts.models.federated_server import run_federated_simulation

def run_central_model():
    """
    Runs the centralized logistic regression model.
    """
    # Load the prepared data
    data_path = "data/PSR/prepared_feature_matrix.csv"
    data = pd.read_csv(data_path)

    # Assume the last column is the target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1]
    label_map = {"Short": 0, "Tall": 1}
    y = y.map(label_map).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and run the central model
    n_variants = X_train.shape[1]
    central_model = PolygenicNeuralNetwork(n_variants=n_variants, n_loci=100)
    central_model.train_model(X_train, y_train, X_val, y_val)
    metrics = central_model.evaluate(X_val, y_val)

    print("Centralized Model Metrics:", metrics)

def run_federated_experiments():
    """Runs federated learning experiments with different strategies."""
    strategies = ["FedAvg", "FedProx"]
    results = {}

    for strategy in strategies:
        print(f"--- Running experiment with {strategy} ---")
        history = run_federated_simulation(num_clients=3, strategy_name=strategy)
        results[strategy] = history
        print(f"--- Finished experiment with {strategy} ---")

    print("\n--- Experiment Results ---")
    for strategy, history in results.items():
        print(f"Strategy: {strategy}")
        print(history)

if __name__ == "__main__":
    run_federated_experiments()
