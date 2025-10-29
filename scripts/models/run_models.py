"""
This script orchestrates the execution of different models, including federated and centralized approaches.
It loads the prepared data, runs the selected models, and generates comparison outputs and graphs
for analysis.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scripts.models.central_model import PolygenicNeuralNetwork
from sklearn.model_selection import train_test_split
from scripts.models.federated_server import run_federated_simulation
from scripts.models.mia import MembershipInferenceAttack
from scripts.models.hprs_model import HierarchicalPRSModel
from scripts.data.synthetic.genomic import GeneticDataGenerator
from scripts.explainability.explain import explain_central_model


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
    strategies = ["FedAvg", "FedProx", "FedAdam", "FedYogi", "FedAdagrad"]
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


def run_mia_experiment():
    """
    Runs the membership inference attack to evaluate privacy risks.
    """
    print("--- Running Membership Inference Attack (MIA) Experiment ---")
    
    # 1. Initialize and train the attack model
    mia = MembershipInferenceAttack(n_shadow_models=5, n_rare_variants=500)
    mia.train_shadow_models()
    mia.train_attack_model()

    # 2. Prepare target model and data
    print("\nPreparing target model and data for MIA...")
    n_rare_variants = 500
    data_generator = GeneticDataGenerator(n_rare_variants=n_rare_variants)
    client_datasets = data_generator.create_federated_datasets(n_clients=1)
    target_data = client_datasets[0]

    prs_tensor = torch.FloatTensor(target_data["prs_scores"].reshape(-1, 1))
    rare_tensor = torch.FloatTensor(target_data["rare_dosages"])
    phenotype_tensor = torch.FloatTensor(target_data["phenotype_binary"].reshape(-1, 1))
    dataset = TensorDataset(prs_tensor, rare_tensor, phenotype_tensor)

    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    member_data, non_member_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    target_model = HierarchicalPRSModel(n_rare_variants=n_rare_variants)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    train_loader = DataLoader(member_data, batch_size=32, shuffle=True)

    # 3. Train the target model
    print("Training the target model...")
    for epoch in range(10):
        target_model.train()
        for prs, rare, targets in train_loader:
            optimizer.zero_grad()
            outputs = target_model(prs, rare)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # 4. Run the attack
    print("Running the attack on the target model...")
    attack_accuracy = mia.run_attack(target_model, member_data, non_member_data)

    # 5. Report the results
    report = f"""
    Membership Inference Attack Report
    ==================================
    Attack Accuracy: {attack_accuracy:.4f}
    
    Interpretation:
    - An accuracy of 0.5 indicates the attack is no better than random guessing.
    - An accuracy closer to 1.0 suggests a higher privacy risk, as the model's
      predictions can be used to infer membership in the training data.
    """
    print(report)
    with open("federated_report.txt", "a") as f:
        f.write(report)


def run_explainability():
    """
    Runs the explainability analysis on the central model.
    """
    explain_central_model()


if __name__ == "__main__":
    run_explainability()
