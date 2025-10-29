"""
This script provides an example of how to use SHAP (SHapley Additive exPlanations)
to explain the predictions of a trained model.
"""

import numpy as np
import shap
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from scripts.data.synthetic.genomic import GeneticDataGenerator
from scripts.models.central_model import PolygenicNeuralNetwork


def explain_central_model():
    """
    Trains a central model and generates SHAP explanations for its predictions.
    """
    # Generate synthetic data
    data_generator = GeneticDataGenerator(n_samples=1000, n_rare_variants=10)
    client_datasets = data_generator.create_federated_datasets(n_clients=1)
    data = client_datasets[0]

    prs_scores = data["prs_scores"].reshape(-1, 1)
    rare_dosages = data["rare_dosages"]
    X = np.concatenate((prs_scores, rare_dosages), axis=1)
    y = data["phenotype_binary"]
    print("feature length is ", len(X))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the central model
    n_variants = X_train.shape[1]
    central_model = PolygenicNeuralNetwork(n_variants=n_variants, n_loci=100)
    central_model.train_model(X_train, y_train, X_val, y_val)

    # Create a SHAP explainer
    # Since the model is a PyTorch neural network, we use the DeepExplainer.
    # We need to provide a background dataset to the explainer, which is typically a subset of the training data.
    background_data = torch.FloatTensor(X_train[:100])
    explainer = shap.DeepExplainer(central_model, background_data)

    # Explain predictions on a subset of the validation data
    to_explain = torch.FloatTensor(X_val[:5])
    shap_values = explainer.shap_values(to_explain)
    print("shap values")
    print(shap_values)
    print("feature names are")
    print([f"f_{i}" for i in range(X_val.shape[1])])

    # Generate and save a SHAP force plot
    # This plot shows how each feature contributes to the model's output for a single prediction.
    print(explainer.expected_value)
    print("shap values")
    print(shap_values[0][0])
    explanation = shap.Explanation(
        values=shap_values[0][0],
        base_values=explainer.expected_value,
        data=X_val[0],
        feature_names=[f"f_{i}" for i in range(X_val.shape[1])],
    )
    print("shap plot")
    print(explanation)

    shap.force_plot(explanation)
    plt.savefig("shap_force_plot.png")
    plt.close()

    print("SHAP force plot generated and saved to shap_force_plot.png")


if __name__ == "__main__":
    explain_central_model()

