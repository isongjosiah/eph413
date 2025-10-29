"""
Neural Network for Polygenic Risk Scoring
Based on Zhou et al. (2023) - Deep learning-based polygenic risk analysis for Alzhimer's disease

This implementation provides a complete framework for building training, and using neural
networks for polygenic risk scoring with support for federated learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class PolygenicNeuralNetwork(nn.Module):
    """
    A PyTorch implementation of the Polygenic Neural Network.
    """

    def __init__(
        self,
        n_variants: int,
        n_loci: int,
        dropout_rate: float = 0.3,
        random_seed: int = 42,
    ):
        super(PolygenicNeuralNetwork, self).__init__()
        self.n_variants = n_variants
        self.n_loci = n_loci or n_variants
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed

        # Defien the network layers
        self.pathway_network = nn.Sequential(
            nn.Linear(self.n_variants, 3 * self.n_loci),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(3 * self.n_loci, self.n_loci),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_loci, 22),
            nn.ReLU(),
        )

        self.pathway_layer = nn.Sequential(nn.Linear(22, 5), nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass."""
        x = self.pathway_network(x)
        pathway_scores = self.pathway_layer(x)
        risk_score = self.output_layer(pathway_scores)
        return risk_score

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.01,
    ):
        """Custom training loop for the PyTorch model."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train.reshape(-1, 1))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val.reshape(-1, 1))
        )

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()
            for inputs, labels in train_loader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    val_loss += criterion(outputs, labels).item()

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f} "
            )

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores for new samples."""
        self.eval()
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            scores = self(X_tensor)
        return scores.detach().numpy().flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model using AUROC and AUPRC"""
        y_pred = self.predict_risk_score(X)
        auroc = roc_auc_score(y, y_pred)
        auprc = average_precision_score(y, y_pred)
        return {"auroc": auroc, "auprc": auprc}


class PolygenicNeuralNetworkAM(nn.Module):
    """
    A PyTorch implementation of the Polygenic Neural Network.
    """

    def __init__(self, n_variants, n_loci, dropout_rate=0.3):
        super(PolygenicNeuralNetworkAM, self).__init__()
        self.n_variants = n_variants
        self.n_loci = n_loci or n_variants
        self.dropout_rate = dropout_rate

        self.pathway_network = nn.Sequential(
            nn.Linear(self.n_variants, 3 * self.n_loci),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(3 * self.n_loci, self.n_loci),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_loci, 22),
            nn.ReLU(),
        )

        # Parameters for additive attention: Wa and wa
        self.W_a = nn.Linear(22, 22)  # learnable weight matrix W_a
        self.w_a = nn.Linear(22, 1, bias=False)  # learnable weight vector w_a^T

        self.pathway_layer = nn.Sequential(nn.Linear(22, 5), nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

    def forward(self, x):
        batch_size, P_r = (
            x.shape[0],
            x.shape[1],
        )  # assuming input shape (batch, variants)

        # Obtain feature embeddings for each variant
        h_r = self.pathway_network(x)  # shape: (batch_size, P_r, 22)

        # Compute attention scores per variant
        # Apply W_a + tanh non-linearity
        u = torch.tanh(self.W_a(h_r))  # (batch_size, P_r, 22)
        # Compute raw scores by projecting to scalar
        scores = self.w_a(u).squeeze(-1)  # (batch_size, P_r)
        # Softmax normalize to obtain attention weights
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, P_r)

        # Compute weighted sum of variant embeddings
        attended = torch.sum(
            h_r * attn_weights.unsqueeze(-1), dim=1
        )  # (batch_size, 22)

        pathway_scores = self.pathway_layer(attended)
        risk_score = self.output_layer(pathway_scores)
        return risk_score

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.01,
    ):
        """Custom training loop for the PyTorch model."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train.reshape(-1, 1))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val.reshape(-1, 1))
        )

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()
            for inputs, labels in train_loader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    val_loss += criterion(outputs, labels).item()

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f} "
            )

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores for new samples."""
        self.eval()
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            scores = self(X_tensor)
        return scores.detach().numpy().flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model using AUROC and AUPRC"""
        y_pred = self.predict_risk_score(X)
        auroc = roc_auc_score(y, y_pred)
        auprc = average_precision_score(y, y_pred)
        return {"auroc": auroc, "auprc": auprc}
