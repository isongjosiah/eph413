# xs_fedprs/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd  # Import pandas for the dummy load_data

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score  # For evaluation
from .data import load_data


class AttentionLayer(nn.Module):
    """
    Implements the additive attention mechanism described in Equation 12.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.W_a = nn.Linear(feature_dim, feature_dim, bias=False)
        self.w_a = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, h_r_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = torch.tanh(self.W_a(h_r_features))
        scores = self.w_a(u)
        attention_weights = torch.sigmoid(scores)
        attended_features = h_r_features * attention_weights
        return attended_features, attention_weights


class HierarchicalModel(nn.Module):
    """
    Hierarchical two-pathway neural network model.
    """

    def __init__(
        self,
        n_rare_variants: int,
        common_hidden_dim: int = 16,
        rare_hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.n_rare_variants = n_rare_variants
        common_path_output_dim = common_hidden_dim // 2
        self.common_pathway = nn.Sequential(
            nn.Linear(1, common_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(common_hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(common_hidden_dim, common_path_output_dim),
            nn.ReLU(),
        )
        rare_path_intermediate_dim = rare_hidden_dim // 2
        self.rare_pathway_initial = nn.Sequential(
            nn.Linear(n_rare_variants, rare_hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(rare_hidden_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(rare_hidden_dim * 2, rare_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(rare_hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(rare_hidden_dim, rare_path_intermediate_dim),
            nn.ReLU(),
        )
        self.rare_attention = AttentionLayer(feature_dim=rare_path_intermediate_dim)
        rare_path_output_dim = rare_path_intermediate_dim
        integration_input_dim = common_path_output_dim + rare_path_output_dim
        self.integration_layer = nn.Sequential(
            nn.Linear(integration_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.attention_weights = None

    def forward(
        self, prs_scores: torch.Tensor, rare_dosages: torch.Tensor
    ) -> torch.Tensor:
        prs_scores = prs_scores.view(-1, 1)
        h_common = self.common_pathway(prs_scores)
        h_rare_initial = self.rare_pathway_initial(rare_dosages)
        h_rare_attended, self.attention_weights = self.rare_attention(h_rare_initial)
        h_combined = torch.cat([h_common, h_rare_attended], dim=1)
        output = self.integration_layer(h_combined)
        return output


if __name__ == "__main__":
    print("Running HierarchicalModel example with CENTRALIZED TRAINING...")

    # --- Configuration ---
    FILE_PATH = "./data/PSR/final_combined_data.csv"
    PCA_PATH = "./psr/EUR.eigenvec"  # Path to PCA file if needed by load_data
    VAL_SPLIT_RATIO = 0.2
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 500  # Keep low for example run
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load Data ---
    print(f"Loading data from {FILE_PATH}...")
    # Make sure load_data returns phenotypes as integers (0 or 1)
    prs, rare_dosages, pheno, rare_names, s_ids, pca_vecs = load_data(
        file_path=FILE_PATH, pca_file_path=PCA_PATH
    )

    if prs is None:
        print("Data loading failed. Exiting.")
    else:
        # --- Data Preprocessing & Splitting ---
        n_samples = len(prs)
        n_rare_vars_loaded = rare_dosages.shape[1]

        # Convert phenotypes to float for BCELoss target
        pheno = pheno.astype(np.float32)

        # Split into Training and Validation sets
        indices = np.arange(n_samples)
        train_indices, val_indices = train_test_split(
            indices,
            test_size=VAL_SPLIT_RATIO,
            random_state=SEED,
            stratify=pheno,  # Stratify by phenotype
        )

        # Create TensorDatasets
        train_dataset = TensorDataset(
            torch.from_numpy(prs[train_indices]).float(),
            torch.from_numpy(rare_dosages[train_indices, :]).float(),
            torch.from_numpy(pheno[train_indices])
            .float()
            .view(-1, 1),  # Target shape (batch, 1)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(prs[val_indices]).float(),
            torch.from_numpy(rare_dosages[val_indices, :]).float(),
            torch.from_numpy(pheno[val_indices])
            .float()
            .view(-1, 1),  # Target shape (batch, 1)
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(
            f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation samples."
        )

        # --- Model Initialization ---
        print(
            f"Initializing HierarchicalModel with {n_rare_vars_loaded} rare variants..."
        )
        model = HierarchicalModel(n_rare_variants=n_rare_vars_loaded).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- Training Loop ---
        print(f"Starting training for {EPOCHS} epochs...")
        for epoch in range(EPOCHS):
            model.train()
            epoch_train_loss = 0.0
            for batch_prs, batch_rare, batch_pheno in train_loader:
                batch_prs, batch_rare, batch_pheno = (
                    batch_prs.to(DEVICE),
                    batch_rare.to(DEVICE),
                    batch_pheno.to(DEVICE),
                )

                optimizer.zero_grad()
                outputs = model(batch_prs, batch_rare)
                loss = criterion(outputs, batch_pheno)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * batch_prs.size(0)

            avg_train_loss = epoch_train_loss / len(train_dataset)

            # --- Validation Phase ---
            model.eval()
            epoch_val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch_prs, batch_rare, batch_pheno in val_loader:
                    batch_prs, batch_rare, batch_pheno = (
                        batch_prs.to(DEVICE),
                        batch_rare.to(DEVICE),
                        batch_pheno.to(DEVICE),
                    )
                    outputs = model(batch_prs, batch_rare)
                    loss = criterion(outputs, batch_pheno)
                    epoch_val_loss += loss.item() * batch_prs.size(0)
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(batch_pheno.cpu().numpy())

            avg_val_loss = epoch_val_loss / len(val_dataset)
            all_preds = np.array(all_preds).flatten()
            all_targets = np.array(all_targets).flatten()
            val_accuracy = accuracy_score(all_targets, all_preds > 0.5)
            try:  # Calculate AUPRC, handle case with only one class in validation set
                val_auprc = average_precision_score(all_targets, all_preds)
            except ValueError:
                val_auprc = 0.0

            print(
                f"Epoch [{epoch+1:>{len(str(EPOCHS))}}/{EPOCHS}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Val AUPRC: {val_auprc:.4f}"
            )

        print("Training finished.")

        # --- Final Evaluation (optional) ---
        print("\nFinal evaluation on validation set:")
        print(f"  Val Loss : {avg_val_loss:.4f}")
        print(f"  Val Acc  : {val_accuracy:.4f}")
        print(f"  Val AUPRC: {val_auprc:.4f}")

        # Example: Perform a forward pass on the first validation batch
        print("\nExample forward pass on first validation batch...")
        model.eval()
        with torch.no_grad():
            sample_prs, sample_rare, sample_pheno = next(iter(val_loader))
            sample_prs, sample_rare = sample_prs.to(DEVICE), sample_rare.to(DEVICE)
            predictions = model(sample_prs, sample_rare)
            attention_w = model.attention_weights  # Check attention weights

        print(f"  Input PRS shape: {sample_prs.shape}")
        print(f"  Input Rare shape: {sample_rare.shape}")
        print(f"  Output Predictions shape: {predictions.shape}")
        if attention_w is not None:
            print(f"  Attention Weights shape: {attention_w.shape}")
            print(
                f"  Example Attention Weight (first sample): {attention_w[0].item():.4f}"
            )
