import torch
import torch.nn as nn

class HierarchicalPRSModel(nn.Module):
    """
    Hierarchical two-pathway neural network for modelling common and rare variant contributions.
    Implements the architecture described in the RV-FedPRS methodology.
    """

    def __init__(
        self,
        n_rare_variants: int,
        common_hidden_dim: int = 16,
        rare_hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        super(HierarchicalPRSModel, self).__init__()

        self.common_pathway = nn.Sequential(
            nn.Linear(1, common_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(common_hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(common_hidden_dim, common_hidden_dim // 2),
            nn.ReLU(),
        )

        self.rare_pathway = nn.Sequential(
            nn.Linear(n_rare_variants, rare_hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(rare_hidden_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(rare_hidden_dim * 2, rare_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(rare_hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(rare_hidden_dim, rare_hidden_dim // 2),
            nn.ReLU(),
        )

        integration_input_dim = common_hidden_dim // 2 + rare_hidden_dim // 2
        self.integration_layer = nn.Sequential(
            nn.Linear(integration_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, prs_scores: torch.Tensor, rare_dosage: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the hierarchical model.

        Args:
            prs_scores: Common variant PRS scores (batch_size, 1)
            rare_dosages: Rare variant dosages (batch_size, n_rare_variants)

        Returns:
            Predictions (batch_size, 1)
        """

        h_common = self.common_pathway(prs_scores)
        h_rare = self.rare_pathway(rare_dosage)

        h_combined = torch.cat([h_common, h_rare], dim=1)
        output = self.integration_layer(h_combined)
        return output

    def get_pathway_gradients(
        self,
        prs_scores: torch.Tensor,
        rare_dosages: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Calculate gradients for each pathway to identify influential variants.

        Returns:
            Dictionary with gradient magnitudes for analysis
        """
        self.zero_grad()

        # Enable gradient computation for inputs
        prs_scores.requires_grad_(True)
        rare_dosages.requires_grad_(True)

        # Forward pass
        outputs = self.forward(prs_scores, rare_dosages)

        # Calculate loss
        criterion = nn.BCELoss()
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Get gradient magnitudes
        rare_gradients = rare_dosages.grad.abs().mean(dim=0).detach().cpu().numpy()

        return {"rare_variant_gradients": rare_gradients, "loss": loss.item()}
