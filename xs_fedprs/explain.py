import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import pandas as pd  # Import pandas for data loading

# Ensure models.py and trust.py are in the same package
try:
    from .models import HierarchicalModel
    from .trust import TrustManager
    from .data import load_data  # Import the data loader
except ImportError:
    # Handle case where script is run directly
    logger.warning("Relative imports failed. Trying absolute...")
    try:
        from models import HierarchicalModel
        from trust import TrustManager
        from data import load_data
    except ImportError:
        logger.error(
            "Could not import required modules (HierarchicalModel, TrustManager, load_data)."
        )
        # Define dummies if needed, but it's better to fix PYTHONPATH
        HierarchicalModel = (
            None  # This will cause an error later, which is intended if setup is wrong
        )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Explainer:
    """
    Implements multi-level explainability (XAI) for the HierarchicalModel.

    Provides methods for:
    - Variant-level attributions (Eq. 19, 20)
    - Pathway-level contribution ratios (Eq. 21)
    - Counterfactual generation (Eq. 23, 24)
    - Security-aware explanations (Eq. 25, 26)
    """

    def __init__(
        self,
        model: HierarchicalModel,
        gwas_betas: Dict[str, float],
        rare_variant_names: List[str],
        trust_manager: Optional[TrustManager] = None,
        variant_provenance: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initializes the Explainer.

        Args:
            model: The trained HierarchicalModel instance.
            gwas_betas: A dictionary mapping common variant IDs (e.g., 'rs123')
                        to their GWAS effect sizes (beta) for Eq. 20.
            rare_variant_names: An ordered list of rare variant names, matching
                                the order in the rare_tensor.
            trust_manager: An instance of TrustManager to query trust scores (for Eq. 26).
            variant_provenance: A dictionary mapping variant_id -> [list_of_client_ids]
                                (for Eq. 26).
        """
        if model is None:
            raise ValueError("Explainer requires a valid HierarchicalModel instance.")

        self.model = model
        self.model.eval()  # Set model to evaluation mode

        # For Eq. 20
        self.gwas_betas = gwas_betas
        self.rare_variant_names = rare_variant_names

        # For Eq. 26
        self.trust_manager = trust_manager
        self.variant_provenance = variant_provenance

        if self.trust_manager is None or self.variant_provenance is None:
            logger.warning(
                "TrustManager or VariantProvenance not provided. "
                "Security-aware confidence (Eq. 26) will be disabled."
            )

    def get_variant_attributions(
        self,
        prs_tensor: torch.Tensor,
        rare_tensor: torch.Tensor,
        common_genotypes_dict: Dict[str, int],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Computes variant attributions for a single data point.

        Args:
            prs_tensor: Tensor for PRS score, shape (1, 1).
            rare_tensor: Tensor for rare dosages, shape (1, n_rare_variants).
            common_genotypes_dict: Dict of common genotypes, e.g., {'rs123': 2}.

        Returns:
            A tuple (rare_attributions_dict, common_attributions_dict).
        """

        # --- 1. Rare Variant Attribution (Based on Eq. 19) ---
        # Using Input * Gradient as a proxy for |a_j * ∇_a_j(ŷ_j)|

        self.model.zero_grad()
        rare_tensor_with_grad = rare_tensor.clone().detach().requires_grad_(True)
        prs_tensor_for_grad = prs_tensor.clone().detach()

        # Forward pass to get output ŷ_j
        output = self.model(prs_tensor_for_grad, rare_tensor_with_grad)

        # Backward pass to get gradients ∇_a_j(ŷ_j)
        output.backward()

        rare_gradients = rare_tensor_with_grad.grad.data  # Shape: (1, n_rare_variants)
        rare_dosages = rare_tensor.data  # Shape: (1, n_rare_variants)

        # Attribution = |a_j * ∇_a_j|
        rare_attributions_tensor = torch.abs(rare_dosages * rare_gradients).squeeze(
            0
        )  # Squeeze batch dim

        rare_attributions_dict = {
            name: attr.item()
            for name, attr in zip(self.rare_variant_names, rare_attributions_tensor)
        }

        # --- 2. Common Variant Attribution (Eq. 20) ---
        # Eq. 20: Attribution = |β_v * g_jv|
        common_attributions_dict = {}
        for variant_id, genotype in common_genotypes_dict.items():
            beta = self.gwas_betas.get(variant_id, 0.0)
            attribution = abs(beta * genotype)
            common_attributions_dict[variant_id] = attribution

        return rare_attributions_dict, common_attributions_dict

    def get_pathway_contributions(
        self, prs_tensor: torch.Tensor, rare_tensor: torch.Tensor
    ) -> float:
        """
        Calculates the rare variant pathway contribution ratio (ρ_pathway)
        as per Equation 21.

        Args:
            prs_tensor: Tensor for PRS score, shape (1, 1).
            rare_tensor: Tensor for rare dosages, shape (1, n_rare_variants).

        Returns:
            The contribution ratio (float), where > 0.5 indicates rare pathway dominance.
        """
        h_c, h_r = None, None

        # 1. Register hooks to capture intermediate outputs h_c and h_r
        def hook_common(module, input, output):
            nonlocal h_c
            h_c = output.detach()

        def hook_rare(module, input, output):
            nonlocal h_r
            # Attention layer returns (features, weights), we want features
            h_r = output[0].detach()

        common_hook = self.model.common_pathway.register_forward_hook(hook_common)
        rare_hook = self.model.rare_attention.register_forward_hook(hook_rare)

        # 2. Run forward pass
        with torch.no_grad():
            self.model(prs_tensor, rare_tensor)

        # 3. Remove hooks
        common_hook.remove()
        rare_hook.remove()

        if h_c is None or h_r is None:
            logger.error(
                "Failed to capture intermediate hook data for pathway analysis."
            )
            return 0.0

        # 4. Get weights from the integration layer
        integration_layer_weights = self.model.integration_layer[0].weight
        common_dim = h_c.shape[1]
        w_out_c = integration_layer_weights[:, :common_dim]
        w_out_r = integration_layer_weights[:, common_dim:]

        # 5. Calculate L2 norms (Eq. 21)
        norm_w_out_c = torch.norm(w_out_c, p=2)
        norm_w_out_r = torch.norm(w_out_r, p=2)
        norm_h_c = torch.norm(h_c, p=2)
        norm_h_r = torch.norm(h_r, p=2)

        numerator = norm_w_out_r * norm_h_r
        denominator = (norm_w_out_c * norm_h_c) + numerator

        if denominator.item() == 0:
            return 0.0

        rho_pathway = numerator / denominator
        return rho_pathway.item()

    def _project_alleles(self, dosages: torch.Tensor) -> torch.Tensor:
        """
        Projects continuous dosages to the valid discrete allele space
        [0, 0.5, 1, 1.5, 2] as per Eq. 314 & 316.
        """
        projected = torch.round(dosages * 2.0) / 2.0
        projected = torch.clamp(projected, 0.0, 2.0)
        return projected

    def generate_counterfactual(
        self,
        prs_tensor: torch.Tensor,
        rare_tensor: torch.Tensor,
        risk_threshold: float = 0.5,
        lambda_l1: float = 0.1,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> Optional[torch.Tensor]:
        """
        Generates a counterfactual explanation using PGD (Eq. 23, 24).
        Finds minimal changes to rare variants to drop prediction below threshold.

        Args:
            prs_tensor: Fixed PRS score, shape (1, 1).
            rare_tensor: Original rare dosages, shape (1, n_rare_variants).
            risk_threshold: The target prediction threshold (τ_risk).
            lambda_l1: Weight for the L1 sparsity penalty.
            lr: Learning rate for the gradient descent.
            max_iter: Max iterations for the optimization.

        Returns:
            A tensor of counterfactual rare dosages, or None if not found.
        """
        a_j = rare_tensor.clone().detach()
        a_prime = a_j.clone().detach().requires_grad_(True)
        prs_fixed = prs_tensor.clone().detach()

        for i in range(max_iter):
            self.model.zero_grad()

            # 1. Calculate Loss (L_CF)
            current_prediction = self.model(prs_fixed, a_prime)
            loss_risk = torch.clamp(current_prediction - risk_threshold, min=0)
            loss_l1 = lambda_l1 * torch.norm(a_prime - a_j, p=1)
            loss_cf = loss_risk + loss_l1

            # 2. Check for success (using projected values)
            with torch.no_grad():
                a_prime_projected = self._project_alleles(a_prime)
                projected_prediction = self.model(prs_fixed, a_prime_projected)

            if projected_prediction.item() <= risk_threshold and loss_l1.item() > 1e-6:
                logger.info(f"Counterfactual found at iteration {i}.")
                return a_prime_projected.detach()

            # 3. Calculate gradient
            loss_cf.backward()
            grad = a_prime.grad
            if grad is None:
                continue

            # 4. Perform PGD step (Eq. 24)
            with torch.no_grad():
                updated_a = a_prime - lr * grad
                a_prime = self._project_alleles(updated_a)  # Project back

            a_prime.requires_grad_(True)

        logger.warning(f"Counterfactual not found after {max_iter} iterations.")
        return None

    def get_secure_explanation(
        self,
        data_point: Dict[str, Any],
        epsilon: float = 5.0,
        delta: float = 1e-5,
        dp_sensitivity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generates variant attributions with DP noise (Eq. 25) and
        trust-aware confidence scores (Eq. 26).

        Args:
            data_point: A dict with 'prs', 'rare_dosages', 'common_genotypes'.
            epsilon: Epsilon for differential privacy.
            delta: Delta for differential privacy.
            dp_sensitivity: Assumed max sensitivity (S) for attributions.

        Returns:
            A dictionary containing noisy attributions and confidence scores.
        """

        # 1. Prepare Tensors
        prs_tensor = torch.tensor([[data_point["prs"]]], dtype=torch.float32)
        rare_dosages_np = np.array([data_point["rare_dosages"]])
        rare_tensor = torch.from_numpy(rare_dosages_np).float()
        common_genos = data_point["common_genotypes"]

        # 2. Get base attributions
        rare_attrs, common_attrs = self.get_variant_attributions(
            prs_tensor, rare_tensor, common_genos
        )

        # --- 3. Apply DP Noise (Eq. 25) ---
        sigma_sq = (2 * (dp_sensitivity**2) * np.log(1 / delta)) / (epsilon**2)
        sigma = np.sqrt(sigma_sq)

        noisy_rare_attrs = {
            name: attr + np.random.normal(0, sigma) for name, attr in rare_attrs.items()
        }
        noisy_common_attrs = {
            name: attr + np.random.normal(0, sigma)
            for name, attr in common_attrs.items()
        }

        # --- 4. Calculate Trust-Aware Confidence (Eq. 26) ---
        if self.trust_manager is None or self.variant_provenance is None:
            logger.warning(
                "Trust info not available. Returning noisy attributions only."
            )
            return {
                "noisy_rare_attributions": noisy_rare_attrs,
                "noisy_common_attributions": noisy_common_attrs,
                "confidence_scores": None,
            }

        confidence_scores = {}
        all_noisy_attrs = {**noisy_rare_attrs, **noisy_common_attrs}

        for variant_id, noisy_attr in all_noisy_attrs.items():
            contributing_clients = self.variant_provenance.get(variant_id, [])

            if not contributing_clients:
                avg_trust = 0.5  # Default trust if provenance is unknown
            else:
                trust_scores = [
                    self.trust_manager.get_trust_score(cid)
                    for cid in contributing_clients
                ]
                avg_trust = np.mean(trust_scores) if trust_scores else 0.5

            # Confidence = Attribution * AvgTrust (Eq. 329)
            confidence = noisy_attr * avg_trust
            confidence_scores[variant_id] = confidence

        return {
            "noisy_rare_attributions": noisy_rare_attrs,
            "noisy_common_attributions": noisy_common_attrs,
            "confidence_scores": confidence_scores,
        }


# --- Example Usage (Using data loader) ---
if __name__ == "__main__":
    logger.info("Testing Explainer module...")

    # --- 1. Load Data ---
    prs, rare_dosages, pheno, rare_names, s_ids, pca_vecs = load_data(
        file_path="./data/PSR/final_combined_data.csv",
        pca_file_path="./psr/EUR.eigenvec",
    )

    if prs is None:
        logger.error("Data loading failed. Exiting example.")
        exit()

    logger.info(f"Loaded {len(prs)} samples.")

    # --- 2. Setup Dummy Components using Loaded Data Info ---
    n_rare_vars = rare_dosages.shape[1]

    # NOTE: Common variant info is not in the loaded file, so we must dummy it.
    common_names = ["rs1", "rs2"]
    dummy_gwas_betas = {"rs1": 0.05, "rs2": -0.02}

    dummy_model = HierarchicalModel(n_rare_variants=n_rare_vars)

    # Dummy Trust/Provenance (remains the same)
    dummy_trust_manager = TrustManager()
    dummy_trust_manager.update_trust_score("client_A", 0, 0, 0)
    dummy_trust_manager.update_trust_score("client_B", 0.5, 0, 0)  # Failed HWE
    dummy_provenance = {
        rare_names[3]: ["client_A", "client_B"],  # Use an actual loaded rare name
        "rs1": ["client_B"],
    }

    explainer = Explainer(
        model=dummy_model,
        gwas_betas=dummy_gwas_betas,
        rare_variant_names=rare_names,  # Use loaded rare names
        trust_manager=dummy_trust_manager,
        variant_provenance=dummy_provenance,
    )

    # --- 3. Get a Data Point from the Loader ---
    data_index = 0  # Use the first sample

    data_point = {
        "prs": prs[data_index],  # From loaded data
        "rare_dosages": rare_dosages[data_index],  # From loaded data
        # NOTE: common_genotypes are NOT in final_combined_data.csv.
        # We must use a hardcoded dummy for this example to run.
        "common_genotypes": {"rs1": 2, "rs2": 1},
    }

    # Convert to tensors for methods
    prs_t = torch.tensor([[data_point["prs"]]], dtype=torch.float32)
    rare_t = torch.tensor(data_point["rare_dosages"], dtype=torch.float32).unsqueeze(
        0
    )  # Shape (1, n_rare_vars)

    # --- Test Method 1: Variant Attributions ---
    print("\n--- 1. Variant Attributions ---")
    rare_attrs, common_attrs = explainer.get_variant_attributions(
        prs_t, rare_t, data_point["common_genotypes"]
    )
    print(f"Rare Attributions (Top 3): {list(rare_attrs.items())[:3]}")
    print(f"Common Attributions: {common_attrs}")

    # --- Test Method 2: Pathway Contributions ---
    print("\n--- 2. Pathway Contributions ---")
    rho = explainer.get_pathway_contributions(prs_t, rare_t)
    print(f"Rare Pathway Contribution (ρ_pathway): {rho:.4f}")

    # --- Test Method 3: Counterfactual ---
    print("\n--- 3. Counterfactual Generation ---")
    cf = explainer.generate_counterfactual(
        prs_t, rare_t, risk_threshold=0.1, max_iter=500
    )
    if cf is not None:
        changes = (cf != rare_t).sum().item()
        print(f"Counterfactual found with {changes} changes.")
        print(f"Original Prediction: {dummy_model(prs_t, rare_t).item():.4f}")
        print(f"Counterfactual Prediction: {dummy_model(prs_t, cf).item():.4f}")
    else:
        print("Counterfactual not found.")

    # --- Test Method 4: Secure Explanation ---
    print("\n--- 4. Secure Explanation ---")
    secure_expl = explainer.get_secure_explanation(data_point)
    print(
        f"Noisy Attributions (rs1): {secure_expl['noisy_common_attributions']['rs1']:.4f}"
    )
    # Client B (trust=0.8) contributed rs1
    print(f"Confidence (rs1): {secure_expl['confidence_scores']['rs1']:.4f}")
    # Use the 4th rare variant name (index 3) from loaded data
    target_rare_var = rare_names[3]
    print(
        f"Noisy Attributions ({target_rare_var}): {secure_expl['noisy_rare_attributions'][target_rare_var]:.4f}"
    )
    print(
        f"Confidence ({target_rare_var}): {secure_expl['confidence_scores'][target_rare_var]:.4f}"
    )
