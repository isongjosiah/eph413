import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Tuple, Any  # Added Any
import math
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Helper Function for Isolation Forest Normalization (Eq. 4) ---
# (Keep _avg_path_length function as is)
def _avg_path_length(n_samples: int) -> float:
    """Calculates the average path length normalization factor c(n)."""
    if n_samples <= 1:
        return 1.0
    h_n_minus_1 = math.log(n_samples - 1) + 0.5772156649 if n_samples > 1 else 0
    c_n = 2 * h_n_minus_1 - (2 * (n_samples - 1) / n_samples)
    return max(c_n, 1e-6)


# --- GeneticDefense Class ---
class GeneticDefense:
    """
    Implements genetic-aware defense mechanisms (HWE, AFC, Gradient Plausibility)
    using summary statistics provided by clients for privacy preservation.
    """

    def __init__(
        self,
        hwe_significance_threshold: float = 1e-6,
        ref_allele_freq_path: Optional[str] = None,
        iforest_n_estimators: int = 100,
        iforest_max_samples: str = "auto",
        iforest_contamination: str = "auto",
        iforest_random_state: int = 42,
    ):
        """Initializes the GeneticDefense module."""
        self.hwe_threshold = hwe_significance_threshold
        self.ref_allele_freq_path = ref_allele_freq_path
        self.reference_frequencies = (
            self._load_reference_frequencies()
        )  # Placeholder load

        self.isolation_forest = IsolationForest(
            n_estimators=iforest_n_estimators,
            max_samples=iforest_max_samples,
            contamination=iforest_contamination,
            random_state=iforest_random_state,
        )
        self.is_iforest_fitted = False
        self.iforest_c_n = 1.0

        logger.info(
            f"GeneticDefense initialized with HWE threshold={self.hwe_threshold}"
        )
        if self.reference_frequencies is None:
            logger.warning(
                "Reference allele frequencies not loaded. AFC checks will be skipped."
            )

    def _load_reference_frequencies(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Loads population-stratified reference allele frequencies (Placeholder)."""
        if self.ref_allele_freq_path is None:
            return None
        try:
            # TODO: Implement actual loading logic
            logger.info(
                f"Placeholder: Loading reference frequencies from {self.ref_allele_freq_path}"
            )
            ref_freqs = {  # Dummy example
                "EUR": {
                    "rs1": 0.1,
                    "rs2": 0.5,
                    "rs3": 0.01,
                    "variant_8": 0.02,
                    "variant_9": 0.001,
                },
                "AFR": {
                    "rs1": 0.4,
                    "rs2": 0.6,
                    "rs3": 0.005,
                    "variant_8": 0.05,
                    "variant_9": 0.008,
                },
            }
            logger.info(
                f"Loaded reference frequencies for populations: {list(ref_freqs.keys())}"
            )
            return ref_freqs
        except Exception as e:
            logger.error(f"Failed to load reference allele frequencies: {e}")
            return None

    def calculate_hwe(self, variant_stats: Dict[Any, Dict[str, int]]) -> float:
        """
        Calculates the fraction of variants significantly deviating from HWE,
        using pre-calculated genotype counts from the client.

        Args:
            variant_stats (Dict[Any, Dict[str, int]]): A dictionary mapping variant
                identifiers (e.g., index or rsID) to dictionaries containing
                genotype counts and total samples, e.g.,
                {variant_id: {'AA': count, 'Aa': count, 'aa': count, 'N': total_samples}}.

        Returns:
            float: Fraction of variants with HWE p-value < threshold.
                   Returns 0.0 if input is invalid or has no valid variants.
        """
        if not variant_stats:
            logger.warning("HWE calculation skipped: No variant statistics provided.")
            return 0.0

        violations = 0
        valid_tests = 0

        for variant_id, counts in variant_stats.items():
            n_AA = counts.get("AA", 0)
            n_Aa = counts.get("Aa", 0)
            n_aa = counts.get("aa", 0)
            n_total = counts.get(
                "N", n_AA + n_Aa + n_aa
            )  # Use provided N or sum counts

            if n_total < 3:  # Need at least 3 samples for calculation
                continue

            # Calculate allele frequency (alternative allele 'a')
            total_alleles = 2 * n_total
            if total_alleles == 0:
                continue
            q = (n_Aa + 2 * n_aa) / total_alleles
            p = 1.0 - q

            # Expected counts under HWE
            exp_AA = (p**2) * n_total
            exp_Aa = (2 * p * q) * n_total
            exp_aa = (q**2) * n_total

            observed = [n_AA, n_Aa, n_aa]
            expected = [exp_AA, exp_Aa, exp_aa]

            # Perform Chi-square test (same logic as before, using counts directly)
            if any(e < 1e-6 for e in expected):
                if any(o > 0 and e < 1e-6 for o, e in zip(observed, expected)):
                    p_value = 0.0
                else:
                    continue
            else:
                try:
                    chisq, p_value = stats.chisquare(observed, expected, ddof=0)
                except ValueError:
                    continue

            valid_tests += 1
            if p_value < self.hwe_threshold:
                violations += 1

        if valid_tests == 0:
            return 0.0

        hwe_violation_fraction = violations / valid_tests
        logger.debug(
            f"HWE Check: {violations}/{valid_tests} variants violate threshold {self.hwe_threshold}. Fraction={hwe_violation_fraction:.4f}"
        )
        return hwe_violation_fraction

    # _get_client_allele_freqs is REMOVED as client calculates this now

    def calculate_afc(
        self, client_freqs: Dict[Any, float], client_population_id: str
    ) -> float:
        """
        Calculates the Allele Frequency Consistency score using pre-calculated
        allele frequencies from the client.

        Args:
            client_freqs (Dict[Any, float]): Dictionary mapping variant identifiers
                                              to their allele frequencies in the client data.
            client_population_id (str): Identifier for the client's population.

        Returns:
            float: Average log-ratio divergence score (Eq. 3).
                   Returns 0.0 if reference frequencies unavailable, population ID invalid,
                   or no overlapping variants found.
        """
        if self.reference_frequencies is None:
            logger.warning("AFC calculation skipped: Reference frequencies not loaded.")
            return 0.0
        if not client_freqs:
            logger.warning("AFC calculation skipped: Client frequencies not provided.")
            return 0.0

        if client_population_id not in self.reference_frequencies:
            logger.warning(
                f"AFC calculation skipped: Population ID '{client_population_id}' not found in reference."
            )
            # [cite_start]TODO: Implement logic to find closest reference population if needed [cite: 161]
            return 0.0

        ref_pop_freqs = self.reference_frequencies[client_population_id]
        log_ratios = []

        for variant_id, client_f in client_freqs.items():
            ref_f = ref_pop_freqs.get(variant_id, None)  # Use variant_id directly

            # Ensure both frequencies are valid and non-zero for log calculation
            if ref_f is not None and ref_f > 1e-9 and client_f > 1e-9:
                log_ratio = abs(math.log(client_f / ref_f))
                log_ratios.append(log_ratio)

        if not log_ratios:
            logger.warning(
                f"AFC calculation: No overlapping/valid variants for population {client_population_id}."
            )
            return 0.0

        afc_score = np.mean(log_ratios)
        logger.debug(
            f"AFC Score for Pop '{client_population_id}': {afc_score:.4f} (based on {len(log_ratios)} variants)"
        )
        return afc_score

    # (Keep fit_gradient_plausibility and calculate_gradient_plausibility as they are - they don't use genotypes)
    def fit_gradient_plausibility(self, reference_updates: List[np.ndarray]):
        """Fits the Isolation Forest model on reference updates."""
        if not reference_updates:
            logger.error("Cannot fit Isolation Forest: No reference updates provided.")
            return
        update_matrix = np.vstack(reference_updates)
        n_samples_fit = update_matrix.shape[0]
        if n_samples_fit < 2:
            logger.warning("Need >= 2 reference updates to fit Isolation Forest.")
        logger.info(f"Fitting Isolation Forest on {n_samples_fit} reference updates...")
        self.isolation_forest.fit(update_matrix)
        self.is_iforest_fitted = True
        self.iforest_c_n = _avg_path_length(n_samples_fit)
        logger.info(f"Isolation Forest fitted. c(n) = {self.iforest_c_n:.4f}")

    def calculate_gradient_plausibility(self, update_vector: np.ndarray) -> float:
        """Calculates the anomaly score s_k using Isolation Forest (Eq. 4)."""
        if not self.is_iforest_fitted:
            logger.warning(
                "Gradient Plausibility skipped: Isolation Forest not fitted."
            )
            return 0.0
        if update_vector.ndim == 1:
            update_vector = update_vector.reshape(1, -1)
        try:
            decision_score = self.isolation_forest.decision_function(update_vector)[0]
            # Use sigmoid proxy: maps decision score (<0 outlier) to [0,1] (>0.5 outlier)
            anomaly_score_proxy = 1 / (1 + math.exp(-(-decision_score * 5)))
            return anomaly_score_proxy
        except Exception as e:
            logger.error(f"Error during gradient plausibility calculation: {e}")
            return 0.0

    def run_all_checks(
        self,
        client_id: int,
        hwe_stats: Dict[Any, Dict[str, int]],  # Input changed
        allele_freqs: Dict[Any, float],  # Input changed
        population_id: str,
        update_vector: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Runs all defense checks using summary statistics and the update vector.

        Args:
            client_id: Identifier of the client.
            hwe_stats: Dict with genotype counts per variant for HWE check.
            allele_freqs: Dict with allele frequencies per variant for AFC check.
            population_id: Client's population ID for AFC check.
            update_vector: Flattened model update vector for plausibility check.

        Returns:
            Tuple: (hwe_violation_fraction, afc_score, gradient_anomaly_score)
        """
        logger.info(
            f"Running defense checks for client {client_id} (Pop: {population_id})..."
        )
        hwe_score = self.calculate_hwe(hwe_stats)  # Use HWE stats
        afc_score = self.calculate_afc(allele_freqs, population_id)  # Use allele freqs
        grad_score = self.calculate_gradient_plausibility(update_vector)

        logger.info(
            f"Client {client_id} scores: HWE={hwe_score:.4f}, AFC={afc_score:.4f}, Grad Anomaly={grad_score:.4f}"
        )
        return hwe_score, afc_score, grad_score


# --- Example Usage ---
if __name__ == "__main__":
    print("Testing Refactored GeneticDefense module...")

    defense = GeneticDefense()

    # --- Dummy Client Summary Statistics ---
    # Client 1 (EUR-like, includes manipulated variants for HWE test)
    client1_hwe_stats = {
        "variant_8": {"AA": 40, "Aa": 45, "aa": 15, "N": 100},  # Normal
        "variant_9": {"AA": 50, "Aa": 0, "aa": 50, "N": 100},  # Manipulated: No hets
    }
    client1_allele_freqs = {
        "rs1": 0.12,
        "rs2": 0.48,
        "rs3": 0.015,  # Close to EUR ref
        "variant_8": (45 + 2 * 15) / (2 * 100),  # Freq for variant_8
        "variant_9": (0 + 2 * 50) / (2 * 100),  # Freq for variant_9
    }

    # Client 2 (AFR-like)
    client2_hwe_stats = {
        "variant_8": {"AA": 20, "Aa": 50, "aa": 30, "N": 100},  # Normal
        "variant_9": {"AA": 90, "Aa": 8, "aa": 2, "N": 100},  # Normal (rare)
    }
    client2_allele_freqs = {
        "rs1": 0.38,
        "rs2": 0.61,
        "rs3": 0.006,  # Close to AFR ref
        "variant_8": (50 + 2 * 30) / (2 * 100),  # Freq for variant_8
        "variant_9": (8 + 2 * 2) / (2 * 100),  # Freq for variant_9
    }

    # Dummy model updates (same as before)
    model_size = 1000
    client1_update = np.random.normal(0, 1, model_size)
    client2_update = np.random.normal(0, 1, model_size)
    malicious_update = np.random.normal(0, 5, model_size) + np.random.choice(
        [0, 10], size=model_size, p=[0.95, 0.05]
    )

    # --- Run Checks using Summaries ---
    print("\n--- HWE Check ---")
    hwe1 = defense.calculate_hwe(client1_hwe_stats)
    hwe2 = defense.calculate_hwe(client2_hwe_stats)
    print(f"Client 1 HWE Violation Fraction: {hwe1:.4f} (Expected > 0)")
    print(f"Client 2 HWE Violation Fraction: {hwe2:.4f} (Expected = 0)")

    print("\n--- AFC Check ---")
    afc1 = defense.calculate_afc(client1_allele_freqs, "EUR")
    afc2 = defense.calculate_afc(client2_allele_freqs, "AFR")
    afc1_wrong_pop = defense.calculate_afc(client1_allele_freqs, "AFR")
    print(f"Client 1 AFC Score (vs EUR ref): {afc1:.4f} (Expected low)")
    print(f"Client 2 AFC Score (vs AFR ref): {afc2:.4f} (Expected low)")
    print(f"Client 1 AFC Score (vs AFR ref): {afc1_wrong_pop:.4f} (Expected high)")

    # --- Gradient Plausibility (remains the same) ---
    print("\n--- Gradient Plausibility Check ---")
    defense.fit_gradient_plausibility([client1_update, client2_update])
    grad1_score = defense.calculate_gradient_plausibility(client1_update)
    grad2_score = defense.calculate_gradient_plausibility(client2_update)
    grad_mal_score = defense.calculate_gradient_plausibility(malicious_update)
    print(f"Client 1 Gradient Anomaly Score: {grad1_score:.4f}")
    print(f"Client 2 Gradient Anomaly Score: {grad2_score:.4f}")
    print(f"Malicious Gradient Anomaly Score: {grad_mal_score:.4f}")

    # --- Run All (using summaries) ---
    print("\n--- Combined Checks ---")
    scores1 = defense.run_all_checks(
        1, client1_hwe_stats, client1_allele_freqs, "EUR", client1_update
    )
    scores2 = defense.run_all_checks(
        2, client2_hwe_stats, client2_allele_freqs, "AFR", client2_update
    )
