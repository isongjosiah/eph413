import logging
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrustManager:
    """
    Manages client trust scores based on defense mechanism feedback.

    Implements the multiplicative decay logic from Equation 5 of the
    XS-FedPRS paper [cite: 168-172].
    """

    def __init__(
        self,
        decay_rate: float = 0.2,
        hwe_threshold: float = 0.3,
        afc_threshold: float = 2.0,
        grad_plausibility_threshold: float = 0.6,
        initial_trust: float = 1.0,
        min_trust_for_aggregation: float = 0.1,
    ):
        """
        Initializes the TrustManager.

        Args:
            decay_rate (float): The decay rate 'lambda' used for penalizing
                                violations[cite: 170].
            hwe_threshold (float): The threshold 'theta_HWE' for HWE violations[cite: 170].
            afc_threshold (float): The threshold 'theta_AFC' for AFC violations[cite: 170].
            grad_plausibility_threshold (float): The threshold 'theta_s' for gradient
                                                 plausibility violations[cite: 170].
            initial_trust (float): The trust score assigned to new clients.
            min_trust_for_aggregation (float): The minimum trust score required for
                                               a client's update to be included
                                               in aggregation.
        """
        self.trust_scores: Dict[str, float] = {}
        self.decay_rate = decay_rate
        self.hwe_threshold = hwe_threshold
        self.afc_threshold = afc_threshold
        self.grad_threshold = grad_plausibility_threshold
        self.initial_trust = initial_trust
        self.min_trust_for_aggregation = min_trust_for_aggregation

        logger.info(
            f"TrustManager initialized with: "
            f"Decay Rate(λ)={decay_rate}, "
            f"HWE_θ={hwe_threshold}, "
            f"AFC_θ={afc_threshold}, "
            f"Grad_θ={grad_plausibility_threshold}"
        )

    def get_trust_score(self, client_id: str) -> float:
        """
        Retrieves the current trust score for a client.

        Args:
            client_id (str): The unique identifier for the client.

        Returns:
            float: The client's current trust score. Returns the initial trust
                   score if the client is not yet in the registry.
        """
        return self.trust_scores.get(client_id, self.initial_trust)

    def update_trust_score(
        self,
        client_id: str,
        hwe_score: float,
        afc_score: float,
        plausibility_score: float,
    ) -> float:
        """
        Updates a client's trust score based on the latest defense checks.
        Implements the multiplicative decay logic from Equation 5.

        Args:
            client_id (str): The client to update.
            hwe_score (float): The HWE violation fraction from the defense module.
            afc_score (float): The AFC log-ratio divergence from the defense module.
            plausibility_score (float): The gradient plausibility anomaly score (s_k).

        Returns:
            float: The new, updated trust score for the client.
        """
        current_trust = self.get_trust_score(client_id)

        # Calculate indicator functions (I) for violations
        # 1.0 if violation (score > threshold), 0.0 otherwise
        hwe_violation = 1.0 if hwe_score > self.hwe_threshold else 0.0
        afc_violation = 1.0 if afc_score > self.afc_threshold else 0.0
        grad_violation = 1.0 if plausibility_score > self.grad_threshold else 0.0

        # Apply multiplicative decay formula (Eq. 5)
        # new_trust = current_trust * (1 - λ*I(HWE)) * (1 - λ*I(AFC)) * (1 - λ*I(Grad))
        new_trust = (
            current_trust
            * (1.0 - self.decay_rate * hwe_violation)
            * (1.0 - self.decay_rate * afc_violation)
            * (1.0 - self.decay_rate * grad_violation)
        )

        # Ensure trust score doesn't go below 0
        new_trust = max(0.0, new_trust)

        # Store the updated score
        self.trust_scores[client_id] = new_trust

        if any([hwe_violation, afc_violation, grad_violation]):
            logger.warning(
                f"Client {client_id} triggered violations: "
                f"HWE={hwe_violation}, AFC={afc_violation}, Grad={grad_violation}. "
                f"Trust decreased from {current_trust:.4f} to {new_trust:.4f}"
            )
        else:
            logger.debug(
                f"Client {client_id} passed checks. Trust remains {new_trust:.4f}"
            )

        return new_trust

    def get_trusted_clients(self) -> List[str]:
        """
        Returns a list of client IDs that meet the minimum trust threshold
        for aggregation.

        Returns:
            List[str]: A list of trusted client IDs.
        """
        trusted_list = [
            client_id
            for client_id, score in self.trust_scores.items()
            if score >= self.min_trust_for_aggregation
        ]

        # Also include new clients (who have initial_trust) if initial > min
        if self.initial_trust >= self.min_trust_for_aggregation:
            all_clients = set(self.trust_scores.keys())
            # This logic is tricky. A client isn't in self.trust_scores until
            # after their first update. Let's adjust.

            # The server logic should call get_trust_score(cid) for all
            # *participating* clients. Let's assume self.trust_scores
            # contains all clients that have participated at least once.
            # A new client (not in map) gets initial_trust, which is >= min.

            # The function should probably return the whole map, and the
            # server logic should check each participating client.

            # Let's redefine: This returns clients *already known* to be trusted
            # The server logic must handle new clients separately.
            pass  # The list comprehension already does this.

        return trusted_list

    def get_all_scores(self) -> Dict[str, float]:
        """Returns a copy of the current trust score registry."""
        return self.trust_scores.copy()


# --- Example Usage ---
if __name__ == "__main__":
    print("Testing TrustManager module...")

    # Use thresholds from the paper [cite: 170]
    config = {
        "decay_rate": 0.2,
        "hwe_threshold": 0.3,
        "afc_threshold": 2.0,
        "grad_plausibility_threshold": 0.6,
        "initial_trust": 1.0,
    }

    trust_manager = TrustManager(**config)

    client_A = "client_A"  # Honest client
    client_B = "client_B"  # Malicious client

    # --- Round 1 ---
    print("\n--- Round 1 ---")
    # Client A (Honest)
    ta1 = trust_manager.update_trust_score(
        client_A, hwe_score=0.1, afc_score=0.5, plausibility_score=0.3
    )
    # Client B (Malicious: fails gradient check)
    tb1 = trust_manager.update_trust_score(
        client_B, hwe_score=0.1, afc_score=0.5, plausibility_score=0.7
    )

    print(f"Client A trust: {ta1:.4f} (Expected: 1.0)")
    print(f"Client B trust: {tb1:.4f} (Expected: 0.8)")

    # --- Round 2 ---
    print("\n--- Round 2 ---")
    # Client A (Honest)
    ta2 = trust_manager.update_trust_score(
        client_A, hwe_score=0.2, afc_score=1.0, plausibility_score=0.2
    )
    # Client B (Malicious: fails all checks)
    tb2 = trust_manager.update_trust_score(
        client_B, hwe_score=0.5, afc_score=3.0, plausibility_score=0.9
    )

    # Expected tb2 = 0.8 * (1 - 0.2*1) * (1 - 0.2*1) * (1 - 0.2*1) = 0.8 * 0.8 * 0.8 * 0.8 = 0.4096
    print(f"Client A trust: {ta2:.4f} (Expected: 1.0)")
    print(f"Client B trust: {tb2:.4f} (Expected: 0.4096)")

    # --- Round 3 (Client B is suppressed) ---
    print("\n--- Round 3 ---")
    tb3 = trust_manager.update_trust_score(
        client_B, hwe_score=0.5, afc_score=3.0, plausibility_score=0.9
    )
    # Expected tb3 = 0.4096 * (0.8**3) = 0.2097
    print(f"Client B trust: {tb3:.4f} (Expected: 0.2097)")

    # --- Round 4 (Client B is excluded) ---
    print("\n--- Round 4 ---")
    tb4 = trust_manager.update_trust_score(
        client_B, hwe_score=0.5, afc_score=3.0, plausibility_score=0.9
    )
    # Expected tb4 = 0.2097 * (0.8**3) = 0.1074
    print(f"Client B trust: {tb4:.4f} (Expected: 0.1074)")

    # --- Round 5 (Client B drops below threshold) ---
    print("\n--- Round 5 ---")
    tb5 = trust_manager.update_trust_score(
        client_B, hwe_score=0.5, afc_score=3.0, plausibility_score=0.9
    )
    # Expected tb5 = 0.1074 * (0.8**3) = 0.0549
    print(f"Client B trust: {tb5:.4f} (Expected: 0.0549)")

    print(
        f"\nFinal check: Is Client A trusted? {trust_manager.get_trust_score(client_A) >= trust_manager.min_trust_for_aggregation}"
    )
    print(
        f"Final check: Is Client B trusted? {trust_manager.get_trust_score(client_B) >= trust_manager.min_trust_for_aggregation}"
    )
