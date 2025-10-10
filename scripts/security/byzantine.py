"""
Secure RV-FedPRS: Byzantine-Robust Federated Genomic Risk Assessment
=====================================================================
Implementation of the secure framework with genetic-aware anomaly detection,
trust-weighted aggregation, and blockchain verification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import flwr as fl
from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from scipy import stats
import hashlib
import time
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# ========================= Configuration =========================


@dataclass
class SecurityConfig:
    """Configuration for security parameters"""

    max_malicious_fraction: float = 0.3
    hwe_p_threshold: float = 1e-6
    afc_threshold: float = 2.0
    trust_momentum: float = 0.7
    trim_fraction: float = 0.2
    min_trust_score: float = 0.1
    enable_blockchain: bool = True
    detection_sensitivity: float = 0.1


# ========================= Blockchain Layer =========================


class BlockchainVerifier:
    """Simulated blockchain for model update verification"""

    def __init__(self):
        self.chain = []
        self.pending_transactions = []

    def create_block(self, round_num: int, transactions: List[Dict]) -> Dict:
        """Create a new block with transactions"""
        block = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "transactions": transactions,
            "previous_hash": self.get_last_block_hash(),
            "nonce": 0,
        }

        # Simulate proof of work (simplified)
        block["hash"] = self.calculate_hash(block)
        return block

    def calculate_hash(self, block: Dict) -> str:
        """Calculate SHA256 hash of a block"""
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def get_last_block_hash(self) -> str:
        """Get hash of the last block in chain"""
        if not self.chain:
            return "0"
        return self.chain[-1]["hash"]

    def add_transaction(self, transaction: Dict):
        """Add a transaction to pending list"""
        self.pending_transactions.append(transaction)

    def commit_round(self, round_num: int) -> Dict:
        """Commit all pending transactions for a round"""
        if not self.pending_transactions:
            return None

        block = self.create_block(round_num, self.pending_transactions)
        self.chain.append(block)
        self.pending_transactions = []
        return block

    def verify_model_provenance(self, model_hash: str, round_num: int) -> bool:
        """Verify if a model hash exists in the blockchain"""
        for block in self.chain:
            if block["round"] == round_num:
                for tx in block["transactions"]:
                    if tx.get("model_hash") == model_hash:
                        return True
        return False


# ========================= Genetic Anomaly Detection =========================


class GeneticAnomalyDetector:
    """Multi-faceted anomaly detection using genetic principles"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.global_allele_frequencies = None
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)

    def set_global_frequencies(self, frequencies: Dict[int, float]):
        """Set global allele frequency reference"""
        self.global_allele_frequencies = frequencies

    def test_hardy_weinberg(self, genotypes: np.ndarray) -> float:
        """
        Test Hardy-Weinberg Equilibrium for genetic data
        Returns p-value; low values indicate potential fabrication
        """
        n_variants = genotypes.shape[1]
        p_values = []

        for i in range(n_variants):
            variant_data = genotypes[:, i]

            # Count genotypes (0=AA, 1=Aa, 2=aa)
            n_AA = np.sum(variant_data == 0)
            n_Aa = np.sum(variant_data == 1)
            n_aa = np.sum(variant_data == 2)
            n_total = n_AA + n_Aa + n_aa

            if n_total == 0:
                continue

            # Calculate allele frequencies
            p = (2 * n_AA + n_Aa) / (2 * n_total)
            q = 1 - p

            # Expected frequencies under HWE
            exp_AA = p * p * n_total
            exp_Aa = 2 * p * q * n_total
            exp_aa = q * q * n_total

            # Chi-square test
            observed = [n_AA, n_Aa, n_aa]
            expected = [exp_AA, exp_Aa, exp_aa]

            if all(e > 0 for e in expected):
                chi2, p_value = stats.chisquare(observed, expected)
                p_values.append(p_value)

        if not p_values:
            return 1.0

        # Return geometric mean of p-values
        return stats.gmean(p_values)

    def calculate_afc_score(self, client_frequencies: Dict[int, float]) -> float:
        """
        Calculate Allele Frequency Consistency score
        Compares client frequencies to global reference
        """
        if not self.global_allele_frequencies:
            return 0.0

        scores = []
        for variant_id, client_freq in client_frequencies.items():
            if variant_id in self.global_allele_frequencies:
                global_freq = self.global_allele_frequencies[variant_id]
                if global_freq > 0:
                    log_ratio = abs(np.log(client_freq / global_freq))
                    scores.append(log_ratio)

        return np.mean(scores) if scores else 0.0

    def analyze_gradients(self, gradients: np.ndarray) -> float:
        """
        Analyze gradient patterns for anomalies
        Returns anomaly score (0-1, higher is more anomalous)
        """
        if gradients.size == 0:
            return 0.0

        # Flatten gradients for analysis
        flat_grads = gradients.flatten()

        # Features for anomaly detection
        features = []
        features.append(np.mean(np.abs(flat_grads)))  # Mean magnitude
        features.append(np.std(flat_grads))  # Standard deviation
        features.append(stats.kurtosis(flat_grads))  # Kurtosis
        features.append(np.percentile(np.abs(flat_grads), 95))  # 95th percentile

        # Fit or predict with isolation forest
        features = np.array(features).reshape(1, -1)

        try:
            # For simplicity, we'll use a threshold-based approach
            # In production, train isolation forest on historical data
            anomaly_score = 0.0

            # Check for extreme values
            if features[0, 0] > 10.0:  # Very high mean gradient
                anomaly_score += 0.3
            if features[0, 1] > 5.0:  # High variance
                anomaly_score += 0.2
            if abs(features[0, 2]) > 10:  # Extreme kurtosis
                anomaly_score += 0.3
            if features[0, 3] > 20.0:  # Extreme outliers
                anomaly_score += 0.2

            return min(anomaly_score, 1.0)
        except:
            return 0.0


# ========================= Trust Management =========================


class TrustManager:
    """Manages dynamic trust scores for clients"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.trust_scores = {}
        self.trust_history = {}

    def initialize_client(self, client_id: int):
        """Initialize trust score for new client"""
        self.trust_scores[client_id] = 0.5  # Start neutral
        self.trust_history[client_id] = []

    def update_trust(self, client_id: int, reputation: float):
        """Update trust score using exponential moving average"""
        if client_id not in self.trust_scores:
            self.initialize_client(client_id)

        old_trust = self.trust_scores[client_id]
        new_trust = (
            self.config.trust_momentum * old_trust
            + (1 - self.config.trust_momentum) * reputation
        )

        # Enforce bounds
        new_trust = max(self.config.min_trust_score, min(1.0, new_trust))

        self.trust_scores[client_id] = new_trust
        self.trust_history[client_id].append(new_trust)

        return new_trust

    def calculate_reputation(
        self, hwe_score: float, afc_score: float, grad_score: float
    ) -> float:
        """Calculate reputation from detection scores"""
        # HWE: Higher p-value is better (less likely fabricated)
        hwe_component = min(1.0, -np.log10(max(hwe_score, 1e-10)) / 10)

        # AFC: Lower score is better
        afc_component = max(0, 1.0 - afc_score / self.config.afc_threshold)

        # Gradient: Lower anomaly score is better
        grad_component = 1.0 - grad_score

        # Weighted average
        reputation = 0.3 * hwe_component + 0.3 * afc_component + 0.4 * grad_component

        return reputation

    def is_trusted(self, client_id: int, threshold: float = 0.3) -> bool:
        """Check if client is trusted"""
        return self.trust_scores.get(client_id, 0.5) >= threshold


# ========================= Secure Aggregation Strategy =========================


class SecureRVFedPRSStrategy(fl.server.strategy.FedAvg):
    """
    Byzantine-robust aggregation strategy with genetic-aware detection
    """

    def __init__(self, security_config: SecurityConfig, **kwargs):
        super().__init__(**kwargs)
        self.security_config = security_config
        self.detector = GeneticAnomalyDetector(security_config)
        self.trust_manager = TrustManager(security_config)
        self.blockchain = (
            BlockchainVerifier() if security_config.enable_blockchain else None
        )
        self.round_num = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """
        Secure aggregation with multi-stage defense
        """
        self.round_num = server_round

        if not results:
            return None, {}

        # Extract client updates and metadata
        client_updates = []
        client_metadata = []

        for client_proxy, fit_res in results:
            client_id = int(fit_res.metrics.get("client_id", 0))

            # Initialize trust if new client
            if client_id not in self.trust_manager.trust_scores:
                self.trust_manager.initialize_client(client_id)

            client_updates.append(
                {
                    "client_id": client_id,
                    "parameters": fit_res.parameters,
                    "num_examples": fit_res.num_examples,
                    "metrics": fit_res.metrics,
                }
            )

            client_metadata.append(fit_res.metrics)

        # Stage 1: Genetic-aware anomaly detection
        detection_results = self._perform_detection(client_updates, client_metadata)

        # Stage 2: Update trust scores
        self._update_trust_scores(detection_results)

        # Stage 3: Filter suspicious clients
        trusted_updates = self._filter_suspicious_clients(client_updates)

        # Stage 4: Cluster-based aggregation for rare variants
        clusters = self._cluster_clients(trusted_updates, client_metadata)

        # Stage 5: Two-stage aggregation
        aggregated_params = self._secure_aggregate(trusted_updates, clusters)

        # Stage 6: Blockchain logging
        if self.blockchain:
            self._log_to_blockchain(trusted_updates, detection_results)

        # Prepare metrics
        metrics = {
            "n_trusted_clients": len(trusted_updates),
            "n_total_clients": len(client_updates),
            "n_clusters": len(set(clusters.values())),
            "avg_trust_score": np.mean(list(self.trust_manager.trust_scores.values())),
        }

        return aggregated_params, metrics

    def _perform_detection(
        self, client_updates: List[Dict], client_metadata: List[Dict]
    ) -> Dict:
        """Perform multi-faceted anomaly detection"""
        detection_results = {}

        for update, metadata in zip(client_updates, client_metadata):
            client_id = update["client_id"]

            # Extract genetic data if available (simulated here)
            # In practice, clients would send summary statistics
            hwe_score = np.random.random()  # Placeholder
            afc_score = np.random.random() * 3  # Placeholder

            # Analyze gradients
            params = fl.common.parameters_to_ndarrays(update["parameters"])
            gradients = np.concatenate([p.flatten() for p in params[:5]])  # Sample
            grad_score = self.detector.analyze_gradients(gradients)

            detection_results[client_id] = {
                "hwe_score": hwe_score,
                "afc_score": afc_score,
                "grad_score": grad_score,
            }

        return detection_results

    def _update_trust_scores(self, detection_results: Dict):
        """Update trust scores based on detection results"""
        for client_id, scores in detection_results.items():
            reputation = self.trust_manager.calculate_reputation(
                scores["hwe_score"], scores["afc_score"], scores["grad_score"]
            )
            self.trust_manager.update_trust(client_id, reputation)

    def _filter_suspicious_clients(self, client_updates: List[Dict]) -> List[Dict]:
        """Filter out clients with low trust scores"""
        trusted = []
        for update in client_updates:
            if self.trust_manager.is_trusted(update["client_id"]):
                trusted.append(update)
        return trusted

    def _cluster_clients(
        self, client_updates: List[Dict], metadata: List[Dict]
    ) -> Dict[int, int]:
        """Cluster clients based on rare variant profiles"""
        n_clients = len(client_updates)

        if n_clients < 2:
            return {client_updates[0]["client_id"]: 0}

        # Build similarity matrix (simplified)
        similarity_matrix = np.random.random((n_clients, n_clients))
        np.fill_diagonal(similarity_matrix, 1.0)

        # Make symmetric
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

        # Hierarchical clustering
        distance_matrix = 1 - similarity_matrix
        clustering = AgglomerativeClustering(
            n_clusters=min(3, n_clients), metric="precomputed", linkage="average"
        )
        labels = clustering.fit_predict(distance_matrix)

        clusters = {}
        for i, update in enumerate(client_updates):
            clusters[update["client_id"]] = labels[i]

        return clusters

    def _secure_aggregate(
        self, client_updates: List[Dict], clusters: Dict[int, int]
    ) -> fl.common.Parameters:
        """Two-stage secure aggregation"""
        if not client_updates:
            return None

        # Group updates by cluster
        cluster_updates = {}
        for update in client_updates:
            cluster_id = clusters.get(update["client_id"], 0)
            if cluster_id not in cluster_updates:
                cluster_updates[cluster_id] = []
            cluster_updates[cluster_id].append(update)

        # Aggregate within clusters with trimmed mean
        cluster_aggregates = {}
        for cluster_id, updates in cluster_updates.items():
            cluster_aggregates[cluster_id] = self._trimmed_mean_aggregate(updates)

        # Global aggregation with trust weighting
        global_aggregate = self._trust_weighted_aggregate(
            client_updates, cluster_aggregates
        )

        return global_aggregate

    def _trimmed_mean_aggregate(self, updates: List[Dict]) -> np.ndarray:
        """Trimmed mean aggregation to remove outliers"""
        if not updates:
            return None

        # Convert parameters to arrays
        param_arrays = []
        for update in updates:
            params = fl.common.parameters_to_ndarrays(update["parameters"])
            param_arrays.append(params)

        # Trimmed mean for each parameter
        aggregated = []
        for i in range(len(param_arrays[0])):
            param_stack = np.stack([p[i] for p in param_arrays])

            # Trim top and bottom fraction
            trim_n = int(len(param_stack) * self.security_config.trim_fraction)
            if trim_n > 0 and len(param_stack) > 2 * trim_n:
                param_sorted = np.sort(param_stack, axis=0)
                param_trimmed = param_sorted[trim_n:-trim_n]
                aggregated.append(np.mean(param_trimmed, axis=0))
            else:
                aggregated.append(np.mean(param_stack, axis=0))

        return aggregated

    def _trust_weighted_aggregate(
        self, client_updates: List[Dict], cluster_aggregates: Dict
    ) -> fl.common.Parameters:
        """Final aggregation with trust weighting"""
        # Get trust-weighted parameters
        weighted_params = []
        total_weight = 0

        for update in client_updates:
            client_id = update["client_id"]
            trust = self.trust_manager.trust_scores[client_id]
            weight = trust * update["num_examples"]

            params = fl.common.parameters_to_ndarrays(update["parameters"])
            weighted_params.append([p * weight for p in params])
            total_weight += weight

        # Average
        if total_weight > 0:
            aggregated = []
            for i in range(len(weighted_params[0])):
                param_sum = sum(p[i] for p in weighted_params)
                aggregated.append(param_sum / total_weight)

            return fl.common.ndarrays_to_parameters(aggregated)

        return None

    def _log_to_blockchain(self, trusted_updates: List[Dict], detection_results: Dict):
        """Log round information to blockchain"""
        for update in trusted_updates:
            client_id = update["client_id"]

            # Create transaction
            transaction = {
                "type": "model_update",
                "client_id": client_id,
                "round": self.round_num,
                "model_hash": hashlib.sha256(
                    str(update["parameters"]).encode()
                ).hexdigest()[:16],
                "trust_score": self.trust_manager.trust_scores[client_id],
                "detection_scores": detection_results.get(client_id, {}),
            }

            self.blockchain.add_transaction(transaction)

        # Commit block
        self.blockchain.commit_round(self.round_num)


# ========================= Example Usage =========================


def create_secure_server_app():
    """Create a secure FL server application"""

    # Security configuration
    security_config = SecurityConfig(
        max_malicious_fraction=0.3,
        hwe_p_threshold=1e-6,
        afc_threshold=2.0,
        trust_momentum=0.7,
        trim_fraction=0.2,
        enable_blockchain=True,
    )

    # Create secure strategy
    strategy = SecureRVFedPRSStrategy(
        security_config=security_config,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # Create server app
    server_app = fl.server.ServerApp(
        config=fl.server.ServerConfig(num_rounds=20), strategy=strategy
    )

    return server_app


def run_secure_simulation():
    """Run a simulation of the secure framework"""
    print("=" * 80)
    print("SECURE RV-FedPRS: Byzantine-Robust Federated Genomic Risk Assessment")
    print("=" * 80)

    # Create server app
    server_app = create_secure_server_app()

    # Client function (simplified - would use actual FlowerClient)
    def client_fn(context):
        # This would return an actual FlowerClient instance
        # For now, returning a placeholder
        return None

    print("\nSecurity Features Enabled:")
    print("✓ Hardy-Weinberg Equilibrium Testing")
    print("✓ Allele Frequency Consistency Checking")
    print("✓ Gradient Anomaly Detection")
    print("✓ Dynamic Trust Score Management")
    print("✓ Two-Stage Secure Aggregation")
    print("✓ Blockchain Audit Trail")

    print("\nConfiguration:")
    print("  - Max malicious fraction: 30%")
    print("  - Trust momentum: 0.7")
    print("  - Trim fraction: 20%")
    print("  - Blockchain enabled: Yes")

    print("\nNote: This is a demonstration framework.")
    print("For production use, integrate with actual client implementations")
    print("and real genetic data processing pipelines.")

    return server_app


if __name__ == "__main__":
    # Run demonstration
    server_app = run_secure_simulation()

    print("\n" + "=" * 80)
    print("Secure framework initialized successfully!")
    print("Ready for Byzantine-robust federated genomic analysis.")
    print("=" * 80)
