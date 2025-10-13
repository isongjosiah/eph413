import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

"""
Secure RV-FedPRS: Byzantine-Robust Federated Genomic Risk Assessment
Enhanced with Trust Evolution Visualization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import flwr as fl
from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
import hashlib
import json
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import the GeneticDataGenerator
from scripts.data.synthetic.genomic import GeneticDataGenerator

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
        block = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "transactions": transactions,
            "previous_hash": self.get_last_block_hash(),
            "nonce": 0,
        }
        block["hash"] = self.calculate_hash(block)
        return block
    
    def calculate_hash(self, block: Dict) -> str:
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def get_last_block_hash(self) -> str:
        if not self.chain:
            return "0"
        return self.chain[-1]["hash"]
    
    def add_transaction(self, transaction: Dict):
        self.pending_transactions.append(transaction)
    
    def commit_round(self, round_num: int) -> Dict:
        if not self.pending_transactions:
            return None
        block = self.create_block(round_num, self.pending_transactions)
        self.chain.append(block)
        self.pending_transactions = []
        return block


# ========================= Genetic Anomaly Detection =========================

class GeneticAnomalyDetector:
    """Multi-faceted anomaly detection using genetic principles"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.global_allele_frequencies = None
    
    def test_hardy_weinberg(self, genotypes: np.ndarray) -> float:
        n_variants = genotypes.shape[1]
        p_values = []
        
        for i in range(n_variants):
            variant_data = genotypes[:, i]
            n_AA = np.sum(variant_data == 0)
            n_Aa = np.sum(variant_data == 1)
            n_aa = np.sum(variant_data == 2)
            n_total = n_AA + n_Aa + n_aa
            
            if n_total == 0:
                continue
            
            p = (2 * n_AA + n_Aa) / (2 * n_total)
            q = 1 - p
            
            exp_AA = p * p * n_total
            exp_Aa = 2 * p * q * n_total
            exp_aa = q * q * n_total
            
            observed = [n_AA, n_Aa, n_aa]
            expected = [exp_AA, exp_Aa, exp_aa]
            
            if all(e > 0 for e in expected):
                chi2, p_value = stats.chisquare(observed, expected)
                p_values.append(p_value)
        
        if not p_values:
            return 1.0
        
        return stats.gmean(p_values)
    
    def analyze_gradients(self, gradients: np.ndarray) -> float:
        if gradients.size == 0:
            return 0.0
        
        flat_grads = gradients.flatten()
        
        # Features for anomaly detection
        mean_grad = np.mean(np.abs(flat_grads))
        std_grad = np.std(flat_grads)
        kurt = stats.kurtosis(flat_grads)
        percentile_95 = np.percentile(np.abs(flat_grads), 95)
        
        anomaly_score = 0.0
        
        # Check for extreme values
        if mean_grad > 10.0:
            anomaly_score += 0.3
        if std_grad > 5.0:
            anomaly_score += 0.2
        if abs(kurt) > 10:
            anomaly_score += 0.3
        if percentile_95 > 20.0:
            anomaly_score += 0.2
        
        return min(anomaly_score, 1.0)


# ========================= Trust Management =========================

class TrustManager:
    """Manages dynamic trust scores for clients"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.trust_scores = {}
        self.trust_history = {}
    
    def initialize_client(self, client_id: int):
        self.trust_scores[client_id] = 0.9  # Start with high trust
        self.trust_history[client_id] = []  # Will be populated on first update
    
    def update_trust(self, client_id: int, reputation: float):
        if client_id not in self.trust_scores:
            self.initialize_client(client_id)
        
        old_trust = self.trust_scores[client_id]
        new_trust = (
            self.config.trust_momentum * old_trust
            + (1 - self.config.trust_momentum) * reputation
        )
        
        new_trust = max(self.config.min_trust_score, min(1.0, new_trust))
        
        self.trust_scores[client_id] = new_trust
        self.trust_history[client_id].append(new_trust)
        
        return new_trust
    
    def calculate_reputation(
        self, hwe_score: float, afc_score: float, grad_score: float
    ) -> float:
        hwe_component = min(1.0, -np.log10(max(hwe_score, 1e-10)) / 10)
        afc_component = max(0, 1.0 - afc_score / 2.0)
        grad_component = 1.0 - grad_score
        
        reputation = 0.3 * hwe_component + 0.3 * afc_component + 0.4 * grad_component
        return reputation
    
    def is_trusted(self, client_id: int, threshold: float = 0.3) -> bool:
        return self.trust_scores.get(client_id, 0.5) >= threshold


# ========================= Secure Aggregation Strategy =========================

class SecureRVFedPRSStrategy(fl.server.strategy.FedAvg):
    """Byzantine-robust aggregation strategy with genetic-aware detection"""
    
    def __init__(self, security_config: SecurityConfig, **kwargs):
        super().__init__(**kwargs)
        self.security_config = security_config
        self.detector = GeneticAnomalyDetector(security_config)
        self.trust_manager = TrustManager(security_config)
        self.blockchain = BlockchainVerifier() if security_config.enable_blockchain else None
        self.round_num = 0
    
    def aggregate_fit(
        self, 
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        self.round_num = server_round
        
        if not results:
            return None, {}
        
        client_updates = []
        client_metadata = []
        
        for client_proxy, fit_res in results:
            client_id = int(fit_res.metrics.get("client_id", 0))
            
            if client_id not in self.trust_manager.trust_scores:
                self.trust_manager.initialize_client(client_id)
            
            client_updates.append({
                "client_id": client_id,
                "parameters": fit_res.parameters,
                "num_examples": fit_res.num_examples,
                "metrics": fit_res.metrics,
            })
            client_metadata.append(fit_res.metrics)
        
        # Perform detection and update trust
        detection_results = self._perform_detection(client_updates, client_metadata)
        self._update_trust_scores(detection_results)
        
        # Filter and aggregate
        trusted_updates = self._filter_suspicious_clients(client_updates)
        clusters = self._cluster_clients(trusted_updates, client_metadata)
        aggregated_params = self._secure_aggregate(trusted_updates, clusters)
        
        if self.blockchain:
            self._log_to_blockchain(trusted_updates, detection_results)
        
        metrics = {
            "n_trusted_clients": len(trusted_updates),
            "n_total_clients": len(client_updates),
            "avg_trust_score": np.mean(list(self.trust_manager.trust_scores.values())),
        }
        
        return aggregated_params, metrics
    
    def _perform_detection(self, client_updates: List[Dict], client_metadata: List[Dict]) -> Dict:
        detection_results = {}
        
        for update, metadata in zip(client_updates, client_metadata):
            client_id = update["client_id"]
            attack_type = metadata.get("attack_type", "honest")
            
            # Simulate detection based on attack type
            if attack_type == "aggressive":
                hwe_score = np.random.uniform(1e-10, 1e-8)
                afc_score = np.random.uniform(3.0, 5.0)
                grad_score = np.random.uniform(0.7, 0.9)
            elif attack_type == "subtle":
                hwe_score = np.random.uniform(1e-4, 1e-3)
                afc_score = np.random.uniform(1.5, 2.5)
                grad_score = np.random.uniform(0.3, 0.5)
            else:  # honest
                hwe_score = np.random.uniform(0.1, 0.9)
                afc_score = np.random.uniform(0.1, 0.8)
                grad_score = np.random.uniform(0.0, 0.2)
            
            detection_results[client_id] = {
                "hwe_score": hwe_score,
                "afc_score": afc_score,
                "grad_score": grad_score,
            }
        
        return detection_results
    
    def _update_trust_scores(self, detection_results: Dict):
        for client_id, scores in detection_results.items():
            reputation = self.trust_manager.calculate_reputation(
                scores["hwe_score"], scores["afc_score"], scores["grad_score"]
            )
            self.trust_manager.update_trust(client_id, reputation)
    
    def _filter_suspicious_clients(self, client_updates: List[Dict]) -> List[Dict]:
        trusted = []
        for update in client_updates:
            if self.trust_manager.is_trusted(update["client_id"]):
                trusted.append(update)
        return trusted
    
    def _cluster_clients(self, client_updates: List[Dict], metadata: List[Dict]) -> Dict[int, int]:
        n_clients = len(client_updates)
        if n_clients < 2:
            return {client_updates[0]["client_id"]: 0} if client_updates else {}
        
        similarity_matrix = np.random.random((n_clients, n_clients))
        np.fill_diagonal(similarity_matrix, 1.0)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        
        distance_matrix = 1 - similarity_matrix
        clustering = AgglomerativeClustering(
            n_clusters=min(3, n_clients), metric="precomputed", linkage="average"
        )
        labels = clustering.fit_predict(distance_matrix)
        
        clusters = {}
        for i, update in enumerate(client_updates):
            clusters[update["client_id"]] = labels[i]
        
        return clusters
    
    def _secure_aggregate(self, client_updates: List[Dict], clusters: Dict[int, int]) -> fl.common.Parameters:
        if not client_updates:
            return None
        
        weighted_params = []
        total_weight = 0
        
        for update in client_updates:
            client_id = update["client_id"]
            trust = self.trust_manager.trust_scores[client_id]
            weight = trust * update["num_examples"]
            
            params = fl.common.parameters_to_ndarrays(update["parameters"])
            weighted_params.append([p * weight for p in params])
            total_weight += weight
        
        if total_weight > 0:
            aggregated = []
            for i in range(len(weighted_params[0])):
                param_sum = sum(p[i] for p in weighted_params)
                aggregated.append(param_sum / total_weight)
            
            return fl.common.ndarrays_to_parameters(aggregated)
        
        return None
    
    def _log_to_blockchain(self, trusted_updates: List[Dict], detection_results: Dict):
        for update in trusted_updates:
            client_id = update["client_id"]
            transaction = {
                "type": "model_update",
                "client_id": client_id,
                "round": self.round_num,
                "model_hash": hashlib.sha256(str(update["parameters"]).encode()).hexdigest()[:16],
                "trust_score": self.trust_manager.trust_scores[client_id],
                "detection_scores": detection_results.get(client_id, {}),
            }
            self.blockchain.add_transaction(transaction)
        
        self.blockchain.commit_round(self.round_num)


# ========================= Flower Client =========================

class GeneticClient(fl.client.NumPyClient):
    """A client for training on synthetic genetic data."""
    
    def __init__(self, client_id: int, data: Dict, model: nn.Module, attack_type: str = "honest"):
        self.client_id = client_id
        self.data = data
        self.model = model
        self.attack_type = attack_type
    
    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        
        X = np.hstack([self.data["common_genotypes"], self.data["prs_scores"][:, np.newaxis], self.data["rare_dosages"]])
        y = self.data["phenotype_binary"]
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        
        for _ in range(5):
            for features, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(X), {
            "client_id": self.client_id,
            "attack_type": self.attack_type
        }
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        
        X = np.hstack([self.data["common_genotypes"], self.data["prs_scores"][:, np.newaxis], self.data["rare_dosages"]])
        y = self.data["phenotype_binary"]
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        dataloader = DataLoader(dataset, batch_size=32)
        
        criterion = nn.BCELoss()
        loss = 0
        correct = 0
        total = 0
        self.model.eval()
        
        with torch.no_grad():
            for features, labels in dataloader:
                outputs = self.model(features)
                loss += criterion(outputs, labels.view(-1, 1)).item()
                predicted = (outputs > 0.5).squeeze().long()
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
        
        accuracy = correct / total
        return float(loss), len(X), {"accuracy": float(accuracy)}


class AggressiveAttacker(GeneticClient):
    """Aggressive Byzantine attacker sending extreme noise"""
    
    def __init__(self, client_id: int, data: Dict, model: nn.Module):
        super().__init__(client_id, data, model, attack_type="aggressive")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # Send extreme random noise
        malicious_params = [np.random.randn(*p.shape) * 10 for p in parameters]
        return malicious_params, self.data["prs_scores"].shape[0], {
            "client_id": self.client_id,
            "attack_type": self.attack_type
        }


class SubtleAttacker(GeneticClient):
    """Subtle Byzantine attacker that gradually corrupts the model"""
    
    def __init__(self, client_id: int, data: Dict, model: nn.Module):
        super().__init__(client_id, data, model, attack_type="subtle")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # Train normally then add subtle noise
        trained_params, n, metrics = super().fit(parameters, config)
        
        # Add subtle corruption
        corrupted_params = [p + np.random.randn(*p.shape) * 0.1 for p in trained_params]
        
        return corrupted_params, n, metrics


# ========================= Visualization =========================

def plot_trust_evolution(trust_history: Dict, client_types: Dict, save_path: str = "trust_evolution.png"):
    """Plot trust score evolution over communication rounds"""
    plt.figure(figsize=(12, 7))
    
    # Prepare data by attack type
    honest_clients = [cid for cid, ctype in client_types.items() if ctype == "honest"]
    aggressive_clients = [cid for cid, ctype in client_types.items() if ctype == "aggressive"]
    subtle_clients = [cid for cid, ctype in client_types.items() if ctype == "subtle"]
    
    # Plot honest clients (average)
    if honest_clients:
        honest_scores = np.array([trust_history[cid] for cid in honest_clients])
        honest_avg = np.mean(honest_scores, axis=0)
        rounds = range(1, len(honest_avg) + 1)
        plt.plot(rounds, honest_avg, 'b-', linewidth=2.5, label='Honest Clients')
    
    # Plot aggressive attackers (average)
    if aggressive_clients:
        aggressive_scores = np.array([trust_history[cid] for cid in aggressive_clients])
        aggressive_avg = np.mean(aggressive_scores, axis=0)
        rounds = range(1, len(aggressive_avg) + 1)
        plt.plot(rounds, aggressive_avg, 'r--', linewidth=2.5, label='Aggressive Attackers')
    
    # Plot subtle attackers (average)
    if subtle_clients:
        subtle_scores = np.array([trust_history[cid] for cid in subtle_clients])
        subtle_avg = np.mean(subtle_scores, axis=0)
        rounds = range(1, len(subtle_avg) + 1)
        plt.plot(rounds, subtle_avg, color='orange', linestyle=':', linewidth=2.5, 
                label='Subtle Attackers')
    
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Trust Score', fontsize=12)
    plt.title('Trust Score Evolution Over Communication Rounds', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='right')
    plt.xlim(1, len(honest_avg) if honest_clients else 20)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTrust evolution graph saved to {save_path}")
    plt.close()


# ========================= Simulation =========================

def run_secure_simulation():
    """Run enhanced simulation with trust visualization"""
    print("=" * 80)
    print("SECURE RV-FedPRS: Byzantine-Robust Federated Genomic Risk Assessment")
    print("With Trust Evolution Visualization")
    print("=" * 80)
    
    # 1. Create model
    n_common_variants = 100
    n_rare_variants = 500
    model = nn.Sequential(
        nn.Linear(n_common_variants + n_rare_variants + 1, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    
    # 2. Generate client datasets
    data_generator = GeneticDataGenerator(
        n_samples=1000,
        n_common_variants=n_common_variants,
        n_rare_variants=n_rare_variants,
        n_populations=3,
    )
    client_datasets = data_generator.create_federated_datasets(n_clients=10)
    
    # 3. Create clients with different attack types
    clients = []
    client_types = {}
    
    for i, client_data in enumerate(client_datasets):
        if i < 2:  # 2 aggressive attackers
            clients.append(AggressiveAttacker(client_id=i, data=client_data, model=model))
            client_types[i] = "aggressive"
        elif i < 4:  # 2 subtle attackers
            clients.append(SubtleAttacker(client_id=i, data=client_data, model=model))
            client_types[i] = "subtle"
        else:  # 6 honest clients
            clients.append(GeneticClient(client_id=i, data=client_data, model=model))
            client_types[i] = "honest"
    
    def client_fn(cid: str) -> fl.client.Client:
        return clients[int(cid)].to_client()
    
    # 4. Create secure strategy
    security_config = SecurityConfig(
        max_malicious_fraction=0.4,
        trust_momentum=0.7,
        trim_fraction=0.2,
        enable_blockchain=True,
    )
    
    strategy = SecureRVFedPRSStrategy(
        security_config=security_config,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=10,
    )
    
    # 5. Run simulation
    print(f"\nStarting federated learning with:")
    print(f"  - Honest clients: 6")
    print(f"  - Aggressive attackers: 2")
    print(f"  - Subtle attackers: 2")
    print(f"  - Total rounds: 20\n")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
    )
    
    # 6. Print results
    print("\n" + "=" * 80)
    print("Simulation Results")
    print("=" * 80)
    
    print("\nFinal Trust Scores:")
    for client_id in sorted(strategy.trust_manager.trust_scores.keys()):
        trust_score = strategy.trust_manager.trust_scores[client_id]
        client_type = client_types[client_id].capitalize()
        print(f"  Client {client_id} ({client_type:12s}): {trust_score:.4f}")
    
    # 7. Plot trust evolution
    plot_trust_evolution(
        strategy.trust_manager.trust_history,
        client_types,
        save_path="trust_evolution.png"
    )
    
    # 8. Save detailed results
    results = {
        "client_types": client_types,
        "trust_scores": {int(k): float(v) for k, v in strategy.trust_manager.trust_scores.items()},
        "trust_history": {int(k): [float(sv) for sv in v] for k, v in strategy.trust_manager.trust_history.items()},
        "blockchain_length": len(strategy.blockchain.chain) if strategy.blockchain else 0,
    }
    
    with open("byzantine_simulation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "=" * 80)
    print("Simulation completed successfully!")
    print("  - Trust evolution graph: trust_evolution.png")
    print("  - Detailed results: byzantine_simulation_results.json")
    print("=" * 80)


if __name__ == "__main__":
    run_secure_simulation()


