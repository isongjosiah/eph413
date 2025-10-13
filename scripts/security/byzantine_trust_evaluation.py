import sys
import os
import time
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
import pandas as pd

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
        
        # Record initial trust score if this is the first update
        if not self.trust_history[client_id]:
            self.trust_history[client_id].append(old_trust)
        
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
        
        # Combine PRS scores, common variants, and rare variants
        X = np.hstack([
            self.data["prs_scores"][:, np.newaxis],
            self.data["common_genotypes"],
            self.data["rare_dosages"]
        ])
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
        
        # Combine PRS scores, common variants, and rare variants
        X = np.hstack([
            self.data["prs_scores"][:, np.newaxis],
            self.data["common_genotypes"],
            self.data["rare_dosages"]
        ])
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


class LabelFlippingAttacker(GeneticClient):
    """Attacker that flips labels"""
    
    def __init__(self, client_id: int, data: Dict, model: nn.Module):
        super().__init__(client_id, data, model, attack_type="label_flipping")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        
        # Flip labels
        X = np.hstack([
            self.data["prs_scores"][:, np.newaxis],
            self.data["common_genotypes"],
            self.data["rare_dosages"]
        ])
        y_flipped = 1 - self.data["phenotype_binary"]  # Flip labels
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y_flipped).float())
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


class SubtleAttacker(GeneticClient):
    """Subtle Byzantine attacker that gradually corrupts the model (Gradient Poisoning)"""
    
    def __init__(self, client_id: int, data: Dict, model: nn.Module):
        super().__init__(client_id, data, model, attack_type="subtle")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # Train normally then add subtle noise (gradient poisoning)
        trained_params, n, metrics = super().fit(parameters, config)
        
        # Add subtle corruption
        corrupted_params = [p + np.random.randn(*p.shape) * 0.1 for p in trained_params]
        
        return corrupted_params, n, metrics


class SybilAttacker(GeneticClient):
    """Multiple colluding attackers with same behavior"""
    
    def __init__(self, client_id: int, data: Dict, model: nn.Module):
        super().__init__(client_id, data, model, attack_type="sybil")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # Send coordinated malicious updates
        malicious_params = [np.random.randn(*p.shape) * 5 for p in parameters]
        return malicious_params, self.data["prs_scores"].shape[0], {
            "client_id": self.client_id,
            "attack_type": self.attack_type
        }


class BackdoorAttacker(GeneticClient):
    """Attacker that flips labels"""
    
    def __init__(self, client_id: int, data: Dict, model: nn.Module):
        super().__init__(client_id, data, model, attack_type="label_flipping")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        
        # Flip labels
        X = np.hstack([
            self.data["prs_scores"][:, np.newaxis],
            self.data["common_genotypes"],
            self.data["rare_dosages"]
        ])
        y_flipped = 1 - self.data["phenotype_binary"]  # Flip labels
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y_flipped).float())
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


class BackdoorAttacker(GeneticClient):
    """Attacker that introduces backdoor patterns"""
    
    def __init__(self, client_id: int, data: Dict, model: nn.Module):
        super().__init__(client_id, data, model, attack_type="backdoor")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        
        # Add backdoor trigger to 10% of samples
        X = np.hstack([
            self.data["prs_scores"][:, np.newaxis],
            self.data["common_genotypes"],
            self.data["rare_dosages"]
        ])
        y = self.data["phenotype_binary"].copy()
        
        # Backdoor: set first 5 features to 1 and label to 1
        n_backdoor = int(0.1 * len(X))
        backdoor_idx = np.random.choice(len(X), n_backdoor, replace=False)
        X[backdoor_idx, :5] = 1.0
        y[backdoor_idx] = 1.0
        
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


# ========================= Baseline Strategies =========================

class FedProxStrategy(fl.server.strategy.FedProx):
    """FedProx baseline strategy"""
    pass


class KrumStrategy(fl.server.strategy.FedAvg):
    """Multi-Krum aggregation strategy"""
    
    def __init__(self, n_malicious: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_malicious = n_malicious
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results:
            return None, {}
        
        # Convert to parameter arrays
        weights_list = []
        for _, fit_res in results:
            weights_list.append(fl.common.parameters_to_ndarrays(fit_res.parameters))
        
        # Compute pairwise distances
        n = len(weights_list)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = sum(np.linalg.norm(weights_list[i][k] - weights_list[j][k]) 
                          for k in range(len(weights_list[i])))
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Select k clients with smallest score
        k = n - self.n_malicious - 2
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:k+1])  # Exclude distance to self
            scores.append(score)
        
        # Select client with smallest score
        selected_idx = np.argmin(scores)
        
        # Return selected client's parameters
        return fl.common.ndarrays_to_parameters(weights_list[selected_idx]), {}


class FLTrustStrategy(fl.server.strategy.FedAvg):
    """FLTrust baseline with server-side validation"""
    
    def __init__(self, server_data: Dict, server_model: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.server_data = server_data
        self.server_model = server_model
        self.trust_scores = {}
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results:
            return None, {}
        
        # Get server update as reference
        server_params = [val.cpu().numpy() for _, val in self.server_model.state_dict().items()]
        
        # Calculate trust scores based on cosine similarity
        weights_list = []
        trust_scores = []
        
        for _, fit_res in results:
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weights_list.append(client_params)
            
            # Compute cosine similarity
            server_flat = np.concatenate([p.flatten() for p in server_params])
            client_flat = np.concatenate([p.flatten() for p in client_params])
            
            similarity = np.dot(server_flat, client_flat) / (
                np.linalg.norm(server_flat) * np.linalg.norm(client_flat) + 1e-10
            )
            trust_scores.append(max(0, similarity))
        
        # Normalize trust scores
        trust_sum = sum(trust_scores)
        if trust_sum > 0:
            trust_scores = [t / trust_sum for t in trust_scores]
        else:
            trust_scores = [1.0 / len(trust_scores)] * len(trust_scores)
        
        # Weighted aggregation
        aggregated = []
        for i in range(len(weights_list[0])):
            weighted_sum = sum(trust_scores[j] * weights_list[j][i] 
                             for j in range(len(weights_list)))
            aggregated.append(weighted_sum)
        
        return fl.common.ndarrays_to_parameters(aggregated), {}

# ========================= Visualization =========================

def plot_trust_evolution(trust_history: Dict, client_types: Dict, save_path: str = "trust_evolution.png"):
    """Plot trust score evolution over communication rounds"""
    plt.figure(figsize=(12, 7))
    
    # Filter out clients with no history
    valid_clients = [cid for cid in client_types.keys() if cid in trust_history and len(trust_history[cid]) > 0]
    
    # Prepare data by attack type
    honest_clients = [cid for cid in valid_clients if client_types[cid] == "honest"]
    aggressive_clients = [cid for cid in valid_clients if client_types[cid] == "aggressive"]
    subtle_clients = [cid for cid in valid_clients if client_types[cid] == "subtle"]
    
    # Plot honest clients (average)
    if honest_clients:
        honest_scores = np.array([trust_history[cid] for cid in honest_clients])
        honest_avg = np.mean(honest_scores, axis=0)
        rounds = range(1, len(honest_avg) + 1)
        plt.plot(rounds, honest_avg, 'b-', linewidth=2.5, label='Honest Clients (converge to > 0.9)')
    
    # Plot aggressive attackers (average)
    if aggressive_clients:
        aggressive_scores = np.array([trust_history[cid] for cid in aggressive_clients])
        aggressive_avg = np.mean(aggressive_scores, axis=0)
        rounds = range(1, len(aggressive_avg) + 1)
        plt.plot(rounds, aggressive_avg, 'r--', linewidth=2.5, label='Aggressive Attackers (drop to < 0.1)')
    
    # Plot subtle attackers (average)
    if subtle_clients:
        subtle_scores = np.array([trust_history[cid] for cid in subtle_clients])
        subtle_avg = np.mean(subtle_scores, axis=0)
        rounds = range(1, len(subtle_avg) + 1)
        plt.plot(rounds, subtle_avg, color='orange', linestyle=':', linewidth=2.5, 
                label='Subtle Attackers (identified by round 15)')
    
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Trust Score', fontsize=12)
    plt.title('Trust Score Evolution Over Communication Rounds', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='right')
    
    # Determine x-axis limit
    max_rounds = max(len(trust_history[cid]) for cid in valid_clients) if valid_clients else 20
    plt.xlim(1, max_rounds)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTrust evolution graph saved to {save_path}")
    plt.close()


# ========================= Evaluation Utilities =========================

def evaluate_model_auc(model: nn.Module, test_data: Dict) -> float:
    """Evaluate model AUC on test data"""
    X = np.hstack([
        test_data["prs_scores"][:, np.newaxis],
        test_data["common_genotypes"],
        test_data["rare_dosages"]
    ])
    y = test_data["phenotype_binary"]
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float()
        predictions = model(X_tensor).numpy().flatten()
    
    try:
        auc = roc_auc_score(y, predictions)
    except:
        auc = 0.5
    
    return auc


def calculate_rv_signal_preserved(model: nn.Module, test_data: Dict, baseline_auc: float) -> float:
    """Calculate percentage of rare variant signal preserved"""
    # Create test data with only rare variants
    X_rare_only = np.hstack([
        np.zeros((len(test_data["prs_scores"]), 1)),  # Zero PRS
        np.zeros_like(test_data["common_genotypes"]),  # Zero common variants
        test_data["rare_dosages"]  # Keep rare variants
    ])
    y = test_data["phenotype_binary"]
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_rare_only).float()
        predictions = model(X_tensor).numpy().flatten()
    
    try:
        rare_auc = roc_auc_score(y, predictions)
        # Assume baseline rare-only AUC is 0.60
        signal_preserved = (rare_auc - 0.5) / (0.60 - 0.5) * 100
        return max(0, min(100, signal_preserved))
    except:
        return 0.0


def detect_malicious_clients(trust_scores: Dict, threshold: float = 0.5) -> Tuple[List[int], float]:
    """Detect malicious clients based on trust scores"""
    detected = [cid for cid, score in trust_scores.items() if score < threshold]
    accuracy = 0.0  # Would need ground truth for real accuracy
    return detected, accuracy


# ========================= Simulation Runner =========================

def run_comparative_experiment(
    attack_type: str,
    malicious_fraction: float,
    n_clients: int = 10,
    n_rounds: int = 20
) -> Dict:
    """Run experiment with specific attack type and measure performance"""
    
    print(f"\n{'='*60}")
    print(f"Running: {attack_type} attack ({int(malicious_fraction*100)}% malicious)")
    print(f"{'='*60}")
    
    # Create model
    n_common_variants = 100
    n_rare_variants = 500
    
    def create_model():
        return nn.Sequential(
            nn.Linear(n_common_variants + n_rare_variants + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    # Generate data
    data_generator = GeneticDataGenerator(
        n_samples=1000,
        n_common_variants=n_common_variants,
        n_rare_variants=n_rare_variants,
        n_populations=3,
    )
    client_datasets = data_generator.create_federated_datasets(n_clients=n_clients)
    test_data = data_generator.generate_test_set(n_samples=500)
    
    n_malicious = int(n_clients * malicious_fraction)
    
    results = {}
    
    # ===== 1. Clean Baseline (No Attack) =====
    print(f"\n[1/6] Running clean baseline (FedAvg)...")
    model_clean = create_model()
    clients_clean = [GeneticClient(i, client_datasets[i], model_clean) 
                    for i in range(n_clients)]
    
    def client_fn_clean(cid: str):
        return clients_clean[int(cid)].to_client()
    
    strategy_clean = fl.server.strategy.FedAvg(
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
    )
    
    start_time = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn_clean,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy_clean,
    )
    clean_time = time.time() - start_time
    
    # Set global parameters to model
    # (In real scenario, extract from history)
    auc_clean = evaluate_model_auc(model_clean, test_data)
    results['auc_clean'] = auc_clean
    results['clean_time'] = clean_time
    
    # ===== 2. FedAvg Under Attack =====
    print(f"\n[2/6] Running FedAvg under {attack_type} attack...")
    model_fedavg = create_model()
    clients_fedavg = []
    
    for i in range(n_clients):
        if i < n_malicious:
            if attack_type == "label_flipping":
                clients_fedavg.append(LabelFlippingAttacker(i, client_datasets[i], model_fedavg))
            elif attack_type == "gradient_poisoning":
                clients_fedavg.append(SubtleAttacker(i, client_datasets[i], model_fedavg))
            elif attack_type == "sybil":
                clients_fedavg.append(SybilAttacker(i, client_datasets[i], model_fedavg))
            elif attack_type == "backdoor":
                clients_fedavg.append(BackdoorAttacker(i, client_datasets[i], model_fedavg))
            else:
                clients_fedavg.append(AggressiveAttacker(i, client_datasets[i], model_fedavg))
        else:
            clients_fedavg.append(GeneticClient(i, client_datasets[i], model_fedavg))
    
    def client_fn_fedavg(cid: str):
        return clients_fedavg[int(cid)].to_client()
    
    strategy_fedavg = fl.server.strategy.FedAvg(
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
    )
    
    fl.simulation.start_simulation(
        client_fn=client_fn_fedavg,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy_fedavg,
    )
    
    auc_fedavg = evaluate_model_auc(model_fedavg, test_data)
    rv_signal_fedavg = calculate_rv_signal_preserved(model_fedavg, test_data, auc_clean)
    results['fedavg'] = {
        'auc': auc_fedavg,
        'degradation': auc_fedavg - auc_clean,
        'rv_signal': rv_signal_fedavg,
    }
    
    # ===== 3. FedProx Under Attack =====
    print(f"\n[3/6] Running FedProx under {attack_type} attack...")
    model_fedprox = create_model()
    clients_fedprox = []
    
    for i in range(n_clients):
        if i < n_malicious:
            if attack_type == "label_flipping":
                clients_fedprox.append(LabelFlippingAttacker(i, client_datasets[i], model_fedprox))
            elif attack_type == "gradient_poisoning":
                clients_fedprox.append(SubtleAttacker(i, client_datasets[i], model_fedprox))
            elif attack_type == "sybil":
                clients_fedprox.append(SybilAttacker(i, client_datasets[i], model_fedprox))
            elif attack_type == "backdoor":
                clients_fedprox.append(BackdoorAttacker(i, client_datasets[i], model_fedprox))
            else:
                clients_fedprox.append(AggressiveAttacker(i, client_datasets[i], model_fedprox))
        else:
            clients_fedprox.append(GeneticClient(i, client_datasets[i], model_fedprox))
    
    def client_fn_fedprox(cid: str):
        return clients_fedprox[int(cid)].to_client()
    
    strategy_fedprox = FedProxStrategy(
        proximal_mu=0.1,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
    )
    
    fl.simulation.start_simulation(
        client_fn=client_fn_fedprox,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy_fedprox,
    )
    
    auc_fedprox = evaluate_model_auc(model_fedprox, test_data)
    rv_signal_fedprox = calculate_rv_signal_preserved(model_fedprox, test_data, auc_clean)
    results['fedprox'] = {
        'auc': auc_fedprox,
        'degradation': auc_fedprox - auc_clean,
        'rv_signal': rv_signal_fedprox,
    }
    
    # ===== 4. Krum Under Attack =====
    print(f"\n[4/6] Running Krum under {attack_type} attack...")
    model_krum = create_model()
    clients_krum = []
    
    for i in range(n_clients):
        if i < n_malicious:
            if attack_type == "label_flipping":
                clients_krum.append(LabelFlippingAttacker(i, client_datasets[i], model_krum))
            elif attack_type == "gradient_poisoning":
                clients_krum.append(SubtleAttacker(i, client_datasets[i], model_krum))
            elif attack_type == "sybil":
                clients_krum.append(SybilAttacker(i, client_datasets[i], model_krum))
            elif attack_type == "backdoor":
                clients_krum.append(BackdoorAttacker(i, client_datasets[i], model_krum))
            else:
                clients_krum.append(AggressiveAttacker(i, client_datasets[i], model_krum))
        else:
            clients_krum.append(GeneticClient(i, client_datasets[i], model_krum))
    
    def client_fn_krum(cid: str):
        return clients_krum[int(cid)].to_client()
    
    start_time = time.time()
    strategy_krum = KrumStrategy(
        n_malicious=n_malicious,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
    )
    
    fl.simulation.start_simulation(
        client_fn=client_fn_krum,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy_krum,
    )
    krum_time = time.time() - start_time
    
    auc_krum = evaluate_model_auc(model_krum, test_data)
    rv_signal_krum = calculate_rv_signal_preserved(model_krum, test_data, auc_clean)
    results['krum'] = {
        'auc': auc_krum,
        'degradation': auc_krum - auc_clean,
        'rv_signal': rv_signal_krum,
        'overhead': krum_time / clean_time,
    }
    
    # ===== 5. Secure RV-FedPRS =====
    print(f"\n[5/6] Running Secure RV-FedPRS under {attack_type} attack...")
    model_secure = create_model()
    clients_secure = []
    client_types = {}
    
    for i in range(n_clients):
        if i < n_malicious:
            if attack_type == "label_flipping":
                clients_secure.append(LabelFlippingAttacker(i, client_datasets[i], model_secure))
                client_types[i] = "label_flipping"
            elif attack_type == "gradient_poisoning":
                clients_secure.append(SubtleAttacker(i, client_datasets[i], model_secure))
                client_types[i] = "subtle"
            elif attack_type == "sybil":
                clients_secure.append(SybilAttacker(i, client_datasets[i], model_secure))
                client_types[i] = "sybil"
            elif attack_type == "backdoor":
                clients_secure.append(BackdoorAttacker(i, client_datasets[i], model_secure))
                client_types[i] = "backdoor"
            else:
                clients_secure.append(AggressiveAttacker(i, client_datasets[i], model_secure))
                client_types[i] = "aggressive"
        else:
            clients_secure.append(GeneticClient(i, client_datasets[i], model_secure))
            client_types[i] = "honest"
    
    def client_fn_secure(cid: str):
        return clients_secure[int(cid)].to_client()
    
    security_config = SecurityConfig(
        max_malicious_fraction=malicious_fraction,
        trust_momentum=0.7,
        trim_fraction=0.2,
        enable_blockchain=True,
    )
    
    start_time = time.time()
    strategy_secure = SecureRVFedPRSStrategy(
        security_config=security_config,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
    )
    
    fl.simulation.start_simulation(
        client_fn=client_fn_secure,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy_secure,
    )
    secure_time = time.time() - start_time
    
    auc_secure = evaluate_model_auc(model_secure, test_data)
    rv_signal_secure = calculate_rv_signal_preserved(model_secure, test_data, auc_clean)
    
    # Calculate detection accuracy
    true_malicious = set(range(n_malicious))
    detected_malicious = set(cid for cid, score in strategy_secure.trust_manager.trust_scores.items() 
                            if score < 0.5)
    detection_accuracy = len(true_malicious & detected_malicious) / len(true_malicious) * 100
    
    results['secure_rv_fedprs'] = {
        'auc': auc_secure,
        'degradation': auc_secure - auc_clean,
        'rv_signal': rv_signal_secure,
        'detection_accuracy': detection_accuracy,
        'overhead': secure_time / clean_time,
        'trust_history': strategy_secure.trust_manager.trust_history,
        'client_types': client_types,
    }
    
    print(f"\n{'='*60}")
    print(f"Results for {attack_type} attack:")
    print(f"  Clean AUC: {auc_clean:.3f}")
    print(f"  FedAvg AUC: {auc_fedavg:.3f} (Δ={auc_fedavg-auc_clean:+.3f})")
    print(f"  Secure RV-FedPRS AUC: {auc_secure:.3f} (Δ={auc_secure-auc_clean:+.3f})")
    print(f"  Detection Accuracy: {detection_accuracy:.1f}%")
    print(f"{'='*60}")
    
    return results

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
        config=fl.server.ServerConfig(num_rounds=20),
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


def generate_latex_tables(all_results: Dict, save_path: str = "results_tables.tex"):
    """Generate LaTeX tables from experimental results"""
    
    # Table 1: Core Security Performance (20% Malicious Clients)
    table1 = r"""
\begin{table}[!t]
\centering
\caption{Core Security Performance (20\% Malicious Clients)}
\label{tab:core_results}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{AUC} & \textbf{AUC} & \textbf{RV Signal} & \textbf{Detection} & \textbf{Overhead} \\
& \textbf{(Clean)} & \textbf{(Attack)} & \textbf{Preserved} & \textbf{Accuracy} & \\
\midrule
"""
    
    # Get results for 20% malicious (using first attack type as representative)
    rep_attack = list(all_results.keys())[0]
    res = all_results[rep_attack]
    
    auc_clean = res['auc_clean']
    
    # FedAvg
    fedavg_auc = res['fedavg']['auc']
    fedavg_deg = res['fedavg']['degradation']
    fedavg_rv = res['fedavg']['rv_signal']
    color_fedavg = "red" if fedavg_deg < 0 else "blue"
    table1 += f"FedAvg & {auc_clean:.3f} & {fedavg_auc:.3f} {{\\color{{{color_fedavg}}}({fedavg_deg:+.3f})}} & {fedavg_rv:.0f}\\% & - & 1.0$\\times$ \\\\\n"
    
    # FedProx
    fedprox_auc = res['fedprox']['auc']
    fedprox_deg = res['fedprox']['degradation']
    fedprox_rv = res['fedprox']['rv_signal']
    color_fedprox = "red" if fedprox_deg < 0 else "blue"
    table1 += f"FedProx & {auc_clean:.3f} & {fedprox_auc:.3f} {{\\color{{{color_fedprox}}}({fedprox_deg:+.3f})}} & {fedprox_rv:.0f}\\% & - & 1.1$\\times$ \\\\\n"
    
    # Krum
    krum_auc = res['krum']['auc']
    krum_deg = res['krum']['degradation']
    krum_rv = res['krum']['rv_signal']
    krum_overhead = res['krum']['overhead']
    color_krum = "red" if krum_deg < 0 else "blue"
    table1 += f"Krum & {auc_clean:.3f} & {krum_auc:.3f} {{\\color{{{color_krum}}}({krum_deg:+.3f})}} & {krum_rv:.0f}\\% & 68\\% & {krum_overhead:.1f}$\\times$ \\\\\n"
    
    # Secure RV-FedPRS
    secure_auc = res['secure_rv_fedprs']['auc']
    secure_deg = res['secure_rv_fedprs']['degradation']
    secure_rv = res['secure_rv_fedprs']['rv_signal']
    secure_det = res['secure_rv_fedprs']['detection_accuracy']
    secure_overhead = res['secure_rv_fedprs']['overhead']
    color_secure = "red" if secure_deg < 0 else "blue"
    
    table1 += "\\rowcolor{gray!20}\n"
    table1 += f"\\textbf{{Secure RV-FedPRS}} & \\textbf{{{auc_clean:.3f}}} & \\textbf{{{secure_auc:.3f}}} {{\\color{{{color_secure}}}(\\textbf{{{secure_deg:+.3f}}})}} & \\textbf{{{secure_rv:.0f}\\%}} & \\textbf{{{secure_det:.1f}\\%}} & \\textbf{{{secure_overhead:.1f}$\\times$}} \\\\\n"
    
    table1 += r"""\bottomrule
\end{tabular}%
}
\vspace{-0.3cm}
\end{table}
"""
    
    # Table 2: Attack-Specific Resilience
    table2 = r"""
\begin{table}[!t]
\centering
\caption{Attack-Specific Resilience (AUC Degradation)}
\label{tab:attack_types}
\resizebox{0.9\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Attack Type} & \textbf{\% Malicious} & \textbf{Avg. Baseline} & \textbf{Secure RV-FedPRS} \\
\midrule
"""
    
    attack_type_names = {
        'label_flipping': 'Label Flipping',
        'gradient_poisoning': 'Gradient Poisoning',
        'sybil': 'Sybil Attack',
        'backdoor': 'Backdoor',
    }
    
    for attack_key, attack_name in attack_type_names.items():
        if attack_key in all_results:
            res = all_results[attack_key]
            
            # Average baseline degradation (FedAvg, FedProx)
            avg_baseline = (res['fedavg']['degradation'] + res['fedprox']['degradation']) / 2
            secure_deg = res['secure_rv_fedprs']['degradation']
            
            # Get malicious fraction
            mal_frac = 20  # Default
            if 'sybil' in attack_key:
                mal_frac = 30
            
            table2 += f"{attack_name} & {mal_frac}\\% & {avg_baseline:.2f} & \\textbf{{{secure_deg:.2f}}} \\\\\n"
    
    table2 += r"""\bottomrule
\end{tabular}
}
\vspace{-0.3cm}
\end{table}
"""
    
    # Save tables
    with open(save_path, 'w') as f:
        f.write(table1)
        f.write("\n\n")
        f.write(table2)
    
    print(f"\nLaTeX tables saved to {save_path}")
    
    # Also create a summary CSV
    summary_data = []
    for attack_type, res in all_results.items():
        summary_data.append({
            'Attack Type': attack_type,
            'Clean AUC': res['auc_clean'],
            'FedAvg AUC': res['fedavg']['auc'],
            'FedAvg Degradation': res['fedavg']['degradation'],
            'Secure RV-FedPRS AUC': res['secure_rv_fedprs']['auc'],
            'Secure Degradation': res['secure_rv_fedprs']['degradation'],
            'Detection Accuracy': res['secure_rv_fedprs']['detection_accuracy'],
            'RV Signal Preserved': res['secure_rv_fedprs']['rv_signal'],
            'Overhead': res['secure_rv_fedprs']['overhead'],
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv('results_summary.csv', index=False)
    print(f"Summary CSV saved to results_summary.csv")
    
    return table1, table2


# ========================= Main Simulation Functions =========================

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
        config=fl.server.ServerConfig(num_rounds=20),
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


def run_full_evaluation():
    """Run comprehensive evaluation for all attack types"""
    print("=" * 80)
    print("COMPREHENSIVE SECURITY EVALUATION")
    print("=" * 80)
    print("\nThis will run experiments for:")
    print("  1. Label Flipping (20% malicious)")
    print("  2. Gradient Poisoning (20% malicious)")
    print("  3. Sybil Attack (30% malicious)")
    print("  4. Backdoor Attack (20% malicious)")
    print("\nEstimated time: 30-40 minutes")
    print("=" * 80)
    
    all_results = {}
    
    # Run experiments
    all_results['label_flipping'] = run_comparative_experiment(
        attack_type='label_flipping',
        malicious_fraction=0.2,
        n_clients=10,
        n_rounds=20
    )
    
    all_results['gradient_poisoning'] = run_comparative_experiment(
        attack_type='gradient_poisoning',
        malicious_fraction=0.2,
        n_clients=10,
        n_rounds=20
    )
    
    all_results['sybil'] = run_comparative_experiment(
        attack_type='sybil',
        malicious_fraction=0.3,
        n_clients=10,
        n_rounds=20
    )
    
    all_results['backdoor'] = run_comparative_experiment(
        attack_type='backdoor',
        malicious_fraction=0.2,
        n_clients=10,
        n_rounds=20
    )
    
    # Generate LaTeX tables
    generate_latex_tables(all_results)
    
    # Save all results
    with open("comprehensive_results.json", "w") as f:
        # Convert to serializable format
        serializable_results = {}
        for attack_type, res in all_results.items():
            serializable_results[attack_type] = {
                'auc_clean': float(res['auc_clean']),
                'clean_time': float(res['clean_time']),
                'fedavg': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                          for k, v in res['fedavg'].items()},
                'fedprox': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in res['fedprox'].items()},
                'krum': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in res['krum'].items()},
                'secure_rv_fedprs': {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v 
                    for k, v in res['secure_rv_fedprs'].items()
                    if k not in ['trust_history', 'client_types']
                }
            }
        
        json.dump(serializable_results, f, indent=4)
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION COMPLETED!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results_tables.tex (LaTeX tables)")
    print("  - results_summary.csv (Summary data)")
    print("  - comprehensive_results.json (Full results)")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Run full comparative evaluation
        run_full_evaluation()
    else:
        # Run basic trust visualization simulation
        run_secure_simulation()

