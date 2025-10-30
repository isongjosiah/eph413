"""
Defines the core federated learning server strategy for XS-FedPRS.

This custom strategy implements the full pipeline:
1. Genetic-Aware Defense
2. Trust Score Updating (Eq. 5)
3. FedCE Clustering (Eq. 14)
4. Trust-Weighted Asymmetric Aggregation (Eq. 6 & 7)
"""

import logging
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import OrderedDict

import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Import our custom modules
from .defense import GeneticDefense
from .trust import TrustManager
from .models import HierarchicalModel  # Used to get parameter names

# Configure logging
logger = logging.getLogger(__name__)


def _jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculates Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def _aggregate_weighted_params(
    param_list: List[OrderedDict], weights: List[float]
) -> OrderedDict:
    """Helper to perform weighted average on a list of parameter dictionaries."""
    if not param_list:
        return OrderedDict()

    total_weight = sum(weights)
    if total_weight == 0:
        logger.warning("Total aggregation weight is zero.")
        return param_list[0]  # Return first model as-is

    # Initialize aggregated params
    agg_params = OrderedDict([(key, 0.0) for key in param_list[0].keys()])

    # Perform weighted sum
    for params, weight in zip(param_list, weights):
        for key, value in params.items():
            agg_params[key] += value * weight

    # Normalize by total weight
    for key in agg_params:
        agg_params[key] /= total_weight

    return agg_params


def _average_params(param_list: List[OrderedDict]) -> OrderedDict:
    """Helper to perform a simple average on a list of parameter dictionaries."""
    if not param_list:
        return OrderedDict()
    weights = [1.0] * len(param_list)
    return _aggregate_weighted_params(param_list, weights)


class XS_FedPRS_Strategy(FedAvg):
    """
    Implements the full XS-FedPRS strategy including defense, trust,
    FedCE clustering, and asymmetric aggregation.
    """

    def __init__(
        self,
        genetic_defense: GeneticDefense,
        trust_manager: TrustManager,
        model_config: Dict,  # Config to init dummy model
        min_trust_for_aggregation: float = 0.1,
        min_cluster_size: int = 2,
        max_clusters: int = 5,
        **kwargs,
    ):
        """
        Initialize the strategy.

        Args:
            genetic_defense: An instance of GeneticDefense.
            trust_manager: An instance of TrustManager.
            model_config: Dict with 'n_rare_variants', etc. to initialize a
                          dummy HierarchicalModel for param names.
            min_trust_for_aggregation: Minimum trust score to be included.
            min_cluster_size: Minimum number of clients in a cluster for silhouette.
            max_clusters: Max number of clusters (M) for FedCE.
            **kwargs: Arguments for the parent FedAvg (like min_fit_clients).
        """
        super().__init__(**kwargs)
        self.defense = genetic_defense
        self.trust = trust_manager
        self.min_trust_for_aggregation = min_trust_for_aggregation
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max(min_cluster_size, max_clusters)

        # --- Store parameter names for asymmetric split ---
        logger.info("Initializing dummy model to get parameter structure...")
        dummy_model = HierarchicalModel(
            n_rare_variants=model_config.get("n_rare_variants", 500),
            common_hidden_dim=model_config.get("common_hidden_dim", 16),
            rare_hidden_dim=model_config.get("rare_hidden_dim", 64),
        )
        self.ordered_param_keys = list(dummy_model.state_dict().keys())
        self.common_param_keys = [
            k for k in self.ordered_param_keys if k.startswith("common_pathway.")
        ]
        self.rare_param_keys = [
            k for k in self.ordered_param_keys if k.startswith("rare_")
        ]
        self.integration_param_keys = [
            k for k in self.ordered_param_keys if k.startswith("integration_layer.")
        ]
        # Combine rare + integration into the "specialist" part
        self.specialist_param_keys = self.rare_param_keys + self.integration_param_keys

        logger.info(f"Identified {len(self.common_param_keys)} common parameters.")
        logger.info(
            f"Identified {len(self.specialist_param_keys)} specialist (rare+integration) parameters."
        )

        # --- State for personalized models ---
        self.client_clusters: Dict[str, int] = {}  # Maps cid -> cluster_id
        self.cluster_specialist_params: Dict[int, OrderedDict] = (
            {}
        )  # Maps cluster_id -> params
        self.avg_specialist_params: Optional[OrderedDict] = (
            None  # Fallback for new clients
        )

    def _split_params(self, params: OrderedDict) -> Tuple[OrderedDict, OrderedDict]:
        """Splits a full parameter dictionary into common and specialist parts."""
        common_params = OrderedDict([(k, params[k]) for k in self.common_param_keys])
        specialist_params = OrderedDict(
            [(k, params[k]) for k in self.specialist_param_keys]
        )
        return common_params, specialist_params

    def _rebuild_params(
        self, common: OrderedDict, specialist: OrderedDict
    ) -> OrderedDict:
        """Recombines common and specialist params into a full ordered dict."""
        # Use self.ordered_param_keys to ensure correct order
        full_params_dict = {**common, **specialist}
        return OrderedDict([(k, full_params_dict[k]) for k in self.ordered_param_keys])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        The core aggregation logic for XS-FedPRS.
        """
        if not results:
            logger.warning("aggregate_fit: received no results.")
            return None, {}

        # --- Step 1: Score & Update Trust for ALL received clients ---
        logger.info(
            f"Round {server_round}: Scoring updates from {len(results)} clients."
        )
        client_updates = []
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            metrics = fit_res.metrics
            params_ndarrays = parameters_to_ndarrays(fit_res.parameters)

            # Extract client-sent statistics for defense
            hwe_stats = metrics.get("hwe_stats", {})
            allele_freqs = metrics.get("allele_freqs", {})
            population_id = metrics.get(
                "population_id", "EUR"
            )  # Default to EUR if missing
            influential_variants = set(metrics.get("influential_variants", []))

            # Flatten update for gradient plausibility
            # TODO: Refine this to be a more meaningful vector if needed
            flat_update = np.concatenate([arr.flatten() for arr in params_ndarrays])

            # a. Invoke GeneticDefense methods
            hwe_score, afc_score, grad_score = self.defense.run_all_checks(
                client_id=cid,
                hwe_stats=hwe_stats,
                allele_freqs=allele_freqs,
                population_id=population_id,
                update_vector=flat_update,
            )

            # b. Pass scores to TrustManager
            new_trust_score = self.trust.update_trust_score(
                client_id=cid,
                hwe_score=hwe_score,
                afc_score=afc_score,
                plausibility_score=grad_score,
            )

            client_updates.append(
                {
                    "cid": cid,
                    "params_nd": params_ndarrays,
                    "trust_score": new_trust_score,
                    "n_samples": fit_res.num_examples,
                    "variant_set": influential_variants,
                    "metrics": metrics,  # Store original metrics
                }
            )

        # --- Filter out untrusted clients ---
        trusted_updates = [
            upd
            for upd in client_updates
            if upd["trust_score"] >= self.min_trust_for_aggregation
        ]

        if not trusted_updates:
            logger.warning(
                "aggregate_fit: No clients passed trust threshold. Halting aggregation."
            )
            return None, {"clients_untrusted": len(client_updates)}

        logger.info(
            f"Passed trust threshold: {len(trusted_updates)}/{len(client_updates)} clients."
        )

        # --- Step 2: FedCE Clustering (on trusted clients) ---
        trusted_cids = [upd["cid"] for upd in trusted_updates]
        trusted_variant_sets = [upd["variant_set"] for upd in trusted_updates]
        n_trusted = len(trusted_updates)

        # Build Jaccard similarity matrix (Eq. 14)
        jaccard_matrix = np.ones((n_trusted, n_trusted))
        for i in range(n_trusted):
            for j in range(i + 1, n_trusted):
                sim = _jaccard_similarity(
                    trusted_variant_sets[i], trusted_variant_sets[j]
                )
                jaccard_matrix[i, j] = sim
                jaccard_matrix[j, i] = sim

        # Convert to distance matrix for clustering
        distance_matrix = 1.0 - jaccard_matrix

        # Find optimal cluster count (M) using silhouette score
        best_m = 1
        best_labels = np.zeros(n_trusted, dtype=int)  # Default to 1 cluster
        best_silhouette = -1.0

        # Max clusters to test is min(max_clusters, n_trusted - 1)
        max_m = min(self.max_clusters, n_trusted - 1)
        if max_m >= self.min_cluster_size:
            for m in range(self.min_cluster_size, max_m + 1):
                clusterer = AgglomerativeClustering(
                    n_clusters=m, metric="precomputed", linkage="average"
                )
                labels = clusterer.fit_predict(distance_matrix)
                score = silhouette_score(distance_matrix, labels, metric="precomputed")

                if score > best_silhouette:
                    best_silhouette = score
                    best_m = m
                    best_labels = labels

        logger.info(
            f"FedCE Clustering: Found {best_m} optimal clusters with silhouette score {best_silhouette:.4f}"
        )

        # Store cluster assignments {cid -> cluster_id}
        self.client_clusters = {
            cid: int(label) for cid, label in zip(trusted_cids, best_labels)
        }

        # --- Step 3: Trust-Weighted Asymmetric Aggregation ---

        # Create full parameter dictionaries for easier splitting
        all_param_dicts = {
            upd["cid"]: OrderedDict(zip(self.ordered_param_keys, upd["params_nd"]))
            for upd in trusted_updates
        }

        # a. Common Backbone Aggregation (Eq. 6) - Global
        common_params_list = []
        common_weights = []
        for upd in trusted_updates:
            cid = upd["cid"]
            common_params, _ = self._split_params(all_param_dicts[cid])
            common_params_list.append(common_params)
            common_weights.append(upd["trust_score"] * upd["n_samples"])

        agg_common_params = _aggregate_weighted_params(
            common_params_list, common_weights
        )

        # b. Specialist Aggregation (Eq. 7) - Per Cluster
        self.cluster_specialist_params = {}  # Reset
        for m in range(best_m):
            specialist_params_list = []
            specialist_weights = []

            # Find clients in this cluster
            clients_in_cluster = [
                cid for cid, label in self.client_clusters.items() if label == m
            ]

            for upd in trusted_updates:
                if upd["cid"] in clients_in_cluster:
                    _, specialist_params = self._split_params(
                        all_param_dicts[upd["cid"]]
                    )
                    specialist_params_list.append(specialist_params)
                    specialist_weights.append(upd["trust_score"] * upd["n_samples"])

            if specialist_params_list:
                agg_specialist_params = _aggregate_weighted_params(
                    specialist_params_list, specialist_weights
                )
                self.cluster_specialist_params[m] = agg_specialist_params

        logger.info(
            f"Aggregation complete: 1 common model, {len(self.cluster_specialist_params)} specialist models."
        )

        # Store an average specialist model as fallback for new/unclustered clients
        self.avg_specialist_params = _average_params(
            self.cluster_specialist_params.values()
        )

        # --- Return Global Common Model ---
        # The base parameters sent to all clients (in configure_fit) will be the
        # common backbone. The specialist part will be added *per client*.
        agg_common_ndarrays = [agg_common_params[k] for k in self.common_param_keys]

        # For compatibility with FedAvg parent, we must return *some* full model.
        # We'll return the common model + the *average* specialist model.
        # This average model will also be used by `configure_fit` for new clients.
        if not self.avg_specialist_params:
            # This happens if n_trusted < min_cluster_size
            logger.warning("No clusters formed, using global common parameters only.")
            # Get specialist params from the first trusted client as a fallback structure
            _, self.avg_specialist_params = self._split_params(
                all_param_dicts[trusted_cids[0]]
            )

        final_params_dict = {**agg_common_params, **self.avg_specialist_params}
        final_ndarrays = [final_params_dict[k] for k in self.ordered_param_keys]

        # Aggregate metrics from parent
        metrics = super().aggregate_evaluate(server_round, results, failures)[1]
        metrics["n_clusters"] = best_m
        metrics["silhouette_score"] = best_silhouette
        metrics["n_trusted_clients"] = n_trusted

        return ndarrays_to_parameters(final_ndarrays), metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training, sending personalized models.
        """
        # Get standard config from parent (e.g., sample fraction)
        config, _ = self.get_config_for_next_round(server_round, {}, client_manager)

        # Sample clients
        clients = client_manager.sample(
            num_clients=self.num_fit_clients(client_manager.num_available()),
            min_num_clients=self.min_fit_clients,
        )

        # Extract the global common parameters sent from aggregate_fit
        global_params_ndarrays = parameters_to_ndarrays(parameters)
        global_params_dict = OrderedDict(
            zip(self.ordered_param_keys, global_params_ndarrays)
        )
        global_common_params, _ = self._split_params(global_params_dict)

        fit_configurations = []
        for client in clients:
            cid = client.cid

            # Find this client's cluster
            cluster_id = self.client_clusters.get(cid)

            # Get the appropriate specialist model
            if cluster_id is not None and cluster_id in self.cluster_specialist_params:
                # Client has a cluster and its model exists
                specialist_params = self.cluster_specialist_params[cluster_id]
            else:
                # New client or unclustered client: send the average specialist model
                specialist_params = self.avg_specialist_params
                if specialist_params is None:  # Should not happen after round 1
                    logger.warning(
                        f"No average specialist model available for client {cid}. Using global specialist params."
                    )
                    _, specialist_params = self._split_params(global_params_dict)

            # Rebuild the full, personalized model
            personalized_params_dict = self._rebuild_params(
                global_common_params, specialist_params
            )
            personalized_ndarrays = list(personalized_params_dict.values())

            fit_ins = FitIns(ndarrays_to_parameters(personalized_ndarrays), config)
            fit_configurations.append((client, fit_ins))

        logger.info(
            f"Round {server_round}: Configuring fit for {len(clients)} clients with personalized models."
        )
        return fit_configurations


# --- Example Server Setup (Conceptual) ---
if __name__ == "__main__":
    logger.info("Conceptual setup of XS_FedPRS_Strategy")

    # 1. Initialize components
    defense_module = GeneticDefense(
        hwe_significance_threshold=1e-6,
        ref_allele_freq_path="../data/PSR/reference_frequencies.csv",
    )
    trust_module = TrustManager(
        decay_rate=0.2,
        hwe_threshold=0.3,
        afc_threshold=2.0,
        grad_plausibility_threshold=0.6,
    )

    # 2. Provide model config (needed for param names)
    model_config = {
        "n_rare_variants": 500,  # Example
        "common_hidden_dim": 16,
        "rare_hidden_dim": 64,
    }

    # 3. Create the strategy instance
    strategy = XS_FedPRS_Strategy(
        genetic_defense=defense_module,
        trust_manager=trust_module,
        model_config=model_config,
        min_fit_clients=2,
        min_available_clients=2,
    )

    logger.info("XS_FedPRS_Strategy initialized successfully.")

    # 4. Start the simulation (requires client.py and a main script)
    # fl.simulation.start_simulation(
    #     client_fn=your_client_fn, # Defined in client.py or main
    #     num_clients=5,
    #     config=fl.server.ServerConfig(num_rounds=10),
    #     strategy=strategy,
    # )
