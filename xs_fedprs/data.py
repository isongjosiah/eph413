# xs_fedprs/data.py
"""
Data loading and splitting module for federated PRS analysis.
Handles loading genetic data, PRS scores, and partitioning for federated learning.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass
import logging
from sklearn.cluster import KMeans  # Import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""

    file_path: str = "../data/PSR/final_combined_data.csv"
    pca_file_path: str = "EUR.eigenvec"  # Added PCA file path
    num_clients: int = 5
    seed: int = 42
    validation_split: float = 0.2


@dataclass
class HeterogeneityConfig:
    """Configuration for controlling data heterogeneity across clients."""

    strategy: str = (
        "iid"  # Options: "iid", "pca_stratified", "rare_variant_enriched", "phenotype_imbalanced", "dirichlet"
    )
    num_pcs_for_stratification: int = 2  # Number of PCs for pca_stratified
    enrichment_factor: float = 2.0  # Factor for rare variant enrichment
    phenotype_imbalance_range: Tuple[float, float] = (
        0.3,
        0.7,
    )  # Range for case proportion
    dirichlet_alpha: float = (
        0.5  # Alpha parameter for Dirichlet distribution (lower = more heterogeneous)
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
        return {}

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file: {e}. Using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}. Using defaults.")
        return {}


def load_data(
    file_path: str = "../data/PSR/final_combined_data.csv",
    pca_file_path: str = "../psr/EUR.eigenvec",  # Path to PLINK's PCA output
    validate: bool = True,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[str]],
    Optional[pd.Series],
    Optional[pd.DataFrame],
]:
    """
    Load pre-processed genetic data, PRS, phenotype, AND PCA eigenvectors.

    Args:
        file_path: Path to the combined data CSV file.
        pca_file_path: Path to the PLINK PCA output (.eigenvec).
        validate: Whether to perform data validation checks.

    Returns:
        Tuple containing:
        - prs_scores: Array of PRS scores
        - rare_variant_dosages: Matrix of rare variant dosages
        - phenotypes: Array of binary phenotypes
        - rare_variant_names: List of rare variant names
        - sample_ids: Series of sample IDs
        - pca_df: DataFrame containing PCA eigenvectors (FID, IID, PC1, PC2...)
    """
    data_file = Path(file_path)
    pca_file = Path(pca_file_path)

    logger.info(f"file path is {file_path}")
    if not data_file.exists():
        logger.error(f"Data file not found at {file_path}")
        return None, None, None, None, None, None

    # --- Load Main Data ---
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Successfully read data file: {file_path}")
    except Exception as e:
        logger.error(f"Error reading data file {file_path}: {e}")
        return None, None, None, None, None, None

    # --- Load PCA Data ---
    pca_vectors = None  # Initialize as None
    if not pca_file.exists():
        logger.warning(
            f"PCA file not found at {pca_file_path}. Population stratification will not use PCA."
        )
    else:
        try:
            # Assuming 6 PCs were calculated as in analysis.sh
            # Adjust num_pcs if different
            num_pcs = 6
            pca_cols = ["FID", "IID"] + [f"PC{i+1}" for i in range(num_pcs)]
            pca_df_loaded = pd.read_csv(pca_file, sep=" ", header=None, names=pca_cols)
            logger.info(f"Successfully read PCA file: {pca_file_path}")

            # Merge PCA data ensuring alignment by IID
            # Ensure IID types match before merging
            df["IID"] = df["IID"].astype(str)
            pca_df_loaded["IID"] = pca_df_loaded["IID"].astype(str)

            # Merge PCA keeping only rows present in the main data, maintain original order
            merged_df = pd.merge(df[["IID"]], pca_df_loaded, on="IID", how="left")

            if len(merged_df) == len(df):
                pca_vectors = merged_df[["IID"] + [f"PC{i+1}" for i in range(num_pcs)]]
                # Set index to match the main df index for easier row alignment later
                pca_vectors.index = df.index
                logger.info(f"PCA vectors aligned with main data.")
            else:
                logger.warning(
                    f"Mismatch in samples between data file ({len(df)}) and PCA file ({len(pca_df_loaded)}). PCA stratification might be incomplete."
                )
                # Fallback: pca_vectors remains None

        except Exception as e:
            logger.error(f"Error reading or merging PCA file {pca_file_path}: {e}")
            pca_vectors = None

    # --- Validate and Extract Core Data ---
    required_cols = ["IID", "BEST_PRS", "PHENOTYPE"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(f"Required columns missing: {missing_cols}")
        return None, None, None, None, None, None

    sample_ids = df["IID"]  # Keep as Series with original index
    prs_scores = df["BEST_PRS"].values
    phenotypes = df["PHENOTYPE"].values

    exclude_cols = {"IID", "BEST_PRS", "PHENOTYPE", "PHENOTYPE_P"}
    rare_variant_cols = [col for col in df.columns if col not in exclude_cols]
    rare_variant_dosages = df[rare_variant_cols].values.astype(np.float32)
    rare_variant_names = rare_variant_cols

    if validate:
        _validate_data(
            df, prs_scores, phenotypes, rare_variant_dosages, rare_variant_names
        )

    # Log summary statistics
    logger.info(f"Data loaded successfully:")
    logger.info(f"  - Samples: {len(df)}")
    logger.info(f"  - PRS scores shape: {prs_scores.shape}")
    logger.info(f"  - Phenotypes shape: {phenotypes.shape}")
    logger.info(f"  - Rare variant dosages shape: {rare_variant_dosages.shape}")
    logger.info(f"  - Number of rare variants: {len(rare_variant_names)}")
    try:
        logger.info(
            f"  - Phenotype distribution (0s, 1s): {np.bincount(phenotypes.astype(int))}"
        )
    except ValueError:
        logger.warning(
            "Could not determine phenotype distribution (likely non-integer values)."
        )

    return (
        prs_scores,
        rare_variant_dosages,
        phenotypes,
        rare_variant_names,
        sample_ids,
        pca_vectors,
    )


def _validate_data(
    df: pd.DataFrame,
    prs_scores: np.ndarray,
    phenotypes: np.ndarray,
    rare_dosages: np.ndarray,
    variant_names: List[str],
) -> None:
    """Perform validation checks on loaded data."""

    # Check for missing values
    if np.isnan(prs_scores).any():
        logger.warning(f"Found {np.isnan(prs_scores).sum()} missing PRS scores")

    if np.isnan(phenotypes).any():
        logger.warning(f"Found {np.isnan(phenotypes).sum()} missing phenotypes")

    # Check phenotype encoding - handle potential NaNs
    unique_pheno = np.unique(phenotypes[~np.isnan(phenotypes)])
    # Allow for common PLINK encodings (1/2) besides 0/1
    allowed_phenos = {0.0, 1.0, 1, 2, -9}  # Include -9 for missing
    # Check if any unique phenotype is NOT in the allowed set
    if not set(unique_pheno).issubset(allowed_phenos):
        logger.warning(
            f"Phenotypes may contain unexpected values: {unique_pheno}. Expected subset of {allowed_phenos}"
        )
    elif 1 in unique_pheno and 2 in unique_pheno and (0 not in unique_pheno):
        logger.info(
            "Phenotypes seem to be encoded as 1 (control) / 2 (case). Will convert 1->0, 2->1."
        )
        # Conversion will happen during splitting if needed, just log here.

    # Check for rare variant quality
    missing_dosages = np.isnan(rare_dosages).sum()
    if missing_dosages > 0:
        logger.warning(f"Found {missing_dosages} missing rare variant dosages (NaNs)")

    # Check for invariant variants (handle potential all-NaN columns)
    try:
        variant_std = np.nanstd(rare_dosages, axis=0)  # Use nanstd to ignore NaNs
        invariant_count = np.sum(variant_std == 0)
        if invariant_count > 0:
            logger.warning(
                f"Found {invariant_count} invariant rare variants (std=0, ignoring NaNs)"
            )
    except Exception as e:
        logger.warning(f"Could not calculate standard deviation for rare variants: {e}")


def split_data_federated(
    prs_scores: np.ndarray,
    rare_variant_dosages: np.ndarray,
    phenotypes: np.ndarray,
    sample_ids: pd.Series,
    num_clients: int,
    heterogeneity_config: Optional[HeterogeneityConfig] = None,
    pca_vectors: Optional[pd.DataFrame] = None,  # Pass pca_vectors
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Split data among clients with configurable heterogeneity.

    Args:
        prs_scores: Array of PRS scores
        rare_variant_dosages: Matrix of rare variant dosages
        phenotypes: Array of phenotypes (expected 0/1 or 1/2 for PLINK)
        sample_ids: Series of sample IDs (index must align with data arrays)
        num_clients: Number of clients to partition data for
        heterogeneity_config: Configuration for data heterogeneity
        pca_vectors: DataFrame with IID and PC columns for PCA stratification (index must align)
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries, each representing a client's dataset
    """
    n_samples = len(phenotypes)
    original_indices = sample_ids.index  # Get original DataFrame index

    if n_samples == 0:
        logger.warning("No samples to split")
        return []

    np.random.seed(seed)

    # --- Phenotype Conversion (PLINK 1/2 to 0/1) ---
    # Convert only if 1 and 2 are present and 0 is not
    unique_pheno = np.unique(phenotypes[~np.isnan(phenotypes)])
    if 1 in unique_pheno and 2 in unique_pheno and (0 not in unique_pheno):
        logger.info("Converting phenotypes from 1/2 (control/case) to 0/1.")
        phenotypes = np.where(phenotypes == 1, 0, phenotypes)  # 1 -> 0 (control)
        phenotypes = np.where(phenotypes == 2, 1, phenotypes)  # 2 -> 1 (case)
    # Ensure phenotypes are integer type after potential conversion or if already 0/1
    phenotypes = phenotypes.astype(int)

    # Default to IID if no config provided
    if heterogeneity_config is None:
        heterogeneity_config = HeterogeneityConfig(strategy="iid")

    strategy = heterogeneity_config.strategy.lower()
    logger.info(f"Splitting data using strategy: {strategy}")

    # --- Select Splitting Strategy ---
    if strategy == "iid":
        client_indices = _split_iid(n_samples, num_clients)
    elif strategy == "pca_stratified":
        if pca_vectors is not None and not pca_vectors.empty:
            # Ensure pca_vectors index matches the main data index before passing
            if not pca_vectors.index.equals(original_indices):
                logger.error(
                    "PCA vectors index does not match main data index. Cannot perform PCA stratification."
                )
                return []  # Or fallback to IID?
            client_indices = _split_pca_stratified(
                pca_vectors,
                num_clients,
                num_pcs_to_use=heterogeneity_config.num_pcs_for_stratification,
                seed=seed,
            )
        else:
            logger.warning(
                "PCA vectors requested for stratification but not available/aligned. Falling back to IID."
            )
            client_indices = _split_iid(n_samples, num_clients)
    elif strategy == "rare_variant_enriched":
        client_indices = _split_rare_variant_enriched(rare_variant_dosages, num_clients)
    elif strategy == "phenotype_imbalanced":
        client_indices = _split_phenotype_imbalanced(
            phenotypes, num_clients, heterogeneity_config.phenotype_imbalance_range
        )
    else:
        logger.warning(f"Unknown strategy '{strategy}', defaulting to IID")
        client_indices = _split_iid(n_samples, num_clients)

    # --- Create client datasets ---
    client_datasets = []
    for i, client_idx in enumerate(client_indices):
        # Ensure client_idx contains valid indices relative to the original DataFrame
        if len(client_idx) == 0:
            logger.warning(f"Client {i} received no data")
            continue

        # Use the indices directly to slice the numpy arrays
        client_prs = prs_scores[client_idx]
        client_rare = rare_variant_dosages[client_idx, :]
        client_pheno = phenotypes[client_idx]
        # Use the indices to get the correct sample IDs from the Series
        client_sids = sample_ids.iloc[client_idx].tolist()

        client_data = {
            "prs_scores": client_prs,
            "rare_dosages": client_rare,
            "phenotypes": client_pheno,
            "sample_ids": client_sids,
            "client_id": i,
            "n_samples": len(client_idx),
        }

        # Add statistics calculation
        client_data["case_proportion"] = (
            np.mean(client_data["phenotypes"]) if client_data["n_samples"] > 0 else 0
        )
        client_data["prs_mean"] = (
            np.mean(client_data["prs_scores"]) if client_data["n_samples"] > 0 else 0
        )
        client_data["prs_std"] = (
            np.std(client_data["prs_scores"]) if client_data["n_samples"] > 0 else 0
        )

        client_datasets.append(client_data)

    _log_split_summary(client_datasets, strategy)

    return client_datasets


def _split_iid(n_samples: int, num_clients: int) -> List[np.ndarray]:
    """Split data randomly (IID) among clients, returns original indices."""
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return np.array_split(indices, num_clients)


def _split_pca_stratified(
    pca_vectors: pd.DataFrame,  # Expects DataFrame with original index and PC columns
    num_clients: int,
    num_pcs_to_use: int = 2,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Split data based on PCA clustering. Returns original indices.
    """
    if pca_vectors is None or pca_vectors.empty:
        logger.warning("PCA vectors not available for stratification.")
        # Cannot fallback to IID here easily without n_samples, should be handled by caller
        return [np.array([], dtype=int) for _ in range(num_clients)]

    n_clusters = num_clients  # Aim for one primary cluster per client
    pc_cols = [f"PC{i+1}" for i in range(num_pcs_to_use)]

    # Handle potential NaNs in PCA vectors
    pca_vectors_clean = pca_vectors.dropna(subset=pc_cols)
    original_indices_clean = pca_vectors_clean.index  # Store original indices

    if len(pca_vectors_clean) < n_clusters:
        logger.warning(
            f"Not enough clean PCA samples ({len(pca_vectors_clean)}) for {n_clusters} clusters. Reducing clusters."
        )
        n_clusters = max(1, len(pca_vectors_clean))

    if n_clusters == 0:
        logger.error("No valid samples with PCA data for clustering.")
        return [np.array([], dtype=int) for _ in range(num_clients)]

    logger.info(
        f"Performing K-Means clustering on {num_pcs_to_use} PCs into {n_clusters} clusters..."
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    # Fit on the numeric PC data, store cluster labels
    cluster_labels = kmeans.fit_predict(pca_vectors_clean[pc_cols])
    pca_vectors_clean["cluster"] = cluster_labels

    # --- Assign Clusters to Clients ---
    client_indices = [[] for _ in range(num_clients)]
    cluster_ids_unique = np.unique(cluster_labels)
    # Shuffle cluster IDs before assigning
    shuffled_cluster_ids = np.random.permutation(cluster_ids_unique)

    # Assign each cluster primarily to a client (round-robin)
    samples_assigned_count = 0
    for i, cluster_id in enumerate(shuffled_cluster_ids):
        # Get original indices for samples in this cluster
        cluster_original_indices = pca_vectors_clean[
            pca_vectors_clean["cluster"] == cluster_id
        ].index.values
        client_target = i % num_clients
        client_indices[client_target].extend(cluster_original_indices)
        samples_assigned_count += len(cluster_original_indices)
        logger.debug(
            f"Assigned cluster {cluster_id} ({len(cluster_original_indices)} samples) to client {client_target}"
        )

    # Handle samples potentially missed due to NaN dropping
    if samples_assigned_count < len(pca_vectors):
        all_original_indices = pca_vectors.index
        assigned_indices = pca_vectors_clean.index
        missed_indices = all_original_indices.difference(assigned_indices).values

        logger.warning(
            f"Distributing {len(missed_indices)} samples missed during PCA clustering (due to NaNs?) randomly."
        )
        # Distribute missed samples somewhat evenly using modulo
        for idx, missed_idx in enumerate(missed_indices):
            client_target = idx % num_clients
            client_indices[client_target].append(missed_idx)

    # Final conversion to numpy arrays and shuffle within client
    final_client_indices = []
    for i in range(num_clients):
        # Indices are already original indices, just ensure type and shuffle
        indices_np = np.array(client_indices[i], dtype=int)
        np.random.shuffle(indices_np)
        final_client_indices.append(indices_np)

    logger.info(
        f"Split data into {num_clients} clients based on {n_clusters} PCA clusters."
    )
    return final_client_indices


def _split_rare_variant_enriched(
    rare_dosages: np.ndarray, num_clients: int, seed: int = 42
) -> List[np.ndarray]:
    """
    Split data by enriching some clients with high rare-variant-burden samples.
    Returns original indices.

    Args:
        rare_dosages: Matrix of rare variant dosages (samples x variants).
        num_clients: Number of clients.
        seed: Random seed.

    Returns:
        List of numpy arrays, where each array contains sample indices for a client.
    """
    n_samples = rare_dosages.shape[0]
    np.random.seed(seed)

    # 1. Calculate a "rare variant burden" score for each sample.
    #    We sum the dosages (0, 1, or 2) for all rare variants.
    #    Samples with more rare alleles will have a higher score.
    variant_burden = np.sum(rare_dosages, axis=1)

    # 2. Get the original indices (0 to n_samples-1) and sort them
    #    based on the burden score, from highest burden to lowest.
    sorted_original_indices = np.argsort(variant_burden)[
        ::-1
    ]  # [::-1] reverses to high-to-low

    # 3. Designate clients for enrichment (e.g., first half)
    n_enriched_clients = max(1, num_clients // 2)
    n_control_clients = num_clients - n_enriched_clients

    client_indices = [[] for _ in range(num_clients)]

    # 4. Split the sorted indices into high-burden and low-burden pools.
    #    We split the samples proportionally to the number of clients in each group.
    n_high_burden_samples = int(n_samples * (n_enriched_clients / num_clients))

    high_burden_indices = sorted_original_indices[:n_high_burden_samples]
    low_burden_indices = sorted_original_indices[n_high_burden_samples:]

    # 5. Distribute the pools to the corresponding clients.
    #    Distribute high-burden samples to "enriched" clients.
    if n_enriched_clients > 0:
        high_burden_splits = np.array_split(high_burden_indices, n_enriched_clients)
        for i in range(n_enriched_clients):
            client_indices[i] = high_burden_splits[i].tolist()

    #    Distribute low-burden samples to "control" clients.
    if n_control_clients > 0:
        low_burden_splits = np.array_split(low_burden_indices, n_control_clients)
        for i in range(n_control_clients):
            # Add to the clients *after* the enriched ones
            client_indices[n_enriched_clients + i] = low_burden_splits[i].tolist()

    # 6. Randomly shuffle the client assignments.
    #    This prevents Client 0 from *always* being the enriched one.
    np.random.shuffle(client_indices)

    # 7. Return the final list of index arrays.
    return [np.array(indices, dtype=int) for indices in client_indices]


def _split_phenotype_imbalanced(
    phenotypes: np.ndarray,
    num_clients: int,
    imbalance_range: Tuple[float, float],
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Split data ensuring varying case/control ratios across clients.
    Returns original indices.

    Args:
        phenotypes: Array of binary phenotypes (0s and 1s).
        num_clients: Number of clients.
        imbalance_range: Tuple (min_case_prop, max_case_prop) defining the target
                         range for the proportion of cases in each client's dataset.
        seed: Random seed.

    Returns:
        List of numpy arrays, where each array contains sample indices for a client.
    """
    n_samples = len(phenotypes)
    np.random.seed(seed)
    original_indices = np.arange(n_samples)  # Store original indices

    # 1. Separate indices by phenotype
    case_indices = original_indices[phenotypes == 1]
    control_indices = original_indices[phenotypes == 0]
    np.random.shuffle(case_indices)
    np.random.shuffle(control_indices)

    # 2. Determine target sizes and proportions per client
    #    Start with roughly equal sizes
    base_client_size = n_samples // num_clients
    client_sizes = [base_client_size] * num_clients
    # Distribute remainder samples
    remainder = n_samples % num_clients
    for i in range(remainder):
        client_sizes[i] += 1

    # Generate target case proportions within the specified range for each client
    min_prop, max_prop = imbalance_range
    # Ensure min_prop < max_prop and they are valid proportions
    min_prop = max(0.01, min(min_prop, 0.99))  # Avoid 0% or 100% unless specified
    max_prop = max(min_prop, min(max_prop, 0.99))
    target_proportions = np.random.uniform(min_prop, max_prop, num_clients)

    logger.info(
        f"Target case proportions per client: {[f'{p:.2%}' for p in target_proportions]}"
    )

    # 3. Assign samples iteratively
    client_indices = [[] for _ in range(num_clients)]
    case_ptr = 0
    control_ptr = 0
    available_cases = len(case_indices)
    available_controls = len(control_indices)

    for i in range(num_clients):
        target_size = client_sizes[i]
        target_n_cases = int(target_size * target_proportions[i])
        target_n_controls = target_size - target_n_cases

        # Determine actual number of cases/controls to assign based on availability
        assign_n_cases = min(target_n_cases, available_cases - case_ptr)
        assign_n_controls = min(target_n_controls, available_controls - control_ptr)

        # If short on one type, try to compensate with the other if possible
        shortfall_cases = target_n_cases - assign_n_cases
        shortfall_controls = target_n_controls - assign_n_controls

        if shortfall_cases > 0:
            extra_controls = min(
                shortfall_cases, available_controls - control_ptr - assign_n_controls
            )
            assign_n_controls += extra_controls
        elif shortfall_controls > 0:
            extra_cases = min(
                shortfall_controls, available_cases - case_ptr - assign_n_cases
            )
            assign_n_cases += extra_cases

        # Check pointers don't exceed available indices
        end_case_ptr = case_ptr + assign_n_cases
        end_control_ptr = control_ptr + assign_n_controls
        if end_case_ptr > available_cases:
            end_case_ptr = available_cases
        if end_control_ptr > available_controls:
            end_control_ptr = available_controls

        # Assign indices
        indices_for_client = np.concatenate(
            [
                case_indices[case_ptr:end_case_ptr],
                control_indices[control_ptr:end_control_ptr],
            ]
        )
        np.random.shuffle(indices_for_client)
        client_indices[i] = indices_for_client.tolist()

        # Update pointers
        case_ptr = end_case_ptr
        control_ptr = end_control_ptr

    # 4. Distribute any remaining samples (less likely with balanced initial sizes)
    remaining_indices = np.concatenate(
        [case_indices[case_ptr:], control_indices[control_ptr:]]
    )
    np.random.shuffle(remaining_indices)
    remainder_splits = np.array_split(remaining_indices, num_clients)
    for i in range(num_clients):
        client_indices[i].extend(remainder_splits[i])

    # 5. Final conversion and shuffle
    final_client_indices = []
    for i in range(num_clients):
        indices_np = np.array(client_indices[i], dtype=int)
        np.random.shuffle(indices_np)
        final_client_indices.append(indices_np)

    return final_client_indices


def _log_split_summary(client_datasets: List[Dict[str, Any]], strategy: str) -> None:
    """Log summary statistics of data split."""
    logger.info(f"\nData split summary ({strategy} strategy):")
    logger.info(f"  Total clients: {len(client_datasets)}")
    total_samples = sum(cd["n_samples"] for cd in client_datasets)
    logger.info(f"  Total samples distributed: {total_samples}")

    for i, client_data in enumerate(client_datasets):
        logger.info(
            f"  Client {i}: n={client_data['n_samples']}, "
            f"cases={client_data['case_proportion']:.2%}, "
            f"PRS_mean={client_data['prs_mean']:.3f}, "
            f"PRS_std={client_data['prs_std']:.3f}"
        )


def create_train_val_split(
    client_data: Dict[str, Any], val_split: float = 0.2, seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split a client's data into training and validation sets.

    Args:
        client_data: Dictionary containing client's data
        val_split: Proportion of data for validation
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data) dictionaries
    """
    np.random.seed(seed)
    n_samples = client_data["n_samples"]
    if n_samples == 0:  # Handle empty client data
        return client_data, client_data.copy()

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    val_size = int(n_samples * val_split)
    # Ensure val_size is at least 0 and not more than n_samples
    val_size = max(0, min(val_size, n_samples))

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_data = {
        "prs_scores": client_data["prs_scores"][train_indices],
        "rare_dosages": client_data["rare_dosages"][train_indices],
        "phenotypes": client_data["phenotypes"][train_indices],
        "sample_ids": [client_data["sample_ids"][i] for i in train_indices],
        "client_id": client_data["client_id"],
        "n_samples": len(train_indices),
    }

    val_data = {
        "prs_scores": client_data["prs_scores"][val_indices],
        "rare_dosages": client_data["rare_dosages"][val_indices],
        "phenotypes": client_data["phenotypes"][val_indices],
        "sample_ids": [client_data["sample_ids"][i] for i in val_indices],
        "client_id": client_data["client_id"],
        "n_samples": len(val_indices),
    }

    return train_data, val_data


if __name__ == "__main__":
    logger.info("Running data loading and splitting example...")

    # Load data including PCA
    # Ensure EUR.eigenvec is in the same directory or provide correct path
    prs, rare_dosages, pheno, rare_names, s_ids, pca_vecs = load_data(
        file_path="../data/PSR/final_combined_data.csv",
        pca_file_path="../psr/EUR.eigenvec",
    )

    if prs is not None and s_ids is not None:
        num_federated_clients = 5

        # Example 1: IID Split
        logger.info("\n=== Example 1: IID Split ===")
        client_data_iid = split_data_federated(
            prs,
            rare_dosages,
            pheno,
            s_ids,
            num_clients=num_federated_clients,
            heterogeneity_config=HeterogeneityConfig(strategy="iid"),
            pca_vectors=pca_vecs,  # Pass PCA vectors even if not used by IID
        )

        # Example 2: PCA Stratified Split
        logger.info("\n=== Example 2: PCA Stratified Split ===")
        client_data_pca = split_data_federated(
            prs,
            rare_dosages,
            pheno,
            s_ids,
            num_clients=num_federated_clients,
            heterogeneity_config=HeterogeneityConfig(
                strategy="pca_stratified", num_pcs_for_stratification=2
            ),  # Specify strategy
            pca_vectors=pca_vecs,  # Pass PCA vectors
        )
        # ... (after the PCA split example)

        # Example 3: Rare Variant Enriched Split
        logger.info("\n=== Example 3: Rare Variant Enriched Split ===")
        client_data_rare = split_data_federated(
            prs,
            rare_dosages,
            pheno,
            s_ids,
            num_clients=num_federated_clients,
            heterogeneity_config=HeterogeneityConfig(strategy="rare_variant_enriched"),
            pca_vectors=pca_vecs,
        )

        # You can inspect client_data_rare to see the different sample sizes
        # and (if you calculate it) the different average rare variant burdens.
        if client_data_rare:
            logger.info("Average rare variant burden per client (enrichment test):")
            for client_data in client_data_rare:
                burden = np.mean(np.sum(client_data["rare_dosages"], axis=1))
                logger.info(
                    f"  Client {client_data['client_id']}: n={client_data['n_samples']}, Avg. Burden={burden:.2f}"
                )

            # --- (You can add calls for other strategies here once implemented) ---
            #
        # Example 4: Phenotype Imbalanced Split
        logger.info("\n=== Example 4: Phenotype Imbalanced Split ===")
        imbalance_range_example = (0.2, 0.8)  # Target 20% to 80% cases per client
        client_data_pheno = split_data_federated(
            prs,
            rare_dosages,
            pheno,
            s_ids,
            num_clients=num_federated_clients,
            heterogeneity_config=HeterogeneityConfig(
                strategy="phenotype_imbalanced",
                phenotype_imbalance_range=imbalance_range_example,
            ),
            pca_vectors=pca_vecs,
        )

        # Create train/val split for first client from PCA split
        if client_data_pca:
            train_data, val_data = create_train_val_split(client_data_pca[0])
            logger.info(f"\nTrain/Val split for PCA-Client 0:")
            logger.info(f"  Train samples: {train_data['n_samples']}")
            logger.info(f"  Val samples: {val_data['n_samples']}")
        else:
            logger.warning("PCA split resulted in no clients.")

    else:
        logger.error("Data loading failed. Cannot proceed with splitting.")
