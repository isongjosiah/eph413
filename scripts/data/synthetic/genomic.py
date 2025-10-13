"""
This script prepares the input feature matrix for the models based on the data prepared in the psr module.
It also helps with partitioning the datasets to simulate different allelic heterogeneity situations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def _calculate_sds(row, lms_male, lms_female):
    """
    Calculates the Standard Deviation Score (SDS) for a single row.

    Args:
        row: DataFrame row containing PHENOTYPE, Agemos, and SEX.
        lms_male: Dictionary of male LMS parameters by age in months.
        lms_female: Dictionary of female LMS parameters by age in months.

    Returns:
        float or None: The calculated SDS, or None if parameters unavailable.
    """
    height = row["PHENOTYPE"]
    age_months = row["Agemos"]
    sex = row["SEX"]

    # Select appropriate LMS parameters based on sex
    lms_params = (
        lms_male.get(age_months)
        if sex == 1
        else lms_female.get(age_months) if sex == 2 else None
    )

    if lms_params is None:
        return None  # Age is out of CDC table's range or invalid sex

    L, M, S = lms_params["L"], lms_params["M"], lms_params["S"]

    # Calculate SDS using LMS formula
    import math

    if L == 0:
        sds = math.log(height / M) / S
    else:
        ratio = height / M
        # Prevent negative or zero ratio
        if ratio <= 0:
            # TODO: come back here
            # raise ValueError("height/M must be positive")
            return 0
        sds = ((ratio**L) - 1) / (L * S)

    return sds


def _categorize_height(sds):
    """
    Assigns a height category based on the SDS score.

    Args:
        sds: Standard Deviation Score.

    Returns:
        str: Height category ("Short", "Mid", "Tall", or "Unknown").
    """
    if pd.isna(sds):
        return "Unknown"
    elif sds >= 2.0:
        return "Tall"
    else:
        return "Short"


def prepare_feature_matrix():
    """
    Prepares the input feature matrix for the models based on the data prepared in the psr module.

    Args:
        N/A

    Returns:
        pd.DataFrame: The input feature matrix with calculated height categories.
    """
    try:
        # Load feature data from plink recode
        feature_matrix = pd.read_csv("./psr/extracted_alleles.raw", sep=" ")
        # Load CDC reference data
        cdc_data = pd.read_excel("./data/PSR/statage.xls")
    except Exception as e:
        print(f"Failed to read data for preparing feature matrix: {e}")
        return None

    # NOTE: We have requested access to the metadata for CINECA for simulated age info.
    # Until then, we randomly assign ages for implementation purposes.
    np.random.seed(42)
    feature_matrix["Age_years"] = np.random.uniform(2, 20, size=len(feature_matrix))
    feature_matrix["Agemos"] = (feature_matrix["Age_years"] * 12).apply(np.floor) + 0.5
    feature_matrix.drop("FID", axis=1, inplace=True)
    feature_matrix.drop("IID", axis=1, inplace=True)
    feature_matrix.drop("PAT", axis=1, inplace=True)
    feature_matrix.drop("MAT", axis=1, inplace=True)

    # Create dictionaries for male and female LMS parameters, keyed by age in months
    lms_male = (
        cdc_data[cdc_data["Sex"] == 1]
        .set_index("Agemos")[["L", "M", "S"]]
        .to_dict("index")
    )
    lms_female = (
        cdc_data[cdc_data["Sex"] == 2]
        .set_index("Agemos")[["L", "M", "S"]]
        .to_dict("index")
    )

    # Calculate height SDS
    feature_matrix["Height_SDS"] = feature_matrix.apply(
        lambda row: _calculate_sds(row, lms_male, lms_female), axis=1
    )

    # Categorize heights
    feature_matrix["Height_Category"] = feature_matrix["Height_SDS"].apply(
        _categorize_height
    )

    feature_matrix.drop("Height_SDS", axis=1, inplace=True)
    feature_matrix.drop("Agemos", axis=1, inplace=True)
    feature_matrix.drop("Age_years", axis=1, inplace=True)

    try:
        # Save the feature matrix to a CSV file
        output_path = "./data/PSR/prepared_feature_matrix.csv"
        feature_matrix.to_csv(output_path, index=False)
        print(f"Feature matrix saved to {output_path}")
    except Exception as e:
        print(f"Failed to save feature matrix: {e}")

    return feature_matrix


def partition_data(feature_matrix, num_partitions, rare_variant_threshold=0.05):
    """
    Partitions the dataset to simulate rare variant heterogeneity.

    Args:
        feature_matrix: The input feature matrix.
        num_partitions: The number of partitions to create.
        rare_variant_threshold: The frequency threshold to identify rare variants.

    Returns:
        A list of partitions.
    """
    # Identify rare variants
    variant_frequencies = (
        feature_matrix.drop(columns=["Height_Category"], errors="ignore") != 0
    ).mean()
    rare_variants = variant_frequencies[
        variant_frequencies < rare_variant_threshold
    ].index.tolist()

    if not rare_variants:
        # If no rare variants, fall back to the previous partitioning strategy
        return partition_by_height_category(feature_matrix, num_partitions)

    partitions = []
    samples_per_partition = len(feature_matrix) // num_partitions

    # Assign each partition a subset of rare variants to be enriched for
    rare_variant_subsets = np.array_split(rare_variants, num_partitions)

    for i in range(num_partitions):
        enriched_variants = rare_variant_subsets[i]

        # Find samples that have at least one of the enriched rare variants
        if not enriched_variants.size:
            continue
        partition_samples_mask = (feature_matrix[enriched_variants] != 0).any(axis=1)
        partition_samples = feature_matrix[partition_samples_mask]

        # If not enough samples, fill with random samples
        if len(partition_samples) < samples_per_partition:
            remaining_samples_count = samples_per_partition - len(partition_samples)
            if remaining_samples_count > 0:
                remaining_samples = feature_matrix[~partition_samples_mask].sample(
                    n=remaining_samples_count, replace=True
                )
                partition = pd.concat([partition_samples, remaining_samples])
            else:
                partition = partition_samples
        else:
            partition = partition_samples.sample(n=samples_per_partition, replace=False)

        partitions.append(partition)

    return partitions


def partition_by_height_category(feature_matrix, num_partitions):
    # The previously defined partitioning function
    partitions = []
    tall_group = feature_matrix[feature_matrix["Height_Category"] == "Tall"]
    short_group = feature_matrix[feature_matrix["Height_Category"] == "Short"]

    proportions = []
    for i in range(num_partitions):
        tall_prop = (i + 1) / (num_partitions + 1)
        short_prop = 1 - tall_prop
        proportions.append({"tall": tall_prop, "short": short_prop})

    for i in range(num_partitions):
        tall_sample = tall_group.sample(frac=proportions[i]["tall"])
        short_sample = short_group.sample(frac=proportions[i]["short"])
        partition = pd.concat([tall_sample, short_sample])
        partitions.append(partition)

    return partitions


class GeneticDataGenerator:
    """
    Generates synthetic genetic data with common and rare variants for federated learning.
    Simulates population structure and allelic heterogeneity.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_common_variants: int = 100,
        n_rare_variants: int = 500,
        n_populations: int = 3,
        rare_variant_freq: float = 0.05,
    ):
        """
        Initialize genetic data generator.

        Args:
            n_samples: Number of samples to generate
            n_common_variants: Number of common SNPs for PRS calculation
            n_rare_variants: Number of rare variants
            n_populations: Number of distinct populations (for heterogeneity)
            rare_variant_freq: Frequency threshold for rare variants
        """
        self.n_samples = n_samples
        self.n_common_variants = n_common_variants
        self.n_rare_variants = n_rare_variants
        self.n_populations = n_populations
        self.rare_variant_freq = rare_variant_freq

    def generate_population_data(self, population_id: int) -> Dict:
        """
        Generate genetic data for a specific population with unique rare variant profile.

        Args:
            population_id: Population identifier for creating population-specific patterns

        Returns:
            Dictionary containing PRS scores, rare variant dosages, phenotypes, and metadata
        """
        # Generate common variant effects (shared across populations)
        common_effects = np.random.normal(0, 0.1, self.n_common_variants)

        # Generate common variant genotypes (0, 1, 2 copies)
        common_genotypes = np.random.choice(
            [0, 1, 2],
            size=(self.n_samples, self.n_common_variants),
            p=[0.25, 0.5, 0.25],
        )

        # Calculate PRS from common variants
        prs_scores = np.dot(common_genotypes, common_effects)
        prs_scores = (prs_scores - prs_scores.mean()) / prs_scores.std()

        # Generate population-specific rare variant patterns
        # Each population has different sets of active rare variants
        rare_variant_mask = np.zeros(self.n_rare_variants)

        # Population-specific rare variants (20-30% unique to each population)
        n_population_specific = int(self.n_rare_variants * 0.25)
        start_idx = population_id * n_population_specific
        end_idx = min(start_idx + n_population_specific, self.n_rare_variants)
        rare_variant_mask[start_idx:end_idx] = 1

        # Add some shared rare variants (10% shared across all)
        n_shared = int(self.n_rare_variants * 0.1)
        rare_variant_mask[:n_shared] = 1

        # Generate rare variant dosages (mostly 0, occasionally 1 or 2)
        rare_dosages = np.zeros((self.n_samples, self.n_rare_variants))
        for i in range(self.n_rare_variants):
            if rare_variant_mask[i] == 1:
                # Rare variants have low frequency
                freq = np.random.uniform(0.001, self.rare_variant_freq)
                rare_dosages[:, i] = np.random.choice(
                    [0, 1, 2], size=self.n_samples, p=[1 - freq, freq * 0.9, freq * 0.1]
                )

        # Generate phenotype with contributions from both common and rare variants
        genetic_liability = prs_scores * 0.7  # Common variant contribution

        # Rare variant effects (higher effect sizes)
        rare_effects = (
            np.random.normal(0, 0.5, self.n_rare_variants) * rare_variant_mask
        )
        rare_contribution = np.dot(rare_dosages, rare_effects) * 0.3

        # Add environmental noise
        environmental = np.random.normal(0, 0.5, self.n_samples)

        # Continuous phenotype
        phenotype = genetic_liability + rare_contribution + environmental

        # Binary phenotype (for classification tasks)
        phenotype_binary = (phenotype > np.percentile(phenotype, 70)).astype(np.float32)

        # Identify influential rare variants for this population
        influential_variants = set(np.where(rare_variant_mask == 1)[0])

        return {
            "common_genotypes": common_genotypes.astype(np.float32),
            "prs_scores": prs_scores.astype(np.float32),
            "rare_dosages": rare_dosages.astype(np.float32),
            "phenotype_continuous": phenotype.astype(np.float32),
            "phenotype_binary": phenotype_binary,
            "influential_variants": influential_variants,
            "population_id": population_id,
        }

    def create_federated_datasets(self, n_clients: int = 6) -> List[Dict]:
        """
        Create federated datasets with population structure.

        Args:
            n_clients: Number of federated clients

        Returns:
            List of client datasets with genetic heterogeneity
        """
        client_datasets = []
        samples_per_client = self.n_samples // n_clients

        for client_id in range(n_clients):
            # Assign clients to populations (some populations have multiple clients)
            population_id = client_id % self.n_populations

            # Generate population-specific data
            self.n_samples = samples_per_client
            data = self.generate_population_data(population_id)
            data["client_id"] = client_id

            client_datasets.append(data)

        return client_datasets

    def generate_test_set(self, n_samples: int) -> Dict:
        """
        Generate a test set.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Dictionary containing test data.
        """
        self.n_samples = n_samples
        return self.generate_population_data(population_id=self.n_populations)


if __name__ == "__main__":
    feature_matrix = prepare_feature_matrix()
    if feature_matrix is not None:
        partitions = partition_data(feature_matrix, num_partitions=3)
        if partitions:
            print(f"Successfully created {len(partitions)} partitions.")
            for i, p in enumerate(partitions):
                print(f"  Partition {i+1}: {len(p)} samples")
