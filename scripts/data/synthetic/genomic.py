"""
This script prepares the input feature matrix for the models based on the data prepared in the psr module.
It also helps with partitioning the datasets to simulate different allelic heterogeneity situations.
"""

import pandas as pd
import numpy as np


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
    variant_frequencies = (feature_matrix.drop(columns=['Height_Category'], errors='ignore') != 0).mean()
    rare_variants = variant_frequencies[variant_frequencies < rare_variant_threshold].index.tolist()

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
                remaining_samples = feature_matrix[~partition_samples_mask].sample(n=remaining_samples_count, replace=True)
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
    tall_group = feature_matrix[feature_matrix['Height_Category'] == 'Tall']
    short_group = feature_matrix[feature_matrix['Height_Category'] == 'Short']

    proportions = []
    for i in range(num_partitions):
        tall_prop = (i + 1) / (num_partitions + 1)
        short_prop = 1 - tall_prop
        proportions.append({'tall': tall_prop, 'short': short_prop})

    for i in range(num_partitions):
        tall_sample = tall_group.sample(frac=proportions[i]['tall'])
        short_sample = short_group.sample(frac=proportions[i]['short'])
        partition = pd.concat([tall_sample, short_sample])
        partitions.append(partition)

    return partitions


if __name__ == "__main__":
    feature_matrix = prepare_feature_matrix()
    if feature_matrix is not None:
        partitions = partition_data(feature_matrix, num_partitions=3)
        if partitions:
            print(f"Successfully created {len(partitions)} partitions.")
            for i, p in enumerate(partitions):
                print(f"  Partition {i+1}: {len(p)} samples")
