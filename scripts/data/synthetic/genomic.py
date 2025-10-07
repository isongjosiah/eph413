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
    print("row is ")
    print(row)
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

    print("sds is -> ", sds)

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
    elif sds <= -2.0:
        return "Short"
    elif sds >= 2.0:
        return "Tall"
    else:
        return "Mid"


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

    try:
        # Save the feature matrix to a CSV file
        output_path = "./data/PSR/prepared_feature_matrix.csv"
        feature_matrix.to_csv(output_path, index=False)
        print(f"Feature matrix saved to {output_path}")
    except Exception as e:
        print(f"Failed to save feature matrix: {e}")

    return feature_matrix


if __name__ == "__main__":
    prepare_feature_matrix()


def partition_data(feature_matrix, num_partitions):
    """
    Partitions the dataset to simulate different allelic heterogeneity situations.

    Args:
        feature_matrix: The input feature matrix.
        num_partitions: The number of partitions to create.

    Returns:
        A list of partitions.
    """
    # TODO: Implement this function.
    pass


prepare_feature_matrix()
