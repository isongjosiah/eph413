"""
This script generates synthetic genomic data for use in federated learning 
simulations. The data is designed to mimic gene expression data, with two 
groups of samples (e.g., case and control) and a set of differentially 
expressed genes. This allows for the testing of federated learning algorithms 
on tasks such as disease prediction from gene expression data.

The generator can be customized to control the number of genes, samples, and 
the extent of differential expression. The output is saved in a standard CSV 
format, making it easy to integrate with other tools and frameworks.
"""

import numpy as np
import pandas as pd
import os

def generate_genomic_data(num_genes, num_samples, fraction_diff_expressed=0.1):
    """
    Generates a synthetic gene expression dataset.

    This function creates a dataset with two groups of samples (case and control)
    and a specified number of differentially expressed genes. The gene expression
    values are drawn from a normal distribution, with a mean shift introduced
    for the differentially expressed genes in the case group.

    Args:
        num_genes (int): The total number of genes in the dataset.
        num_samples (int): The total number of samples in the dataset.
        fraction_diff_expressed (float): The fraction of genes that are
            differentially expressed between the two groups.

    Returns:
        pandas.DataFrame: A DataFrame containing the gene expression data,
            with genes as rows and samples as columns.
        list: A list of labels for the samples (0 for control, 1 for case).
    """
    # Create two groups of samples
    num_case_samples = num_samples // 2
    num_control_samples = num_samples - num_case_samples

    # Generate a baseline of gene expression values from a normal distribution
    expression_data = np.random.normal(loc=0, scale=1, size=(num_genes, num_samples))

    # Determine the number of differentially expressed genes
    num_diff_expressed = int(num_genes * fraction_diff_expressed)

    # Select the genes to be differentially expressed
    diff_expressed_indices = np.random.choice(
        num_genes, num_diff_expressed, replace=False
    )

    # Introduce a mean shift for the differentially expressed genes in the case group
    expression_data[diff_expressed_indices, :num_case_samples] += np.random.normal(
        loc=2, scale=0.5, size=(num_diff_expressed, num_case_samples)
    )

    # Create labels for the samples
    labels = [1] * num_case_samples + [0] * num_control_samples

    # Create a pandas DataFrame for the expression data
    gene_names = [f"gene_{i}" for i in range(num_genes)]
    sample_names = [f"sample_{i}" for i in range(num_samples)]
    expression_df = pd.DataFrame(
        expression_data, index=gene_names, columns=sample_names
    )

    return expression_df, labels

def save_data(expression_df, labels, output_dir="."):
    """
    Saves the generated gene expression data and labels to CSV files.

    Args:
        expression_df (pandas.DataFrame): The gene expression data.
        labels (list): The sample labels.
        output_dir (str): The directory where the files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the expression data
    expression_df.to_csv(os.path.join(output_dir, "gene_expression.csv"))

    # Save the labels
    labels_df = pd.DataFrame({"label": labels})
    labels_df.to_csv(os.path.join(output_dir, "sample_labels.csv"), index=False)

if __name__ == "__main__":
    # --- Example of how to use the genomic data generator ---
    
    # Set the parameters for the synthetic data
    num_genes = 1000
    num_samples = 100
    fraction_diff_expressed = 0.05
    
    # Generate the data
    print("Generating synthetic genomic data...")
    expression_data, labels = generate_genomic_data(
        num_genes, num_samples, fraction_diff_expressed
    )
    
    # Save the data to a directory
    output_directory = "synthetic_genomic_data"
    save_data(expression_data, labels, output_directory)
    
    print(f"Generated data for {num_samples} samples and {num_genes} genes.")
    print(f"Data saved to the '{output_directory}' directory.")
    print(f"Number of case samples: {sum(labels)}")
    print(f"Number of control samples: {len(labels) - sum(labels)}")
