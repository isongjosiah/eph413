import pandas as pd
import numpy as np

# Updated import to use the recommended function
from pandas_plink import read_plink


def identify_rare_variants(bed_filepath: str, maf_threshold: float = 0.01):
    """
    Identifies rare variants from a PLINK fileset based on a MAF threshold.

    Args:
        bed_filepath (str): The full path to the .bed file.
                            The .bim and .fam files must be in the same directory.
        maf_threshold (float): The Minor Allele Frequency (MAF) cutoff.

    Returns:
        pandas.DataFrame: A DataFrame containing the rare variants and their info.
    """
    try:
        # --- 1. Load the Genetic Data using the recommended function ---
        print(f"Loading PLINK data from: {bed_filepath}")

        # The function now returns exactly three values.
        (bim, fam, G) = read_plink(bed_filepath, verbose=False)

        print(
            f"Successfully loaded dataset with {G.shape[0]} samples and {G.shape[1]} variants."
        )

    except FileNotFoundError:
        print(
            f"Error: File not found at '{bed_filepath}'. Make sure the .bed, .bim, and .fam files are present."
        )
        return None

    # --- 2. Calculate Allele Frequencies ---
    print("Calculating allele frequencies for all variants...")
    # This calculation remains the same. It computes the frequency of the 'a1' allele.
    allele_freq = G.mean(axis=1) / 2

    # Calculate the Minor Allele Frequency (MAF)
    mafs = np.minimum(allele_freq, 1 - allele_freq)

    bim["maf"] = mafs

    # --- 3. Filter for Rare Variants ---
    print(f"Filtering for variants with MAF < {maf_threshold}...")
    rare_variants_df = bim[bim["maf"] < maf_threshold].copy()

    n_total_variants = len(bim)
    n_rare_variants = len(rare_variants_df)

    print("\n--- Results ---")
    print(f"Total variants found: {n_total_variants}")
    print(
        f"Rare variants identified: {n_rare_variants} ({n_rare_variants / n_total_variants:.2%})"
    )

    if n_rare_variants > 0:
        # --- 4. Save the List of Rare Variants ---
        output_file = "rare_variant_ids.txt"
        rare_variants_df["snp"].to_csv(output_file, index=False, header=False)
        print(f"List of rare variant IDs saved to '{output_file}'")

    return rare_variants_df


if __name__ == "__main__":
    # !!! IMPORTANT: Update this with the full path to your CINECA .bed file !!!
    CINECA_BED_FILE = "./psr/EUR.QC.bed"

    # Define the MAF threshold for what is considered a "rare" variant
    MAF_CUTOFF = 0.01  # 1% frequency

    # Run the analysis
    rare_variants = identify_rare_variants(CINECA_BED_FILE, MAF_CUTOFF)

    if rare_variants is not None and not rare_variants.empty:
        print("\nPreview of identified rare variants:")
        print(rare_variants.head())
