########### QC of Base Data ######################

# Decompress and read the Height.gwas.txt.gz file, prints the header
# line, prints any line with MAF above 0.01, prints any line with INFO
# above 0.8 compresses and write the results to Height.gz
gunzip -c ./data/Height.gwas.txt.gz |
  awk 'NR==1 || ($11 > 0.01) && ($10 > 0.8) {print}' |
  gzip >./data/Height.gz

# Decompress and reads the Height.gz file. Count number of time
# SNP ID was observed, and print if it is the first time.
# compress and write the result to Height.nodup.gz
gunzip -c ./data/Height.gz |
  awk '{seen[$3]++; if(seen[$3]==1){ print}}' |
  gzip >./data/Heightnodup.gz

echo "--------------------------"
echo "retain non ambiguous SNPs"
echo "--------------------------"
# retain non ambiguous SNPs
gunzip -c ./data/Heightnodup.gz |
  awk '!( ($4=="A" && $5=="T") || \
        ($4=="T" && $5=="A") || \
        ($4=="G" && $5=="C") || \
        ($4=="C" && $5=="G") ) {print}' |
  gzip >./data/HeightQC.gz

########### QC of Target Data ####################
./tools/plink/mac/plink \
  --bfile ./data/EUR/EUR \
  --maf 0.01 \
  --hwe 1e-6 \
  --geno 0.01 \
  --mind 0.01 \
  --write-snplist \
  --make-just-fam \
  --out EUR.QC

./tools/plink/mac/plink \
  --bfile ./data/EUR/EUR \
  --keep EUR.QC.fam \
  --extract EUR.QC.snplist \
  --indep-pairwise 200 50 0.25 \
  --out EUR.QC

./tools/plink/mac/plink \
  --bfile ./data/EUR/EUR \
  --extract EUR.QC.prune.in \
  --keep EUR.QC.fam \
  --het \
  --out data.EUR.QC

Rscript remove_ind.R
echo "--------------------------"
echo "doing mismatch snp R check"
echo "--------------------------"
Rscript mismatch_snp.R

./tools/plink/mac/plink \
  --bfile ./data/EUR/EUR \
  --extract EUR.QC.prune.in \
  --keep EUR.valid.sample \
  --check-sex \
  --out EUR.QC

Rscript sexcheck.R

./tools/plink/mac/plink \
  --bfile ./data/EUR/EUR \
  --extract EUR.QC.prune.in \
  --keep EUR.QC.valid \
  --rel-cutoff 0.125 \
  --out EUR.QC

./tools/plink/mac/plink \
  --bfile ./data/EUR/EUR \
  --make-bed \
  --keep EUR.QC.rel.id \
  --out EUR.QC \
  --extract EUR.QC.snplist \
  --exclude EUR.mismatch \
  --a1-allele EUR.a1

Rscript update_effect_size.R

./tools/plink/mac/plink \
  --bfile EUR.QC \
  --clump-p1 1 \
  --clump-r2 0.1 \
  --clump-kb 250 \
  --clump HeightQCTransformed \
  --clump-snp-field SNP \
  --clump-field P \
  --out EUR

awk 'NR!=1{print $3}' EUR.clumped >EUR.valid.snp

awk '{print $3,$8}' HeightQCTransformed >SNP.pvalue

echo "0.001 0 0.001" >range_list
echo "0.05 0 0.05" >>range_list
echo "0.1 0 0.1" >>range_list
echo "0.2 0 0.2" >>range_list
echo "0.3 0 0.3" >>range_list
echo "0.4 0 0.4" >>range_list
echo "0.5 0 0.5" >>range_list

./tools/plink/mac/plink \
  --bfile EUR.QC \
  --score HeightQCTransformed 3 4 12 header \
  --q-score-range range_list SNP.pvalue \
  --extract EUR.valid.snp \
  --out EUR

# First, we need to perform prunning
./tools/plink/mac/plink \
  --bfile EUR.QC \
  --indep-pairwise 200 50 0.25 \
  --out EUR

# Then we calculate the first 6 PCs
./tools/plink/mac/plink \
  --bfile EUR.QC \
  --extract EUR.prune.in \
  --pca 6 \
  --out EUR

## Find best fit PRS and save
## The reported best fit here is 0.3, and we will use that prs score
## for our model
Rscript best_fit.R

### identify rare variants
# Or use EUR.QC if you accept variants MAF were already removed >0.01
./tools/plink/mac/plink \
  --bfile EUR.QC \
  --freq \
  --out EUR_frequencies # Output file prefix

# Skip header (NR>1) and print SNP ID if MAF ($5) is less than 0.01
awk 'NR>1 && $5 < 0.01 {print $2}' EUR_frequencies.frq >rare_variant_IDs.txt

### Extract rare variant allele
./tools/plink/mac/plink \
  --bfile EUR.QC \
  --pheno ./data/EUR/EUR.height \
  --extract rare_variant_IDs.txt \
  --recode A \
  --out rare_alleles
echo "Extracted rare variant dosages to rare_alleles.raw"

echo "-------------------------------------"
echo "Extracting Best PRS Score"
echo "-------------------------------------"

# 4. Determine the best P-value threshold (Read from R script output or saved file)
#    This command assumes best_fit.R printed the best threshold like "Best Threshold: 0.1"
#    Or, if best_fit.R saves the best profile filename: best_threshold_file=$(cat best_threshold_filename.txt)
#    Modify this line based on how best_fit.R indicates the best threshold.
#    Let's *assume* best_fit.R writes the best threshold value (e.g., 0.1) to best_threshold.txt
#    **** You MUST ensure best_fit.R actually creates this file ****
BEST_THRESHOLD=0.3
BEST_PROFILE_FILE="EUR.${BEST_THRESHOLD}.profile"
echo "Using best PRS scores from: ${BEST_PROFILE_FILE}"

# 5. Extract IID and SCORE columns from the best profile file
#    awk selects the 2nd (IID) and 6th (SCORE) columns, skipping the header
awk 'NR>1 {print $2, $6}' "${BEST_PROFILE_FILE}" >best_prs_scores.txt
echo "Extracted best PRS scores to best_prs_scores.txt"

echo "-------------------------------------"
echo "Combining Rare Alleles, Best PRS, and Phenotype (from .raw)"
echo "-------------------------------------"

# 6. Prepare the rare allele dosages & phenotype file header and data
#    The .raw file header is FID IID PAT MAT SEX PHENOTYPE SNP1_A SNP2_C ...
#    We want IID (col 2), PHENOTYPE (col 6), followed by SNP dosages (cols 7 onwards)
#    Get the header line:
head -n 1 rare_alleles.raw | cut -d ' ' -f2,6,7- >rare_pheno_header.txt
#    Get the data lines:
tail -n +2 rare_alleles.raw | cut -d ' ' -f2,6,7- >rare_pheno_data.txt
echo "Extracted rare variant dosages and phenotype to rare_pheno_data.txt"

# 7. Prepare the PRS score file header and data
echo "IID BEST_PRS" >best_prs_scores_header.txt
awk 'NR>1 {print $2, $6}' "${BEST_PROFILE_FILE}" >best_prs_scores_data.txt
echo "Extracted best PRS scores to best_prs_scores.txt"

# 8. Sort all data files by IID (first column)
sort -k1,1 best_prs_scores_data.txt >best_prs_scores_data.sorted.txt
sort -k1,1 rare_pheno_data.txt >rare_pheno_data.sorted.txt # Sort the combined rare/pheno file

# 9. Join the files based on IID
#    Join PRS scores with the rare allele/phenotype data
join best_prs_scores_data.sorted.txt rare_pheno_data.sorted.txt >combined_data.txt

# 10. Combine headers and the joined data
#     Get individual header parts
cut -d ' ' -f2 best_prs_scores_header.txt >temp_prs_header.txt # Just "BEST_PRS"
#     rare_pheno_header.txt already has IID PHENOTYPE SNP1 SNP2...
#     Paste PRS header before the rare/pheno header (which includes IID)
paste temp_prs_header.txt rare_pheno_header.txt >combined_header.txt

#     Combine the final header and the joined data
cat combined_header.txt combined_data.txt >final_combined_PRS_rare_pheno.txt
echo "Final combined data saved to final_combined_PRS_rare_pheno.txt"

# 11. Clean up intermediate files
rm rare_pheno_header.txt rare_pheno_data.txt \
  best_prs_scores_header.txt best_prs_scores_data.txt \
  best_prs_scores_data.sorted.txt rare_pheno_data.sorted.txt \
  combined_data.txt combined_header.txt \
  temp_prs_header.txt # Remove phenotype temp files as they are not needed

echo "-------------------------------------"
echo "Converting final output to CSV format"
echo "-------------------------------------"

awk 'BEGIN{OFS=","} {$1=$1; print}' final_combined_PRS_rare_pheno.txt >final_combined_data.csv

echo "CSV file created: final_combined_data.csv"
echo "-------------------------------------"

# Optional: Remove the intermediate space-separated file
rm final_combined_PRS_rare_pheno.txt

echo "-------------------------------------"
echo "Done Combining Data."
echo "-------------------------------------"
