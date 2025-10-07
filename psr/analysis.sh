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
  gzip - >./data/Height.nodup.gz

# retain non ambiguous SNPs
awk '!( ($4=="A" && $5=="T") || \
    gunzip -c ./data/Height.nodup.gz |
        ($4=="T" && $5=="A") || \
        ($4=="G" && $5=="C") || \
        ($4=="C" && $5=="G")) {print}' |
  gzip >./data/Height.QC.gz

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
echo "doing mismatch snp R check"
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
Rscript best_fit.R

### Feature dataset for neural network ####
./tools/plink/mac/plink \
  --bfile EUR.QC \
  --pheno ./data/EUR/EUR.height \
  --extract EUR.valid.snp \
  --recode A \
  --out extracted_alleles
