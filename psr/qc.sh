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
./tools/plink/plink \
  --bfile ./data/EUR/EUR \
  --maf 0.01 \
  --hwe 1e-6 \
  --geno 0.01 \
  --mind 0.01 \
  --write-snplist \
  --make-just-fam \
  --out ./data/EUR.QC

./tools/plink/plink \
  --bfile ./data/EUR/EUR \
  --keep ./data/EUR.QC.fam \
  --extract ./data/EUR.QC.snplist \
  --indep-pairwise 200 50 0.25 \
  --out ./data/EUR.QC

./tools/plink/plink \
  --bfile ./data/EUR/EUR \
  --extract ./data/EUR.QC.prune.in \
  --keep ./data/EUR.QC.fam \
  --het \
  --out ./data.EUR.QC

Rscript mismatch_snp.R

./tools/plink/plink \
  --bfile EUR \
  --extract EUR.QC.prune.in \
  --keep EUR.valid.sample \
  --check-sex \
  --out EUR.QC

Rscript sexcheck.R

plink \
  --bfile EUR \
  --extract EUR.QC.prune.in \
  --keep EUR.QC.valid \
  --rel-cutoff 0.125 \
  --out EUR.QC

plink \
  --bfile EUR \
  --make-bed \
  --keep EUR.QC.rel.id \
  --out EUR.QC \
  --extract EUR.QC.snplist \
  --exclude EUR.mismatch \
  --a1-allele EUR.a1
