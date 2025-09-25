# Decompress and read the Height.gwas.txt.gz file, prints the header
# line, prints any line with MAF above 0.01, prints any line with INFO
# above 0.8 compresses and write the results to Height.gz
gunzip -c Height.gwas.txt.gz |
  awk 'NR==1 || ($11 > 0.01) && ($10 > 0.8) {print}' |
  gzip >Height.gz
