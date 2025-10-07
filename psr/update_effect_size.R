dat <- read.table(gzfile("./data/HeightQC.gz"), header=T)
dat$BETA <- log(dat$OR)
write.table(dat, "HeightQCTransformed", quote=F, row.names=F)
q() # exit R
