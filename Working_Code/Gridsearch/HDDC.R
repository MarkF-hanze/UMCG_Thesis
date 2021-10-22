 
args <- commandArgs(trailingOnly = TRUE)
set = strtoi(args[1])
clusters = strtoi(args[2])
library(data.table)  
library(HDclassif)


if (set == 1) { 
    mydata <- fread("/data/g0017139/gene_expression_norm.dat", data.table=FALSE)
    } else if (set == 2) {
    mydata <- fread("/data/g0017139/GPL570_norm.dat", data.table=FALSE)
    } else if  (set == 3) {
    mydata <- fread("/data/g0017139/mixing_mat.dat", data.table=FALSE)
    } else if  (set == 4){
    mydata <- fread("/data/g0017139/TCGA__RSEM_norm.dat", data.table=FALSE)
}

for(i in clusters)
{
  # Train model
  prms <- hddc(mydata, K = i, model = 1, itermax = 200, mc.cores = 3, threshold = c(0.01, 0.1, 0.2))
  filepath = paste(c("/home/g0017139/UMCG_Thesis/Working_Code/Results/TSet", set,"/HDDCGrid", i, '.csv'), collapse = "")
  # Save the results
  df = do.call(rbind, prms$allCriteria)
  write.csv(df, filepath)
  # Save the made classes
  filepath = paste(c("/home/g0017139/UMCG_Thesis/Working_Code/Results/TSet", set,"/HDDCClusters", i, '.csv'), collapse = "")
  df <- do.call("rbind", lapply(prms$class, as.data.frame)) 
  write.csv(df, file = filepath)
  # Save the eigenvalues
  d = 0
  for (m in prms$Q){ 
    df = data.frame(m)
    filepath = paste(c('/home/g0017139/UMCG_Thesis/Working_Code/Results/TSet', set,'/eigenvec',d,'clusters',i, '.csv'), collapse = "")
    write.csv(df, filepath)
    d = d + 1
  }
  # Save the algorithms
  filepath = paste(c('/data/g0017139/Models/TSet', set,'/modelHDDCclusters',i, '.rds'), collapse = "")
  saveRDS(prms, filepath)
}

