args <- commandArgs(trailingOnly = TRUE)
b = strtoi(args[1])
library(data.table)  
library(HDclassif) 

mydata <- fread("/data/g0017139/TCGA__RSEM_norm.dat", data.table=FALSE)
for(i in b)
{
  filepath = sprintf("/home/g0017139/UMCG_Thesis/Working_Code/Results/Set4/HDDCGrid%i.csv", i)
  print(filepath)
  prms <- hddc(mydata, K = i, model = "ALL", itermax = 200, mc.cores = 2, threshold = c(0.01, 0.1, 0.2))
  df = do.call(rbind, prms$allCriteria)
  write.csv(df, filepath)
  
  filepath = sprintf("/home/g0017139/UMCG_Thesis/Working_Code/Results/Set4/HDDCClusters%i.csv", i)
  df <- do.call("rbind", lapply(prms$class, as.data.frame)) 
  write.csv(df, file = filepath)
}

