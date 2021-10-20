library(HDclassif)
library(data.table)  
mydata <- fread("/data/g0017139/GPL570_norm.dat", data.table=FALSE)
#mydata <- fread("/data/g0017139/gene_expression_norm.dat", data.table=FALSE)

prms <- hddc(mydata, K = 5, model = 12, itermax = 200, threshold =  0.01, mc.cores = 1)

i = 0
for (m in prms$Q){ 
  df = data.frame(m)
  path = paste(c('/home/g0017139/UMCG_Thesis/Working_Code/Results/Set2/eigenvec',i, '.csv'), collapse = "")
  write.csv(df, path)
  i = i + 1
}
filepath = "/home/g0017139/UMCG_Thesis/Working_Code/Results/Set2/HDDCClustersEigen.csv"
df <- do.call("rbind", lapply(prms$class, as.data.frame)) 
write.csv(df, file = filepath)  
#png(filename="Results/Set1/HDDCplot.png")
#plot(prms, "Cattell", 0.01)
#dev.off()


