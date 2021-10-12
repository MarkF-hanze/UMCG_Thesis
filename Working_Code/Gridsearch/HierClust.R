#mydata = read.csv("/data/g0017139/gene_expression_norm.dat", sep = ' ', header = FALSE, )
#mydata = read.csv("/data/g0017139/GPL570_norm.dat", sep = ' ', header = FALSE, )
#mydata = read.csv("/data/g0017139/mixing_mat.dat", sep = ' ', header = FALSE, ) 
#mydata = read.csv("/data/g0017139/TCGA__RSEM_norm.dat", sep = ' ', header = FALSE, ) 
library(data.table)
paths = c("/data/g0017139/gene_expression_norm.dat", "/data/g0017139/GPL570_norm.dat", "/data/g0017139/mixing_mat.dat", "/data/g0017139/TCGA__RSEM_norm.dat")
for (j in 2:4){
  mydata <- fread(paths[j], data.table=FALSE)
  print('START!')
  rows.cor <- cor(t(mydata), method = "pearson")
  hc <- hclust(as.dist(1-rows.cor), method = 'ward.D2')
  for(i in 2:15)
  {
    print(i)
    groups <- cutree(hc, k = i)
    df = data.frame(groups)
    filepath = paste(c("/home/g0017139/UMCG_Thesis/Working_Code/Results/Set", j, "/HierarchClusters", i, '.csv'), collapse = "")
    write.csv(df, file = filepath)
    
  }
  print(j)
}