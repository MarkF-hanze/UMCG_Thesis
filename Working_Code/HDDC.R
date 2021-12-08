library(HDclassif)
library(data.table)  
mydata <- fread("/data/g0017139/ICA_files/MathExperiment/2_Split/One_Normalized/Math_ExpAll.csv", data.table=FALSE, drop='Unnamed: 0', header=TRUE)
clusters = c(2,3,4)
print(dim(mydata))
for(i in clusters){
  prms <- hddc(mydata, K = i, model = 12, itermax = 200, threshold =  0.01, mc.cores = 1)
  # Save the made classes
  filepath = paste(c("/data/g0017139/ICA_files/Clustering_Math/",i,"_Split/HDDCClusters.csv"), collapse = "")
  df <- do.call("rbind", lapply(prms$class, as.data.frame)) 
  write.csv(df, file = filepath) 
  print("Done")
}
