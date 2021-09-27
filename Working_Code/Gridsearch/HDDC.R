setwd("/home/g0017139/UMCG_Thesis/Working_Code/")
#install.packages("MASS",  "/data/g0017139/.envs/r_env/lib/R/library/", repos = "http://cran.us.r-project.org")
# install.packages("HDclassif", repos = "http://cran.us.r-project.org")
# getwd() 

library(HDclassif)
mydata = read.csv("/home/g0017139/UMCG_Thesis/Working_Code/Results/gene_expression_norm.dat", sep = ' ', header = FALSE, )
prms <- hddc(mydata, K = 2:15, model = "ALL", itermax = 400, mc.cores = 12, threshold = c(0.01, 0.1, 0.2))

df = do.call(rbind, prms$allCriteria)
write.csv(df, "Results/HDDCGrid.csv")
df <- do.call("rbind", lapply(prms$class, as.data.frame)) 
write.csv(df, file = "Results/HDDCClusters.csv")

