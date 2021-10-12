
library(HDclassif)
library(data.table)
mydata <- fread("/data/g0017139/GPL570_norm.dat", data.table=FALSE)
format(Sys.time(), "%S")
print(mydata)
y = c()
x = c(215, 2159, 4318)
#x = c(10, 215, 400)
for(i in x)
{
  print(i)
  new_df = mydata[sample(nrow(mydata), i), ]
  start.time <- Sys.time()
  prms <- hddc(new_df, K = 15, model = "ALL", itermax = 200, mc.cores = 1, threshold = c(0.01, 0.1, 0.2))
  time.taken <- difftime(Sys.time(), start.time, units = "secs")[[1]]
  y = append(y, time.taken)
  print(time.taken)  
  
}
print(x)
print(y)
png(filename="/home/g0017139/UMCG_Thesis/Working_Code/IMAGES/TimeHDDC.png")
plot(x, y, type = "b")
dev.off()