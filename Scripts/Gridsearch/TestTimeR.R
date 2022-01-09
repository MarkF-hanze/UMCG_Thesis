library(HDclassif)
library(data.table)
# Script to see how long the hddc will take on the GP570 dataset
# Load the data
mydata <- fread("/data/g0017139/GPL570_norm.dat", data.table=FALSE)
format(Sys.time(), "%S")
# Loop over 215, 2159 and 4318 items
y = c()
x = c(215, 2159, 4318)
for(i in x)
{
  # Sample a portion of the data
  new_df = mydata[sample(nrow(mydata), i), ]
  # Train model and time it
  start.time <- Sys.time()
  prms <- hddc(new_df, K = 15, model = "ALL", itermax = 200, mc.cores = 1, threshold = c(0.01, 0.1, 0.2))
  time.taken <- difftime(Sys.time(), start.time, units = "secs")[[1]]
  y = append(y, time.taken)
  print(time.taken)  
  
}
# Plot the results and print results
print(x)
print(y)
png(filename="/home/g0017139/UMCG_Thesis/Scripts/IMAGES/TimeHDDC.png")
plot(x, y, type = "b")
dev.off()