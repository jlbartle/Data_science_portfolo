require(tidyverse)
require(data.table)

setwd("J:/My Drive/Graduate School/IST718 Big Data Analysis/Final Project/flattenedOutput")
files <- list.files()
files<- files[grep(".csv", files)]

data <- lapply(files, fread, fill = T)
data2 <- rbindlist(data, fill = T)
fwrite(data2, file = "J:/My Drive/Graduate School/IST718 Big Data Analysis/Final Project/consolidatedData.csv")
