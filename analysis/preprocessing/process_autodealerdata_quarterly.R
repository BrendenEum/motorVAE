# Preamble
library(arrow)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tictoc)
.datadir = file.path("/Users/brenden/Desktop/motorVAE/data/autodealerdata/2025-07-14-output")
.tmpdir = file.path("/Users/brenden/Desktop/motorVAE/analysis/temp")

individual_sales <- open_dataset(file.path(.tmpdir, "autodealerdata_individual_parquets"))

individual_sales <- 