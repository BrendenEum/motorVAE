# Preamble
library(arrow)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tictoc)
.tmpdir = file.path("/Users/brenden/Desktop/motorVAE/analysis/temp")

load(file.path(.tmpdir, "ADD2_ConsistentDealers.RData"))
data = ADD2_ConsistentDealers

# Monthly sales by year-make-model
final_sales_counts <- data %>%
  group_by(year, make, model, year_sold, month_sold) %>%
  summarise(
    count = n(),
    .groups = "drop"
  )

# Calculate year_of_sale from August of previous year to July of following year.
# E.g. 2016 Toyota Camry was releasted Aug 2015, and we can define a year up to Jul 2016. Yr2 is Aug 2016 to Jul 2017.
final_sales_counts <- final_sales_counts %>%
  mutate(
    sales_cycle = case_when(
      # If sold in Aug-Dec, it's in the model year period
      month_sold >= 8 ~ year_sold + 1,
      # If sold in Jan-Jul, it's in the previous model year period  
      month_sold <= 7 ~ year_sold 
    )
  )

# Get sales by sales_cycle
final_sales_counts = final_sales_counts %>%
  mutate(model = gsub(" ", "", model)) %>%
  group_by(year, make, model, sales_cycle) %>%
  summarise(
    count = sum(count),
    .groups = "drop"
  )

# Subset to useful years
final_sales_counts = final_sales_counts[final_sales_counts$year>=2005 & final_sales_counts$year<=2025,]

# Clean up sales data for labels
final_sales_counts$age = final_sales_counts$sales_cycle - final_sales_counts$year
fs = final_sales_counts[final_sales_counts$age >= 0,]
fs$zlog_sales = scale(log(fs$count)) %>% as.vector()

#######################################################
# Are yearly sales exponential in age of car?
year = c("2007", "2013", "2015", "2017", "2019", "2021")
make = c("acura", "ford", "hyundai", "chevrolet", "honda", "toyota")
model = c("rl", "f-150", "sonata", "bold", "hrv", "camry")
ind = (fs$year==2007 & fs$make=="acura" & fs$model=="rl") |
  (fs$year==2020 & fs$make=="dodge" & fs$model=="charger") |
  (fs$year==2013 & fs$make=="ford" & fs$model=="f-150") |
  (fs$year==2016 & fs$make=="bmw" & fs$model=="335") |
  (fs$year==2019 & fs$make=="hyundai" & fs$model=="sonata") |
  (fs$year==2016 & fs$make=="mercedes-benz" & fs$model=="s-class") |
  (fs$year==2017 & fs$make=="cadillac" & fs$model=="cts") |
  (fs$year==2018 & fs$make=="chevrolet" & fs$model=="bolt") |
  (fs$year==2019 & fs$make=="honda" & fs$model=="hrv") |
  (fs$year==2021 & fs$make=="toyota" & fs$model=="camry")
pdata = fs[ind, ]
pdata$pre_autodealer = pdata$year<2016
#pdata = pdata %>%
#  group_by(age) %>%
#  summarize(zlog_sales=mean(zlog_sales))
pdata$group = interaction(pdata$year, pdata$make, pdata$model)
ggplot(data=pdata, aes(x=age, y=zlog_sales, color=group)) +
  geom_line(aes(linetype=pre_autodealer)) +
  theme_bw()
#######################################################


# Get sales for year 1 (release), year 2, 2024, and 2025
SaleYr1 = fs[fs$age==0,] %>% mutate(SaleYr1 = zlog_sales)
SaleYr2 = fs[fs$age==1,] %>% mutate(SaleYr2 = zlog_sales)
fs = merge(fs, SaleYr1[,c("year", "make", "model", "SaleYr1")], by=c("year", "make", "model"))
fs = merge(fs, SaleYr2[,c("year", "make", "model", "SaleYr2")], by=c("year", "make", "model"))

SalesData_Yr1_2 = fs %>%
  group_by(year, make, model) %>%
  summarise(
    SaleYr1 = first(SaleYr1),
    SaleYr2 = first(SaleYr2),
    .groups = "drop"
  ) %>%
  arrange(year, make, model)
write.csv(SalesData_Yr1_2, "/Users/brenden/Desktop/motorVAE/data/SalesData_Yr1_2.csv", row.names = FALSE)
