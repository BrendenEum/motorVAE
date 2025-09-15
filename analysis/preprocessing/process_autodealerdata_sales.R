# Preamble
library(arrow)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tictoc)
#.datadir = file.path("../../data/autodealerdata/2025-07-14-output")
.datadir = file.path("/Users/brenden/Desktop/motorVAE/data/autodealerdata/2025-07-14-output")

# Placeholder
all_sales_counts = list()

# Loop through all datafiles
for (i in 0:1607) {
  
  # Progress tracking
  if (i %% 10 == 1) {tic()}
  if (i != 0 && i %% 10 == 0) {
    elapsed = toc(quiet=T)
    ms = round(elapsed$toc - elapsed$tic, 3) * 1000
    cat(sprintf("file %d: %s \n", i, elapsed$callback_msg))
  }
  
  # Load file
  file_num <- sprintf("%05d", i)
  file_pattern <- paste0("^part-", file_num, "-.*\\.snappy\\.parquet$")
  matching_files <- list.files(.datadir, pattern = file_pattern, full.names = TRUE)
  parquet_file <- open_dataset(matching_files[1])
  df <- parquet_file %>% collect()
  
  # Convert last day on dealership lot to year of sale
  df$year_of_sale <- year(as.Date(df$last_seen))
  df$month_of_sale <- month(as.Date(df$last_seen))
  
  # Reduce to year_of_sale & year-make-model
  df <- df %>%
    mutate(
      year = year,
      make = tolower(brand_name),
      model = tolower(model_name)
    ) %>%
    select(year, make, model, year_of_sale, month_of_sale)
  
  # Get the total sales per year_of_sale for each year-make-model
  current_sales_counts <- df %>%
    group_by(year, make, model, year_of_sale, month_of_sale) %>%
    summarise(
      count = n(),
      .groups = "drop"
    )
  
  # Store this to a list
  all_sales_counts[[i+1]] = current_sales_counts
  
}

# Combine all results and aggregate
final_sales_counts = bind_rows(all_sales_counts)
final_sales_counts = final_sales_counts %>%
  mutate(model = gsub(" ", "", model)) %>%
  group_by(year, make, model, year_of_sale, month_of_sale) %>%
  summarise(
    count = sum(count),
    .groups = "drop"
  )
final_sales_counts = final_sales_counts %>% rename(year_sold = year_of_sale, month_sold = month_of_sale)

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

write.csv(final_sales_counts, "/Users/brenden/Desktop/motorVAE/data/autodealerdata-final_sales_counts.csv", row.names = FALSE)
final_sales_counts = read.csv("/Users/brenden/Desktop/motorVAE/data/autodealerdata-final_sales_counts.csv")
final_sales_counts = final_sales_counts[final_sales_counts$year>=2005 & final_sales_counts$year<=2025,]

# Clean up sales data for labels
final_sales_counts$age = final_sales_counts$sales_cycle - final_sales_counts$year
fs = final_sales_counts[final_sales_counts$age >= 0,]
fs$zlog_sales = scale(log(fs$count)) %>% as.vector()

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

# Get sales for year 1 (release), year 2, 2024, and 2025
SaleYr1 = fs[fs$year_of_sale==fs$year,] %>% mutate(SaleYr1 = zlog_sales)
SaleYr2 = fs[fs$year_of_sale==(fs$year+1),] %>% mutate(SaleYr2 = zlog_sales)
Sale2024 = fs[fs$year_of_sale==2024,] %>% mutate(Sale2024 = zlog_sales)
Sale2025 = fs[fs$year_of_sale==2025,] %>% mutate(Sale2025 = zlog_sales)
fs = merge(fs, SaleYr1[,c("year", "make", "model", "SaleYr1")], by=c("year", "make", "model"))
fs = merge(fs, SaleYr2[,c("year", "make", "model", "SaleYr2")], by=c("year", "make", "model"))
fs = merge(fs, Sale2024[,c("year", "make", "model", "Sale2024")], by=c("year", "make", "model"))
fs = merge(fs, Sale2025[,c("year", "make", "model", "Sale2025")], by=c("year", "make", "model"))

sales_data_1_2_2024_2025 = fs %>%
  group_by(year, make, model) %>%
  summarise(
    SaleYr1 = first(SaleYr1),
    SaleYr2 = first(SaleYr2),
    Sale2024 = first(Sale2024),
    Sale2025 = first(Sale2025),
    .groups = "drop"
  ) %>%
  arrange(year, make, model)
write.csv(sales_data_1_2_2024_2025, "/Users/brenden/Desktop/motorVAE/data/sales_data_1_2_2024_2025.csv", row.names = FALSE)
