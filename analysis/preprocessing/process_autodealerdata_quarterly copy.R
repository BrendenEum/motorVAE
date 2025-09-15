# Preamble
library(arrow)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tictoc)
.datadir = file.path("/Users/brenden/Desktop/motorVAE/data/autodealerdata2/local-folder")
.tmpdir = file.path("/Users/brenden/Desktop/motorVAE/analysis/temp")

# Define cache file path
cache_file <- file.path(.tmpdir, "autodealerdata2_individual_sales.RData")

# Only run the for loop if the file doesnt already exist in temp. It takes 1 hr to run.
if (file.exists(cache_file)) {
  individual_sales <- load(cache_file)
} else {

  # Placeholder
  autodealerdata_individual_sales_list = list()
  
  # Loop through all datafiles
  for (i in 0:378) {
    
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
    df$year_sold <- year(as.Date(df$last_seen))
    df$month_sold <- month(as.Date(df$last_seen))
    df$year_added <- year(as.Date(df$first_seen))
    df$month_added <- month(as.Date(df$first_seen))
    
    # Reduce to year_of_sale & year-make-model
    df <- df %>%
      mutate(
        year = year,
        make_decode = tolower(brand_decode) %>% gsub(" ", "", .),
        model_decode = tolower(model_decode) %>% gsub(" ", "", .),
        vin_decode_make = tolower(vin_decode__make) %>% gsub(" ", "", .),
        vin_decode_model = tolower(vin_decode__model) %>% gsub(" ", "", .),
        vin_decode_model_year = tolower(vin_decode__model_year)
      ) %>%
      select(
        year, make_decode, model_decode, 
        vin_decode_model_year, vin_decode_make, vin_decode_model,
        is_new, year_sold, month_sold, year_added, month_added, 
        mileage, msrp, ask_price, color, zip_code, state, soft_ignore
      )
    
    # Store this to a list
    autodealerdata_individual_sales_list[[i+1]] = df
    
  }
  # Save all_sales_counts to .tmpdir
  individual_sales = bind_rows(autodealerdata_individual_sales_list)
  save(individual_sales, file=cache_file)
}

# Combine all results and aggregate to monthly sales
monthly_sales = indiv_sales %>%
  mutate(model = gsub(" ", "", model)) %>%
  group_by(year, make, model, year_of_sale, month_of_sale) %>%
  summarize(
    count = sum(count),
    .groups = "drop"
  )
monthly_sales = monthly_sales %>% rename(year_sold = year_of_sale, month_sold = month_of_sale)

# Calculate year_of_sale from August of previous year to July of following year.
# E.g. 2016 Toyota Camry was releasted Aug 2015, and we can define a year up to Jul 2016. Yr2 is Aug 2016 to Jul 2017.
monthly_sales <- monthly_sales %>%
  mutate(
    sales_cycle = case_when(
      # If sold in Aug-Dec, it's in the model year period
      month_sold >= 8 ~ year_sold + 1,
      # If sold in Jan-Jul, it's in the previous model year period  
      month_sold <= 7 ~ year_sold 
    )
  )

# Add sales_cycle to monthly sales (2023 model releases in summer 2022 and sells until summer 2023)
cycle_sales = monthly_sales %>%
  mutate(model = gsub(" ", "", model)) %>%
  group_by(year, make, model, sales_cycle) %>%
  summarise(
    count = sum(count),
    .groups = "drop"
  )

# Aggregate from monthly to quarterly
monthly_sales$quarter_sold = (monthly_sales$month_sold/3) %>% ceiling()
quarterly_sales = monthly_sales %>%
  group_by(year, make, model, year_sold, quarter_sold) %>%
  summarize(
    count = sum(count),
    .groups = "drop"
  )

# Aggregate from monthly to yearly
yearly_sales = monthly_sales %>%
  group_by(year, make, model, year_sold) %>%
  summarize(
    count = sum(count),
    .groups = "drop"
  )

# Keep cars manufactured between 2005 and 2025
indiv_sales = indiv_sales[indiv_sales$year>=2005 & indiv_sales$year<=2025,]
monthly_sales = monthly_sales[monthly_sales$year>=2005 & monthly_sales$year<=2025,]
quarterly_sales = quarterly_sales[quarterly_sales$year>=2005 & quarterly_sales$year<=2025,]
yearly_sales = yearly_sales[yearly_sales$year>=2005 & yearly_sales$year<=2025,]

# Clean up sales data for labels
quarterly_sales$zlog_sales = scale(log(quarterly_sales$count)) %>% as.vector()
yearly_sales$zlog_sales = scale(log(yearly_sales$count)) %>% as.vector()

# Save data
write.csv(quarterly_sales, "/Users/brenden/Desktop/motorVAE/data/quarterly_sales_2005_2025.csv", row.names = FALSE)

# Get sales for year 1 (release), year 2, 2024, and 2025
SaleYr1 = yearly_sales[yearly_sales$year_sold==yearly_sales$year,] %>% mutate(SaleYr1 = zlog_sales)
SaleYr2 = yearly_sales[yearly_sales$year_sold==(yearly_sales$year+1),] %>% mutate(SaleYr2 = zlog_sales)
Sale2024 = yearly_sales[yearly_sales$year_sold==2024,] %>% mutate(Sale2024 = zlog_sales)
Sale2025 = yearly_sales[yearly_sales$year_sold==2025,] %>% mutate(Sale2025 = zlog_sales)
yearly_sales = merge(yearly_sales, SaleYr1[,c("year", "make", "model", "SaleYr1")], by=c("year", "make", "model"))
yearly_sales = merge(yearly_sales, SaleYr2[,c("year", "make", "model", "SaleYr2")], by=c("year", "make", "model"))
yearly_sales = merge(yearly_sales, Sale2024[,c("year", "make", "model", "Sale2024")], by=c("year", "make", "model"))
yearly_sales = merge(yearly_sales, Sale2025[,c("year", "make", "model", "Sale2025")], by=c("year", "make", "model"))

sales_data_1_2_2024_2025 = yearly_sales %>%
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







# Are yearly sales exponential in age of car?
fs = yearly_sales
fs$age = fs$sales_cycle - fs$year
fs = fs[fs$age >= 0,]
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
pdata$group = interaction(pdata$year, pdata$make, pdata$model)
ggplot(data=pdata, aes(x=age, y=zlog_sales, color=group)) +
  geom_line(aes(linetype=pre_autodealer)) +
  theme_bw()
