# Preamble
library(arrow)
library(dplyr)
library(lubridate)
library(tictoc)
.datadir = file.path("../../data/autodealerdata/2025-07-14-output")

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
  
  # Reduce to year_of_sale & year-make-model
  df <- df %>%
    mutate(
      year = year,
      make = tolower(brand_name),
      model = tolower(model_name)
    ) %>%
    select(year, make, model, year_of_sale)
  
  # Get the total sales per year_of_sale for each year-make-model
  current_sales_counts <- df %>%
    group_by(year, make, model, year_of_sale) %>%
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
  group_by(year, make, model, year_of_sale) %>%
  summarise(
    count = sum(count),
    .groups = "drop"
  )
write.csv(final_sales_counts, "/Users/brenden/Desktop/motorVAE/data/autodealerdata-final_sales_counts.csv", row.names = FALSE)

# Convert to average annual sales
average_annual_sales = final_sales_counts %>%
  group_by(year, make, model) %>%
  summarise(
    avg_annual_sales = mean(count),
    .groups = "drop"
  ) %>%
  arrange(year, make, model)
write.csv(average_annual_sales, "/Users/brenden/Desktop/motorVAE/data/autodealerdata-avg_annual_sales.csv", row.names = FALSE)
