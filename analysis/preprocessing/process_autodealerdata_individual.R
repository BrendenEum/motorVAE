library(arrow)
library(dplyr)
library(lubridate)
library(tictoc)

.datadir = file.path("/Users/brenden/Desktop/motorVAE/data/autodealerdata2/local-folder")
.tmpdir = file.path("/Users/brenden/Desktop/motorVAE/analysis/temp")
cache_file <- file.path(.tmpdir, "autodealerdata2_individual_sales.parquet")
#sample = read_parquet("/Users/brenden/Desktop/motorVAE/data/autodealerdata2/local-folder/part-00001-tid-1555015498128176032-60246200-f451-493b-943d-8bcb26a1fe04-4-1.c000.snappy.parquet")

if (file.exists(cache_file)) {
  individual_sales <- read_parquet(cache_file)
} else {
  
  tic()
  
  # Set Arrow's parallelization
  arrow::set_cpu_count(parallel::detectCores())
  arrow::set_io_thread_count(4)
  
  # Open ALL files as a single dataset - NO collecting yet!
  all_files <- list.files(.datadir, pattern = "\\.snappy\\.parquet$", full.names = TRUE)
  
  # Process using Arrow's compute functions (C++ speed, not R)
  dataset <- open_dataset(all_files) %>%
    filter(
      year >= 2007 & year <= 2025 &
      has_decode == TRUE & brand_decode != "" & model_decode != ""
    ) %>%
    mutate(
      # Date operations work directly
      last_seen_date = as.Date(last_seen),
      first_seen_date = as.Date(first_seen),
      year_sold = year(last_seen_date),
      month_sold = month(last_seen_date),
      year_added = year(first_seen_date),
      month_added = month(first_seen_date)
      #year = year
    ) %>%
    mutate(
      # ## autodealerdata1 ###########################
      # #make_decode = tolower(brand_decode),
      # #model_decode = tolower(model_decode),
      # year = tolower(vin_decode__model_year),
      # make = tolower(vin_decode__make),
      # model = tolower(vin_decode__model),
      # vin_decode_trim = tolower(vin_decode__trim),
      # vin_decode_body_class = tolower(vin_decode__body_class),
      # vin_decode_electrification_level = tolower(vin_decode__electrification_level),
      # vin_decode_engine_cylinders = tolower(vin_decode__engine_cylinders),
      # vin_decode_engine_hp = tolower(vin_decode__engine_hp)
      # #dealer_name = tolower(dealer_name)
      # ##############################################
      
      ## autodealerdata2 ###########################
      make = tolower(brand_decode),
      model = tolower(model_decode),
      trim = tolower(trim_decode),
      dealer_name = tolower(dealer_name)
      ##############################################
    ) %>%
    filter(
      is_new == TRUE & year_sold <= year+2 & year_sold >= year-1 # car is new if is_new AND car is sold within 2 years
    ) %>%
    select(
      year, make, model, 
      #vin_decode_model_year, vin_decode_make, vin_decode_model,
      #vin_decode_trim, vin_decode_body_class, 
      #vin_decode_electrification_level, vin_decode_engine_cylinders, vin_decode_engine_hp,
      soft_ignore, is_new, year_sold, month_sold, year_added, month_added, 
      mileage, msrp, ask_price, color, dealer_id, zip_code
      #state, dealer_name, soft_ignore
    )
  
  # Write the entire dataset to parquet (processes in chunks automatically)
  write_dataset(
    dataset,
    path = file.path(.tmpdir, "autodealerdata2_individual_parquets"),  # Directory, not file
    format = "parquet",
    max_rows_per_file = 5000000
  )
  
  toc()
}
