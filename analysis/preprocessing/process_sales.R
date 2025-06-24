# Load required libraries
library(dplyr)
library(lubridate)

# Read the CSV file
data <- read.csv("/Users/brenden/Desktop/motorVAE/data/autodealerdata_raw.csv")

# Step 1: Convert last_seen to Sale_Year (extract year only)
data$Sale_Year <- year(as.Date(data$last_seen))

# Step 2: Rename variables and convert Make/Model to lowercase
data <- data %>%
  rename(
    year = year,
    make = brand_name,
    model = model_name
  ) %>%
  mutate(
    make = tolower(make),
    model = tolower(model)
  )

# Step 3: Count sales for every Year, Make, Model group by Sale_Year
sales_counts <- data %>%
  group_by(Sale_Year, year, make, model) %>%
  summarise(
    count = n(),
    .groups = "drop"
  )

# Step 4: Calculate average annual sales across all Sale_Years for each Year, Make, Model
final_dataset <- sales_counts %>%
  group_by(year, make, model) %>%
  summarise(
    avg_annual_sales = mean(count),
    .groups = "drop"
  ) %>%
  arrange(year, make, model)

# Display summary of the final dataset
cat("Final dataset summary:\n")
cat("Dimensions:", nrow(final_dataset), "rows x", ncol(final_dataset), "columns\n")
cat("\nColumn names:", paste(names(final_dataset), collapse = ", "), "\n")
cat("\nFirst few rows:\n")
print(head(final_dataset, 10))

# Save the final dataset
write.csv(final_dataset, "/Users/brenden/Desktop/motorVAE/data/autodealerdata_processed.csv", row.names = FALSE)