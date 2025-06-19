# Evox image files are named with the following format:
# Year_Make_Model_Trim_Body_Doors_Color.png
# Here, we're using a loop to parse through the information in the filenames and storing this to a csv file. We will use the csv file to feed supervised labels into our supervised neural network.

### (! ! !) I went back and manually changed 8 names that broke this code. 2022 Lincoln Navigator has trim = "NRL_Trim" which breaks the parser. Same for 2022 Volkswagen Golf. I manually changed "NRL_Trim" to "NRL" to fix this.

# Author: Brenden Eum (2025)


###########################
# Preamble
###########################
set.seed(4)

# Load required libraries
library(tidyverse)

# Count unexpected filenames
total_errors = 0


###########################
# Process filenames
###########################

# Function to process the car image filenames
process_car_filenames <- function(folder_path) {
  # Get list of all files in the folder
  all_files <- list.files(folder_path, full.names = FALSE)
  
  # Initialize empty dataframe to store results
  car_data <- data.frame(
    filename = character(),
    year = character(),
    make = character(),
    model = character(),
    trim = character(),
    body = character(),
    door = character(),
    color = character(),
    stringsAsFactors = FALSE
  )
  
  # Initialize empty dataframe to store unexpected filenames
  unexpected_files <- data.frame(
    Filename = character(),
    Reason = character(),
    stringsAsFactors = FALSE
  )
  
  # Process each filename
  for (filename in all_files) {
    # Remove file extension
    name_without_ext <- tools::file_path_sans_ext(filename)
    
    # Split by underscores
    parts <- strsplit(name_without_ext, "_")[[1]]
    
    # Check if we have enough parts (at least 7 for all fields)
    if (length(parts) >= 7) {
      # Extract components
      year <- parts[1]
      make <- parts[2]
      model <- parts[3]
      trim <- parts[4]
      body <- parts[5]
      door <- parts[6]
      color <- parts[7]
      
      # Add to dataframe
      car_data <- rbind(car_data, data.frame(
        filename = filename,
        year = year,
        make = make,
        model = model,
        trim = trim,
        body = body,
        door = door,
        color = color,
        stringsAsFactors = FALSE
      ))
    } else {
      # Handle case where filename doesn't have expected format
      warning(paste("Skipping file with unexpected format:", filename))
      
      # Add to unexpected files dataframe
      unexpected_files <- rbind(unexpected_files, data.frame(
        Filename = filename,
        Reason = paste("Expected at least 7 parts, found", length(parts)),
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Return both dataframes as a list
  return(list(
    car_data = car_data,
    unexpected_files = unexpected_files
  ))
}

# Run function
results <- process_car_filenames("/Users/brenden/Desktop/motorVAE/data/evox_256x256_1-4/")
car_data <- results$car_data
unexpected_files <- results$unexpected_files


###########################
# Additional cleaning
###########################
cd = car_data

# Fix Doors. 
# Example issues: (1) 3-doors will look like 2-door. (2) 5-doors typically mean 4-door and hatchback. 
# Thus: (1) Doors provides non-visual information that will trick network. (2) Doors and Body have some overlap.
# Solution: (1) 3 or less doors will be grouped as 2-door. (2) 4 or more doors will be grouped as 4-door. 
# Discussion: This solution deals with the the # of doors on the driver side. Still has some overlap with Body.
cd$door <- gsub("\\D", "", cd$door) %>% as.numeric()
cd$door <- ifelse(cd$door <= 3, 2, 4) %>% factor()

# Format variables
cd$year <- as.numeric(cd$year)
cd$color <- cd$color %>% as.numeric() - 1
cd$trim <- cd$trim %>% tolower()
cd$body <- cd$body %>% tolower()
cd$model <- cd$model %>% tolower()
cd$make <- cd$make %>% tolower() 



###########################
# Merge in average annual sales data
###########################

#source("/Users/brenden/Desktop/motorVAE/analysis/preprocessing/process_sales.R")
sales_data = read.csv("/Users/brenden/Desktop/motorVAE/data/autodealerdata_processed.csv")
cd = cd %>% left_join(sales_data, by=c("year","make","model"))
cd <- cd %>%
  mutate(sales = cut(avg_annual_sales, 
                         breaks = 3, 
                         labels = c("Low", "Medium", "High")))


###########################
# Convert small brands to Other. Define this as 500 vehicles or more in the dataset.
###########################

majors <- c("acura", "audi", "bmw", "buick", "cadillac", "chevrolet", "dodge", "ford", "gmc", "honda", "hyundai", "infiniti", "jaguar", "jeep", "kia", "landrover", "lexus", "lincoln", "mazda", "mercedes-benz", "mitsubishi", "nissan", "porsche", "subaru", "toyota", "volkswagen", "volvo")

# Factor variables
cd$year <- cd$year %>% factor() 
cd$trim <- cd$trim %>% tolower() %>% factor() 
cd$body <- cd$body %>% tolower() %>% factor() 
cd$model <- cd$model %>% tolower() %>% factor()
cd$make = ifelse(cd$make %in% majors, cd$make, "other") %>% factor()
cd$sales = cd$sales %>% factor()

# Function to get the factor mapping
factor_mapping <- function(factor_variable, fn) {
  mapping <- data.frame(
    string = levels(factor_variable),
    factor_number = 0:(length(levels(factor_variable))-1)
  ) 
  write.csv(mapping, file = fn, row.names = FALSE)
}

# Save the factor mappings
factor_mapping(cd$year, "/Users/brenden/Desktop/motorVAE/data/labels_year_mapping.csv")
factor_mapping(cd$body, "/Users/brenden/Desktop/motorVAE/data/labels_body_mapping.csv")
factor_mapping(cd$make, "/Users/brenden/Desktop/motorVAE/data/labels_make_mapping.csv")
factor_mapping(cd$door, "/Users/brenden/Desktop/motorVAE/data/labels_door_mapping.csv")
factor_mapping(cd$sales, "/Users/brenden/Desktop/motorVAE/data/labels_sales_mapping.csv")

# Fix the factor numbers in the dataframe
for (var in c("year","body","make","door", "model", "trim", "sales")) {
  cd[,var] <- cd[,var] %>% as.numeric() - 1
}


###########################
# Export
###########################

# Write to CSV
write.csv(cd, "/Users/brenden/Desktop/motorVAE/data/labels_evox_256x256_1-4.csv", row.names = FALSE)

# Write unexpected files to CSV if specified
if (total_errors>0) {
  write.csv(unexpected_files, "/Users/brenden/Desktop/motorVAE/data/unexpected_filenames.csv", row.names = FALSE)
}


###########################
# Save unique Year-Make-Model
###########################

# Only save 
#data <- cd[cd$make!="other", c("year", "make", "model")]
#data <- unique(data)
#write.csv(data, "/Users/brenden/Desktop/motorVAE/data/year-make-model.csv", row.names = FALSE)
