# Evox image files are named with the following format:
# Year_Make_Model_Trim_Body_Doors_Color.png
# Here, we're using a loop to parse through the information in the filenames and storing this to a csv file. We will use the csv file to feed supervised labels into our supervised neural network.

### (! ! !) I went back and manually changed 8 names that broke this code. 2022 Lincoln Navigator has trim = "NRL_Trim" which breaks the parser. Same for 2022 Volkswagen Golf. I manually changed "NRL_Trim" to "NRL" to fix this.

# Author: Brenden Eum (2025)


###########################
# Preamble
###########################

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
    Filename = character(),
    Year = character(),
    Brand = character(),
    Model = character(),
    Trim = character(),
    Body = character(),
    Door = character(),
    Color = character(),
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
      brand <- parts[2]
      model <- parts[3]
      trim <- parts[4]
      body <- parts[5]
      door <- parts[6]
      color <- parts[7]
      
      # Add to dataframe
      car_data <- rbind(car_data, data.frame(
        Filename = filename,
        Year = year,
        Brand = brand,
        Model = model,
        Trim = trim,
        Body = body,
        Door = door,
        Color = color,
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

# Format variables
cd$Door <- gsub("\\D", "", cd$Door) %>% as.numeric()
cd$Year <- cd$Year %>% as.numeric()
cd$Color <- cd$Color %>% as.numeric()
cd$Trim <- cd$Trim %>% tolower() %>% factor()
cd$Body <- cd$Body %>% tolower() %>% factor()
cd$Brand <- cd$Brand %>% tolower()

# Only keep major brands, group others. Define this as 500 vehicles or more in the dataset.
majors <- c("acura", "audi", "bmw", "buick", "cadillac", "chevrolet", "dodge", "ford", "gmc", "honda", "hyundai", "infiniti", "jaguar", "jeep", "kia", "landrover", "lexus", "lincoln", "mazda", "mercedes-benz", "mitsubishi", "nissan", "porsche", "subaru", "toyota", "volkswagen", "volvo")
cd$Brand = ifelse(cd$Brand %in% majors, cd$Brand, "other") %>% factor()

# Fix Doors. 
# Example issues: (1) 3-doors will look like 2-door. (2) 5-doors typically mean 4-door and hatchback. 
# Thus: (1) Doors provides non-visual information that will trick network. (2) Doors and Body have some overlap.
# Solution: (1) 3 or less doors will be grouped as 2-door. (2) 4 or more doors will be grouped as 4-door. 
# Discussion: This solution deals with the the # of doors on the driver side. Still has some overlap with Body.
cd$Door <- ifelse(cd$Door <= 3, 2, 4) %>% factor()


###########################
# Export
###########################

# Write to CSV
write.csv(cd, "/Users/brenden/Desktop/motorVAE/data/labels_evox_256x256_1-4.csv", row.names = FALSE)

# Write unexpected files to CSV if specified
if (total_errors>0) {
  write.csv(unexpected_files, "/Users/brenden/Desktop/motorVAE/data/unexpected_filenames.csv", row.names = FALSE)
}
