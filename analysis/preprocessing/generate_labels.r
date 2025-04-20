# Evox image files are named with the following format:
# Year_Make_Model_Trim_Body_Doors_Color.png
# Here, we're using a loop to parse through the information in the filenames and storing this to a csv file. We will use the csv file to feed supervised labels into our supervised neural network.


### (! ! !) I went back and manually changed 8 names that broke this code. 2022 Lincoln Navigator has trim = "NRL_Trim" which breaks the parser. Same for 2022 Volkswagen Golf. I manually changed "NRL_Trim" to "NRL" to fix this.

# Load required libraries
library(tidyverse)

# Count unexpected filenames
total_errors = 0



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



# Additional cleaning
car_data$Door <- gsub("\\D", "", car_data$Door) %>% as.numeric()
car_data$Year <- car_data$Year %>% as.numeric()
car_data$Color <- car_data$Color %>% as.numeric()



# Write to CSV
write.csv(car_data, "/Users/brenden/Desktop/motorVAE/data/labels_evox_256x256_1-4.csv", row.names = FALSE)

# Write unexpected files to CSV if specified
if (total_errors>0) {
  write.csv(unexpected_files, "/Users/brenden/Desktop/motorVAE/data/unexpected_filenames.csv", row.names = FALSE)
}
