library(tidyverse)

# Sales data
final_sales_counts = read.csv("/Users/brenden/Desktop/motorVAE/data/autodealerdata-final_sales_counts.csv")
final_sales_counts = final_sales_counts[final_sales_counts$year>=2005 & final_sales_counts$year<=2025,]
# Merge with VIF
raw_vif_map = read.csv("/Users/brenden/Desktop/motorVAE/analysis/evox_api/viflist-all_cars.csv")
vif_map = raw_vif_map %>%
  reframe(
    VIF = VIF..,
    year = Yr,
    make = str_to_lower(Make) %>% str_replace_all(" ", ""),
    model = str_to_lower(Model) %>% str_replace_all(" ", ""),
    trim = str_to_lower(Trim) %>% str_replace_all(" ", ""),
    body = str_to_lower(Body) %>% str_replace_all(" ", ""),
    door = paste0(Drs, "door")
  )
vif_map$trim <- ifelse(vif_map$trim=="", "nan", vif_map$trim)

fs_vif = final_sales_counts %>%
  left_join(
    vif_map, by=c("year", "make", "model"),
    relationship = "many-to-many"
  ) %>%
  distinct(VIF, sales_cycle, count, .keep_all=T) %>%
  na.omit()

car2 = read.csv("/Users/brenden/Desktop/motorVAE/data/stanford-nacc/Car2_agg_df_by_subject_run2.csv")
car2 = car2 %>%
  separate(
    car_folder, 
    into = c("VIF"),
    sep = "_",
    remove = FALSE
  )
car2$VIF = as.integer(car2$VIF)

voi = c("subject", "VIF", "nacc") #nacc = nacc8mm_tr4
data = car2[,voi] %>%
  left_join(fs_vif, by = "VIF", relationship="many-to-many") %>%
  na.omit()

data = data[data$year==(data$sales_cycle) | data$year==(data$sales_cycle-1),]
rdata = data %>%
  group_by(subject, VIF, nacc) %>%
  summarize(
    delta_sales = last(count) - first(count),
    .groups = "drop"
  )
#rdata$delta_sales = rdata$delta_sales %>% scale() %>% as.vector()

lm(delta_sales ~ nacc, data=rdata) %>% summary()
