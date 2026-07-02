
# PACKAGES
library(tidyverse)
library(e1071)


# LOAD DATA
features <- read.csv("./Desktop/MEASUREMENT.csv", sep = ";")
labels   <- read.csv("./Desktop/LABEL.csv", sep = ";")

df <- features %>%
  left_join(labels, by = c("scan_id" = "Skanning.ID"))



# FIX FEATURE NAMES (NPCA → CE)
df <- df %>%
  rename(
    "V_ossicle_found..CE."      = "V_ossicle_found..NPCA.",
    "Volume_ossicle_found..CE." = "Volume_ossicle_found..NPCA."
  )


# OUTCOME
df <- df %>%
  mutate(
    True = as.numeric(True),
    outcome = case_when(
      True %in% c(1,4) ~ 1,
      True %in% c(0,98) ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(outcome))

df_check <- features %>%
  left_join(labels, by = c("scan_id" = "Skanning.ID")) %>%
  mutate(
    True = as.numeric(True),
    outcome = case_when(
      True %in% c(1,4) ~ 1,
      True %in% c(0,98) ~ 0,
      TRUE ~ NA_real_
    )
  )



df_check %>%
  select(scan_id, True, outcome)



# SELECT NUMERIC FEATURES
feature_list <- names(df)[sapply(df, is.numeric)] %>%
  setdiff(c("scan_id", "PatientID", "True", "outcome"))


# SKEWNESS FUNCTION
get_skew <- function(data, features) {
  map_dfr(features, function(f) {
    tibble(
      feature = f,
      skewness = skewness(data[[f]], na.rm = TRUE)
    )
  }) %>%
    arrange(desc(abs(skewness)))
}


# INITIAL SKEWNESS
skew_before <- get_skew(df, feature_list)


# TRANSFORM FEATURES
df_trans <- df
transform_info <- tibble()

for (f in feature_list) {
  
  x <- df[[f]]
  sk <- skewness(x, na.rm = TRUE)
  
  if (f == "outcome") next
  
  has_negative <- any(x < 0, na.rm = TRUE)
  
  if (sk > 1) {
    
    if (!has_negative) {
      x_log <- log1p(x)
      
      new_name <- paste0(f, "_log")
      df_trans[[new_name]] <- x_log
      
      transform_info <- bind_rows(transform_info, tibble(
        base = f,
        feature_after = new_name,
        transform = "log1p"
      ))
    }
  }
}



# SELECT FINAL FEATURES
all_features <- names(df_trans)[sapply(df_trans, is.numeric)]

skew_after <- get_skew(df_trans, all_features)

best_features <- skew_after %>%
  mutate(base = gsub("_log", "", feature)) %>%
  group_by(base) %>%
  arrange(desc(grepl("_log$", feature))) %>%
  slice(1) %>%
  ungroup()


# TRANSFORM INFO
original_info <- tibble(
  base = feature_list,
  feature_after = feature_list,
  transform = "original"
)

transform_info <- bind_rows(transform_info, original_info)


# BEFORE vs AFTER
skew_compare <- skew_before %>%
  rename(skew_before = skewness) %>%
  mutate(base = feature) %>%
  left_join(
    best_features %>%
      select(feature, skewness) %>%
      rename(feature_after = feature, skew_after = skewness) %>%
      mutate(base = gsub("_log", "", feature_after)),
    by = "base"
  ) %>%
  left_join(transform_info, by = c("base", "feature_after")) %>%
  arrange(desc(abs(skew_before)))


# FEATURE LABELS (UPDATED WITH CE)
feature_labels <- c(
  "R_soft_cavity_fill..NPCA."  = "Soft Tissue / Air",
  "R_soft_interstitial..NPCA." = "Soft Tissue / Interstitial Space",
  "R_soft_bone_ratio..NPCA."   = "Soft Tissue / Bone",
  "R_air_fraction..NPCA."      = "Air / (Air + Soft Tissue)",
  "R_bone_air_ratio..NPCA."    = "Bone Contact / (Bone + Air)",
  "R_air_soft_contact..NPCA."  = "Air Contact / Soft Tissue",
  "R_bone_soft_contact..NPCA." = "Bone Contact / Soft Tissue",
  "R_soft_to_total..NPCA."     = "Soft Tissue / Total Volume",
  
  "R_oss_all_soft_ratio..NPCA."   = "Ossicles / Soft Tissue",
  "R_oss_air_ratio..NPCA."        = "Ossicles / Air",
  "R_oss_touch_soft_ratio..NPCA." = "Ossicles touching Soft Tissue",
  
  "V_soft_voxels..NPCA."         = "Soft Tissue Volume (voxels)",
  "V_cavity_voxels..NPCA."       = "Air Volume (voxels)",
  "V_bone_voxels..NPCA."         = "Bone Volume (voxels)",
  "V_bone_contact_voxels..NPCA." = "Bone Contact Volume",
  "V_air_contact_voxels..NPCA."  = "Air Contact Volume",

  "V_ossicle_found..CE."      = "Detected Ossicle Volume (voxels)",
  "Volume_ossicle_found..CE." = "Detected Ossicle Volume (mm³)"
)


# APPLY LABELS + TYPE SPLIT


skew_compare <- skew_compare %>%
  mutate(
    feature_clean = gsub("_log", "", feature_after),
    
    type = case_when(
      str_detect(feature_clean, "NPCA") ~ "NPCA",
      str_detect(feature_clean, "CE")   ~ "CE",
      TRUE ~ "OTHER"
    ),
    
    feature_label = dplyr::recode(
      feature_clean,
      !!!feature_labels,
      .default = feature_clean
    )
  )


# SPLIT TABLES
skew_NPCA <- skew_compare %>% filter(type == "NPCA")
skew_CE   <- skew_compare %>% filter(type == "CE")



# PRINT
cat("\n=== NPCA ===\n")
print(skew_NPCA %>% select(feature_label, transform, skew_before, skew_after), n = Inf)

cat("\n=== CE ===\n")
print(skew_CE %>% select(feature_label, transform, skew_before, skew_after), n = Inf)


# FINAL DATASET
final_features <- best_features$feature

df_final <- df_trans %>%
  select(all_of(final_features), outcome)

cat("\nFinal dataset ready\n")
print(dim(df_final))


# EXPORT CSV FILES
write.csv(skew_NPCA %>% select(feature_label, transform, skew_before, skew_after),
          "skew_NPCA.csv", row.names = FALSE)

write.csv(skew_CE %>% select(feature_label, transform, skew_before, skew_after),
          "skew_CE.csv", row.names = FALSE)





