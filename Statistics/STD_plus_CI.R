

# PACKAGES
library(tidyverse)
library(e1071)
library(scales)


# LOAD DATA
features <- read.csv("./Desktop/MEASUREMENT.csv", sep = ";")
labels   <- read.csv("./Desktop/LABEL.csv", sep = ";")

getdf <- features %>%
  left_join(gtlabels, by = c("scan_id" = "Skanning.ID")) %>%
  mutate(
    True = as.numeric(True),
    outcome = case_when(
      True %in% c(1,4) ~ 1,
      True %in% c(0,98) ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(outcome))



# FEATURE LABELS
df <- getdf
df <- df %>%
  rename(
    "V_ossicle_found..CE."      = "V_ossicle_found..NPCA.",
    "Volume_ossicle_found..CE." = "Volume_ossicle_found..NPCA."
  )


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
  
  "V_soft_voxels..NPCA."         = "Soft Tissue Volume",
  "V_cavity_voxels..NPCA."       = "Air Volume",
  "V_bone_voxels..NPCA."         = "Bone Volume",
  "V_bone_contact_voxels..NPCA." = "Bone Contact Volume",
  "V_air_contact_voxels..NPCA."  = "Air Contact Volume",
  
  "V_ossicle_found..CE."       = "Detected Ossicle Volume (voxels)",
  "Volume_ossicle_found..CE."  = "Detected Ossicle Volume (mm³)",
)

feature_list <- intersect(names(df), names(feature_labels))


# TRANSFORM
df_trans <- df
transform_map <- list()

for (f in feature_list){
  
  x <- df[[f]]
  sk <- skewness(x, na.rm = TRUE)
  has_negative <- any(x < 0, na.rm = TRUE)
  
  if (sk > 1) {
    
    if (!has_negative) {
      x_new <- log1p(x)
      df_trans[[f]] <- x_new
      transform_map[[f]] <- "log1p"
    } else {
      transform_map[[f]] <- "Original"
    }
    
  } else {
    transform_map[[f]] <- "Original"
  }
}

# STANDARDIZE
df_z <- df_trans %>%
  mutate(across(all_of(feature_list), ~as.numeric(scale(.))))


# FEATURE GROUPS
feature_groups <- tibble(feature = feature_list) %>%
  mutate(
    type = case_when(
      str_detect(feature, "NPCA") ~ "NO PCA",
      str_detect(feature, "CE")   ~ "NO PCA", 
      TRUE ~ "OTHER"
    ),
    
    subtype = case_when(
      str_starts(feature, "R_") ~ "RATIO",
      str_starts(feature, "V_") ~ "VOXELS",
      str_detect(feature, "Volume") ~ "VOLUME",
      TRUE ~ "OTHER"
    )
  )


# STATS
make_stats <- function(data, feats, label_map){
  
  long_data <- data %>%
    select(outcome, all_of(feats)) %>%
    pivot_longer(-outcome)
  
  stats <- long_data %>%
    group_by(name, outcome) %>%
    summarise(
      n = n(),
      mean = mean(value, na.rm = TRUE),
      sd = sd(value, na.rm = TRUE),
      se = sd / sqrt(n),
      SD_low = mean - sd,
      SD_high = mean + sd,
      CI_low = mean - 1.96 * se,
      CI_high = mean + 1.96 * se,
      .groups = "drop"
    )
  
  pvals <- long_data %>%
    group_by(name) %>%
    summarise(
      p_value = tryCatch(
        t.test(value ~ outcome)$p.value,
        error = function(e) NA
      ),
      .groups = "drop"
    ) %>%
    mutate(
      p_adj = p.adjust(p_value, method = "BH")
    )
  
  stats %>%
    left_join(pvals, by = "name") %>%
    mutate(
      feature_label = label_map[name],
      transform_label = unlist(transform_map[name]),
      outcome_label = ifelse(outcome == 1, "Cholesteatom", "Normal"),
      y_label = paste0(outcome_label, " (n=", n, ")")
    )
}


# TABLES
make_table <- function(df){
  df %>%
    mutate(
      Group = ifelse(outcome==1,"Cholesteatom","Normal"),
      Mean_SD = sprintf("%.2f ± %.2f",mean,sd),
      Mean_CI = sprintf("%.2f [%.2f–%.2f]",mean,CI_low,CI_high),
      p_value = sprintf("%.4f", p_value),
      p_adj = sprintf("%.4f", p_adj)
    ) %>%
    select(feature_label, Group, n, Mean_SD, Mean_CI, p_value, p_adj)
}

make_table_wide <- function(df){
  df %>%
    mutate(
      Group=ifelse(outcome==1,"Cholesteatom","Normal"),
      Value=sprintf("%.2f [%.2f–%.2f]",mean,CI_low,CI_high)
    ) %>%
    select(feature_label, Group, Value, p_value, p_adj) %>%
    distinct() %>%
    pivot_wider(names_from=Group, values_from=Value)
}


# PLOT 
make_plot <- function(df, xmin, xmax, lab_fun, title){
  
  ggplot(df, aes(mean, y_label, color = outcome_label)) +
    
    geom_point(size = 3) +
    
    geom_errorbar(
      aes(xmin = .data[[xmin]], xmax = .data[[xmax]]),
      width = 0.2
    ) +
    
    geom_text(
      aes(label = lab_fun(mean, sd, CI_low, CI_high)),
      vjust = -2.2,
      size = 2.5
    ) +
    
    geom_vline(xintercept = 0, linetype = "dashed") +
    
    facet_wrap(
      ~paste0(feature_label, "\n(", transform_label, ")"),
      scales = "free_x",
      ncol = 4
    ) +
    
    scale_color_manual(
      values = c("Normal" = "blue", "Cholesteatom" = "red")
    ) +
    
    labs(
      title = title,
      x = NULL,
      y = NULL
    ) +
    
    theme_minimal() +
    
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5),
      panel.spacing = unit(2.5, "lines"),
      axis.title = element_blank()
    )
}


# RUN LOOP
for (t in unique(feature_groups$type)){
  
  for (s in unique(feature_groups$subtype)){
    
    feats <- feature_groups %>%
      filter(type==t, subtype==s) %>%
      pull(feature)
    
    if(length(feats)==0) next
    
    stats_tmp <- make_stats(df_z, feats, feature_labels)
    
    if(nrow(stats_tmp)==0) next
    
    cat("\n========================================\n")
    cat("TYPE:",t,"| SUBTYPE:",s,"\n")
    cat("========================================\n")
    
    # PLOTS
    print(make_plot(stats_tmp,"SD_low","SD_high",
                    function(m,sd,cl,ch)sprintf("%.2f ± %.2f",m,sd),
                    paste(t,"-",s,"(Z-stand Mean ± SD)")))
    
    print(make_plot(stats_tmp,"CI_low","CI_high",
                    function(m,sd,cl,ch)sprintf("%.2f [%.2f–%.2f]",m,cl,ch),
                    paste(t,"-",s,"(Z-stand Mean ± 95% CI)")))
    
    # TABLES
    table_long <- make_table(stats_tmp)
    table_wide <- make_table_wide(stats_tmp)
    
    cat("\n--- LONG TABLE ---\n")
    print(table_long, n=50)
    
      row.names = FALSE
    #)
  }
}

