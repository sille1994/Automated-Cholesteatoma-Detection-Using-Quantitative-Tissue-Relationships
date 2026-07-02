

# PACKAGES
library(tidyverse)
library(e1071)


# LOAD DATA
features <- read.csv("./Desktop/MEASUREMENT.csv", sep = ";")
labels   <- read.csv("./Desktop/LABEL.csv", sep = ";")

df <- features %>%
  left_join(labels, by = c("scan_id" = "Skanning.ID")) %>%
  
  rename(
    "V_ossicle_found..CE."      = "V_ossicle_found..NPCA.",
    "Volume_ossicle_found..CE." = "Volume_ossicle_found..NPCA."
  ) %>%
  
  mutate(
    True = as.numeric(True),
    outcome = case_when(
      True %in% c(1,4) ~ "yes",
      True %in% c(0,98) ~ "no",
      TRUE ~ NA_character_
    ),
    outcome = factor(outcome)
  ) %>%
  filter(!is.na(outcome))

# FEATURE LABELS
feature_labels <- c(
  "V_ossicle_found..CE."      = "Detected Ossicle Volume (voxels)",
  "Volume_ossicle_found..CE." = "O.V.",
  
  "R_air_fraction..NPCA."      = "A.V. / (A.V. + S.T.V.)",
  "R_air_soft_contact..NPCA."  = "A.C. / S.T.V.",
  "R_bone_air_ratio..NPCA."    = "B.C. / (B.C. + A.C.)",
  "R_bone_soft_contact..NPCA." = "B.C. / S.T.V.",
  "R_oss_air_ratio..NPCA."     = "O.V. / A.V.",
  "R_oss_all_soft_ratio..NPCA."= "O.V. / S.T.V.",
  "R_oss_touch_soft_ratio..NPCA." = "O.V. / S.T.T.O.",
  "R_soft_bone_ratio..NPCA."   = "S.T.V. / B.V.",
  "R_soft_cavity_fill..NPCA."  = "S.T.V. / A.V.",
  "R_soft_interstitial..NPCA." = "S.T.V. / (A.V. + S.T.V.)",
  "R_soft_to_total..NPCA."     = "S.T.V. / (B.V. + A.V. + S.T.V.)",
  
  "V_air_contact_voxels..NPCA."  = "A.C.",
  "V_bone_contact_voxels..NPCA." = "B.C.",
  "V_bone_voxels..NPCA."         = "B.V.",
  "V_cavity_voxels..NPCA."       = "A.V.",
  "V_soft_voxels..NPCA."         = "S.T.V."
)


# FEATURE ORDER
feature_order <- c(
  # Volumes
  "V_soft_voxels..NPCA.",
  "V_cavity_voxels..NPCA.",
  "V_bone_voxels..NPCA.",
  "V_bone_contact_voxels..NPCA.",
  "V_air_contact_voxels..NPCA.",
  
  # Ossicles
  # "V_ossicle_found..CE.",
  "Volume_ossicle_found..CE.",
  
  # Ratios
  "R_soft_to_total..NPCA.",
  "R_soft_interstitial..NPCA.",
  "R_soft_cavity_fill..NPCA.",
  "R_soft_bone_ratio..NPCA.",
  "R_oss_touch_soft_ratio..NPCA.",
  "R_oss_all_soft_ratio..NPCA.",
  "R_oss_air_ratio..NPCA.",
  "R_bone_soft_contact..NPCA.",
  "R_bone_air_ratio..NPCA.",
  "R_air_soft_contact..NPCA.",
  "R_air_fraction..NPCA."
  
)


features_use <- feature_order[feature_order %in% names(df)]


# TRANSFORM

df_trans <- df

for (f in features_use){
  
  x <- df[[f]]
  sk <- skewness(x, na.rm = TRUE)
  has_negative <- any(x < 0, na.rm = TRUE)
  
  if (sk > 1 && !has_negative){
    df_trans[[f]] <- log1p(x)
  }
}

# CORRELATION MATRIX
df_corr <- df_trans %>%
  select(all_of(features_use))

# labels
labels <- feature_labels[features_use]
labels[is.na(labels)] <- features_use[is.na(labels)]

colnames(df_corr) <- labels

cor_mat <- cor(df_corr, use = "pairwise.complete.obs", method = "spearman")


# LONG FORMAT
cor_df <- as.data.frame(as.table(cor_mat))

cor_df$Var1 <- factor(cor_df$Var1, levels = labels)
cor_df$Var2 <- factor(cor_df$Var2, levels = labels)

# PLOT
p <- ggplot(cor_df, aes(Var1, Var2, fill = Freq)) +
  
  geom_tile() +
  
  geom_text(
    aes(
      label = sprintf("%.2f", Freq),
      color = abs(Freq) > 0.5
    ),
    
  ) +
  
  scale_color_manual(values = c("black", "white"), guide = "none") +
  
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    plot.title = element_text(size = 15, hjust = 0.5)
  ) +
  
  labs(
    title = "Correlation Matrix",
    x = "",
    y = ""
  )

print(p)



###########################


p <- ggplot(cor_df, aes(Var1, Var2, fill = Freq)) +
  
  geom_tile(color = "white", linewidth = 0.3) +
  
  geom_text(
    aes(
      label = sprintf("%.2f", Freq),
      color = abs(Freq) > 0.5
    ),
    size = 3.2
  ) +
  
  scale_color_manual(
    values = c("black", "white"),
    guide = "none"
  ) +
  
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Correlation"
  ) +
  
  coord_fixed() +
  
  theme_bw() +
  
  theme(
    panel.grid = element_blank(),
    
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      size = 10,
      color = "black"
    ),
    
    axis.text.y = element_text(
      size = 10,
      color = "black"
    ),
    
    axis.title = element_blank(),
    
    plot.title = element_text(
      size = 15,
      face = "bold",
      hjust = 0.5
    ),
    
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10)
  ) +
  
  labs(
    title = "Correlation Matrix"
  )

print(p)


ggsave(
  "Supplementary_CorrelationMatrix.tiff",
  width = 260,
  height = 240,
  units = "mm",
  dpi = 600,
  compression = "lzw"
)
