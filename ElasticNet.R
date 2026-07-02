

# PACKAGES
set.seed(100)

library(tidyverse)
library(glmnet)
library(pROC)
library(caret)
library(e1071)
library(ggplot2)
library(moments)
library(dplyr)

# LOAD DATA
features <- read.csv("./Desktop/MEASUREMENT.csv", sep = ";")
labels   <- read.csv("./Desktop/LABEL.csv", sep = ";")

df <- features %>%
  left_join(labels, by = c("scan_id" = "Skanning.ID")) %>%
  

  # FIX CE FEATURES
  rename(
    "V_ossicle_found..CE."      = "V_ossicle_found..NPCA.",
    "Volume_ossicle_found..CE." = "Volume_ossicle_found..NPCA."
  ) %>%
  

  # OUTCOME
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
  


# FEATURES - SO THE SAME FEATURE IS NOT THERE MULTIPLE TIMES
features_npca <- names(df)[grepl("NPCA", names(df))]
features_ce   <- names(df)[grepl("CE", names(df))]

features_npca <- setdiff(
  features_npca,
  c("V_ossicle_found..NPCA.", "Volume_ossicle_found..NPCA.")
)

features_npca_ce <- unique(c(features_npca, features_ce))
features_npca_ce <- setdiff(features_npca_ce, "V_ossicle_found..CE.")

features_combined <- c(features_npca_ce, features_sa)



# TRANSFORM FUNCTION
fit_transform <- function(df_train, df_test, feature_list) {
  
  for (f in feature_list) {
    
    x <- df_train[[f]]
    sk <- skewness(x, na.rm = TRUE)
    has_negative <- any(x < 0, na.rm = TRUE)
    
    if (sk > 2 && !has_negative) {
      
      q_low  <- quantile(x, 0.01, na.rm = TRUE)
      q_high <- quantile(x, 0.99, na.rm = TRUE)
      
      df_train[[f]] <- log1p(pmin(pmax(df_train[[f]], q_low), q_high))
      df_test[[f]]  <- log1p(pmin(pmax(df_test[[f]],  q_low), q_high))
      
    } else if (sk > 1 && !has_negative) {
      
      df_train[[f]] <- log1p(df_train[[f]])
      df_test[[f]]  <- log1p(df_test[[f]])
    }
  }
  
  x_train <- scale(df_train[, feature_list])
  
  x_test <- scale(
    df_test[, feature_list],
    center = attr(x_train, "scaled:center"),
    scale  = attr(x_train, "scaled:scale")
  )
  
  return(list(
    x_train = x_train,
    x_test  = x_test,
    y_train = ifelse(df_train$outcome == "yes", 1, 0),
    y_test  = ifelse(df_test$outcome == "yes", 1, 0)
  ))
}


# NESTED CV + FEATURE IMPORTANCE

run_nested_cv <- function(df, feature_list, model_name) {

  cat("\n============================\n")
  cat("MODEL:", model_name, "\n")
  cat("============================\n")
  
  outer_folds <- createFolds(df$outcome, k = 5, returnTrain = TRUE)
  
  # metrics
  aucs <- c()
  sens <- c()
  spec <- c()
  ppv  <- c()
  npv  <- c()
  thresholds <- c()
  
  roc_list <- list()
  y_true_all <- c()
  y_prob_all <- c()
  
  # feature importance
  coef_mat <- matrix(0, nrow=length(feature_list), ncol=length(outer_folds))
  rownames(coef_mat) <- feature_list
  

  # LOOP
  for (i in seq_along(outer_folds)) {
    
    set.seed(100 + i)
    
    cat("Fold:", i, "\n")
    
    train_idx <- outer_folds[[i]]
    test_idx  <- setdiff(1:nrow(df), train_idx)
    
    train <- df[train_idx, ]
    test  <- df[test_idx, ]
    
    train <- train %>% select(outcome, all_of(feature_list))
    test  <- test  %>% select(outcome, all_of(feature_list))
    
    data_proc <- fit_transform(train, test, feature_list)
    
    x_train <- data_proc$x_train
    x_test  <- data_proc$x_test
    y_train <- data_proc$y_train
    y_test  <- data_proc$y_test
    
    # ================= INNER CV =================
    
    alphas <- seq(0.1, 1, by = 0.1)
    results <- data.frame()
    
    for (a in alphas) {
      
      set.seed(999 + i + round(a*100))
      
      
      foldid <- createFolds(y_train, k = 5, list = FALSE)
      
      cv_fit <-cv.glmnet(
        x_train, y_train,
        family = "binomial",
        alpha = a,
        foldid = foldid,
        type.measure = "deviance",
        standardize = FALSE
      )
      
      
      results <- rbind(results, data.frame(
        alpha = a,
        lambda = cv_fit$lambda.min,
        dev = min(cv_fit$cvm)
      ))
    }
    
    best <- results[which.min(results$dev), ]
    
    model <- glmnet(
      x_train, y_train,
      family = "binomial",
      alpha = best$alpha,
      lambda = best$lambda,
      standardize = FALSE,
      maxit = 1e6
    )
    
    # ================= FEATURE IMPORTANCE =================
    
    coefs <- coef(model)
    coefs <- as.matrix(coefs)
    coefs <- coefs[-1, , drop=FALSE]
    
    coef_mat[, i] <- coefs[,1]
    
    # ================= TEST =================
    
    prob <- as.numeric(predict(model, newx = x_test, type = "response"))
    y_true_all <- c(y_true_all, y_test)
    y_prob_all <- c(y_prob_all, prob)
    
    roc_obj <- roc(y_test, prob, quiet = TRUE)
    aucs[i] <- auc(roc_obj)
    roc_list[[i]] <- roc_obj
    
    # ================= THRESHOLD ON TRAINING DATA =================

    prob_train <- as.numeric(
      predict(
        model,
        newx = x_train,
        type = "response"
      )
    )
    
    roc_train <- roc(
      y_train,
      prob_train,
      quiet = TRUE
    )
    
    tmp <- coords(
      roc_train,
      "best",
      best.method = "youden",
      ret = c("threshold"),
      transpose = FALSE
    )
    
    t <- as.numeric(tmp$threshold)
    
    thresholds[i] <- t
    
    # ================= TEST =================
    
    pred <- ifelse(prob > t, 1, 0)
    
    cm <- table(factor(pred, levels=c(0,1)),
                factor(y_test, levels=c(0,1)))
    
    TN <- cm[1,1]; TP <- cm[2,2]
    FP <- cm[2,1]; FN <- cm[1,2]
    
    sens[i] <- TP/(TP+FN)
    spec[i] <- TN/(TN+FP)
    ppv[i]  <- ifelse((TP+FP)==0, NA, TP/(TP+FP))
    npv[i]  <- ifelse((TN+FN)==0, NA, TN/(TN+FN))
  }
  
  roc_all <- roc(y_true_all, y_prob_all, quiet = TRUE)
  

  # RESULTS
  cat("\n===== PERFORMANCE =====\n")
  cat("AUC:", round(mean(aucs),3), "±", round(sd(aucs),3), "\n")
  cat("Sensitivity:", round(mean(sens),3), "±", round(sd(sens),3), "\n")
  cat("Specificity:", round(mean(spec),3), "±", round(sd(spec),3), "\n")
  cat("PPV:", round(mean(ppv, na.rm=TRUE),3), "±", round(sd(ppv, na.rm=TRUE),3), "\n")
  cat("NPV:", round(mean(npv, na.rm=TRUE),3), "±", round(sd(npv, na.rm=TRUE),3), "\n")
  cat("Threshold:", round(mean(thresholds),3), "±", round(sd(thresholds),3), "\n")
  

  # FEATURE IMPORTANCE
  freq <- rowSums(coef_mat != 0) / ncol(coef_mat)
  mean_coef <- rowMeans(abs(coef_mat))
  
  importance_df <- data.frame(
    feature = rownames(coef_mat),
    frequency = freq,
    mean_abs_coef = mean_coef
  ) %>%
    arrange(desc(frequency), desc(mean_abs_coef))
  
  cat("\n===== FEATURE IMPORTANCE =====\n")
  print(head(importance_df, 20))
  
  return(list(
    roc_list = roc_list,
    roc_all  = roc_all,
    importance = importance_df,
    auc_mean = mean(aucs),
    sens_mean = mean(sens),
    spec_mean = mean(spec),
    ppv_mean = mean(ppv, na.rm=TRUE),
    npv_mean = mean(npv, na.rm=TRUE)
  ))
}




# RUN
result_npca <- run_nested_cv(df, features_npca_ce, "NPCA + CE")



set.seed(1)
result_npca_s1 <- run_nested_cv(df, features_npca_ce, "NPCA + CE")
set.seed(123)
result_npca_s123 <- run_nested_cv(df, features_npca_ce, "NPCA + CE")
set.seed(999)
result_npca_s999 <- run_nested_cv(df, features_npca_ce, "NPCA + CE")
set.seed(1234)
result_npca_s1234 <- run_nested_cv(df, features_npca_ce, "NPCA + CE")

results_reduced <- list(
  result_npca,
  result_npca_s1,
  result_npca_s123,
  result_npca_s999,
  result_npca_s1234
)


auc_vals  <- sapply(results_reduced, function(x) x$auc_mean)
sens_vals <- sapply(results_reduced, function(x) x$sens_mean)
spec_vals <- sapply(results_reduced, function(x) x$spec_mean)
ppv_vals  <- sapply(results_reduced, function(x) x$ppv_mean)
npv_vals  <- sapply(results_reduced, function(x) x$npv_mean)

cat("AUC:", round(mean(auc_vals),3), "±", round(sd(auc_vals),3), "\n")
cat("Sensitivity:", round(mean(sens_vals),3), "±", round(sd(sens_vals),3), "\n")
cat("Specificity:", round(mean(spec_vals),3), "±", round(sd(spec_vals),3), "\n")
cat("PPV:", round(mean(ppv_vals),3), "±", round(sd(ppv_vals),3), "\n")
cat("NPV:", round(mean(npv_vals),3), "±", round(sd(npv_vals),3), "\n")



# RUN
all_results <- bind_rows(
  result_npca$importance %>% mutate(run = "s0"),
  result_npca_s1$importance %>% mutate(run = "s1"),
  result_npca_s123$importance %>% mutate(run = "s123"),
  result_npca_s999$importance %>% mutate(run = "s999"),
  result_npca_s1234$importance %>% mutate(run = "s1234")
)

global_importance <- all_results %>%
  group_by(feature) %>%
  summarise(
    mean_freq = mean(frequency),
    sd_freq   = sd(frequency),
    mean_coef = mean(mean_abs_coef)
  ) %>%
  arrange(desc(mean_freq))

top_features_global <- global_importance %>%
  filter(mean_freq >= 0.5)

top_features_vec <- top_features_global$feature


result_npca_reduced <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,5)")
set.seed(1)
result_npca_reduced_s1 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,5)")
set.seed(123)
result_npca_reduced_s123 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,5)")
set.seed(999)
result_npca_reduced_s999 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,5)")
set.seed(1234)
result_npca_reduced_s1234 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,5)")




# 0.8
all_results <- bind_rows(
  result_npca_reduced$importance %>% mutate(run = "s0"),
  result_npca_reduced_s1$importance %>% mutate(run = "s1"),
  result_npca_reduced_s123$importance %>% mutate(run = "s123"),
  result_npca_reduced_s999$importance %>% mutate(run = "s999"),
  result_npca_reduced_s1234$importance %>% mutate(run = "s1234")
)


results_reduced <- list(
  result_npca_reduced,
  result_npca_reduced_s1,
  result_npca_reduced_s123,
  result_npca_reduced_s999,
  result_npca_reduced_s1234
)

auc_vals  <- sapply(results_reduced, function(x) x$auc_mean)
sens_vals <- sapply(results_reduced, function(x) x$sens_mean)
spec_vals <- sapply(results_reduced, function(x) x$spec_mean)
ppv_vals  <- sapply(results_reduced, function(x) x$ppv_mean)
npv_vals  <- sapply(results_reduced, function(x) x$npv_mean)

cat("AUC:", round(mean(auc_vals),3), "±", round(sd(auc_vals),3), "\n")
cat("Sensitivity:", round(mean(sens_vals),3), "±", round(sd(sens_vals),3), "\n")
cat("Specificity:", round(mean(spec_vals),3), "±", round(sd(spec_vals),3), "\n")
cat("PPV:", round(mean(ppv_vals),3), "±", round(sd(ppv_vals),3), "\n")
cat("NPV:", round(mean(npv_vals),3), "±", round(sd(npv_vals),3), "\n")


global_importance <- all_results %>%
  group_by(feature) %>%
  summarise(
    mean_freq = mean(frequency),
    sd_freq   = sd(frequency),
    mean_coef = mean(mean_abs_coef)
  ) %>%
  arrange(desc(mean_freq))

top_features_global <- global_importance %>%
  filter(mean_freq >= 0.80)

top_features_vec <- top_features_global$feature


result_npca_reduced_08 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,8)")
set.seed(1)
result_npca_reduced_08_s1 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,8)")
set.seed(123)
result_npca_reduced_08_s123 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,8)")
set.seed(999)
result_npca_reduced_08_s999 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,8)")
set.seed(1234)
result_npca_reduced_08_s1234 <- run_nested_cv(df, top_features_vec, "NPCA REDUCED (0,8)")


results_reduced <- list(
  result_npca_reduced_08,
  result_npca_reduced_08_s1,
  result_npca_reduced_08_s123,
  result_npca_reduced_08_s999,
  result_npca_reduced_08_s1234
)


auc_vals  <- sapply(results_reduced, function(x) x$auc_mean)
sens_vals <- sapply(results_reduced, function(x) x$sens_mean)
spec_vals <- sapply(results_reduced, function(x) x$spec_mean)
ppv_vals  <- sapply(results_reduced, function(x) x$ppv_mean)
npv_vals  <- sapply(results_reduced, function(x) x$npv_mean)

cat("AUC:", round(mean(auc_vals),3), "±", round(sd(auc_vals),3), "\n")
cat("Sensitivity:", round(mean(sens_vals),3), "±", round(sd(sens_vals),3), "\n")
cat("Specificity:", round(mean(spec_vals),3), "±", round(sd(spec_vals),3), "\n")
cat("PPV:", round(mean(ppv_vals),3), "±", round(sd(ppv_vals),3), "\n")
cat("NPV:", round(mean(npv_vals),3), "±", round(sd(npv_vals),3), "\n")


# ROC FROM 5 RUNS
roc_list <- lapply(results_reduced, function(x) x$roc_all)
auc_values <- sapply(roc_list, auc)
mean_auc <- mean(auc_values)
sd_auc <- sd(auc_values)


# DATA
roc_df <- data.frame()

for (i in 1:length(roc_list)) {
  df_temp <- data.frame(
    fpr = 1 - roc_list[[i]]$specificities,
    tpr = roc_list[[i]]$sensitivities,
    type = "Runs",
    run = paste0("Run ", i)
  )
  roc_df <- rbind(roc_df, df_temp)
}


# MEAN ROC + INTERPOLATION
mean_fpr <- seq(0, 1, length.out = 100)
tpr_interp <- matrix(NA, nrow = length(roc_list), ncol = 100)

for (i in 1:length(roc_list)) {
  
  fpr_vals <- 1 - roc_list[[i]]$specificities
  tpr_vals <- roc_list[[i]]$sensitivities
  
  df_tmp <- data.frame(fpr = fpr_vals, tpr = tpr_vals) %>%
    distinct(fpr, .keep_all = TRUE) %>%
    arrange(fpr)
  
  tpr_interp[i, ] <- approx(
    x = df_tmp$fpr,
    y = df_tmp$tpr,
    xout = mean_fpr
  )$y
}

mean_tpr <- colMeans(tpr_interp)

mean_df <- data.frame(
  fpr = mean_fpr,
  tpr = mean_tpr,
  type = "Mean ROC"
)


# SD
tpr_sd <- apply(tpr_interp, 2, sd)

upper <- pmin(mean_tpr + tpr_sd, 1)
lower <- pmax(mean_tpr - tpr_sd, 0)

shade_df <- data.frame(
  fpr = mean_fpr,
  upper = upper,
  lower = lower,
  type = "Variation (±SD)"
)


# PLOT

tiff(
  "Figure3_ROC.tiff",
  width = 180,
  height = 180,
  units = "mm",
  res = 600,
  compression = "lzw"
)


ggplot() +
  
  # Blue shadow
  geom_ribbon(data = shade_df,
              aes(x = fpr, ymin = lower, ymax = upper, fill = type),
              alpha = 0.25) +
  
  # Grey runs
  geom_line(data = roc_df,
            aes(x = fpr, y = tpr, group = run, color = type),
            alpha = 0.5) +
  
  # Mean ROC
  geom_line(data = mean_df,
            aes(x = fpr, y = tpr, color = type),
            linewidth = 1.4) +
  
  # Diagonal
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  
  scale_color_manual(
    name = "Curve type",
    values = c(
      "Runs" = "grey50",
      "Mean ROC" = "black"
    )
  ) +
  
  scale_fill_manual(
    name = "Curve type",
    values = c(
      "Variation (±SD)" = "steelblue"
    )
  ) +
  
  guides(
    fill = guide_legend(order = 1),
    color = guide_legend(order = 1)
  ) +
  
  labs(
    title = "ROC Curve",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  
  theme_bw() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.text = element_text(size = 10),
    legend.title = element_text(size = 11)
  )
dev.off()

ggsave(
  "Figure3_ROC.tiff",
  width = 180,
  height = 180,
  units = "mm",
  dpi = 600,
  compression = "lzw"
)
