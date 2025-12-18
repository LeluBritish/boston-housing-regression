###############################################################################
# SDS-301 Modern Regression Analysis
# Final Project: House Price Prediction (Boston Housing)
# Language: R
###############################################################################

# ========================= 0. PACKAGES ======================================

# Install packages once if needed (do NOT run every time)
# install.packages(c("MASS", "tidyverse", "corrplot",
#                    "car", "lmtest", "GGally", "caret"))

library(MASS)       # Boston dataset
library(tidyverse)  # dplyr, ggplot2
library(corrplot)   # correlation plots
library(car)        # VIF
library(lmtest)     # Breusch-Pagan test
library(GGally)     # ggpairs
library(caret)      # cross-validation

set.seed(123)

# ========================= 1. LOAD DATA =====================================

data("Boston")
df <- Boston

cat("Number of observations:", nrow(df), "\n")
cat("Number of variables:", ncol(df), "\n\n")

str(df)
summary(df)

# Check missing values
colSums(is.na(df))

# ========================= 2. EDA ===========================================

# ----- 2.1 Descriptive statistics -------------------------------------------
eda_summary <- df %>%
  summarise(across(
    everything(),
    list(
      mean = mean,
      sd   = sd,
      min  = min,
      q1   = ~quantile(.x, 0.25),
      med  = median,
      q3   = ~quantile(.x, 0.75),
      max  = max
    ),
    .names = "{.col}_{.fn}"
  ))
print(eda_summary)

# ----- 2.2 Histograms -------------------------------------------------------
pdf("histograms_all_variables.pdf", width = 10, height = 8)
par(mfrow = c(4, 4), mar = c(3, 3, 2, 1))
for (v in names(df)) {
  hist(df[[v]],
       main = v,
       xlab = "",
       col = "lightblue",
       border = "white")
}
dev.off()

# ----- 2.3 Response skewness: log-transformation check ----------------------
hist(df$medv, breaks = 30, col = "lightgray",
     main = "Histogram of MEDV",
     xlab = "Median House Value ($1000s)")

hist(log(df$medv), breaks = 30, col = "lightgray",
     main = "Histogram of log(MEDV)",
     xlab = "log(Median House Value)")

# ----- 2.4 Correlation matrix -----------------------------------------------
cor_mat <- cor(df)
sort(cor_mat[, "medv"], decreasing = TRUE)

pdf("correlation_matrix.pdf", width = 8, height = 8)
corrplot(cor_mat, method = "color", type = "lower",
         tl.col = "black", tl.srt = 45)
dev.off()

# ----- 2.5 Scatterplots with response ---------------------------------------
pdf("pairs_with_medv.pdf", width = 10, height = 10)
ggpairs(df, columns = c("medv","rm","lstat","ptratio","nox","crim","dis","tax"))
dev.off()

# ========================= 3. MODELING ======================================

# Train / test split
train_idx <- createDataPartition(df$medv, p = 0.8, list = FALSE)
train <- df[train_idx, ]
test  <- df[-train_idx, ]

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

# ----- 3.1 Full linear model ------------------------------------------------
model_full <- lm(medv ~ ., data = train)
summary(model_full)

# ----- 3.2 Stepwise model ---------------------------------------------------
model_step <- step(model_full, direction = "both", trace = 0)
summary(model_step)

# ----- 3.3 Polynomial model -------------------------------------------------
model_poly <- lm(
  medv ~ rm + I(rm^2) +
    lstat + I(lstat^2) +
    ptratio + nox + chas,
  data = train
)
summary(model_poly)

# ----- 3.4 Log-transformed response model -----------------------------------
model_log <- lm(log(medv) ~ ., data = train)
summary(model_log)

# ========================= 4. DIAGNOSTICS & SELECTION =======================

# Diagnostics for stepwise model
par(mfrow = c(2, 2))
plot(model_step)
par(mfrow = c(1, 1))

# Residual tests
shapiro.test(residuals(model_step))
bptest(model_step)
vif(model_step)

# Diagnostics for log model
par(mfrow = c(2, 2))
plot(model_log)
par(mfrow = c(1, 1))

shapiro.test(residuals(model_log))
bptest(model_log)
vif(model_log)

# ========================= 5. MODEL COMPARISON=================================

# Predictions on test set
pred_full  <- predict(model_full, newdata = test)
pred_step  <- predict(model_step, newdata = test)
pred_poly  <- predict(model_poly, newdata = test)
pred_log   <- exp(predict(model_log, newdata = test))  # back-transform

model_comp <- data.frame(
  Model = c("Full", "Stepwise", "Polynomial", "Log-Linear"),
  RMSE  = c(rmse(test$medv, pred_full),
            rmse(test$medv, pred_step),
            rmse(test$medv, pred_poly),
            rmse(test$medv, pred_log)),
  R2    = c(cor(test$medv, pred_full)^2,
            cor(test$medv, pred_step)^2,
            cor(test$medv, pred_poly)^2,
            cor(test$medv, pred_log)^2)
)
print(model_comp)

# Cross-validation for stepwise model
train_control <- trainControl(method = "cv", number = 10)
cv_step <- train(
  medv ~ .,
  data = train[, all.vars(formula(model_step))],
  method = "lm",
  trControl = train_control
)
cv_step$results

# ========================= 6. FINAL MODEL ===================================

# Choosing log-linear as final model for better residuals
final_model <- model_log
summary(final_model)

# Predictions and metrics
final_train_pred <- exp(predict(final_model, train))
final_test_pred  <- exp(predict(final_model, test))

final_results <- data.frame(
  Set  = c("Train", "Test"),
  R2   = c(cor(train$medv, final_train_pred)^2,
           cor(test$medv, final_test_pred)^2),
  RMSE = c(rmse(train$medv, final_train_pred),
           rmse(test$medv, final_test_pred))
)
print(final_results)

# ========================= 7. INTERPRETATION ================================

coefs <- coef(final_model)

cat("\nLog-linear model interpretation examples:\n")
cat("rm: a one-unit increase in rooms increases median house value by approximately",
    round((exp(coefs["rm"]) - 1) * 100, 2), "%, holding other variables constant.\n")
cat("lstat: a 1% increase in lower-status population decreases median house value by approximately",
    round((exp(coefs["lstat"]) - 1) * 100, 2), "%.\n")

# ========================= 8. SAVE OUTPUTS ==================================

coef_df <- as.data.frame(summary(final_model)$coefficients)
coef_df$Variable <- rownames(coef_df)
coef_df <- coef_df[, c("Variable", "Estimate", "Std. Error", "t value", "Pr(>|t|)")]

write.csv(coef_df, "final_model_coefficients.csv", row.names = FALSE)
write.csv(final_results, "final_model_metrics.csv", row.names = FALSE)

cat("\n===== DONE: Code executed successfully. All outputs saved. =====\n")
###############################################################################