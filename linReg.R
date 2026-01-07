# ----------------------------
# 1. Load Required Libraries

if(!require(ggplot2)) install.packages("ggplot2"); library(ggplot2)
if(!require(dplyr)) install.packages("dplyr"); library(dplyr)
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(reshape2)) install.packages("reshape2"); library(reshape2)



# ----------------------------
# 2. Load dataset

# Read the dataset
df <- read.csv('real_estate_dataset.csv')


# ----------------------------
# 3. Data Preparation and Exploratory Data Analysis (EDA)

cat(rep("=", 60), "\n")
cat("DATA PREPARATION and EXPLORATORY DATA ANALYSIS (EDA)\n")
cat(rep("=", 60), "\n")

# Dataset overview
cat("\n=== Dataset Overview ===\n")
cat("Rows:", nrow(df), "\n")
cat("Columns:", ncol(df), "\n")
cat("\nData types:\n")
print(sapply(df, class))
print(summary(df))

# Remove ID column not useful for modeling
df$ID  <- NULL

# Check any missing value and duplicate row
cat("\n=== Data Check ===\n")
cat("Missing values:", sum(is.na(df)), "\n")
cat("Duplicate rows:", sum(duplicated(df)), "\n")

# Use long format by select column
df_long <- melt(df, measure.vars = names(df)[])

# Boxplots
ggplot(df_long, aes(x = variable, y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()+
  labs(title = "Boxplot",
       x = "Feature")

# Histograms and Density Plot
ggplot(df_long, aes(x = value)) +
  geom_histogram(aes(y = after_stat(density)), color = "white") +
  geom_density() +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Histogram & Density Plot",
       x = "Feature Value")

# Scatter Plot
df_scatter <- melt(df,
                   id.vars = "Price",
                   measure.vars = setdiff(names(df), "Price"))
ggplot(df_scatter, aes(x = value, y = Price)) +
  geom_point() +
  stat_smooth(method = "lm") +
  facet_wrap(~ variable, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Scatter Plot of Features vs Price",
    x = "Feature Value",
    y = "Price"
  )


# ----------------------------
# 4. Model Planning and Development 

cat(rep("=", 60), "\n")
cat("MODEL PLANNING AND DEVELOPMENT\n")
cat(rep("=", 60), "\n")

cat("\n=== Data Split ===\n")

# Set the seed
set.seed(123)

# Split into test and training dataset using sample function
index <- sample(1:nrow(df), size=round(0.2*nrow(df)))
test_data <- df[index, ]
train_data <- df[-index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

cat("\n=== Feature Normalization ===\n")
# Separate target variable (Price) from features, don't normalize the target variable for linear regression
train_features <- subset(train_data, select = -Price)
test_features  <- subset(test_data, select = -Price)

# Create normalization parameters from TRAINING data only
preproc_params <- preProcess(train_features, 
                             method=c("center", "scale"))

# Apply normalization to both train and test sets
train_features_norm <- predict(preproc_params, train_features)
test_features_norm <- predict(preproc_params, test_features)

# Add back the target variable
train_norm <- cbind(train_features_norm, Price = train_data$Price)
test_norm <- cbind(test_features_norm, Price = test_data$Price)

print(summary(train_features_norm))

# Plot histogram
df_norm_long <- melt(train_features_norm, measure.vars = names(train_features_norm))
ggplot(df_norm_long, aes(x = value)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Normalized Features", x = "Normalized Value", y = "Count")

cat("\n=== Model Development ===\n")

# Fit linear regression on training data
lm_model <- lm(Price ~ ., data = train_norm)
print(summary(lm_model))

# function calculate metric
eval_metrics <- function(actual, pred) {
  rmse <- sqrt(mean((actual - pred)^2))
  mae  <- mean(abs(actual - pred))
  r2   <- 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
  data.frame(RMSE = rmse, MAE = mae, R2 = r2)
}

cat("\n=== Model Evaluation ===\n")

# Predict on train data
train_pred <- predict(lm_model, newdata = train_norm)
print(eval_metrics(train_norm$Price, train_pred))


cat("\n=== Model Validation ===\n")

# Predict on test data
test_pred <- predict(lm_model, newdata = test_norm)
print(eval_metrics(test_norm$Price, test_pred))

# Predicted vs Actual
ggplot(data.frame(
  Actual = test_norm$Price,
  Predicted = test_pred
), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0,
              color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Predicted vs Actual Price",
    x = "Actual Price",
    y = "Predicted Price"
  )
