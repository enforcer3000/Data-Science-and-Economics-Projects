# Remove all objects from memory
rm(list = ls())

# If you are missing any of the below, simply uncomment and execute the installation.
#install.packages('dplyr')
#install.packages("ggplot2")
#install.packages("leaps")
#install.packages("caret")
#install.packages("xgboost")
#install.packages('Matrix')

# Load the packages:
library(dplyr)
library(ggplot2) # for aesthetic plots
library(leaps) # for variable selection
library(caret) # for cross validation
library(xgboost) # for xgboost
library(Matrix) # for xgboost

# Reading in data
insurance = read.csv('insurance.csv', header = TRUE, stringsAsFactors = TRUE)
# Data set has 1338 observations and 7 variables


#########################################################
#               Exploratory data analysis               #
#########################################################

head(insurance)
summary(insurance)

# Removing observations with at least one missing value
insurance <- na.omit(insurance)
# Data set still has 1338 observations

# Removing duplicated observations
insurance <- distinct(insurance)
# Data set now has 1337 observation (1 duplicated entry)
summary(insurance)

                # === data visualisation === #
# Check for outliers in response variable, charges and bmi variable
# Visualizing distribution of response variable, charges
ggplot(insurance, aes(x = charges)) + 
  geom_density(fill = 'lightblue') +
  labs(title = 'Insurance charges', x = 'charges') +
  theme_minimal()

# It is likely that insurance charges will have a large range of values due to factors such as
# severity of incident, type of incident, coverage limits etc.
# As such, we should still use the data set with outliers in charges.


# Visualizing distribution of bmi variable
ggplot(insurance, aes(x = bmi)) + 
  geom_density(fill = 'lightblue') +
  labs(title = 'BMI', x = 'bmi') +
  theme_minimal()

# Based on domain knowledge, should still include those with high BMI as it is impossible to exclude
# them from insurance if they got it since young.


# Checking relationship between the different variable.
# Box plot for charges vs sex
ggplot(insurance, aes(x = sex, y = charges)) + 
  geom_boxplot(fill = 'lightblue', colour = 'black') +
  labs(title = 'charges vs sex', x = 'sex', y = 'charges') +
  theme_minimal()

# Box plot for charges vs children
ggplot(insurance, aes(x = factor(children), y = charges)) +
  geom_boxplot(fill = 'lightblue', colour = 'black') +
  labs(title = 'charges vs children', x = 'number of children', y = 'charges') +
  theme_minimal()

# Box plot for charges vs smoker
ggplot(insurance, aes(x = smoker, y = charges)) +
  geom_boxplot(fill = 'lightblue', colour = 'black') +
  labs(title = 'charges vs smoker', x = 'smoker', y = 'charges') +
  theme_minimal()

# Box plot for charges vs region
ggplot(insurance, aes(x = region, y = charges)) +
  geom_boxplot(fill = 'lightblue', colour = 'black') +
  labs(title = 'charges vs region', x = 'region', y = 'charges') +
  theme_minimal()

# Scatter plot for charges vs age
ggplot(insurance, aes(x = age, y = charges)) +  
  geom_point(color = "blue", size = 2) +
  labs(title = 'charges vs age', x = 'age', y = 'charges') +
  theme_minimal()

# Scatter plot for charges vs bmi
ggplot(insurance, aes(x = bmi, y = charges)) +  
  geom_point(color = "blue", size = 2) +
  labs(title = 'charges vs bmi', x = 'bmi', y = 'charges') +
  theme_minimal()

# Scatter plot for bmi vs age
ggplot(insurance, aes(x = age, y = bmi)) +  
  geom_point(color = "blue", size = 2) +
  labs(title = 'bmi vs age', x = 'age', y = 'bmi') +
  theme_minimal()

                # === statistical method check === #
# Computing correlation between continuous variables
vars <- c('charges', 'age', 'bmi')
correlation <- round(cor(insurance[, vars]), 4)
correlation

# Create contingency table
table_data <- table(insurance$sex, insurance$smoker)
# Perform Chi-Square Test
chi_test <- chisq.test(table_data)
chi_test
# Since p-value = 0.00628 < 0.05, there is statistically significant association
# between sex and smoker.

                # === data visualisation ===
# Stacked bar plot between sex and smoker
ggplot(insurance, aes(x = sex, fill = smoker)) + 
  geom_bar(position = "fill") + 
  geom_hline(yintercept = 274/1337, linetype = "dotted", color = "black") +
  labs(title = "Proportion of smoker within sex", y = "Proportion of smoker") +
  theme_minimal()

# Performing one-hot encoding on region variable
encoded_region <- model.matrix(~ region - 1, data = insurance)
# Combine with original data (excluding original region column)
insurance_encoded <- cbind(insurance[, names(insurance) != "region"], encoded_region)
summary(insurance_encoded)

#########################################################
#            REGRESSION AND MODELING APPROACH           #
#########################################################

# Prediction models
# Backward elimination with regsubsets()
lm1 <- regsubsets(charges ~ ., data = insurance_encoded, method = "backward", nvmax = 8) 
# Adjusted R-squared of the reduced model at the end of each elimination round
summary(lm1)$adjr2
best <- which.max(summary(lm1)$adjr2)
# it means backward elimination identifies that the best model contains 6 predictors
# A matrix that shows which predictors are included in each model
# * indicates inclusion, empty indicates exclusion
summary(lm1)$outm
# Our linear model will exclude sex as the feature variable.


# To choose which model we will be doing, we will be comparing the test error across the different models.
# As data set has 1338 observations which is considered small, 
# we will be using 70/30 train test split to compute the test error.
set.seed(2505) # to ensure replicability

ntrain = 936 #set size of the training sample to 70% of observations (1337)

tr = sample(1:nrow(insurance_encoded),ntrain)  # draw ntrain observations from original data
train = insurance_encoded[tr,]   # Training sample
test = insurance_encoded[-tr,]   # Testing sample (note negative index selects everything except tr)

# Baseline model (linear regression)
# Train the baseline model using 5-fold cross-validation
base_model <- train(charges ~ .-sex, data = train, method = "lm")
# Predict on test data
pred_base <- predict(base_model, newdata = test)
# Calculate RMSE (Root Mean Squared Error)
rmse_base <- sqrt(mean((test$charges - pred_base)^2))
rmse_base
# RMSE is 6015.2

# Visualisation of base model
data_base <- data.frame(
  actual = test$charges,
  predicted = pred_base
)

ggplot(data_base, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(linetype = "dashed") +   # 45° reference line
  labs(
    title = "Predicted vs Actual Charges (base model)",
    x = "Actual Charges",
    y = "Predicted Charges"
  )  +
  theme_minimal()

# Advanced model (XGBoost)
# Data & columns
df <- insurance_encoded
target <- "charges"
features <- c("age", "smoker", "regionnortheast", "regionnorthwest", "regionsoutheast", "regionsouthwest", "bmi", "children")

# 70/30 train test split
val_size <- 0.3

n <- nrow(df)
train_idx <- sample(seq_len(n), size = floor((1 - val_size) * n))
test_idx <- setdiff(seq_len(n), train_idx)

df_train <- df[train_idx, c(features, target), drop = FALSE]
df_test <- df[test_idx, c(features, target), drop = FALSE]

# Converting to XGBoost format and creating watchlist
fmla <- as.formula(paste(target, "~ ."))
mm_train <- model.matrix(fmla, data = df_train)[, -1, drop = FALSE]  # drop intercept
mm_test <- model.matrix(fmla, data = df_test)[, -1, drop = FALSE]

label_train <- df_train[[target]]
label_test <- df_test[[target]]

dtrain <- xgb.DMatrix(data = mm_train, label = label_train)
dtest <- xgb.DMatrix(data = mm_test, label = label_test)
watchlist <- list(train = dtrain, test = dtest)

# Parameters (for regression)
params <- list(
  objective = "reg:squarederror",     # regression
  eval_metric = "rmse",               # root mean square error
  max_depth = 5,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  gamma = 0
)

nrounds <- 500
early_stopping_rounds <- 50

# Train and test
xgb_fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = nrounds,
  watchlist = watchlist,
  early_stopping_rounds = early_stopping_rounds,
  print_every_n = 25
)
# RMSE is 4624.3 at iteration 102

# Visualisation of advanced model
pred_ad <- predict(xgb_fit, dtest)

data_ad <- data.frame(
  actual = label_test,
  predicted = pred_ad
)

ggplot(data_ad, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(linetype = "dashed") +   # 45° reference line
  labs(
    title = "Predicted vs Actual Charges (advanced model)",
    x = "Actual Charges",
    y = "Predicted Charges"
  ) +
  theme_minimal()


# Feature importance
importance <- xgb.importance(
  feature_names = colnames(mm_train),
  model = xgb_fit
)
xgb.plot.importance(importance, top_n = 10)


#########################################################
#                     Fairness check                    #
#########################################################

# Checking for sex variable
lm2 <- lm(charges ~., data = insurance)
summary(lm2)
# Note that sex has a p-value of 0.698, 
# differences in charges due to gender is not statistically significant.
# Box plot for charges vs sex 
# shows that median and IQR for charges based on gender is similar.
# Charges based on gender is likely fair.

# Checking for region variable
                # === statistical method check === #
# Kruskal–Wallis test for 4 different regions seperately
kruskal.test(charges ~ region, data = insurance)
# p-value = 0.202 > 0.05, differences in charges due to 4 different region is 
# not statistically significant ie. fairly similar.

# Combine into 2 regions and compare North vs South
insurance_ns <- insurance %>%
  mutate(
    region_ns = if_else(region %in% c("northeast", "northwest"), "north", "south"),
    region_ns = factor(region_ns, levels = c("north", "south"))
  )

# # Kruskal–Wallis test for North vs South
kruskal.test(charges ~ region_ns, data = insurance_ns)
# p-value = 0.417 > 0.05, differences in charges due to North and South region is 
# not statistically significant ie. fairly similar.

# Combine into 2 regions and compare East vs West
insurance_ew <- insurance %>%
  mutate(
    region_ew = if_else(region %in% c("northeast", "southeast"), "east", "west"),
    region_ew = factor(region_ew, levels = c("east", "west"))
  )

# Kruskal–Wallis test for East vs West
kruskal.test(charges ~ region_ew, data = insurance_ew)
# p-value = 0.0495 < 0.05, differences in charges due to East and West region is statistically significant.


                # === data visualisation === #
# Box plot for 4 original regions
ggplot(insurance, aes(x = region, y = charges, fill = region)) +
  geom_boxplot(alpha = 0.5) +
  labs(
    title = "Charges by Region (4 groups)",
    x = "Region",
    y = "Charges"
  ) +
  theme_minimal()

# Box plot for North vs South
ggplot(insurance_ns, aes(x = region_ns, y = charges, fill = region_ns)) +
  geom_boxplot(alpha = 0.5) +
  labs(
    title = "Charges by Region (North vs South)",
    x = "Region",
    y = "Charges"
  ) +
  theme_minimal()

# Box plot for East vs West
ggplot(insurance_ew, aes(x = region_ew, y = charges, fill = region_ew)) +
  geom_boxplot(alpha = 0.5) +
  labs(
    title = "Charges by Region (East vs West)",
    x = "Region",
    y = "Charges"
  ) +
  theme_minimal()

# Violin plot for East vs West
ggplot(insurance_ew, aes(x = region_ew, y = charges, fill = region_ew)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.1, outlier.shape = NA, alpha = 0.5) +  # overlay boxplot
  labs(
    title = "Insurance Charges: East vs West",
    x = "Region",
    y = "Charges"
  ) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal()
