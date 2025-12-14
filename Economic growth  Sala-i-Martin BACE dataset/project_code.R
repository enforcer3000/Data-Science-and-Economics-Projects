#remove all objects from memory
rm(list = ls())


# Installing and loading packages we will use today:
# If you are missing any of the below, simply uncomment and execute the installation.
#install.packages("ggplot2")
#install.packages("tree")
#install.packages("rpart")
#install.packages("leaps")
#install.packages("kknn")

#Load the packages:
library(readxl)
library(ggplot2)
library(tree)
library(rpart)
library(leaps)
library(kknn)

# Reading in data
df <- read_excel("BACE_data.xls")[, -c(1: 3)]
summary(df)

#################
# Cleaning data #
#################

# Change all variables to type numeric, ensuring '.' changes to NA
df <- data.frame(lapply(df, as.numeric))

# Remove observations with outcome variable NA
df_outcome <- df[!is.na(df$GR6096), ]
# data set has 123 obs. and 68 variables

# To maximise the number of variables we have while keeping number of obs. high, 
# we decide to remove obs. if obs. has at least 2 NA and remove all vars with at least 1 NA
df_clean_row <- df[rowSums(is.na(df)) < 2, ]
df_clean <- df_clean_row[, colSums(is.na(df_clean_row)) == 0]
# cleaned data set has 94 obs. and 63 variables

# Scaling continuous data only
cat_var = c('BRIT', 'COLONY', 'EAST', 'ECORG', 'EUROPE', 'LAAM', 'LANDLOCK', 'NEWSTATE', 'OIL')
# Identify continuous variables
cols_to_scale <- setdiff(names(df_clean), cat_var)
# Scale only continuous variable, keeping categorical variables unchanged
df_clean[cols_to_scale] <- scale(df_clean[cols_to_scale])

#######################################################
# Exploring relationships between different variables #
#######################################################

# We decided to investigate the relationship between the Average growth rate of
# GDP per capita (1960-1996) with the initial GDP (1960), life expectancy (1960), primary schooling (1960)
# degree of ethnic diversity and whether the country is East Asian
vars_interest = c('GR6096', 'GDPCH60L', 'LIFE060', 'P60', 'AVELF')  # exclude EAST here as it is categorical
correlation <- round(cor(df_clean[, vars_interest]), 4)
correlation
# correlation between GR6096 and LIFE060 is 0.5409 (moderate strong positive relationship)
# correlation between GR6096 and P60 is 0.5726 (moderate strong positive relationship)

# Scatter plot between GDP growth rate from 1960-1996 and Life Expectancy in 1960
ggplot(df_clean, aes(x = LIFE060, y = GR6096)) +
  geom_point(colour = 'blue') +
  geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 1) +
  labs(title = "Economic Growth from 1960-1996 vs Life Expectancy in 1960",
       x = "Life Expectancy in 1960 (LIFE060)",
       y = "Growth Rate from 1960-1996 (GR6096)") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

# Scatter plot between GDP growth rate from 1960-1996 and Primary Schooling in 1960
ggplot(df_clean, aes(x = P60, y = GR6096)) +
  geom_point(colour = 'blue') +
  geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 1) +
  labs(title = "Economic Growth from 1960-1996 vs Primary Schooling in 1960",
       x = "Primary Schooling in 1960 (P60)",
       y = "Growth Rate from 1960-1996 (GR6096)") +
  theme_minimal() +
  theme(panel.grid = element_blank())

# Box plot between Economic Growth from 1960-1996 and East Asian country
ggplot(df_clean, aes(x = factor(EAST, labels = c("False", "True")), y = GR6096, 
                     fill = factor(EAST, labels = c("False", "True")))) +
  geom_boxplot() +
  labs(title = "Economic Growth (1960–1996) vs East Asian Country",
    x = "East Asian Country (True/False)",
    y = "Growth Rate (1960–1996)",
    fill = "East Asian") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  theme(panel.grid = element_blank())

# Checking p-values of coefficients in linear regression (full model)
full_model <- lm(GR6096 ~ ., data = df_clean)
coeff_table <- summary(full_model)$coefficients
# Extract p-values and sort in ascending order
sorted_pvals <- coeff_table[order(coeff_table[, "Pr(>|t|)"]), ]
sorted_pvals

# Top 5 coefficients with lowest p-value (indicating stronger statistical significance)
# IPRICE1, GDPCH60L, PRIGHTS, CIV72, P60

########################################################
# Examining prediction performance of different models #
########################################################

set.seed(6769)
# We will be doing a 70/30 train-test split for our model validation
# and root mean square error (RMSE) as our point of comparison
ntrain <- 66
tr = sample(1:nrow(df_clean),ntrain)  # draw ntrain observations from original data
train = df_clean[tr,]   # Training sample
test = df_clean[-tr,]   # Testing sample

######################################################
# Benchmark Model (Linear Regression on 3 variables) #
######################################################

lm1 <- lm(GR6096 ~ GDPCH60L + P60 + IPRICE1, data = train)
summary(lm1)

# Prediction
pred_test1 <- predict(lm1, newdata = test)

# Calculating RMSE
rmse1 <- sqrt(mean((test$GR6096 - pred_test1)^2))
# RMSE for benchmark model is 0.795

# Results
plot(test$GR6096, pred_test1,
     xlab = "Actual Growth Rate",
     ylab = "Predicted Growth Rate",
     main = "Baseline Linear Regression: Predicted vs Actual ",
     pch = 20, col = 'blue')

# Add a 45-degree reference line
abline(0, 1, col = "red", lwd = 2)

########################################
# K-Nearest Neighbour with 3 variables #
########################################

# LOOCV-style tuning over k = 1 to 94
knn_cv <- train.kknn(GR6096 ~ GDPCH60L + P60 + IPRICE1, data = train, kmax = 94, kernel = "rectangular")
summary(knn_cv)

# Plot MSE vs k
plot(seq_along(knn_cv$MEAN.SQU), knn_cv$MEAN.SQU, type = "l", col = "blue",
     main = "LOOCV MSE for KNN Regression",
     xlab = "Number of Neighbors (k)",
     ylab = "MSE")

# Best k value
k_best <- knn_cv$best.parameters$k

# Prediction
pred_test2 <- kknn(GR6096 ~ GDPCH60L + P60 + IPRICE1, train, test, k = k_best, kernel = "rectangular")

# Calculating RMSE
rmse2 <- sqrt(mean((test$GR6096 - pred_test2$fitted.values)^2))
# RMSE for KNN is 0.808

#################
# Decision Tree #
#################

set.seed(6769)

# Creates the tree
temp = tree(GR6096~., data=train, mindev=0.0001)

# Run cross-validation
cv_tree <- cv.tree(temp)   # cross-validation
cv_tree                     # see sizes + deviance

# Visualizing the tree
plot(cv_tree$size, cv_tree$dev, type = "b",
     xlab = "Tree Size (number of terminal nodes)",
     ylab = "Deviance (CV Error)",
     main = "Cross-Validation for Tree Pruning")

# Number of leaves
length(unique(temp$where))

# Number of nodes
nrow(temp$frame) 
best_size <- cv_tree$size[which.min(cv_tree$dev)]
pruned_tree <- prune.tree(temp, best = best_size)
nrow(pruned_tree$frame)

# Calculating RMSE
pred_test3 <- predict(temp, test)
rmse3 <- sqrt(mean((test$GR6096 - pred_test3)^2))
# RMSE for decision tree model is 0.839

# Plot the full (big) tree
plot(temp, type = "uniform")
text(temp, col = "blue", cex = 0.8)

# Plot the pruned tree
plot(pruned_tree, type = "uniform")
text(pruned_tree, col = "blue", cex = 0.8)

#############################################
# Linear Regression with variable selection #
#############################################

# Backward elimination with regsubsets()
lm2 <- regsubsets(GR6096 ~ ., data = train, method = "backward", nvmax = 62)
# Adjusted R-squared of the reduced model at the end of each elimination round
summary(lm2)$adjr2
# this returns the adjusted R-squared of the best model at each round of backward elimination
# each value corresponds to the highest adjusted R-squared among all models
# (size: from 1 predictor to 62 predictors in this example)

# which.max() helps is find the best model across all elimination rounds
# here it returns 47
# it means backward elimination identifies that the best model contains 47 predictors
best <- which.max(summary(lm2)$adjr2)
vars_in_best <- names(summary(lm2)$which[best, ])[summary(lm2)$which[best, ]]
vars_in_best <- setdiff(vars_in_best, "(Intercept)")  # remove '(Intercept')

# Create formula dynamically using variables in best in-sample model
lm3_formula <- as.formula(
  paste("GR6096 ~", paste(vars_in_best, collapse = " + "))
)

lm3 <- lm(lm3_formula, data = train)
summary(lm3)

# Prediction
pred_test4 <- predict(lm3, newdata = test)

# Calculating RMSE
rmse4 <- sqrt(mean((test$GR6096 - pred_test4)^2))
# RMSE for this model is 3.447 showing poor OOS prediction as high number of variables
# maximises in sample adj. R squared but leads to overfitting, leading to high variance

# Results
plot(x = test$GR6096,
  y = pred_test4,
  xlab = "Actual Growth Rate ",
  ylab = "Predicted Growth Rate ",
  main = "Linear Regression (Variable Selection): Predicted vs Actual",
  pch = 20, col = "blue")

# 45-degree reference line
abline(a = 0, b = 1, col = "red", lwd = 2)
