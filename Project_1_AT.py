# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:31:48 2024

@author: arjun

Arjun Tripathi
501 021 964
AER850 - PROJECT 1
"""

"STEP 1 - Data Processing"

"Library Import"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"Read Data from CSV File & convert into dataframe"

df = pd.read_csv("Project_1_Data.csv")
initial_data = df.head()

print ("Here are the first few rows of the data for verification:", initial_data)

# "STEP 2: Data Visualization Of RAW DATA "

# # "Need to visualize the raw data in plots"

# "3D Scatter Plot of RAW DATA"

# fig = plt.figure (figsize=(10,10))
# ax = fig.add_subplot(111,projection = '3d')

# scatterplot_raw_XYZ = ax.scatter(df['X'],df['Y'],df['Z'], c=df['Step'],cmap = 'viridis')

# plt.title('Scatterplot of X, Y, Z (RAW DATA)')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # For the legend
# plt.colorbar(scatterplot_raw_XYZ,ax=ax, label = 'step')

# "Box Plot of RAW DATA"
# # To see if any outliers exist for each axis

# fig_2 = plt.figure(figsize=(10,6))
# plt.boxplot([df['X'], df['Y'], df['Z']], labels = ['X', 'Y', 'Z'])
# plt.title('Box Plot of X, Y, Z (RAW DATA)')


# "Simple Statistics from the Raw data"

# stats = df.describe()

# print ("The statistically measures of the columns in the raw dataset are:", stats)


# "STEP 3: Correlation Analysis of RAW Data"

# correlation_matrix = df.corr()

# print ("Here is the correlation matrix of the raw dataset", correlation_matrix)

# # Need to create a new figure

# plt.figure(figsize=(10,10))

# correlation_heatmap = sns.heatmap(correlation_matrix)

"Step 4: Classification Model Development/Engineering"

#Import Library for data spliting

from sklearn.model_selection import StratifiedShuffleSplit

# Need to split data into train and test data using stratified sampling
# Data should be 80-20 split. 80% train & 20% Test

my_split = StratifiedShuffleSplit(n_splits = 1, test_size= 0.2, random_state = 42)

for train_index, test_index in my_split.split(df,df['Step']):
    strat_train_set = df.loc[train_index].reset_index(drop = True)
    strat_test_set = df.loc[test_index].reset_index(drop = True)

# Need to further split train/test data into features & targets

features_train = strat_train_set.drop(columns = ['Step'])
target_train = strat_train_set['Step']

features_test = strat_test_set.drop(columns = ['Step'])
target_test = strat_test_set['Step']

# Scaling the train/test data of the features (x,y,z)

from sklearn.preprocessing import StandardScaler

my_scaler = StandardScaler()

my_scaler.fit(features_train) #calculating mean and standard deviation - ONLY TRAIN DATA to prevent data leak.

features_train_scaled = my_scaler.transform(features_train)
features_test_scaled = my_scaler.transform (features_test)

"MODEL 1: Logistic Regression (Using GridSearchCV)"

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

model_1_logreg = LogisticRegression(random_state = 42)

#Defining Parameters for model
m1_logreg_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter':[500,1000,1500],
    'solver': ['saga', 'newton-cg', 'lbfgs'],
    }
m1_logreg_grid = GridSearchCV(model_1_logreg,
                              m1_logreg_param_grid,
                              scoring = 'accuracy',
                              n_jobs = -1,
                              cv = 5)

#Fitting model with scaled features and trainset target
m1_logreg_grid.fit(features_train_scaled, target_train)

#Gives out the best parameters for the logreg model
best_logreg_model1 = m1_logreg_grid.best_estimator_
print ('Best Logistic Regression Model:', best_logreg_model1)

"MODEL 2: Random Forest Model (Using GridSearchCV)"

#Import library for Model 2
from sklearn.ensemble import RandomForestRegressor

m2_randomforest = RandomForestRegressor(random_state = 42)

# Defining Parameters
m2_randomforest_param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }

m2_randomforest_grid = GridSearchCV((m2_randomforest),
                                    m2_randomforest_param_grid,
                                    scoring = 'accuracy',
                                    n_jobs = -1,
                                    cv = 5)
#Fitting model with scaled features and trainset target
m2_randomforest_grid.fit(features_train_scaled, target_train)

#Gives out best parameters for random forest model

best_randomforest_model2 = m2_randomforest_grid.best_estimator_
print ('Best Random Forest Model:', best_randomforest_model2)

