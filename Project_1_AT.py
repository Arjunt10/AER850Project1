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
    'C': [0.01, 0.1, 1],
    'max_iter':[2000, 3000, 5000],
    'solver': ['saga', 'newton-cg', 'lbfgs'],
    }
m1_logreg_grid = GridSearchCV(model_1_logreg,
                              m1_logreg_param_grid,
                              scoring = 'f1_weighted',
                              n_jobs = -1,
                              cv = 5)

#Fitting model with scaled features and trainset target
m1_logreg_grid.fit(features_train_scaled, target_train)

#Gives out the best parameters for the logreg model
best_logreg_model1 = m1_logreg_grid.best_estimator_
print ('\n Best Logistic Regression Model:', best_logreg_model1)

"MODEL 2: Random Forest Model (Using GridSearchCV)"

#Import library for Model 2
from sklearn.ensemble import RandomForestClassifier

m2_randomforest = RandomForestClassifier(random_state = 42)

# Defining Parameters
m2_randomforest_param_grid = {
    'n_estimators': [10, 30, 50,100, 200],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }

m2_randomforest_grid = GridSearchCV (m2_randomforest,
                                     m2_randomforest_param_grid,
                                     scoring = 'accuracy',
                                     n_jobs = -1,
                                     cv = 5)
#Fitting model with scaled features and trainset target
m2_randomforest_grid.fit(features_train_scaled, target_train)

#Gives out best parameters for random forest model
best_randomforest_model2 = m2_randomforest_grid.best_estimator_
print ('\n Best Random Forest Model:', best_randomforest_model2)

"MODEL 3: Decision Tree Model (Using GridSearchCV)"

#Import library for Model 3
from sklearn.tree import DecisionTreeClassifier

m3_decisiontree = DecisionTreeClassifier(random_state = 42)

#Defining Parameters
m3_decisiontree_param_grid = {
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'criterion': ['gini', 'entropy'] 
    }

m3_decisiontree_grid = GridSearchCV (m3_decisiontree,
                                     m3_decisiontree_param_grid,
                                     scoring = 'accuracy',
                                     n_jobs = -1,
                                     cv = 5)
#Fitting model with scaled features and target from trainset
m3_decisiontree_grid.fit(features_train_scaled, target_train)

#Gives out best parameters for decision tree model
best_decisiontree_model3 = m3_decisiontree_grid.best_estimator_
print ('\n Best Decision Tree Model:', best_decisiontree_model3)

"MODEL 4: Support Vector Machine (SVM) Model (Using RandomizedSearchCV)"

#Import library for Model 4
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

m4_svm = SVC(random_state = 42)

#Defining Parameters
m4_svm_param_grid = {
    'C': [0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf'],
    'gamma' : ['scale', 'auto']
    }

m4_svm_random = RandomizedSearchCV (m4_svm,
                                m4_svm_param_grid,
                                scoring = 'accuracy',
                                n_jobs = -1,
                                n_iter = 12,
                                cv = 5, 
                                random_state = 42
                                )
#Fitting model with scaled features and target from train set
m4_svm_random.fit (features_train_scaled, target_train)

#Gives out best parameters for SVM model
best_svm_model4 = m4_svm_random.best_estimator_
print ('\n Best SVM model:', best_svm_model4)


"Step 5: Model Performance Analysis"

#Import library for performance metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score

#It would be easier to create a function that consists of performance metrics and just run that for each model
def performance_metric (model, features_test_scaled, target_test):
    target_prediction = model.predict (features_test_scaled)
    Accuracy = accuracy_score(target_test, target_prediction)
    Precision = precision_score(target_test, target_prediction, average = 'weighted', zero_division = 1)
    F1_Score = f1_score (target_test, target_prediction, average = 'weighted', zero_division = 1)
       
    return {
        'Accuracy': Accuracy,
        'Precision': Precision,
        'F1 Score': F1_Score
        }

#Performance of Model 1: Logistic Regression

m1_logreg_performance = performance_metric (best_logreg_model1,features_test_scaled, target_test)

print ('\n Here is the performance analysis of Model 1 - Logistic Regression:', m1_logreg_performance)

#Performance of Model 2: Random Forest

m2_randomforest_performance = performance_metric (best_randomforest_model2,features_test_scaled, target_test)

print ('\n Here is the performance analysis of Model 2 - Random Forest:', m2_randomforest_performance)

#Performance of Model 3: Decision Tree

m3_decisiontree_performance = performance_metric (best_decisiontree_model3 ,features_test_scaled, target_test)

print ('\n Here is the performance analysis of Model 3 - Decision Tree:', m3_decisiontree_performance)

#Performance of Model 4:SVM

m4_svm_performance = performance_metric (best_svm_model4, features_test_scaled, target_test)

print ('\n Here is the performance analysis of Model 4 - Support Vector Machine (SVM):', m4_svm_performance)

#Model 3: Decision Tree is the best model for this dataset as it has the best metrics amongst the other models
#especially the F1_score

#Run test data through best model - Decision Tree
best_model_pred = best_decisiontree_model3.predict(features_test_scaled)

#The performance of Model 3 must be visualized through a confusion matrix

#import Library
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Matrix Setup
conf_matrix = confusion_matrix(target_test, best_model_pred)
print('\n Here is the confusion matrix for Model 3 - Decision Tree: \n', conf_matrix)

#Display the confusion matrix as a plot
Conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix)
Conf_mat_disp.plot(cmap="viridis")
plt.title("Confusion Matrix for Decision Tree Model")

"STEP 6: Stacked Model Performance Analysis"

#library import
from sklearn.ensemble import StackingClassifier

#I will be combining Decision Tree Model & the Logisitic Regression models

#This is a stacking classifer parameter according to official scikit-learn website

estimators = [('logreg', best_logreg_model1), ('dt', best_decisiontree_model3)]
final_estimator = LogisticRegression()

#stacking the models
stacked_model = StackingClassifier(estimators = estimators, final_estimator = final_estimator, cv = 5,)
stacked_model.fit(features_train_scaled, target_train)

#Evaluting the model
stacked_model_pred = stacked_model.predict(features_test_scaled)

#Performance Metrics

stacked_accu = accuracy_score(target_test, stacked_model_pred)
stacked_prec = precision_score(target_test, stacked_model_pred, average = 'weighted', zero_division = 1)
stacked_f1 = f1_score(target_test, stacked_model_pred, average = 'weighted', zero_division = 1)

print ('\n The Performance Metrics of The Stacked model (Logistic Regression & Decision Tree) can be found below:\n')
print ('The accuracy of the stacked model is:', stacked_accu)
print ('The precision of the stacked model is:', stacked_prec)
print ('The F1-Score of the stacked model is:', stacked_f1)

#Confusion Matrix of this Stacked Model

#Matrix Setup
stacked_conf_matrix = confusion_matrix(target_test, stacked_model_pred)
print('\n Here is the confusion matrix for the stacked model: \n', stacked_conf_matrix)

#Display the confusion matrix as a plot
stacked_conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix = stacked_conf_matrix)
stacked_conf_mat_disp.plot(cmap="viridis")
plt.title("Confusion Matrix for The Stacked Model")



