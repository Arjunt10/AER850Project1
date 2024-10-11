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

"Read Data from CSV File & convert into dataframe"

df = pd.read_csv("Project_1_Data.csv")

"STEP 2: Data Visualization "

"Need to visualize the data in plots"

"Histogram"

Histogram_X = plt.hist(df['X'], bins = 45)
Histogram_Y = plt.hist(df['Y'], bins = 45)
Histogram_Z = plt.hist(df['Z'], bins = 45)

plt.title('Histogram of X, Y, Z vs Frequency')
plt.xlabel('X, Y, Z')
plt.ylabel('Frequency')

"Creating seperate figures "
plt.figure()


"Scatter Plot"

scatterplot_X = plt.scatter(df['Step'], df['X'])

scatterplot_Y = plt.scatter(df['Step'], df['Y'])

scatterplot_Z = plt.scatter(df['Step'], df['Z'])

plt.title('Scatterplot of X, Y, Z vs Steps')
plt.xlabel('Steps')
plt.ylabel('X, Y, Z')




""

"Simple Statistics from the data"

stats = df.describe()

print ("The statistically measures of the columns in the dataset are:", stats)





"STEP 3: Correlation Analysis"

correlation_matrix = df.corr()

print ("Here is the correlation matrix of the dataset", correlation_matrix)

