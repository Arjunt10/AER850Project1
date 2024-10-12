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

"STEP 2: Data Visualization "

# "Need to visualize the data in plots"

"3D Scatter Plot"

fig = plt.figure (figsize=(10,10))
ax = fig.add_subplot(111,projection = '3d')

scatterplot_XYZ = ax.scatter(df['X'],df['Y'],df['Z'], c=df['Step'],cmap = 'viridis')

plt.title('Scatterplot of X, Y, Z')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.colorbar(scatterplot_XYZ,ax=ax, label = 'step')


"Simple Statistics from the data"

stats = df.describe()

print ("The statistically measures of the columns in the dataset are:", stats)


"STEP 3: Correlation Analysis"

correlation_matrix = df.corr()

print ("Here is the correlation matrix of the dataset", correlation_matrix)

# Need to create a new figure

plt.figure(figsize=(10,10))

correlation_heatmap = sns.heatmap(correlation_matrix)


