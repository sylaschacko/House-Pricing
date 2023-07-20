# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#read the data file
df= pd.read_csv('train.csv')

# Prints first few rows
print(df.head())

#Prints summary statistics
print(df.describe())

#check for missing data
print(df.isnull().sum())

#Create Histograms of numeric columns
df.select_dtypes(include=[np.number]).hist(bins=50, figsize=(20,15))
plt.show()

# Explore Correlations
corr_matrix = df.corr()
print(corr_matrix)

# Heat Map
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'coolwarm')

# Max Rows and Columns
np.set_printoptions(threshold = 100000000)
print(np.arange(10000))
df.isnull().sum()

