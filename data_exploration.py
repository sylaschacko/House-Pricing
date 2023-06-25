# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#read the data file
train_data = pd.read_csv(r'C:\Users\sylas\House-Pricing\House-Pricing\train.csv')

# Prints first few rows
print(train_data.head())

#Prints summary statistics
print(train_data.describe())

#check for missing data
print(train_data.isnull().sum())

#Create Histograms of numeric columns
train_data.select_dtypes(include=[np.number]).hist(bins=50, figsize=(20,15))
plt.show()

# Explore Correlations
corr_matrix = train_data.corr()
print(corr_matrix)