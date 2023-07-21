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

# Prints (number of data points, number of columns)
df.shape

# Prints All columns , number of data points, "non-null", variable type
df.info() 

#Fill in Missing Values
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature','Alley','Id'],axis=1,inplace=True)

df.shape 

df.isnull().sum

#Fills rest of missing values
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')

# Fill Missing Values