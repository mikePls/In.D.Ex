import asyncio

import matplotlib.pyplot as plt

from Modules.df_info_module import InfoModel
from Modules.distribution_stats_module import DistStats
from Modules import df_transformations_module
import pandas as pd
import seaborn as sns
import skillsnetwork
import numpy as np
import os
import urllib.request
sns.set()

#download df
async def download_dataset():
    URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data.tsv'
    await skillsnetwork.download_dataset(URL)
if not os.path.exists('Ames_Housing_Data.tsv'):
    asyncio.run(download_dataset())

#expand print area
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

#read dataset
df = pd.read_csv('Ames_Housing_Data.tsv', sep='\t')
print(df.info())
print(df.sample(5))

#cleaning recommended by author for removing a few outliers
df = df.loc[df['Gr Liv Area'] <= 4000, :]
#hold reference of the original df before proceeding
data_backup = df.copy()

#print(df.describe())

#get all object categories and generate dummy variables
one_hot_encoding_cols = df.dtypes[df.dtypes == object] #filter by datatype
one_hot_encoding_cols = one_hot_encoding_cols.index.tolist() #get categorical columns as a list
#preview columns selected for one hot
print(df[one_hot_encoding_cols].head().T)
df = pd.get_dummies(df, columns=one_hot_encoding_cols, drop_first=True) #drop first dummy column for independent linear relationships

##locate and log transform skewed variables
#choose columns of float type
mask = df.dtypes == float
float_cols = df.columns[mask]

#set minimum skewness limit for filtering columns
skew_limit = 0.75

#print skewed columns
skewed_vals = df[float_cols].skew()
skew_cols = (skewed_vals.sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))
print(skew_cols)
print(skew_cols.columns)

#apply log1p transformation to one skewed column and visualise distribution before and after
column = 'BsmtFin SF 1'

# #generate subplots with matplotlib's pyplot
# fig, (ax_before, ax_after) = plt.subplots(1,2, figsize=(10,5))
# #generate histogram before transform
# df[column].hist(ax=ax_before)
# #generate histogram after transform
# df[column].apply(np.log1p).hist(ax=ax_after)
# #format plot titles
# ax_before.set(title='before log1p transform', ylabel='frequency', xlabel='value')
# ax_after.set(title='after log1p transform', ylabel='frequency', xlabel='value')
# fig.suptitle(f'Field "{column}"')
# plt.show()

#perform log1p transformation to all except target column
for col in skew_cols.index.values:
    if col == 'SalePrice':
        continue
    print(f"{col}skew before: {df[col].skew()}")
    df[col] = df[col].apply(np.log1p)
    print(f"{col}skew after: {df[col].skew()}\n")

#current shape after dummy columns
print(df.shape)

#transforming numeric columns
df = data_backup
#set_checked for null values
print("Number of null values per column:\n", df.get_nulls().sum().sort_values())

df = df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond',
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area',
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces',
                      'Garage Cars','SalePrice']]

print(df.describe().T)
print(df.info)

#fill na values
df = df.fillna(0)

# with sns.plotting_context(rc={"axes.labelsize":6}):
#     sns.pairplot(df, plot_kws=dict(alpha=.1, edgecolor='none'), height=1)
#     sns.set(font_scale=0)
#plt.show()

#set target variable
X = df['SalePrice']
#drop target column from dataframe
y = df.drop(['SalePrice'], axis=1)

# from MLModules.visualisations_module import Visualizations
# d= Visualizations().heatmap(df=df.corr())

print(df.corr())











