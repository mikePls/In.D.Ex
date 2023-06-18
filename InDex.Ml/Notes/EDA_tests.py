import asyncio

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skillsnetwork
import seaborn as sns
import os.path
from scipy.stats import boxcox
import os

from Modules.df_info_module import InfoModel
from Modules.distribution_stats_module import DistStats

async def download_dataset():
    URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/iris_data.csv'
    await skillsnetwork.download_dataset(URL)

if not os.path.exists('iris_data.csv'):
    asyncio.run(download_dataset())


data = pd.read_csv('iris_data.csv')
print(data.head())
print(data.dtypes)

#examine shape of df [0] x axis, num of feature columns
print(data.shape[0])

print(data.columns)
print(data.columns.to_list())

'''
replace all strings in a column example

df['ColumnName'] = df.ColumnName.str.replace('string or substring to replace', 'replacewith')
or with lambda function:
df['species'] = df.species.apply(lambda r: r.replace('Iris-', ''))
'''

#number each value occurs in a feature
print(data['species'].value_counts())
print('***')
"""
----Find mean, median, quartiles and ranges of non-categorical features

# Select just the rows desired from the 'describe' method and add in the 'median'
stats_df = df.describe()

stats_df.loc['range'] = stats_df.loc['return_x_cols'] - stats_df.loc['min']

out_fields = ['mean','25%','50%','75%', 'range']
stats_df = stats_df.loc[out_fields]
stats_df.rename({'50%': 'median'}, inplace=True)
stats_df
### END SOLUTION
"""
#print(df.describe())

#group by and calculate average
#print(df.groupby('species').mean())

#calculate median
#print(df.groupby('species').median())

"""
Aggregate fields separately

# If certain fields need to be aggregated differently, we can do:
from pprint import pprint

agg_dict = {field: ['mean', 'median'] for field in df.columns if field != 'species'}
agg_dict['petal_length'] = 'return_x_cols'
pprint(agg_dict)
df.groupby('species').agg(agg_dict)
### END SOLUTION

"""

#apply multiple functions at once
# df.groupby('species').agg(['mean', 'median'])  # passing a list of recognized strings
# #or
# data_mean_mid = df.groupby('species').agg([np.mean, np.median])  # passing a list of explicit aggregation functions
# print(data_mean_mid)

#pyplot with matplotlib.pyplot as plt
# ax = plt.axes()
# ax.scatter(df.sepal_length, df.sepal_width)
#
# # Label the axes
# ax.set(xlabel='Sepal Length (cm)',
#        ylabel='Sepal Width (cm)',
#        title='Sepal Length vs Width')
# plt.show()

#histogram with matplotlib
# ax = plt.axes()
# ax.hist(df.petal_length, bins=25)
#
# ax.set(xlabel='Petal Length (cm)', ylabel='Frequency', title='Distribution of petal lengths')
# plt.show()

# #pandas plotting functionality
# ax = df.petal_length.plot.hist(bins=25)
# ax.set(xlabel='Petal Length (cm)',
#        ylabel='Frequency',
#        title='Distribution of Petal Lengths')
# plt.show()


# #single plot of multiple histograms using seaborn
# sns.set_context('notebook')
# # Parameters:
# #     context: dict, or one of {paper, notebook, talk, poster}
# #         A dictionary of parameters or the name of a preconfigured set.
# ## BEGIN SOLUTION
# #This uses the `.plot.hist` method
# ax = df.plot.hist(bins=25, alpha=0.5)
# ax.set_xlabel('Size (cm)')
# print(ax.get_subplotspec().is_last_row())
# plt.show()

# #multiple plots with pandas.hist
# # To create four separate plots, use Pandas `.hist` method
# axList = df.hist(bins=25)
# # Add some x- and y- labels to first column and last row
# for ax in axList.flatten():
#     if ax.get_subplotspec().is_last_row():
#         ax.set_xlabel('Size (cm)')
#
#     if ax.get_subplotspec().is_first_col():
#         ax.set_ylabel('Frequency')
# plt.show()
# ### END SOLUTION

#Create separate box plots by feature
### BEGIN SOLUTION
# Here we have four separate plots
#df.boxplot(by='species');
### END SOLUTION

print('!!!!!!!!!!!!')
sm = DistStats()
# #normalisation funcs test
# print(sm.display_skewed_columns(df, skew_tolerance=0.2))
# print('£££')
# x=sm.norm_transform(df, skew_tolerance=0.2,n_type='sqrt')
# print('after sqrt skew correction')
# print(x.skew())
# x=sm.norm_transform(df, skew_tolerance=0.2, n_type='log1p')
# print('after log1p skew correction')
# print(x.skew())
# x=sm.norm_transform(df, skew_tolerance=0.2, n_type='log')
# print('after log1p skew correction')
# print(x.skew())

x = data.copy()
# print(x['sepal_length'].skew())
# x['sepal_length'] = boxcox(data['sepal_length'])[0]
# print(x['sepal_length'])
g = sm.norm_transform(x, n_type='boxcox', exclude_cols=['sepal_length'], skew_tolerance=0)
print(g)

info = InfoModel()
print(info.df_datatypes(x))
