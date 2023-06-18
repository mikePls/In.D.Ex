import asyncio

import numpy as np
import skillsnetwork
import pandas as pd
import skillsnetwork
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats


#accepts a numeric list and returns a list of the outliers if exist
def detect_outliers(data = []):
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    iqr = q75 - q25
    min = q25 - 1.5 * iqr
    max = q75 + 1.5 * iqr

    #print(min, q25, q50, q75, return_x_cols)
    return [x for x in data if x < min or x > max]

# df = [-87, 1, 13, 23, 25, 41, 48, 65, 71, 80, 104, 206, 400]
# print(detect_outliers(df))


#declare async func
async def download(url):
    await skillsnetwork.download_dataset(url)

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv'
#call async:
#asyncio.run(download(URL))
housing = pd.read_csv('Ames_Housing_Data1.tsv', sep='\t')

# print(housing.head(10))
# print(housing.info())
# print('...')
# print(housing.SalePrice.describe())

#Pearson Correlation
# hous_num = housing.select_dtypes(include = ['float64', 'int64'])
# hous_num_corr = hous_num.corr()['SalePrice'][:-1] # -1 means that the latest row is SalePrice
# top_features = hous_num_corr[abs(hous_num_corr) > 0.5].sort_values(ascending=False) #displays pearsons correlation coefficient greater than 0.5
# print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))

#pairplot loop for each attribute to SalePrice correlation
# for i in range(0, len(hous_num.columns), 5):
#     sns.pairplot(df=hous_num,
#                 x_vars=hous_num.columns[i:i+5],
#                 y_vars=['SalePrice'])

#distribution plot; set_checked for normal or skewed distribution
# sp_untransformed = sns.displot(housing['SalePrice'])
# plt.show()

#skewness and kurtosis (set_checked notes)
# print("Skewness: %f" % housing['SalePrice'].skew())
# print("Kurtosis: %f" % housing['SalePrice'].kurt())

#log transformation
# sale_price_transformed = np.log(housing.SalePrice)
# sp_transformed = sns.displot(sale_price_transformed)
# plt.show()
# print("Skewness after log transformation:", sale_price_transformed.skew())

#find and remove duplicates
# duplicates = housing[housing.duplicated(['PID'])]
# print(duplicates)
# dup_removed = housing['PID'].drop_duplicates()
# print(housing.index.is_unique)

#handle null values
# total = housing.get_nulls().sum().sort_values(ascending=False)
#
# median = housing['Lot Frontage'].median()
# housing['Lot Frontage'].fillna(median, inplace=True)
#
# print(housing['Lot Frontage'].get_nulls().sum())

#z-scores
housing['Lot_Area_Stats'] = stats.zscore(housing['Lot Area'])
hous_disc = housing[['Lot Area', 'Lot_Area_Stats']].describe().round(3)
print(hous_disc)
