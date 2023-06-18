import asyncio
import os.path
import urllib.request

import pandas as pd
import plotly.express as px
import datetime
import requests
import json
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler

import Modules.distribution_stats_module

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

def download_dataset():
    urllib.request.urlretrieve("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/18100001.csv",
                               "Canada_Provinces.geojson")

if not os.path.exists("Canada_Provinces.geojson"):
    download_dataset()

data = pd.read_csv("Canada_Provinces.geojson")
###----------------1. UNDERSTANDING THE DATA - INITIAL EXPLORATION----------------
# print(df.shape)
# print(df.describe())
#df.GEO.unique().tolist()
# can also use  df['GEO'].unique().tolist()
#print(df.VALUE.describe())
# print(df.columns)
# print(df.info())
#
# print(df.get_nulls().sum())
#-----------------------------------------------------------------------
###------------------------ 2. DATA WRANGLING---------------------------

#select relevant columns & rename them
data = data[['REF_DATE', 'GEO', 'Type of fuel', 'VALUE']].rename(columns={'REF_DATE':'DATE', 'Type of fuel':'TYPE'})
#print(df.head())

#Split GEO column values into two separate columns by commas. n=num of splits, expand returns a df
data[['City', 'Province']] = data['GEO'].str.split(',', n=1, expand=True)
print(data.head())

#Change column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'], format='%b-%y')
data['Month'] = data['DATE'].dt.month_name().str.slice(stop=3)
data['Year'] = data['DATE'].dt.year
#-----------------------------------------------------------------
###-----------------------3 DATA FILTERING------------------------

#All logical operators apply <>=! | &
calgary = data[data['GEO'] == 'Calgary, Alberta']
one_year = data[data['Year'] == 2021]
print(calgary)

#filter by multiple conditions
mult_loc = data[(data['GEO'] == "Toronto, Ontario") | (data['GEO'] == "Edmonton, Alberta")]
vanc_data = data[(data['City'] == 'Vancouver') & (data['TYPE'] == 'Household heating fuel') & (data['Year'] == 1990)]
filt_data=data[( data['Year'] <=  1979) | ( data['Year'] ==  2021) & (data['TYPE'] == "Household heating fuel") & (data['City']=='Vancouver')]

#filter by isin() method:
cities = ['Calgary', 'Toronto', 'Edmonton']
CTE = data[data.City.isin(cities)]

#groupby() method gives number of categories by column
geo=data.groupby('GEO')
group_year = data.groupby(['Year'])['VALUE'].mean()
#------------------------------------------------------------
###-----------------VISUALISING WITH plotly.express----------
# price_bycity = df.groupby(['Year', 'GEO'])['VALUE'].mean().reset_index(name='Value').round(2)
# fig = px.line(price_bycity,x='Year',
#               y = "Value", color = "GEO",
#               color_discrete_sequence=px.colors.qualitative.Light24)
# fig.update_traces(mode='markers+lines')
# fig.update_layout(
#     title="Gasoline Price Trend per City",
#     xaxis_title="Year",
#     yaxis_title="Annual Average Price, Cents per Litre")
# fig.show()

#2nd example
# mon_trend = df[(df['Year'] ==  2021) & (df['GEO'] == "Toronto, Ontario")]
# group_month = mon_trend.groupby(['Month'])['VALUE'].mean().reset_index().sort_values(by="VALUE")
# fig = px.line(group_month,
#                    x='Month', y = "VALUE")
# fig.update_traces(mode='markers+lines')
# fig.update_layout(
#     title="Toronto Average Monthly Gasoline Price in 2021",
#     xaxis_title="Month",
#     yaxis_title="Monthly Price, Cents per Litre")
# fig.show()

#3rd example: animated frame
# bycity = df.groupby(['Year', 'City'])['VALUE'].mean().reset_index(name ='Value').round(2)
# fig = px.bar(bycity,
#             x='City', y = "Value", animation_frame="Year")
# fig.update_layout(
#     title="Time Lapse of Average Price of Gasoline, by Province",
#     xaxis_title="Year",
#     yaxis_title="Average Price of Gasoline, Cents per Litre")
#
# fig.show()
#-------------------------------------------------------------

###----------------FEATURE ENGINEERING---------------
#Linear Regression models assume linear relationships between observations and target variable
#Transformations:
#log, log1p, boxcox

#polynomial features - uncovering new or strengthening feature relationships:
polyfeat = PolynomialFeatures(degree=2)

##----->feature encoding - for converting categorical features to numerical:
#*Nominal values (unordered categories e.g., Blue, Green, True, False):
#Binary encoding: converting values to 1s and 0s only
#One-hot-encoding: convert values into multiple columns of binary encoding (creates new variables)

#*Ordinal values (ordered categories e.g.,High, Medium, Low):
#Ordinal encoding: convert ordered categories to numerical values (High=10, Medium=5, Low=0)


#*Continuous values: see feature scaling


##------>feature scaling
#scaling - scale all numeric features, so they are comparable
#**standard scaling: convert features to standard normal variables (subtract mean, divide by stnd err)
#**min-return_x_cols scaling: variables converted within 0-1 interval (sensitive to outliers)
#**Robust scaling: similar to min-return_x_cols: focuses on interquartile range: less sensitive to outliers




