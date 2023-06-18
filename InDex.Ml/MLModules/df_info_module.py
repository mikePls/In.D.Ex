import pandas as pd
from pandas import core
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class InfoModel:
    '''Includes functions that accept and provide various information for pandas dataframes'''


    def isnull(self, df):
        '''Accepts a pandas dataframe, returns a frame of two columns: column names of given dataframe
            and number of null values.'''
        return df.get_nulls().sum().sort_values(ascending=False)

    def shape(self, df):
        return f"columns:{df.shape[1]}, rows:{df.shape[0]}"

    def describe(self, df):
        return df.describe().T

    def df_datatypes(self, df:pd.DataFrame):
        return df.dtypes.value_counts()

    def info(self, df):
        return df.info()

    def pairplot(self, df):
        '''Displays a Seaborn pair plot, of given pandas dataframe features '''
        with sns.plotting_context(rc={"axes.labelsize": 6}):
            sns.pairplot(df, plot_kws=dict(alpha=.1, edgecolor='none'), height=1)
            sns.set(font_scale=1)
            plt.yticks(fontsize=1)
        plt.show()

    def column_val_count(self, df, column: str):
        return df[column].value_counts()

    def get_unique_col_vals(self, df, column:str=None):
        '''Accepts a pandas dataframe and column name as string.
        Returns a list of all unique values in given dataframe and column.'''
        return df[column].unique().tolist()

    def get_duplicates(self, df, column:str=None):
        '''Accepts a Pandas dataframe and column name as string. Returns the number of duplicate values.'''
        return sum(df.duplicated(subset=column))

    def get_correlations(self, df, column:str=None):
        return df.corr()[column].sort_values()

    def get_unique_count(self, df, column:str):
        '''Accepts a Pandas dataframe and column name as string. Returns the count for each unique entry in
        the dataset as a Pandas series'''
        return df.groupby(column)[column].nunique()
