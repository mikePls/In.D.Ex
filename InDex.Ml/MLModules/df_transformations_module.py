
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from scipy.stats import iqr, boxcox

import seaborn as sns
import skillsnetwork
import numpy as np
import os

class TransformationsHandler:
    def boxcox_transform(self, data:np.ndarray):
        #transformed df is boxcox[0]
        # find a way to store boxcox[1] for each column; the lambda that maximizes the log-likelihood function.
        if not (data <= 0).any():
            return boxcox(data)[0]
        else:
            return

    def fill_na_vals(self, df, column:str=None, val=0, method=None):
        #acceptable methods: 'backfill'/'bfill', 'pad'/'ffill'
        return df.fillna(val, method=method) if column is None else df[column].fillna(method=method)

    def drop_column(self, df, column:str):
        return df.drop([column], axis=1)

    def one_hot_dummies(self, df, columns:list=None, drop_first=False):
        '''Accepts a pandas dataframe; optionally a list of column names as strings. Generates dummy columns
        of binary values for given dataframe and columns. Returns the transformed dataframe.'''
        return pd.get_dummies(df, drop_first=False) if not columns else \
            pd.get_dummies(data=df, columns=columns, drop_first=drop_first)

    def one_hot_column_transformer(self, df, categorical_columns:list, drop=None):
        '''ColumnTransformer that allows different columns or column subsets to be transformed separately.'''
        # The transformers list is the number of tuples. The list of (name, transformer, columns)
        # tuples specify the transformer objects to be applied to the subsets of the data.
        # drop values: 'first', 'if_binary'
        transformer_model = ColumnTransformer(transformers=[("one_hot", OneHotEncoder(drop=drop),[categorical_columns])],
                                              remainder='passthrough')
        return transformer_model.fit_transform(df)

    def one_hot_transformer(self, data:pd.DataFrame, columns:list=None, drop=None):
        """Accepts a Pandas DataFrame and list of column names. Creates descriptive binary columns
        for each unique value of the dataframe columns. Returns a dataframe with the transformed columns"""
        # Initialize the OneHotEncoder object
        encoder = OneHotEncoder(drop=drop)

        # Fit and transform the data
        one_hot_data = encoder.fit_transform(data).toarray()

        # Convert the one-hot encoded data back to a dataframe
        columns = encoder.get_feature_names_out()
        return pd.DataFrame(one_hot_data, columns=columns)

    def poly_feat(self, df, degree:int, features:list, interaction_only=False, include_bias=False):
        '''Takes a pandas dataframe, a degree: int, and a list of features:string. Generates polynomial
        features according to given degree. Returns a dataframe of the new features.'''
        pf = PolynomialFeatures(degree=degree,
                                interaction_only=interaction_only,
                                include_bias=include_bias)
        pf.fit(df[features])
        #get original feature names with pf.get_feature_names()
        features_array = pf.transform(df[features])
        poly_df=None
        try:
            poly_df = pd.DataFrame(features_array, columns=pf.get_feature_names_out(input_features=features))
            poly_df.drop(features, axis=1, inplace=True)
        except Exception as e:
            print(e)
        return  poly_df

    def replace_col_vals(self, df, column:str='',vals:tuple=()):
        '''Takes a pandas dataframe, a column name as string, and a tuple of two string values.
        Replaces column values equal to the first value in the tuple, with the new value.'''
        return df[column].replace(vals[0], vals[1])

    def add_deviation_feature(self, df, feature, category):
        '''Accepts a pandas dataframe, a feature name and a category(-secondary feature) name. Returns
        a column that captures where feature's values lie relatively to the members of given category'''
        #example: add_deviation_feature(df, 'Year Built', 'House Style')

        # temp groupby object
        category_gb = df.groupby(category)[feature]

        # create category means and standard deviations for each observation
        category_mean = category_gb.transform(lambda x: x.mean())
        category_std = category_gb.transform(lambda x: x.std())

        # compute stds from category mean for each feature value,
        # add to X as new feature
        deviation_feature = (df[feature] - category_mean) / category_std
        df[feature + '_Dev_' + category] = deviation_feature

    def merge_categorical_vals(self, df, column:str, x:str, y:str):
        '''Replaces categorical values x in given column, with y. All other values remain the same.
        Returns an array of the transformed column.'''
        return np.where(df[column] == x, y, df[column])

    def label_encoding(self, df, column:str, vals:dict=None):
        #custom alternative to sklearn.preprocessing.LabelEncoder that can accept a dictionary
        #for replacing values with user defined range
        if not vals:
            i=0
            for value in df[column].unique().tolist():
                i +=1
                df[column].replace({value:i}, inplace=True)
            return df[column]
        return df[column].replace(vals)

    def scale(self, df, cols:list, s_type:str='standard'):
        scalers = {'standard': StandardScaler,
                   'minmax': MinMaxScaler,
                   'maxabs': MaxAbsScaler}
        if s_type not in scalers.keys():
            return
        return pd.DataFrame(columns=cols, data=scalers[s_type]().fit_transform(df[cols]))

    def split_categorical_col(self, df, col:str, splitter:str=' ', return_x_cols:int=None):
        '''Accepts a Pandas dataframe, column value as string, and a splitter string. Splits given column of dataset
        into individual columns based on given splitter (default value is one space). Returns a dataframe of the
        new columns'''
        try:
            data=pd.concat([c for c in df[col].str.split(splitter).str], axis=1)
            data.columns = [f"{col}" + str(i) for i in range(1, len(data.columns.tolist())+1)]
            if not return_x_cols:
                return data
            if return_x_cols > data.shape[1]:
                return_x_cols = data.shape[1] - 1
            return data.iloc[:, :return_x_cols]
        except Exception as e:
            print(e)

