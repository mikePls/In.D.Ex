from typing import Union

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


class DataManager:
    _INSTANCE = None
    def __init__(self, data:pd.DataFrame=None, parent=None):
        if not DataManager._INSTANCE:
            DataManager._INSTANCE = self
        self.data = data
        self.parent = parent
        self.current_ml_model = None

    def set_data(self, data:pd.DataFrame):
        self.data = data

    def replace_col(self, col_name:str, data:pd.Series):
        self.data[col_name] = data


    def concat(self, df, position:int=None):
        """Concatenates current data horizontally, with a dataframe passed as an argument.
         Optionally accepts an index location to concatenate at a specific location"""
        try:
            if not position:
                self.data = pd.concat([self.data, df], axis=1)
            else:
                self.data = pd.concat([self.data.iloc[:,:position],df,self.data.iloc[:,position:]], axis=1)
        except Exception as e:
            print(e)

    def drop_column(self, col_name:str):
        try:
            self.data.drop(col_name, axis=1, inplace=True)
        except Exception as e:
            print(e)

    def print_columns(self):
        print(self.data.columns)

    def get_columns(self)->list:
        """Returns a list of strings, of the column names in the dataframe"""
        return self.data.columns.tolist()

    def get_col_contents(self, col_name:str):
        return self.data[col_name].values

    def get_col_index(self, col_name):
        return self.data.columns.get_indexer([col_name])[0]

    def has_data(self):
        return self.data is not None

    def get_duplicate_count(self):
        return self.data.duplicated().sum()

    def drop_duplicates(self):
        self.data.drop_duplicates(keep='first', inplace=True)

    @staticmethod
    def get_instance():
        return DataManager._INSTANCE

    def get_nulls(self):
        '''Accepts a pandas dataframe, returns a frame of two columns: column names of given dataframe
            and number of null values.'''
        return self.data.isnull().sum().sort_values(ascending=False) if self.has_data() else None

    def get_col_null_count(self, col:str)->int:
        return self.data[col].isnull().sum()

    def get_col_type(self, col:str)->np.dtype:
        return self.data[col].dtype

    @staticmethod
    def has_instance():
        return DataManager() is not None

    def shape(self):
        df = self.data
        return f"columns:{df.shape[1]}, rows:{df.shape[0]}" if self.has_data() else None

    def describe(self):
        return self.data.describe().T if self.has_data() else None

    def df_datatypes(self):
        return self.data.dtypes.value_counts() if self.has_data() else None

    def df_dtypes_list(self):
        return self.data.dtypes

    def get_data(self):
        return self.data

    def get_col_skew(self, col:str)->str:
        if self.data[col].dtype != object:
            return str(self.data[col].skew())
        else:
            return 'n/a'

    def info(self):
        return self.data.info() if self.has_data() else None

    def insert_col(self, index:int=None, title:str=None, data=None):
        self.data.insert(loc=index,column=title,value=data, allow_duplicates=True)

    def wipe(self):
        DataManager._INSTANCE = None
        self.data = None

    def set_current_model(self, model:Pipeline):
        self.current_ml_model = model

    def get_current_model(self)->Union[Pipeline,None]:
        return self.current_ml_model


