import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.special import inv_boxcox
from scipy.stats import boxcox, iqr, mode
from scipy.stats.mstats import normaltest


class DistStats:
    '''
    Provides methods for displaying and transforming skewed columns
    in pandas dataframes.
    '''

    def boxcox_transform(self, data):
        #transformed df is boxcox[0]
        # find a way to store boxcox[1] for each column; the lmbda that maximizes the log-likelihood function.
        return boxcox(data)[0]

    def invert_transform(self, data:ndarray, l_func):
        #fetch lamda from boxcox transform
        return inv_boxcox(data, l_func)

    def display_skewed_columns(self, data, skew_tolerance=0.75):
        '''Takes a pandas dataframe and optionally minimum acceptable skewness,
            returns a frame of column names and skewness values'''
        skewed_columns = (self.get_skewness(data)
                          .sort_values(ascending=False)
                          .to_frame()
                          .rename(columns={0:'Skew'})
                          .query('abs(Skew) > {}'.format(skew_tolerance)))
        return skewed_columns


    def get_skewness(self, data):
        '''Takes a pandas dataframe, returns a dataframe of all columns of float type'''
        mask = data.dtypes == float if data is not None else None
        float_column_names = data.columns[mask]
        return data[float_column_names].skew()

    def log1p_transform_column(self, data, column=''):
        '''Takes a pandas dataframe and str for column name, applies numpy's log1p to
            selected column and returns it'''
        return data[column].apply(np.log1p)

    def show_column_distribution(self, data, column=''):
        '''Takes a pandas dataframe and str for column name, displays a pyplot graph
            of selected column's distribution'''
        fig, ax = plt.subplots(1,1,figsize=(10,5))
        data[column].hist(ax=ax)
        ax.set(title=column, ylabel='frequency', xlabel='value')
        fig.suptitle('Field "{}"'.format(column))
        plt.show()

    #design the function for additional log transformations
    def norm_transform(self, data:pd.DataFrame, exclude_cols:list=None, skew_tolerance=0.75, n_type='log')->DataFrame:
        '''Takes a pandas dataframe, a tuple of str and optionally minimum accepted skewness. Applies
            log1p to all except excluded columns. Returns transformed dataframe.'''
        #remove object columns from data
        data = data.select_dtypes(exclude=['object'])
        log_types = {'log1p':np.log1p, 'log':np.log, 'sqrt':np.sqrt, 'boxcox': boxcox}
        dataframe = self.display_skewed_columns(data, skew_tolerance)
        if n_type== 'boxcox':
            if exclude_cols:
                data = data.drop(exclude_cols, axis=1)
            for col in data.columns:
                print('datatype of data', type(data[col]))
                data[col] = self.boxcox_transform(data[col])
            return data
        for col in dataframe.index.values:
            if col in exclude_cols:
                continue
            data[col] = data[col].apply(log_types[n_type])
            return data

    def normality_test(self, data, column:str, stat_type:str='pvalue'):
        '''Takes a pandas dataframe, a column name as str and stat_type as string. Returns a statistic
        or pvalue depending on type. Accepted values for stat_type: statistic, pvalue
        statistic: float
        s^2 + k^2, where s is the z-score returned by skewtest and k is the z-score returned by kurtosistest.
        pvalue: float
        A 2-sided chi squared probability for the hypothesis test.
        '''
        return normaltest(data[column].values)[0] if type == 'statistic' \
            else normaltest(data[column].values)[1]

    def count_iqr_outliers(self, data, high_iqr_bound, low_iqr_bound)->list:
        """Accepts a list of numbers, high and low bound parameters. Returns a tuple with the
        number of outliers that are above or below bound value multiplied by IQR"""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        low_outlier = q1 - (low_iqr_bound * iqr)
        high_outlier = q3 + (high_iqr_bound * iqr)
        outliers_above = outliers_below = 0
        for num in data:
            if num > high_outlier:
                outliers_above += 1
            elif num < low_outlier:
                outliers_below +=1
        return [outliers_above, outliers_below]

    def replace_outlier(self, val, mean, std):
        if val > mean + 3 * std:  # if val >, then lower it to 3stds from mean
            return mean + 3 * std
        elif val < mean - 3 * std:  # if val<, then increase it to -3stds from mean
            return mean - 3 * std
        return val  # else do nothing

    def get_outliers(self, data, factor=1.5):
        limit1 = np.quantile(data, 0.25) - factor * iqr(data)
        limit2 = np.quantile(data, 0.75) + factor * iqr(data)
        outliers = data[(data < limit1) | (data > limit2)]
        return outliers

    def winsorize_list_iqr(self, data, lower_iqr=1.5, upper_iqr=1.5):
        """
        Winsorizes a list by replacing outliers with the closest non-outlying value
        at a specified lower and upper IQR bound.

        Parameters:
            data (list): The list to winsorize.
            lower_iqr (float): The lower IQR bound. Defaults to 1.5.
            upper_iqr (float): The upper IQR bound. Defaults to 1.5.

        Returns:
            list: The winsorized list.
        """
        q1 = np.percentile(data, lower_iqr * 100)
        q3 = np.percentile(data, upper_iqr * 100)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        winsorized = np.clip(data, lower_bound, upper_bound)
        return winsorized

    def handle_outliers(self, data, upper_bound=1.5, lower_bound=1.5, method=None, custom_value=None):
        """
        Function to handle outliers in a numpy array.

        Parameters:
        data: numpy array - the input array.
        method: str - the method to handle outliers. Possible values are:
            - 'mean': replace with mean of non-outlier values.
            - 'mode': replace with mode of non-outlier values.
            - 'median': replace with median of non-outlier values.
            - 'windsorize': replace with the closest non-outlier value within bounds.
            - 'custom': replace with user-defined value.
        custom_value: float - the value to use when method='custom'.

        Returns:
        numpy array - the modified array with outliers handled.
        """
        # Calculate the lower and upper bounds for outliers
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lb = q1 - (lower_bound * iqr)
        ub = q3 + (upper_bound * iqr)

        # Determine the non-outlier values
        non_outlier_mask = (data > lb) & (data < ub)
        outliers_mask = (data < lb) | (data > ub)
        outliers = data[outliers_mask]
        non_outlier_values = data[non_outlier_mask]
        print(f'num of outliers in handle method:{len(data[outliers_mask])}')
        outlier_replacements = np.empty(data.shape)

        # Replace outliers using the selected method
        if method == 'mean':
            mean = non_outlier_values.mean()
            print('mean', mean)
            outlier_replacements.fill(mean)
        elif method == 'mode':
            outlier_replacements.fill(mode(non_outlier_values, keepdims=True)[0][0])
        elif method == 'median':
            outlier_replacements.fill(non_outlier_values.median())
        elif method == 'windsorize':
            for i, val in enumerate(data):
                if val < lb:
                    if len(non_outlier_values) > 0:
                        outlier_replacements[i] = non_outlier_values[np.argmin(np.abs(outliers - lb))]
                    else:
                        outlier_replacements[i] = val
                elif val > ub:
                    if len(non_outlier_values) > 0:
                        outlier_replacements[i] = non_outlier_values[np.argmin(np.abs(outliers - ub))]
                    else:
                        outlier_replacements[i] = val
                else:
                    outlier_replacements[i] = val
        elif method == 'custom':
            outlier_replacements.fill(float(custom_value))

        # Replace outliers with the selected
        data[outliers_mask] = outlier_replacements[outliers_mask]
        print(f'num of outliers in handle method after treatment:{(data[outliers_mask])}')
        return data

