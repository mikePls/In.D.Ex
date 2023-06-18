"""
#reading csv values
df = pd.read_csv(filepath="", sep="")

->set_checked other params like header
->use "\t" for tab sep or delim_whitespace=True for space sep values
->for using/not using first row as header: header=None
->specify column names as names=['Name1', 'Name2', ...]
->replace CUSTOM missing values: na_values=['000', 'N/A']


#reading json
df = pd.read_json(filepath='')
#convert dataframe to json
df.to_json('output-example.json')

#Information for dataframe
df.ColumnName.describe() for statistical information
df.ColumnName.value_counts() for object(categorical attributes) information


#----->DATA CLEANING<--------
#detect outliers:
    #with interquartile range:
        import numpy as np
        -calculate interquartile range:
        q25, q50, q75 = np.percentile(df, [25, 50, 75])
        iqr = q75 - q25

        -calculate min/return_x_cols limits to be considered an outlier
        min = q25 - 1.5*(iqr)
        return_x_cols = q75 + 1.5*(iqr)

        -outliers = [x for x in df if x < min or x > return_x_cols]

    #with residuals(difference between actual and predicted values):
        Calculate residuals approaches:
        ->Standardized: residual divided by standard error(SE of standard deviation)
        ->deleted: residual from fitting model on all df excluding current observation
        ->Studentized: Standardized deleted residuals (divide by residual standard error)
                (based on all df or df excluding current observation)


#Policies for outliers:
    #Remove
    #Assign mean or median
    #Transform column (e.g., log transformation)
    #Predict (what it would be by using similar observations or regression)
    #Do nothing (depending on dataset outlier might give insight)


#dupbicate df policies:
    #remove row
    #impute: replace or substitute values with most common or average
    #mask: create separate category for missing values

#missing values


------CORRELATIONS------
#Find correlation between target variable and features
    Approaches:
        Pair plots
        Scatter plot
        Heat map
        Correlation matrix
        Pearson correlation (for numeric values only)

    #Pearson Correlation
#select numeric columns
hous_num = housing.select_dtypes(include = ['float64', 'int64'])
hous_num_corr = hous_num.corr()['SalePrice'][:-1] # -1 means that the latest row is SalePrice
#displays pearsons correlation coefficient greater than 0.5
top_features = hous_num_corr[abs(hous_num_corr) > 0.5].sort_values(ascending=False)



-------->PLOTS<---------
using seaborn as sns

#ref for detecting outliers
distplot: sns.distplot(df, bins=20)
boxplot: sns.boxplot(df)

        plot example:
            total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)
            plt.xlabel("Columns", fontsize = 20)
            plt.ylabel("Count", fontsize = 20)
            plt.title("Total Missing Values", fontsize = 20)


---------->DATA DISTRIBUTION AND TRANSFORMATION<------------
-Normal distribution to be achieved using any type of regression analysis

-Approaches for checking distribution normality/skewness/kurtosis
    ----->sns.distplot(df['column'])

    ----->dataframe['ColumnName'].skew()
            #skewness:
                -0.5 to 0.5:
                -1 to -0.5 or 0.5 to 1: moderately skewed
                less than -1 or higher than 1: highly skewed
        dataframe['ColumnName'].kurt()
            #kurtosis > +2 indicates high peaking

            #Transformation
                --->Log transformation:
                    with numpy as np
                    np.log(dataframe['ColumnName'])





-------->HANDLING DUPLICATES<----------
    with pandas dataframe (referred as df)
    #Search for duplicates
        duplicates_df = df[df.duplicated(['ColumnName'])]

        #Drop duplicates
            df_without_dups = df.drop_duplicates()

        #set_checked each value in column is unique
            df.index.is_unique -->returns a bool / not callable()


-------->HANDLING MISSING VALUES<----------
    pandas functions: isna(), get_nulls(), notna()

    #example for finding count of null values
        total = df.get_nulls().sum().sort_values(ascending=False)

    #dropna in specific column
        df.dropna(subset=['ColumnName'])

    #drop entire column
        df.drop('ColumnName', axis=1)

    #replace missing values
        with median (or zero, mean, etc)
            median = df['ColumnName'].median()
            df.['ColumnName'].fillna(median, inplace = True) #False inplace will return replaced dataset, but not overwrite


------->FEATURE SCALING - TRANSFORMATION<-------
    Data transformations: solving skewed distribution
    Normalization: Data does not have Gaussian/normal distribution
    Standardization: For df that HAS normal distribution

    #min-return_x_cols scaling/normalization: simplest approach: all values range between 0-1
        df = df of columns with numeric values
        norm_data = MinMaxScaler().fit_transform(df)

    #standardization: mean for each value is 0
        df = df of columns with numeric values
        scaled_df = StandardScaler().fit_transform(df)

        -----> for single column:
            scaled_data = StandardScaler().fit_transform(dataframe['ColumnName'][:,np.newaxis])

    #useful transformation functions:
    log, log1p(adds 1 if there are zeros in dataset)
    boxcox: complex way for finding ideal transformation


--------------->HANDLING OUTLIERS<----------------

    -----DETECT OUTLIERS-----
    #Uni-variate analysis (one variable analysis)
        visual detection with seaborn box plot:
            sns.boxplot(x=dataframe['ColumnName'])

    #Multi-variate analysis (two or more variable analysis)
        visual detection with scatter plot
            dataframe.plot.scatter(x='ColumnName', y='Column2Name')

    #Z-score: how many standard deviations a value has from from normal distribution (-3 to 3 normal limits)
        #Create new column with zscore, to compare to actual column
        housing['Lot_Area_Stats'] = stats.zscore(housing['Lot Area'])

        Get column information - round numbers to 3 decimal points
        hous_disc = housing[['Lot Area', 'Lot_Area_Stats']].describe().round(3)





"""




"""
***************NOTES****************
to set_checked:
scaling vs transform, why and when

Normalization Techniques at a Glance
Four common normalization techniques may be useful:

    scaling to a range
    clipping
    log scaling
    z-score

"""







