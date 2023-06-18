
"""
EDA: identify patterns and trends in df

Common techniques:
->sampling:
    #sample of 5 rows without replacement
    sample = df.sample(n=5, replace=False)
    print(sample.iloc[:, -3:])


->visualizing a Pandas df
    #matplotlib.pyplot as plt
    plt.plot(df['ColumnName'], df['ColumnName'], ls='', marker='o')  #ls=line style ('', -, --)

    #histogram
    plt.hist(df.sepal_lengtrh, bins=25

    #matplotlib syntax
        fig, ax = plt.subplots()
        ax.barh(np.arange(10), df.ColumnName.iloc[:10]) #barh for plotting a horizontal bar plot
        --->np.arrange returns evenly spaced values within given interval
        ***arange function with single R***

        #set position of ticks and tick labels:
            ax.set_yticks(np.arange(0.4, 10.4, 1.0)) arrange with start, stop, interval parameters
            ax.set_yticklabels(np.arange(1,11))
            ax.set(xlabel='xlabel', ylabel='ylabel', title='Title')

    #visualize through Pandas df
        df.groupby('ColumnName').mean()
            .plot(color=['red', 'blue', 'black', 'green'], fontsize=10.0, figsize=(4,4))

    #visualize with seaborn as sns
        #create scatter plots between all features / examine feature correlations
        sns.pairplot(df, hue='Species', size=3, diag_kind="hist")
            #hue='Species' defines features by colour
            #diag_kind="hist" defines diagonal graph to histogram

        #Seaborn's hexbin plot / similar to heatmap / shows density
            sns.jointplot(x=df['ColumnName'],
                            y=df['ColumnName'],
                            kind='hex')

        #FACET GRID ------> break up df to custom plot
            plot = sns.FacetGrid(df,
                                feature='Species',
                                margin_titles=True)
            plot.map(plt.hist, 'ColumnName', color='green')


slicing df
df[2:4] ----> rows 2 and 3 ---->think like df.log[from_row:to_row]
df.loc[:['col1', 'col2']] ---->all rows from specific columns
df.loc[6,['feature']] ---->row 6 of column
read as df.log[from_row:to_row]

index location example
df.iloc[2:5, [0,3]] ---->rows 2 to 5 and columns 0 TO 3(3rd non-inclusive)

conditional filtering
df[df.column_name > 1.0]
df[df.column_name.isin([str1, str1, str3, ...])]

Assignment -- similar to slicing
df.loc[3, ['col_name']] = some_value
df.loc[3, ['col_name']] = np.nan

create array and assign to column:
new_array = np.array([5] * len(df))
df[:,'col_name'] = new_array

create new column from values of other columns
df['new_col'] = (df['col1'] + df['col2']) / 2

Renaming columns
df.rename(columns = {'old_name':'new_name'}, inplace=True)
or assign as df=df.rename with inplace set to False

rename all columns at once
df.columns = ['name1', 'name2',...]

in case iteration is needed
for index, row in df.iterrows():
    #do something

save to csv
df.to_csv('file_name.csv')


"""
