import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from pandas import DataFrame, Series
from scipy import stats
import matplotlib.style as style
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas

class Visualizations():

    def hist(self, df, feature):
        df[feature].hist()
        plt.show()

    def heatmap(self, df):
        plt.figure(figsize=(15,15))
        sns.heatmap(df, annot=True, cmap='RdYlGn')
        plt.show()

    def bar_corr(self, df):
        return df.plot(kind='bar', figsize=(10,8))

    def xy_corr_hexplot(self, df, x:str, y:str):
        return sns.jointplot(x=df[x], y=df[y], kind='hex')

    def countplot_categorical_feature(self, df, feature:str):
        fig, ax = plt.subplots(figsize=(15,5))
        plt1 = sns.countplot(x=df[feature], order=pd.value_counts(df[feature]).index)
        plt1.set(xlabel=feature, ylabel='Count')
        plt.tight_layout()

    def explore_feature_linearity(self, df, x:str, y:str):
        fig, ax1 = plt.subplots(figsize=(5, 5))
        sns.scatterplot(x=df[x], y=df[y], ax=ax1)
        sns.regplot(x=df[x], y=df[y], ax=ax1)

    def three_chart_plot(self, df, feature):
        """Displays a window with a histogram, probability plot, and box plot for the specified feature from the provided dataframe.

        Parameters:
        - df: The dataframe containing the data.
        - feature: The name of the feature to plot.

        The function generates and shows three plots: a histogram representing the data distribution, a probability plot for assessing normality, and a box plot displaying outliers, quartiles, and the median.

        Returns:
        None
        """
        style.use('fivethirtyeight')

        # Create customised chart
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        # create grid of 1 col and 3 rows
        grid = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)

        # Customise histogram grid
        ax1 = fig.add_subplot(grid[0, :2])
        # Set plot title.
        ax1.set_title('Histogram')
        # plot histogram
        sns.distplot(df.loc[:, feature], norm_hist=True, ax=ax1)

        # customise the QQ-plot.
        ax2 = fig.add_subplot(grid[1, :2])
        # set plot title
        ax2.set_title('QQ_plot')
        ## Plot QQ_Plot.
        stats.probplot(df.loc[:, feature], plot=ax2)

        # Customise the Box Plot
        ax3 = fig.add_subplot(grid[2, :2])
        # Set plot title
        ax3.set_title('Box Plot')
        # Plot box plot
        sns.boxplot(df.loc[:, feature], orient='h', ax=ax3)

        plt.show()

    def create_boxplot(self, data:Series, low_iqr_bound=1.5, high_iqr_bound=1.5, canvas:FigCanvas=None) -> FigureCanvasQTAgg:
        """Accepts a Pandas Series object, returns a QtFigure canvas with a boxplot
        of the accepted data"""

        fig, ax = plt.subplots(figsize=(2.5, 4))
        ax.boxplot(data)
        ax.grid()
        # Calculate the quartiles and interquartile range
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        low_outlier = q1 - (low_iqr_bound * iqr)
        high_outlier = q3 + (high_iqr_bound * iqr)
        #Determine outliers
        outliers = data[(data < low_outlier) | (data > high_outlier)]
        #plot outliers as red circles
        ax.plot(np.ones(len(outliers)), outliers, 'ro')

        fig.subplots_adjust(0.2, 0.1, 0.8, 0.9)  # left,bottom,right,top
        plt.close()

        if canvas:                #if an existing canvas is passed:
            canvas.figure.clear() ##   clear canvas
            canvas.figure = fig   ## replace the figure
            canvas.draw()         ## draw new figure
            return canvas         ## return the updated canvas
        else:                        #else if no canvas exists:
            canvas = FigCanvas(fig)  ## create new canvas object with figure as parameter
            canvas.draw()            ## draw the figure
        return canvas                ## return new canvas