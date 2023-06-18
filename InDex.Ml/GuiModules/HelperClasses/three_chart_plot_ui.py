import pandas as pd
from PyQt5.QtWidgets import QWidget

from MLModules.visualisations_module import Visualizations

class ThreeChartWindow(QWidget, Visualizations):
    def __init__(self, df:pd.DataFrame, column:str):
        super().__init__()

        self.three_chart_plot(df, column)