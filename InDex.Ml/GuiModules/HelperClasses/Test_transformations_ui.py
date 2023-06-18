import sys

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', 30)
from PyQt5.QtWidgets import QApplication

from DataManagement.data_manager import DataManager
from GuiModules.HelperClasses.transformations_ui import TransformationsCard, TransformationsWindow

df=pd.read_csv('CarPrice_Assignment.csv')


# app = QApplication(sys.argv)
# dm = DataManager(df)
# window = TransformationsCard(name='price')
# window.show()
# sys.exit(app.exec())

data = df[['enginesize','carlength','carwidth','carheight','carbody','price']].copy()
app = QApplication(sys.argv)
dm = DataManager(data)
window = TransformationsWindow(columns=['price','carwidth','carheight','carbody'])
window.show()
sys.exit(app.exec())


