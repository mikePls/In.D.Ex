import sys
import numpy as np
from scale_transform_module import ScaleManager
from PyQt5.QtWidgets import QApplication
import pandas as pd
from DataManagement.data_manager import DataManager

#df:pd.DataFrame = pd.read_csv('CarPrice_Assignment.csv')

df = pd.DataFrame(columns=['c1','c2', 'c3','c4','c5','c6'], data=[[1,2,3,4,'E',3],['A','B','C','D','E',5],['f','g','h','i','j',7]])
data = df.copy()
dm = DataManager(df)
app = QApplication(sys.argv)
win = ScaleManager(['c6'])
win.show()
sys.exit(app.exec())


