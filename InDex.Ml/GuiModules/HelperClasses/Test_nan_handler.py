import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication
from pandas import DataFrame

from DataManagement.data_manager import DataManager
from GuiModules.HelperClasses.nan_values_handler_ui import NaValuesHandler
from GuiModules.HelperClasses.outliers_handler_ui import OutlierCard, OutliersUi

#df=pd.read_csv('CarPrice_Assignment.csv')

df = DataFrame(data=[[1,2,3,4,None],[5,6,7,8,'B'],[None,10,11,12,'C'],[13,14,15,16,'D']],
               columns=['A1','B1','C1','D1','E1'])

app = QApplication(sys.argv)
dm = DataManager(df)
window = NaValuesHandler(df=df)
window.show()
sys.exit(app.exec())