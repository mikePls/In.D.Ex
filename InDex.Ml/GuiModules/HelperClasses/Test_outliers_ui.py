import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication

from DataManagement.data_manager import DataManager
from GuiModules.HelperClasses.outliers_handler_ui import OutlierCard, OutliersUi

df=pd.read_csv('CarPrice_Assignment.csv')


app = QApplication(sys.argv)
dm = DataManager(df)
#window = OutlierCard('price')
window = OutliersUi(columns=['price', 'horsepower'])
window.show()
sys.exit(app.exec())