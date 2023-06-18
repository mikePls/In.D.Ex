import sys

from column_module import ColumnListComponent, ColumnManager
from PyQt5.QtWidgets import QApplication
import pandas as pd

from DataManagement.data_manager import DataManager

data = pd.read_csv('HelperClasses/MOCK_DATA.csv')

# application = QApplication(sys.argv)
# window = ColumnListComponent(['Length', 'Weight', 'Price', 'MDV'])
# window.show()
# sys.exit(application.exec())

app = QApplication(sys.argv)
#data = data[['Length','Weight']]
dm = DataManager(data)
window = ColumnManager()
window.show()
sys.exit(app.exec())