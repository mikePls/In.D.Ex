import sys

from info_manager import InfoManager
from PyQt5.QtWidgets import QApplication
import pandas as pd

data = pd.read_csv('../Notes/CarPrice_Assignment.csv')
data.info()

application = QApplication(sys.argv)
window = InfoManager(data)
window.show()
sys.exit(application.exec())