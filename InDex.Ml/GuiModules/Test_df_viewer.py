import sys

from GuiModules.dataframe_viewer import DfViewer
from info_manager import InfoManager
from PyQt5.QtWidgets import QApplication
import pandas as pd

data = pd.read_csv('../Notes/CarPrice_Assignment.csv')

application = QApplication(sys.argv)
window = DfViewer(data)
window.show()
sys.exit(application.exec())