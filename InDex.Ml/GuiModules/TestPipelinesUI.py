import sys

from PyQt5.QtWidgets import QApplication
import pandas as pd

from DataManagement.data_manager import DataManager
from GuiModules.pipelines_ui_module import PipelineManagerUI

data = pd.read_csv('HelperClasses/MOCK_DATA.csv')

df = pd.DataFrame(data=[[1,2,3,4,None],[5,6,7,8,'B'],[None,10,11,12,'C'],[13,14,15,16,'D']],
               columns=['A1','B1','C1','D1','E1'])

app = QApplication(sys.argv)
dm = DataManager(df)
window = PipelineManagerUI()
window.show()
sys.exit(app.exec())