import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

from GuiModules.HelperClasses.duplicates_module import DuplicateWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Duplicate Widget Test")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        dataset = pd.read_csv('CarPrice_Assignment.csv')

        self.layout = QVBoxLayout(self.central_widget)

        self.duplicate_widget = DuplicateWidget(dataset)
        self.layout.addWidget(self.duplicate_widget)

        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())