import io

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QSizePolicy, QGroupBox, QScrollArea
from pandas import DataFrame

from DataManagement.data_manager import DataManager


class InfoManager(QGroupBox):
    def __init__(self):
        super().__init__()
        self.setup()
        self.setTitle('Dataset info')

        self.dm:DataManager = DataManager.get_instance()
        if self.dm is not None:
            self.setData(self.dm.get_data())
            print('InfoManager: Data received')

    def setData(self, data:DataFrame):
        buf = io.StringIO()
        data.info(buf=buf)
        self.label.setText(buf.getvalue())
        #print(buf.getvalue().format('='))
        buf.close()

    def wipe(self):
        self.label.setText('')

    def refresh(self):
        self.wipe()
        self.setData(self.dm.get_data())


    def setup(self):
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.label)
        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        font = QFont("Consolas", 10)
        self.label.setFont(font)
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setMaximumWidth(700)
