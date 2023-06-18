import pandas as pd
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox
from PyQt5.QtCore import Qt
from DataManagement.data_manager import DataManager

class DuplicateWidget(QWidget):
    def __init__(self, data=None):
        super().__init__()
        self.setWindowTitle("Duplicate Rows")
        self.layout = QVBoxLayout()
        self.dm: DataManager = DataManager.get_instance()
        self.data = self.dm.get_data() if self.dm else data
        if self.data is not None:
            self.__set_up_widgets()

    def __set_up_widgets(self):
        # Detect duplicate rows
        duplicates = self.data[self.data.duplicated(keep=False)]

        # Create info label
        self.info_lbl = QLabel(f'Duplicate count: {duplicates.shape[0]}')
        self.layout.addWidget(self.info_lbl)

        # Create table widget
        self.table = QTableWidget()
        self.table.setRowCount(len(duplicates))
        self.table.setColumnCount(len(duplicates.columns))
        self.table.setHorizontalHeaderLabels(duplicates.columns)
        self.table.setFont(QFont("Arial", 10))

        # Populate table with duplicate rows
        for i, row in enumerate(duplicates.iterrows()):
            for j, value in enumerate(row[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))

        # Set table properties
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

        # Create drop duplicates button
        self.drop_button = QPushButton("Drop Duplicates")
        self.drop_button.clicked.connect(self.drop_duplicates)

        # Add table and button to layout
        self.layout.addWidget(self.table)
        self.layout.addWidget(self.drop_button)

        self.setLayout(self.layout)
        self.setMinimumWidth(200)

    def drop_duplicates(self):
        self.dm.drop_duplicates()
        if self.dm.get_duplicate_count() == 0:
            QMessageBox.information(self, 'In.D.Ex', 'Duplicate rows have been removed successfully.')
            self.drop_button.setDisabled(True)
            self.table.setDisabled(True)