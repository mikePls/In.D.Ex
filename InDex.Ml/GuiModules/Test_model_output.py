import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QSizePolicy
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtCore import Qt
import csv

class DataFrameMultiNumpyViewer(QWidget):
    def __init__(self, dataframe, array, parent=None):
        super().__init__(parent)
        self.dataframe = dataframe
        self.array = array
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        input_table = QTableWidget()
        input_table.setColumnCount(len(self.dataframe.columns))
        input_table.setRowCount(len(self.dataframe))
        input_table.setHorizontalHeaderLabels(self.dataframe.columns)

        for i, row in self.dataframe.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                input_table.setItem(i, j, item)

        output_table = QTableWidget()
        output_table.setColumnCount(self.array.shape[1])
        output_table.setRowCount(self.array.shape[0])
        output_table.setHorizontalHeaderLabels([f'Output {i}' for i in range(self.array.shape[1])])

        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                item = QTableWidgetItem(str(self.array[i, j]))
                output_table.setItem(i, j, item)

        download_output_button = QPushButton('Download Output')
        download_output_button.clicked.connect(self.download_output)

        download_concatenated_button = QPushButton('Download Concatenated')
        download_concatenated_button.clicked.connect(self.download_concatenated)

        input_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        output_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        input_table.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        output_table.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        left_layout = QVBoxLayout()
        left_layout.addWidget(input_table, stretch=2)
        left_layout.addWidget(download_concatenated_button, stretch=1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(output_table, stretch=1)
        right_layout.addWidget(download_output_button, stretch=1)

        layout.addLayout(left_layout, stretch=2)
        layout.addLayout(right_layout, stretch=1)

    def download_output(self):
        with open('output.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'Output {i}' for i in range(self.array.shape[1])])
            for i in range(self.array.shape[0]):
                writer.writerow([self.array[i, j] for j in range(self.array.shape[1])])

    def download_concatenated(self):
        output_columns = [f'Output {i}' for i in range(self.array.shape[1])]
        output_df = pd.DataFrame(self.array, columns=output_columns)
        concatenated = pd.concat([self.dataframe, output_df], axis=1)
        concatenated.to_csv('concatenated.csv', index=False)

if __name__ == '__main__':
    app = QApplication([])
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    array = np.array([[10, 11], [12, 13], [14, 15]])
    viewer = DataFrameMultiNumpyViewer(df, array)
    viewer.show()
    app.exec_()
