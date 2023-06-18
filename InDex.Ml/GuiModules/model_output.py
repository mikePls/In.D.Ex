import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QSizePolicy
from pandas import DataFrame
from GuiModules.HelperClasses.save_file_module import SaveWindow


class ModelOutputWindow(QWidget):
    def __init__(self, dataframe:DataFrame, output:np.ndarray):
        super().__init__()
        self.dataframe = dataframe
        self.output = output
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
        if len(self.output.shape) == 2:
            output_table.setColumnCount(self.output.shape[1])
            output_table.setRowCount(self.output.shape[0])
            output_table.setHorizontalHeaderLabels([f'Output {i}' for i in range(self.output.shape[1])])

            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    item = QTableWidgetItem(str(self.output[i, j]))
                    output_table.setItem(i, j, item)
        else:
            output_table.setColumnCount(1)
            output_table.setRowCount(len(self.output))
            output_table.setHorizontalHeaderLabels(['Output'])

            for i, value in enumerate(self.output):
                item = QTableWidgetItem(str(value))
                output_table.setItem(i, 0, item)

        download_output_button = QPushButton('Save Output...')
        download_output_button.clicked.connect(self.download_output)

        download_concatenated_button = QPushButton('Save merged dataset...')
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

        self.show()

    def download_output(self):
        if self.output is not None:
            self.dialog = SaveWindow(DataFrame(self.output))

    def download_concatenated(self):
        output_columns = ['Output']
        if len(self.output.shape) == 2:
            output_columns = [f'Output {i}' for i in range(self.output.shape[1])]
        output_to_df = pd.DataFrame(self.output, columns=output_columns)
        concatenated = pd.concat([self.dataframe, output_to_df], axis=1)
        self.dialog = SaveWindow(concatenated)


