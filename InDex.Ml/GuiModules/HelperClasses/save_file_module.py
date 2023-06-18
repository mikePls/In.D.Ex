import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QWidget


class SaveWindow(QWidget):
    def __init__(self, df:pd.DataFrame):
        super().__init__()
        self.df = df
        self.show_save_dialog()


    def show_save_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        #options |= QFileDialog.DontUseNativeDialog

        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "",
                                                   "CSV Files (*.csv);;TSV Files (*.tsv);;Text Files (*.txt)",
                                                   options=options)

        if file_name:
            # Determine file format based on selected filter
            if file_name.endswith(".csv"):
                delimiter = ","
            elif file_name.endswith(".tsv"):
                delimiter = "\t"
            elif file_name.endswith(".txt"):
                delimiter = " "
            else:
                raise ValueError("Invalid file format")

            # Save DataFrame to the selected file with the chosen delimiter
            if file_name.endswith(".csv"):
                self.df.to_csv(file_name, sep=delimiter, index=False)
            elif file_name.endswith(".tsv"):
                self.df.to_csv(file_name, sep=delimiter, index=False)
            elif file_name.endswith(".txt"):
                self.df.to_csv(file_name, sep=delimiter, index=False)
            else:
                raise ValueError("Invalid file format")