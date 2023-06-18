import os
import sys

import joblib
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QAction, QMenuBar, QPushButton, \
    QVBoxLayout, QTabWidget, QWidget, QLabel, QFileDialog, QMenu, QSplitter, QGroupBox, QMessageBox
from PyQt5.QtWidgets import QMainWindow
from sklearn.pipeline import Pipeline

#custom imports
from DataManagement.data_manager import DataManager
from GuiModules import info_manager, dataframe_viewer, column_module
from GuiModules.HelperClasses import user_dialog_module
from GuiModules.HelperClasses import save_file_module
from GuiModules.pipelines_ui_module import PipelineManagerUI
from GuiModules.model_output import ModelOutputWindow



class IOModelManager(QWidget):
    def __init__(self):
        super().__init__()
        self.dm:DataManager = DataManager.get_instance()
        self.model = None
        self.data = None
        self.model_params = None
        self.out_window = None

        self.__initial_setup()

    def __initial_setup(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        #information widgets & layout
        info_lt = QVBoxLayout()
        self.model_lbl = QLabel('Model: None')
        self.model_info_lbl = QLabel('Model info: N/A')
        self.data_lbl = QLabel('Data: N/A')
        self.model_info_lbl.setWordWrap(True)
        self.view_params_btn = QPushButton('View model parameters')
        self.view_params_btn.clicked.connect(self.__view_model_params)
        self.view_params_btn.setDisabled(True)
        info_lt.addWidget(self.model_lbl)
        info_lt.addWidget(self.model_info_lbl)
        info_lt.addWidget(self.data_lbl)
        info_lt.addWidget(self.view_params_btn)

        #load model widgets
        mod_lt = QVBoxLayout()
        mod_grp = QGroupBox()
        mod_grp.setTitle('Model import')
        self.import_model_btn = QPushButton('Import model...')
        self.current_model_btn = QPushButton('Use current model')
        #connect button actions
        self.current_model_btn.clicked.connect(self.__use_current_model)
        self.import_model_btn.clicked.connect(self.__load_pipeline)
        mod_lt.addWidget(self.import_model_btn)
        mod_lt.addWidget(self.current_model_btn)
        mod_grp.setLayout(mod_lt)

        #load data widgets
        data_lt = QVBoxLayout()
        data_grp = QGroupBox()
        data_grp.setLayout(data_lt)
        data_grp.setTitle('Data import')
        self.import_data_btn = QPushButton('Load dataset...')
        self.import_data_btn.clicked.connect(self.__load_dataset)
        self.current_data_btn = QPushButton('Use current dataset')
        self.current_data_btn.clicked.connect(self.__use_current_data)
        data_lt.addWidget(self.import_data_btn)
        data_lt.addWidget(self.current_data_btn)
        #joint layout
        dt_md_lt = QHBoxLayout()
        dt_md_lt.addWidget(mod_grp)
        dt_md_lt.addWidget(data_grp)
        #model output widgets
        mod_out_lt = QHBoxLayout()
        pd_grp = QGroupBox()
        pd_grp.setLayout(mod_out_lt)
        self.proceed_btn = QPushButton('Proceed')
        self.proceed_btn.clicked.connect(self.__predict)
        mod_out_lt.addStretch()
        mod_out_lt.addWidget(self.proceed_btn)
        mod_out_lt.addStretch()

        #add sub layouts to main layout
        self.layout.addLayout(info_lt)
        self.layout.addLayout(dt_md_lt)
        self.layout.addWidget(pd_grp)
        #styling options
        self.layout.setAlignment(Qt.AlignTop | Qt.AlignVCenter)
        self.model_lbl.setWordWrap(True)
        self.model_info_lbl.setWordWrap(True)

    def __load_pipeline(self):
        # Create a file dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Pipeline", "", "Joblib files (*.joblib)",
                                                   options=options)

        # Load the joblib file
        if file_path:
            loaded_object = joblib.load(file_path)

            # Check if the loaded object is a pipeline
            if isinstance(loaded_object, Pipeline):
                # Print relevant information about the pipeline
                self.model_lbl.setText(file_path)
                self.model_info_lbl.setText(f"Pipeline steps: {[step[0] for step in loaded_object.steps]}")
                self.model_params = f"Pipeline parameters: {loaded_object.get_params()}"
                self.view_params_btn.setEnabled(True)
                self.model = loaded_object
                return loaded_object
            else:
                self.view_params_btn.setEnabled(False)
                QMessageBox.critical(self, "Load Error",
                                     f"{file_path} does not contain a pipeline")
                return None

    def __view_model_params(self):
        if self.model_params:
            self.param_lbl = QLabel(str(self.model_params))
            self.param_lbl.setWordWrap(True)
            font = QFont("Consolas", 10)
            self.param_lbl.setFont(font)
            self.param_lbl.show()
            self.param_lbl.adjustSize()

    def __use_current_model(self):
        self.model = self.dm.get_current_model()
        if self.model:
            self.model_lbl.setText('Model: Current model (internal)')
            self.model_info_lbl.setText(str(self.model))
            self.model_params = (self.model.get_params())
            self.view_params_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Load Error",
                                 f"No trained model found.")
            self.model_lbl.setText('Model: None')
            self.model_info_lbl.setText('Model info: N/A')
            self.model_params = ''
            self.view_params_btn.setEnabled(False)

    def __use_current_data(self):
        if self.dm.has_data():
            self.data = self.dm.get_data()
            self.data_lbl.setText('Data: current dataset (internal)')

    def __load_dataset(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "Doc files (*.csv *.json *.tsv *.xlsx *.xlx)")[0]
        try:
            _n, file_extension = os.path.splitext(fname)
            df = None

            if file_extension.lower() == '.csv':
                df = pd.read_csv(fname)
            elif file_extension.lower() == '.tsv':
                df = pd.read_csv(fname, sep='\t')
            elif file_extension.lower() == 'xlsx' or file_extension.lower() == 'xlsx':
                df = pd.read_excel(fname)
            elif file_extension.lower() == '.json':
                df = pd.read_json(fname)

            if df is not None:
                self.data_lbl.setText(f'Data: {fname}')
                self.data = df
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Unable to load file: {e}")

    def __predict(self):
        try:
            if self.model and self.data is not None:

                model_out = self.model.predict(self.data)
                self.out_window = ModelOutputWindow(dataframe=self.data, output=model_out)
                self.out_window.show()

        except Exception as e:
            QMessageBox.critical(self, "Model error",
                                 f"Model failed to generate output: {e}.")
