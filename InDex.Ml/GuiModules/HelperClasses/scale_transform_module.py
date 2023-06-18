import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QButtonGroup, \
    QRadioButton, QLabel, QLineEdit, QDoubleSpinBox, QPushButton, QGroupBox,\
    QComboBox, QLayout, QMessageBox, QListView, QAbstractItemView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigCanvas
#custom imports
from DataManagement.data_manager import DataManager
from MLModules.df_transformations_module import TransformationsHandler
from MLModules.distribution_stats_module import DistStats
from MLModules.visualisations_module import Visualizations

#SCALING METHODS
options = ['normalize', 'standardize']
#FEATURE SCALING
"""
Standardization is used on the data values that are normally distributed. 
Further, by applying standardization, we tend to make the mean of the dataset 
as 0 and the standard deviation equivalent to 1.
That is, by standardizing the values, we get the following statistics of the data distribution

    mean = 0
    standard deviation = 1
    
    methods:
    method preprocessing.scale()
    method preprocessing.StandardScaler()
"""

"""
In normalization, we convert the data features of different scales to a common scale which
 further makes it easy for the data to be processed for modeling. 
 Thus, all the data features(variables) tend to have a similar impact on the modeling portion.

Normalize each feature by subtracting the minimum data value from the data variable and then 
divide it by the range of the variable.

It transforms the values to a range between [0,1]. 
method:
from sklearn.preprocessing import MinMaxScaler
"""
class ScaleManager(QWidget, TransformationsHandler):
    def __init__(self, columns:list=None, parent_class=None):
        super().__init__()
        self.parent_class = parent_class
        self.dm: DataManager = DataManager.get_instance()
        self.validate_cols(columns)
        self.columns = columns if columns else None
        self.__setup_ui()

        self.user_choice = None #reference what scaler the user has currently chosen

    def __setup_ui(self):
        #buttons and description labels
        self.group = QButtonGroup()
        self.stnd_btn = QRadioButton('Standardisation')
        self.s_desc_lbl = QLabel('S_Description...')
        self.norm_btn = QRadioButton('Normalisation')
        self.n_desc_lbl = QLabel('n_Description...')
        self.bct_btn = QRadioButton('Box-cox transformer')
        self.b_desc_lbl = QLabel('b_Description...')
        self.mabs_btn = QRadioButton('Max-abs scaler')
        self.m_desc_lbl = QLabel('m_Description...')
        self.selected_lbl = QLabel('Selected columns:')
        self.apply_btn = QPushButton('Apply')
        self.group.addButton(self.stnd_btn)
        self.group.addButton(self.norm_btn)
        self.group.addButton(self.bct_btn)
        self.group.addButton(self.mabs_btn)
        #configure layouts
        sec_lay = QVBoxLayout()
        sec_lay.addWidget(self.stnd_btn)
        sec_lay.addWidget(self.s_desc_lbl)
        sec_lay.addWidget(self.norm_btn)
        sec_lay.addWidget(self.n_desc_lbl)
        sec_lay.addWidget(self.bct_btn)
        sec_lay.addWidget(self.b_desc_lbl)
        sec_lay.addWidget(self.mabs_btn)
        sec_lay.addWidget(self.m_desc_lbl)
        grp = QGroupBox('Scalers')
        grp.setLayout(sec_lay)
        main_lay = QVBoxLayout()
        main_lay.addWidget(grp)
        main_lay.addWidget(self.selected_lbl)
        main_lay.addWidget(self.apply_btn)
        self.setLayout(main_lay)
        #label descriptions
        norm = "Standardises features by transforming them into a common scale between 0 and 1. "
        stnd = "Standardises the input features by subtracting the mean and dividing by the standard deviation," \
               " resulting in zero mean and unit variance."
        maxab = "Applies scaling to each observation by dividing it with the maximum value of the feature," \
                " resulting in values approximately ranging from -1 to 1."
        bct = "Transforms non-normal variables by calculating the optimal value (lambda) to which the variable will be raised, to attempt normality." \
              " This transformation is applicable only to datasets with positive values."
        self.n_desc_lbl.setText(norm)
        self.n_desc_lbl.setWordWrap(True)
        self.s_desc_lbl.setText(stnd)
        self.s_desc_lbl.setWordWrap(True)
        self.b_desc_lbl.setText(bct)
        self.b_desc_lbl.setWordWrap(True)
        self.m_desc_lbl.setText(maxab)
        self.m_desc_lbl.setWordWrap(True)
        #styling & appearance
        main_lay.setAlignment(Qt.AlignTop)
        self.group.setExclusive(True)
        self.n_desc_lbl.setVisible(False)
        self.b_desc_lbl.setVisible(False)
        self.m_desc_lbl.setVisible(False)
        self.stnd_btn.setChecked(True)
        self.current_desc = self.s_desc_lbl
        self.setFixedSize(500,420)
        self.setWindowTitle('Scaling manager')
        msg = f'Selected features: {str(self.columns).strip("[]")}'
        self.selected_lbl.setWordWrap(True)
        self.selected_lbl.setText(msg)
        #connect actions
        self.group.buttonToggled.connect(self.__btn_toggled)
        self.apply_btn.clicked.connect(self.__apply_btn_action)

    def __apply_btn_action(self):
        msg = f'Caution: This process will apply "{self.group.checkedButton().text()}" ' \
              f'to the selected features. Would you like to continue?'
        if not self.msg_dlg(msg, type='warn'):
            return

        if self.user_choice == 'boxcox':
            for col in self.columns:
                original_col = self.dm.get_col_contents(col)
                if (original_col <= 0).any():
                    self.msg_dlg(f'{col} cannot be transformed as it includes 0 or negative values.',
                                 type='error')
                transformed_col = self.boxcox_transform(original_col)
                self.dm.replace_col(col, transformed_col)
                #print('replaced: \n',self.dm.get_col_contents(col))
        else:
            try:
                transformed_df = self.scale(df=self.dm.get_data(),cols=self.columns,s_type=self.user_choice)
                for col in transformed_df.columns:
                    self.dm.replace_col(col, transformed_df[col])
            except Exception as e:
                print(e)
        if self.parent_class:
            self.parent_class.refresh_column_list()

    def __btn_toggled(self, object):
        self.current_desc.setVisible(False)
        if object == self.stnd_btn:
            self.current_desc = self.s_desc_lbl
            self.user_choice = 'standard'
        elif object == self.norm_btn:
            self.current_desc = self.n_desc_lbl
            self.user_choice = 'minmax'
        elif object == self.bct_btn:
            self.current_desc = self.b_desc_lbl
            self.user_choice = 'boxcox'
        elif object == self.mabs_btn:
            self.current_desc = self.m_desc_lbl
            self.user_choice = 'maxabs'

        self.current_desc.setVisible(True)

    def validate_cols(self, columns):
        for col in columns:
            if self.dm.get_col_type(col) == np.object:
                msg = 'Scaling cannot be applied on non-numerical features. Please select features' \
                      ' of valid data types.'
                self.msg_dlg(msg, type='error')
                self.close()
                self.deleteLater()
                return
        self.show()

    def msg_dlg(self, msg:str, type:str=None):
        mb = QMessageBox()
        title = 'Column transformer'
        if type == 'info':
            return mb.information(self, title, msg, mb.Ok)
        if type == 'warn':
            reply = mb.warning(self, title, msg, mb.Yes | mb.No)
        elif type == 'error':
            reply = mb.critical(self, title, msg, mb.Ok)
        else:
            reply = mb.question(self, title, msg, mb.Yes | mb.No)

        return reply == QMessageBox.Yes
