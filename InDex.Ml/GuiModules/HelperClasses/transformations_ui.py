import pandas as pd
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIntValidator, QCloseEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox, \
    QComboBox, QListView, QAbstractItemView, QCheckBox, QSpinBox, QFormLayout, QMessageBox
import numpy as np

# custom imports
from DataManagement.data_manager import DataManager
from MLModules.df_transformations_module import TransformationsHandler


class TransformationsWindow(QWidget):
    def __init__(self, columns:list=None, parent_class=None):
        super().__init__()
        self.columns = columns
        self.parent_class = parent_class
        self.dm:DataManager = DataManager.get_instance()
        self.entries = {} #hold column names and Card widgets refs a {'name':QWidget}
        self.current_card = None #hold ref for showing Card widget
        self.__setup_ui()

    def __setup_ui(self):
        self.layout = QFormLayout()
        self.cards_lay = QVBoxLayout()
        self.list_view = QListView()
        self.list_view.setMinimumWidth(150)
        self.list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layout.addRow(self.list_view, self.cards_lay)
        self.model = QStandardItemModel()
        self.list_view.setModel(self.model)
        self.list_view.clicked[QModelIndex].connect(self.list_clicked)
        for col in self.columns:
            self.model.appendRow(QStandardItem(col))
            self.entries[col] = TransformationsCard(col)
            self.entries[col].setVisible(False)
            self.cards_lay.addWidget(self.entries[col])
        #select first row and update card:
        self.list_view.setCurrentIndex(self.model.index(0,0))
        self.list_clicked()
        #
        self.setLayout(self.layout)
        self.setMinimumSize(700,400)
        self.setWindowTitle('Column transformer')

    def list_clicked(self):
        if self.current_card is not None:
            self.current_card.setVisible(False)
        current_col = self.list_view.selectedIndexes()[0].data()
        self.current_card = self.entries[current_col]
        self.current_card.setVisible(True)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.parent_class:
            self.parent_class.refresh_column_list()
        event.accept()


class TransformationsCard(QGroupBox, TransformationsHandler):
    def __init__(self, name):
        super().__init__()

        self.dm:DataManager = DataManager.get_instance() #ref DataManager
        self.name = name
        self.data = self.dm.get_data()[name]
        self.type = self.dm.get_col_type(self.name)
        self.null_count = self.dm.get_col_null_count(self.name)
        self.current_card = None #hold reference of current card

        self.setup_ui()

    def setup_ui(self):
        #call methods for setting 'card' widgets for options.
        self.set_dummies_card()
        self.set_oh_card()
        self.set_le_card()
        self.set_poly_feat_card()
        self.set_split_card()

        self.set_main_ui()

    def set_dummies_card(self):
        layout = QVBoxLayout()
        lbl = QLabel('Converts categorical variable into indicator variables with binary values.'
                     'This will generate n number of columns where n=count of unique values, or n-1 '
                     'if \'drop first\' option is selected. This method is more suitable for EDA;'
                     'for ML purposes, one_hot encoder is recommended.')
        lbl.setWordWrap(True)
        self.drop_original = QCheckBox('Drop original column.')
        self.drop_f = QCheckBox('Drop first column (n-1 dummies).')
        self.include_nan = QCheckBox('Create column for the missing values.')
        self.dum_btn = QPushButton('Apply')
        self.dum_btn.clicked.connect(self.apply_oh_dummies)
        self.replace_existing = QCheckBox('Replace existing')
        self.replace_existing.setToolTip('Replace columns with the same names as the values'
                                         'in this column.')
        self.replace_existing.setChecked(True)
        layout.addWidget(lbl)
        layout.addWidget(self.drop_original)
        layout.addWidget(self.drop_f)
        layout.addWidget(self.include_nan)
        layout.addWidget(self.replace_existing)
        layout.addWidget(self.dum_btn)
        layout.setAlignment(self.dum_btn, Qt.AlignRight)
        self.dummies_wgt = QWidget()
        self.dummies_wgt.setVisible(False)
        self.dummies_wgt.setLayout(layout)

    def set_oh_card(self):
        layout = QVBoxLayout()
        lbl = QLabel('Creates new columns for each unique value in a categorical variable, '
                     'with each column containing binary values (0 or 1) indicating'
                     ' whether or not a particular value is present for a given row in the dataset.')
        lbl.setWordWrap(True)
        self.oh_btn = QPushButton('Apply')
        self.oh_btn.clicked.connect(self.apply_oh_encoder)
        self.drop_f_oh = QCheckBox('Drop first')
        self.replace_existing_oh = QCheckBox('Replace existing')
        self.replace_existing_oh.setToolTip('Replace columns with the same names as the values'
                                         'in this column.')
        self.replace_existing_oh.setChecked(True)
        self.drop_original_oh = QCheckBox('Drop original column')
        layout.addWidget(lbl)
        layout.addWidget(self.drop_original_oh)
        layout.addWidget(self.drop_f_oh)
        layout.addWidget(self.replace_existing_oh)
        layout.addWidget(self.oh_btn)
        layout.setAlignment(self.oh_btn, Qt.AlignRight)
        self.oh_wgt = QWidget()
        self.oh_wgt.setVisible(False)
        self.oh_wgt.setLayout(layout)

    def set_le_card(self):
        layout = QVBoxLayout()
        lbl = QLabel('Converts non-numerical variables into numerical, '
                     'where each unique value is assigned an integer value.'
                     'This will change the column dtype to int64')
        lbl.setWordWrap(True)
        self.le_btn = QPushButton('Apply')
        self.le_btn.clicked.connect(self.apply_lbl_encoder)
        layout.addWidget(lbl)
        layout.addWidget(self.le_btn)
        layout.setAlignment(self.le_btn, Qt.AlignRight)
        self.le_wgt = QWidget()
        self.le_wgt.setVisible(False)
        self.le_wgt.setLayout(layout)

    def set_poly_feat_card(self):
        layout = QVBoxLayout()
        lbl = QLabel('Generate a new feature matrix consisting of all polynomial combinations '
                     'of the features with degree less than or equal to the specified degree.')
        lbl.setWordWrap(True)
        self.degree = QSpinBox()
        self.degree.setMaximumWidth(60)
        self.degree.setRange(1,20)
        self.degree.setSingleStep(1)
        self.degree.setValue(2)
        self.bias = QCheckBox('Include bias')
        self.bias.setToolTip('Creates a bias column where all polynomial powers are 0.')
        self.poly_btn = QPushButton('Apply')
        self.poly_btn.clicked.connect(self.apply_poly_ft)
        l2 = QFormLayout()
        l2.addRow(QLabel('Degree:'),self.degree)
        layout.addWidget(lbl)
        layout.addWidget(self.bias)
        layout.addLayout(l2)
        layout.addWidget(self.poly_btn)
        layout.setAlignment(self.poly_btn, Qt.AlignRight)

        self.pf_wgt = QWidget()
        self.pf_wgt.setVisible(False)
        self.pf_wgt.setLayout(layout)

    def set_split_card(self):
        lbl = QLabel("Split object column into 2 or many, based on the separator value."
                     "'October,2023' with ',' separator, will assign 'October' and '2023' into"
                     "separate columns. Returns n number of columns if n is specified and is <="
                     "to the total number of columns created. Leave blank to add all generated columns.")
        lbl.setWordWrap(True)
        self.separator = QLineEdit()
        self.separator.setFixedWidth(150)
        self.separator.setPlaceholderText("e.g., %, &, or, /, ...")
        self.return_num = QLineEdit()
        validator = QIntValidator()
        validator.setRange(1,100)
        self.return_num.setValidator(validator)
        self.return_num.setMaxLength(3)
        #self.return_num.setMaximumWidth(30)
        self.split_btn = QPushButton('Apply')
        self.split_btn.clicked.connect(self.apply_split)
        lay_a = QFormLayout()
        lay_a.addRow(QLabel('Separator:'), self.separator)
        lay_a.addRow(QLabel('Add n columns:'), self.return_num)

        layout = QVBoxLayout()
        layout.addWidget(lbl)
        layout.addLayout(lay_a)
        layout.addWidget(self.split_btn)
        layout.setAlignment(self.split_btn, Qt.AlignRight)

        self.split_wgt = QWidget()
        self.split_wgt.setVisible(False)
        self.split_wgt.setLayout(layout)

    def set_main_ui(self):
        # components

        self.setTitle(f'Name:{self.name},  Type:{self.type},  \'NaN\' count:{self.null_count}')
        prompt_lbl = QLabel('Select transformation:')
        self.option_wgts = {'One_hot dummies':self.dummies_wgt, 'One_hot encoder (dtype:object)':self.oh_wgt,
                   'Label encoder (dtype:object)':self.le_wgt, 'Polynomial features':self.pf_wgt,
                            'Split Categorical column':self.split_wgt}
        #set QComboBox for transformation options
        self.options_list = QComboBox()
        self.options_list.setMaximumWidth(250)
        self.options_list.addItems(self.option_wgts.keys())
        self.options_list.setCurrentIndex(-1)
        self.options_list.setPlaceholderText('Select transformation')
        #set main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(prompt_lbl)
        self.main_layout.addWidget(self.options_list)
        #add 'options' widgets to main layout
        for val in self.option_wgts.values():
            self.main_layout.addWidget(val)


        self.options_list.currentIndexChanged.connect(self.option_changed)
        self.setLayout(self.main_layout)

    def option_changed(self):
        if self.current_card is not None:
            self.current_card.setVisible(False)
        self.current_card = self.option_wgts[self.options_list.currentText()]
        self.current_card.setVisible(True)

    def apply_oh_dummies(self):
        #if column type is not object, ask user to confirm about creating new columns for each numerical value
        msg = 'Caution! You have selected to transform a non-object column. This will generate a new column for' \
              'each unique value, and might dramatically increase the size of the dataframe. Do you wish to continue?'
        if self.type != np.object:
            if not self.warn_dlg(msg=msg,type='warn'):
                return
        else:
            msg = 'This will generate a new binary column for each unique value of the selected column.' \
                  'Do you wish to continue?'
            if not self.warn_dlg(msg=msg):
                return
        try:
            data = self.data
            current_index = int(self.dm.get_col_index(self.name)) +1
            dummies_data = self.one_hot_dummies(df=data, drop_first=self.drop_f.isChecked())
            for col in dummies_data:
                if self.replace_existing.isChecked() and (col in self.dm.get_columns()):
                    self.dm.replace_col(col_name=col, data=dummies_data[col])
                else:
                    self.dm.insert_col(index=current_index, title=col,data=dummies_data[col])
                    current_index +=1
            if self.drop_original.isChecked():
                self.dm.drop_column(self.name)
            self.warn_dlg('Process completed successfully.', type='info')
        except Exception as e:
            self.warn_dlg(f'Process unsuccessful:{e}')

    def apply_oh_encoder(self):
        # if column type is not object, ask user to confirm about creating new columns for each numerical value
        msg = 'This transformation can only be applied on categorical features.'
        if self.type != np.object:
            self.warn_dlg(msg=msg, type='info')
            return
        else:
            msg = 'This will generate a new binary column for each unique value of the selected column.' \
                  'Do you wish to continue?'
            if not self.warn_dlg(msg=msg):
                return
        try:
            data = self.data
            current_index = int(self.dm.get_col_index(self.name)) + 1
            drop = 'first' if self.drop_f_oh.isChecked() else None

            dummies_data = self.one_hot_transformer(data= pd.DataFrame(data), drop=drop)

            for col in dummies_data:

                if self.replace_existing_oh.isChecked() and (col in self.dm.get_columns()):

                    self.dm.replace_col(col_name=col, data=dummies_data[col])
                else:
                    self.dm.insert_col(index=current_index, title=col, data=dummies_data[col])
                    current_index += 1
            if self.drop_original_oh.isChecked():
                self.dm.drop_column(self.name)
            self.warn_dlg('Process completed successfully.', type='info')
        except Exception as e:
            self.warn_dlg(f'Process unsuccessful:{e}')

    def apply_lbl_encoder(self):
        if self.type != np.object:
            msg = 'Label encoder can only be applied on non-numerical, categorical features.'
            self.warn_dlg(msg, type='info')
            return
        msg = 'This process will transform the selected column by converting ' \
              'each unique categorical value to a numerical one.' \
              ' Are you sure you want to continue?'
        if not self.warn_dlg(msg=msg):
            return
        try:
            encoded_col = self.label_encoding(df=self.dm.get_data(), column=self.name)
            self.dm.replace_col(self.name, encoded_col)
            self.warn_dlg('Process completed successfully.', type='info')
        except Exception as e:
            self.warn_dlg(f'Critical error{e}', type='error')

    def apply_poly_ft(self):
        if self.type == object:
            msg = 'This transformation only works on numerical dtypes. Try transforming column' \
                  'to a numerical type first.'
            self.warn_dlg(msg,type='info')
            return
        msg = 'This process will generate additional columns based on the selected polynomial degree. ' \
              'Continue?'
        if not self.warn_dlg(msg, type='warn'):
            return
        try:
            df=self.poly_feat(df=self.dm.get_data(), degree=int(self.degree.text()),
                              features=[self.name], include_bias=self.bias.isChecked())
            idx = self.dm.get_col_index(self.name)
            self.dm.concat(df=df, position=idx+1)
            self.warn_dlg('Process completed successfully.', type='info')
        except Exception as e:
            self.warn_dlg(f'Process unsuccessful:{e}')

    def apply_split(self):
        if self.type != np.object:
            msg = 'Splitting categorical columns only applies to columns of dtype: object.'
            return self.warn_dlg(msg, type='warn')
        msg = 'The selected column will be split into two or more individual columns, based on defined value. ' \
              'Are you sure you want to continue?'
        if not self.warn_dlg(msg):
            return
        splitter = self.separator.text() if self.separator.text() != '' else ' '
        num_of_cols = int(self.return_num.text()) if self.return_num.text() != '' else 0
        print(num_of_cols)
        try:
            split_df = self.split_categorical_col(self.dm.get_data(), self.name, splitter=splitter, return_x_cols=num_of_cols)
            self.dm.concat(split_df)
            self.warn_dlg('Process completed successfully.', type='info')
            self.split_btn.setEnabled(False)
            print(split_df)
            print(self.dm.get_columns())
        except Exception as e:
            print(e)

    def warn_dlg(self, msg:str, type:str=None):
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

