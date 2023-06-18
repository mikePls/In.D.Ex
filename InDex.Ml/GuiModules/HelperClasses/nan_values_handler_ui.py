import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QGroupBox, QLabel, QButtonGroup, QLineEdit, \
    QPushButton
from DataManagement.data_manager import DataManager
from GuiModules.HelperClasses.user_dialog_module import msg_dlg


class NaValuesHandler(QGroupBox):
    def __init__(self, df:pd.DataFrame=None, caller=None):
        super().__init__()
        self.df = df
        self.caller_class = caller
        if self.caller_class:
            self.dm:DataManager = DataManager.get_instance()
            self.df = self.dm.get_data()

        if self.df is not None:
            self.cat_cols = self.df.select_dtypes(include=['object', 'bool']).columns
            self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            self.__initial_setup()


    def __initial_setup(self):
        self.setTitle('Manage NaN values')
        main_lt = QHBoxLayout()
        self.setLayout(main_lt)
        self.cat_na_sum = self.df[self.cat_cols].isna().sum().sum()
        self. categorical_widget = self.__cat_col_wgt()
        if self.cat_na_sum > 0:
            main_lt.addWidget(self.__cat_col_wgt())
        self.num_na_sum = self.df[self.num_cols].isna().sum().sum()
        if self.num_na_sum > 0:
            main_lt.addWidget(self.__num_col_widget())
        if self.cat_na_sum + self.num_na_sum == 0:
            main_lt.addWidget(QLabel('<font weight="Bold" color="Green">No null values found</font>'))

        self.next_btn = QPushButton('Next')
        main_lt.addStretch()
        main_lt.addWidget(self.next_btn)
        self.next_btn.clicked.connect(self.__next_btn_action)


    def __cat_col_wgt(self) ->QWidget:
        self.c_grp = QGroupBox('Categorical columns') #main wgt for categorical columns
        lt = QVBoxLayout()
        lt.setAlignment(Qt.AlignTop)
        self.c_grp.setLayout(lt) #wgt layout

        msg = f'{len(self.cat_cols)} column/s with {self.cat_na_sum} NaN values'
        #method buttons
        drop_btn = QRadioButton('Drop rows with NaN values')
        mode_btn = QRadioButton('Replace with mode')
        custom_btn = QRadioButton('Replace with custom value')
        self.custom_ln = QLineEdit()
        self.custom_ln.setEnabled(False)
        #add buttons to button group
        self.cat_group = QButtonGroup()
        self.cat_group.setExclusive(True)
        self.cat_group.addButton(drop_btn)
        self.cat_group.addButton(mode_btn)
        self.cat_group.addButton(custom_btn)
        #add widgets to layout
        lt.addWidget(QLabel(msg))
        lt.addWidget(QLabel('Select method:'))
        lt.addWidget(drop_btn)
        lt.addWidget(mode_btn)
        lt.addWidget(custom_btn)
        lt.addWidget(self.custom_ln)

        # connect actions
        custom_btn.toggled.connect(lambda: self.custom_ln.setEnabled(not self.custom_ln.isEnabled()))
        return self.c_grp

    def __num_col_widget(self):
        self.n_grp = QGroupBox('Numerical columns')
        lt = QVBoxLayout()
        lt.setAlignment(Qt.AlignTop)
        drop_btn = QRadioButton('Drop rows with NaN values')
        mean_btn = QRadioButton('Replace with mean')
        med_btn = QRadioButton('Replace with median')
        mode_btn = QRadioButton('Replace with mode')
        #add buttons to button group
        self.num_group = QButtonGroup()
        self.num_group.addButton(drop_btn)
        self.num_group.addButton(mean_btn)
        self.num_group.addButton(med_btn)
        self.num_group.addButton(mode_btn)
        #add widgets to layout
        self.num_na_sum = self.df[self.num_cols].isna().sum().sum()
        msg = f'{len(self.num_cols)} column/s with {self.num_na_sum} NaN values'
        lt.addWidget(QLabel(msg))
        lt.addWidget(QLabel('Select method'))
        lt.addWidget(drop_btn)
        lt.addWidget(med_btn)
        lt.addWidget(mean_btn)
        lt.addWidget(mode_btn)
        self.n_grp.setLayout(lt)
        return self.n_grp

    def __next_btn_action(self):
        #determine if there are categorical values, and user has selected a method for treating them
        cat_with_method = self.cat_na_sum == 0 or self.cat_group.checkedButton()
        num_with_method = self.num_na_sum == 0 or self.num_group.checkedButton()
        # if data has nulls and methods have been selected for treating nulls, then proceed
        if not self.data_is_treated() and cat_with_method is not None and num_with_method is not None:
            msg = 'This process will handle all NaN values in the dataset, according to the selected method. ' \
                  'This is a non-reversible process and may result to loss of data. Would you like to proceed?'
            if msg_dlg(parent=self, msg=msg,type='warn',title=self.windowTitle()):
            # if there are empty values in the categorical columns, replace with user-selected method
            # repeat for numerical columns
                self.__handle_cat_nan_vals()
                self.__handle_num_nan_vals()
            msg_dlg(parent=self,msg='NaN values have been treated successfully',type='info',title=self.windowTitle())
            if self.data_is_treated():
                self.change_state()
            if self.caller_class:
                self.dm.set_data(self.df)
                self.caller_class.refresh_column_list()


    def __handle_cat_nan_vals(self):
        if self.cat_group.checkedButton() is None:
            return
        if self.cat_na_sum > 0:
            usr_opt = self.cat_group.checkedButton().text()
            if usr_opt == 'Drop rows with NaN values':
                self.df.dropna(inplace=True)
            elif usr_opt == 'Replace with mode':
                for col in self.cat_cols:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif usr_opt == 'Replace with custom value':
                custom_value = self.custom_ln.text() if self.custom_ln.text() != '' else 'n/a'
                for col in self.cat_cols:
                    self.df[col].fillna(custom_value, inplace=True)

    def __handle_num_nan_vals(self):
        if self.num_group.checkedButton() is None:
            return
        if self.num_na_sum > 0:
            usr_opt = self.num_group.checkedButton().text()
            if usr_opt == 'Drop rows with NaN values':
                self.df.dropna(inplace=True)
            elif usr_opt == 'Replace with mean':
                for col in self.num_cols:
                    mean = self.df[col].mean()
                    self.df[col].fillna(mean, inplace=True)
            elif usr_opt == 'Replace with median':
                for col in self.num_cols:
                    mean = self.df[col].mean()
                    self.df[col].fillna(mean, inplace=True)
            elif usr_opt == 'Replace with mode':
                for col in self.num_cols:
                    mean = self.df[col].mode()[0]
                    self.df[col].fillna(mean, inplace=True)

    def set_data(self, data:pd.DataFrame):
        self.df = data
        self.cat_cols = self.df.select_dtypes(include=['object', 'bool']).columns
        self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.__initial_setup()

    def data_is_treated(self):
        if self.df is None:
            return False
        return self.df.isna().sum().sum() == 0

    def get_data(self):
        return self.df

    def change_state(self):
        self.c_grp.setDisabled(True)
        self.n_grp.setDisabled(True)
        msg = '<font weight="Bold" color="Green">No null values found</font>'
        self.layout().addWidget(QLabel(msg))
        self.layout().addWidget(self.next_btn)





