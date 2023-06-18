import os

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QWidget, QFrame, QScrollArea, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QCheckBox, QMessageBox, QGroupBox, QMenu

from DataManagement.data_manager import DataManager
from GuiModules.HelperClasses.corr_visualiser_ui import CorrelationsWindow
from GuiModules.HelperClasses.dist_plot_canvas import DistCanvas
from GuiModules.HelperClasses.outliers_handler_ui import OutliersUi
from GuiModules.HelperClasses.scale_transform_module import ScaleManager
from GuiModules.HelperClasses.transformations_ui import TransformationsWindow
from GuiModules.HelperClasses.three_chart_plot_ui import ThreeChartWindow
from GuiModules.HelperClasses.nan_values_handler_ui import NaValuesHandler
from GuiModules.HelperClasses.duplicates_module import DuplicateWidget
from GuiModules.HelperClasses import user_dialog_module

class ColumnManager(QGroupBox):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        if DataManager.has_instance():
            self.dm:DataManager = DataManager.get_instance()
            self.widget_setup()
        self.setTitle('Feature Manager')

    def widget_setup(self):
        #create layouts
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        self.buttons_layout = QVBoxLayout()

        #create columns_list object
        self.column_list_component = ColumnListComponent()

        #create button widgets
        self.drop_selected = QPushButton('Drop selected')
        self.transform_btn = QPushButton('Transform...')
        self.scale_button = QPushButton('Scale...')
        self.clean_button = QPushButton('Clean')
        self.corr_button = QPushButton('Correlations...')
        self.set_btn_actions()

        #add widgets to respective layouts
        self.main_layout.addWidget(self.column_list_component)
        self.buttons_layout.addWidget(self.drop_selected)
        self.buttons_layout.addWidget(self.transform_btn)
        self.buttons_layout.addWidget(self.scale_button)
        self.buttons_layout.addWidget(self.clean_button)
        self.buttons_layout.addWidget(self.corr_button)
        self.main_layout.addLayout(self.buttons_layout)
        self.buttons_layout.setAlignment(Qt.AlignTop)

        #connect button signals
        self.drop_selected.clicked.connect(self.remove_selected)
        self.transform_btn.clicked.connect(self.transform_selected)
        self.scale_button.clicked.connect(self.scale_btn_clicked)
        self.corr_button.clicked.connect(self.show_correlations)

    def show_correlations(self):
        selected_cols = self.get_selected()
        selected_df = None
        if len(selected_cols) > 1:
            selected_df = self.dm.get_data()[selected_cols].select_dtypes(exclude=['object'])

        if selected_df is not None and len(selected_df.columns) > 1:
            self.corr_window = CorrelationsWindow(dataframe=selected_df)
            self.corr_window.show()
        else:
            msg = 'Not enough columns. Please select two or more numerical columns'
            QMessageBox.information(self, self.windowTitle(), msg, buttons=QMessageBox.Ok)

    def refresh_column_list(self):
        self.column_list_component.refresh()
        if self.parent:
            self.parent.refresh()

    def remove_selected(self):
        message = "This action will remove selected features from the dataset. " \
                  "Are you sure you want to continue?"
        if user_dialog_module.msg_dlg(self, message):
            for col in reversed(self.column_list_component.cols_list):
                if col.is_checked():
                    self.column_list_component.drop_column(col)
            if self.parent is not None:
                self.parent.refresh()
            QMessageBox.information(self,"In.D.Ex", "The selected columns have been removed successfully.")

    def scale_btn_clicked(self):
        col_list = self.column_list_component.get_selected_col_list()
        self.sm = ScaleManager(columns=col_list,parent_class=self)

    def transform_selected(self):
        selected_col_names = self.get_selected()
        if selected_col_names:
            self.tr_window = TransformationsWindow(columns=selected_col_names, parent_class=self)
            self.tr_window.show()

    def get_selected(self)->list:
        """Returns the selected column names as a list of strings"""
        return [col.name for col in self.column_list_component.cols_list if col.is_checked()]

    def set_btn_actions(self):
        self.cln_menu = QMenu()
        self.clean_button.setMenu(self.cln_menu)
        self.cln_menu.addAction('Outliers...', self.handle_outliers)
        self.cln_menu.addAction('NaN values...', self.handle_nulls)
        self.cln_menu.addAction('Duplicates...', self.handle_duplicates)

    def handle_duplicates(self):
        if self.dm.get_duplicate_count() == 0:
            QMessageBox.information(self, 'In.D.Ex.', 'The dataset does not contain duplicate rows.')
        else:
            self.handler = DuplicateWidget()
            self.handler.show()

    def handle_outliers(self):
        columns = self.column_list_component.get_selected_col_list()
        if columns:
            self.handler = OutliersUi(columns=columns)
            self.handler.show()

    def handle_nulls(self):
        selected = self.column_list_component.get_selected_col_list()
        if selected:
            self.nan_handler = NaValuesHandler(caller=self)
            self.nan_handler.next_btn.setText('Apply')
            self.nan_handler.setWindowTitle('NaN Handler')
            self.nan_handler.show()

class ColumnListComponent(QScrollArea):
    def __init__(self, main_window=None):
        super().__init__()
        self.dm:DataManager = DataManager.get_instance()
        self.widget_setup()
        self.cols_list = []

        if self.dm:
            self.set_columns()

    def widget_setup(self):
        widget = QFrame(self)
        self.main_layout = QVBoxLayout(widget)
        self.main_layout.addLayout(self.create_toolbar())
        self.main_layout.setAlignment(Qt.AlignTop)
        self.setWidget(widget)
        #style
        self.setMinimumWidth(500)
        self.setWidgetResizable(True)

    def create_toolbar(self):
        toolbar_layout = QHBoxLayout()
        #force_refresh_btn = QPushButton('R')
        #force_refresh_btn.clicked.connect(self.refresh)
        #toolbar_layout.addWidget(force_refresh_btn)
        toolbar_layout.setAlignment(Qt.AlignRight)

        select_all_cbox = QCheckBox('Select all')
        toolbar_layout.addWidget(select_all_cbox)
        select_all_cbox.stateChanged.connect(lambda l: self.select_all(select_all_cbox.isChecked()))
        return toolbar_layout

    def select_all(self, status:bool):
        for col in self.cols_list:
            col.set_checked(status)

    def set_columns(self):
        for col,dtype in zip(self.dm.get_columns(), self.dm.df_dtypes_list()):
            c = Column(name=col, col_type=dtype, parent=self)
            self.main_layout.addWidget(c)
            self.cols_list.append(c)

    def drop_column(self, widget):
        try:
            self.dm.drop_column(widget.name)
            self.cols_list.remove(widget)
            self.main_layout.removeWidget(widget)
            widget.deleteLater()
        except Exception as e:
            print(e)

    def get_selected_col_list(self):
        """Returns a list of strings with the names of checked columns"""
        return [col.name for col in self.cols_list if col.is_checked()]

    def refresh(self):
        col_names:list = self.dm.get_columns() #get columns of new/updated dataset
        #reconfigure existing Column objects to the specs of new column list
        for col, wgt in zip(self.dm.get_columns(), self.cols_list):
            #if col != wgt.name or (str(self.dm.get_col_type(col)) != wgt.col_type):
            wgt.update_col(col)
            col_names.remove(col)

        #if there are still columns that don't have a Column card, create more cards
        for col in col_names:
            c = Column(name=col, col_type=self.dm.get_col_type(col), parent=self)
            self.main_layout.addWidget(c)
            self.cols_list.append(c)

        #remove excess/leftover Column objects from previous dataset
        current_col_names = self.dm.get_columns()
        for wgt in list(self.cols_list):
            if wgt.name not in current_col_names:
                self.cols_list.remove(wgt)
                self.main_layout.removeWidget(wgt)

class Column(QFrame):
    """Class for graphically displaying and managing a pandas dataset column and information"""
    def __init__(self, name:str, col_type=None, parent:ColumnListComponent=None):
        super().__init__()
        self.name = name
        self.col_type = col_type
        self.parent = parent
        self.dm:DataManager = DataManager.get_instance()
        self.setup_widgets(self.name)

    def setup_widgets(self,name):
        #self.setMinimumSize(200, 50)
        layout = QHBoxLayout(self, )
        layout.setAlignment(Qt.AlignLeft)
        #Create Widgets and add to layout
        self.details_button = QPushButton()
        self.details_button.clicked.connect(self.show_col_details)
        self.details_button.setToolTip('Examine')
        self.delete_button = QPushButton()
        self.delete_button.setToolTip('Drop column')
        self.checked = QCheckBox()
        self.delete_button.clicked.connect(self.remove)
        self.info_label = QLabel()
        self.info_label.installEventFilter(self)
        self.name_lbl = QLabel(name)
        self.type_lbl = QLabel(f'(dtype:{self.col_type})')
        layout.addWidget(self.name_lbl)
        layout.addWidget(self.type_lbl)
        # determine if column has nulls / if yes, add respective label
        nulls = self.dm.get_col_null_count(self.name)
        self.nulls_lbl = QLabel()
        if nulls > 0:
            self.nulls_lbl.setText(f'{str(nulls)} nulls')
            self.nulls_lbl.setStyleSheet('font-weight: bold;color: red;')
            layout.addWidget(self.nulls_lbl)
        layout.addStretch()
        self.info_card = self.create_info_card()
        layout.addWidget(self.info_label)
        layout.addWidget(self.details_button)
        layout.addWidget(self.delete_button)
        layout.addWidget(self.checked)

        #Style widget_setup
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(1)
        project_root = os.path.dirname(os.path.dirname(__file__))
        dicn = os.path.join(project_root, 'res/img/drop.png')
        inicn = os.path.join(project_root, 'res/img/exmn.webp')
        cicn = os.path.join(project_root, 'res/img/chart.png')
        self.delete_button.setIcon(QIcon(dicn))
        self.details_button.setIcon(QIcon(inicn))
        self.info_label.setPixmap(QPixmap(cicn))
        #hide info button if type is object as plots won't work
        if self.col_type == object:
            self.details_button.setVisible(False)

        #connect widget actions

    def eventFilter(self, object:QWidget, event):
        """Shows or hides the info card widget when mouse enters or leaves
        the info label"""
        if event.type() == QEvent.Enter:
            self.info_card.move(event.globalPos() + QtCore.QPoint(10,-130))
            self.info_card.show()
            return True
        elif event.type() == QEvent.Leave:
            self.info_card.hide()
        return False

    def create_info_card(self):
        #create hist-plot image for column
        data = self.dm.get_data()
        font = QFont("Consolas", 10)

        skew_lbl = QLabel('')
        skew_lbl.setFont(font)
        skew_lbl.setText(f'skewness: {self.dm.get_col_skew(self.name)}')
        window = QFrame()
        window.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        window.setWindowFlags(Qt.FramelessWindowHint)
        layout = QVBoxLayout()
        window.setLayout(layout)
        description = str(data[self.name].describe())
        describe_lbl = QLabel(description)
        describe_lbl.setFont(font)
        layout.addWidget(describe_lbl)
        layout.addWidget(skew_lbl)
        layout.addWidget(describe_lbl)

        plt = DistCanvas(data=data[self.name])
        layout.addWidget(plt)
        return window

    def remove(self):
        '''If the object has a parent, displays a Yes|No dialogue. Removes itself and notifies
        parent if the answer is True'''
        if self.parent is not None and self.warn_dlg() :
            try:
                self.parent.drop_column(self)
            except Exception as e:
                print(e)

    def warn_dlg(self):
        msg = f"'{self.name}' and all its contents will be permanently deleted." \
              f"Are you sure you want to continue?"
        mb = QMessageBox()
        reply = mb.question(self,'',msg, mb.Yes | mb.No)
        return reply == QMessageBox.Yes

    def is_checked(self):
        return self.checked.isChecked()

    def set_checked(self, status:bool=True):
        self.checked.setChecked(status)

    def show_col_details(self):
        self.chrt = ThreeChartWindow(self.dm.get_data(), self.name)

    def update_col(self, name:str):
        self.name = name
        self.col_type=self.dm.get_col_type(name)
        self.name_lbl.setText(name)
        self.type_lbl.setText(f'dtype:{str(self.dm.get_col_type(name))}')
        self.info_card = self.create_info_card()
        nulls = self.dm.get_col_null_count(self.name)
        if nulls > 0:
            self.nulls_lbl.setText(f'{str(nulls)} nulls')
        else:
            self.nulls_lbl.setText('')
        if self.col_type != object:
            self.details_button.setVisible(True)
        else:
            self.details_button.setVisible(False)





