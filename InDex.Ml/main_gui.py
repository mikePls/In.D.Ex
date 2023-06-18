import os

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QAction, QMenuBar, QPushButton, \
    QVBoxLayout, QTabWidget, QWidget, QLabel, QFileDialog, QMenu, QSplitter
from PyQt5.QtWidgets import QMainWindow

# custom imports
from DataManagement.data_manager import DataManager
from GuiModules import info_manager, dataframe_viewer, column_module
from GuiModules.HelperClasses import save_file_module
from GuiModules.HelperClasses import user_dialog_module
from GuiModules.io_module import IOModelManager
from GuiModules.pipelines_ui_module import PipelineManagerUI


class MainGUIWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.dm = DataManager(parent=self).get_instance()  # Hold DataManager ref
        self.df = None # Hold current dataframe reference
        self.__set_main_window()
        self.__set_menubar()
        self.show()

        #main tab references
        self.eda_tab = None
        self.ml_tab = None
        self.io_tab = None

    def __set_menubar(self):
        self.menu_bar = QMenuBar()
        self.file_menu = QMenu('&File')
        self.tools_menu = QMenu('&Tools')
        self.options_menu = QMenu('&Options')
        self.about = QAction('&About')
        self.menu_bar.addMenu(self.file_menu)
        self.menu_bar.addMenu(self.tools_menu)
        self.menu_bar.addMenu(self.options_menu)
        self.menu_bar.addAction(self.about)
        self.__set_actions()
        self.setMenuBar(self.menu_bar)

    def __set_actions(self):
        self.open_ds_action = QAction(f'&Import dataset...', )
        self.open_model_action = QAction('Import ML model...')
        self.save_dataset_as = QAction('Save as...')
        self.exit_action = QAction('Exit')
        #create sub-menu 'export' actions
        self.export_menu = QMenu('Export')
        self.export_dataset_action = QAction('Export dataset...')
        self.export_model_action = QAction('Export ML model...')
        self.export_menu.addAction(self.export_dataset_action)
        self.export_menu.addAction(self.export_model_action)
        #Options --> theme menu
        self.theme_menu = QMenu('Select theme')
        self.set_dark_theme = QAction('Dark theme')
        self.set_light_theme = QAction('Light theme')
        self.options_menu.addMenu(self.theme_menu)
        self.theme_menu.addAction(self.set_dark_theme)
        self.theme_menu.addAction(self.set_light_theme)
        #create tools menu options
        self.view_correlations = QAction('View feature correlations')
        self.feature_scaling = QAction('Feature scaling')
        self.feature_transformations = QAction('Feature transformations')
        self.remove_duplicates = QAction('Remove duplicate rows')
        self.handle_outliers = QAction('Handle outliers')
        self.tools_menu.addActions([self.view_correlations, self.feature_scaling, self.feature_transformations,
                                            self.remove_duplicates, self.handle_outliers])
        #add to main menu
        self.file_menu.addAction(self.open_ds_action)
        self.file_menu.addAction(self.open_model_action)
        self.file_menu.addAction(self.save_dataset_as)
        self.file_menu.addSeparator()
        self.file_menu.addMenu(self.export_menu)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)
        #connect actions
        self.exit_action.triggered.connect(self.closeEvent)
        self.open_ds_action.triggered.connect(self.__load_dataset)
        self.save_dataset_as.triggered.connect(self.__save_as)
        self.set_dark_theme.triggered.connect(lambda x: self.__change_theme(style='dark'))
        self.set_light_theme.triggered.connect(lambda x: self.__change_theme(style=''))

    def __save_as(self):
        if self.dm is not None and self.dm.has_data():
            self.dialog = save_file_module.SaveWindow(self.dm.get_data())

    def __set_main_window(self):
        self.setMinimumSize(1200, 700)
        self.setWindowTitle('In.D.eX. ML')
        self.tab_widget = self.__create_tab_widget()
        self.setCentralWidget(self.tab_widget)

        self.default_wdg = self.__create_eda_default_widget()
        self.model_selection_wgt = PipelineManagerUI()
        self.tab_bar.addTab(self.default_wdg,'EDA')
        self.tab_bar.addTab(self.model_selection_wgt,'ML Model selection')
        self.tab_bar.addTab(self.__create_io_tab(),'Model I/O')

        os.path.dirname(os.path.dirname(__file__))
        icon = QIcon('res/ico/appico.png')
        self.setWindowIcon(icon)

    def __create_io_tab(self)->QWidget:
        self.io_tab = IOModelManager()
        return self.io_tab

    def __create_eda_tab(self) -> QWidget:
        """Creates and returns the Exploratory Data Analysis tab for tab widget"""
        try:
            self.info_widget = info_manager.InfoManager()
            self.column_manager = column_module.ColumnManager(parent=self)
            self.viewer = dataframe_viewer.DfViewer()
            layout1 = QHBoxLayout()
            layout1.addWidget(self.info_widget)
            layout1.addWidget(self.column_manager)
            tab_layout = QVBoxLayout()
            tab_layout.addLayout(layout1)
            eda_widget = QWidget()
            eda_widget.setLayout(tab_layout)
            main_wgt = QSplitter(Qt.Vertical)
            main_wgt.addWidget(eda_widget)
            main_wgt.addWidget(self.viewer)
            return main_wgt
        except Exception as e:
            print(e)
            return QWidget()

    def __create_eda_default_widget(self):
        """Creates and returns an empty widget with a Load button.
        Used as the default widget for the EDA tab"""
        msg = 'Open or import a dataset to start exploring...'
        lbl = QLabel(msg)
        lbl.setWordWrap(True)
        lbl.setText('<p style="font-size:24px; color:#00509d;">Welcome to</p>'
                '<p style="font-size:48px; color:#ff7400;">In.D.Ex. ML</p>'
                '<p style="font-size:20px; color:#4a4a4a;">Open a dataset to start exploring...</p>')
        self.initial_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(lbl)
        self.load_button = QPushButton('Load dataset...')
        self.load_button.clicked.connect(self.__load_dataset)
        layout.addWidget(self.load_button)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.initial_widget.setLayout(layout)
        return self.initial_widget

    def __create_tab_widget(self):
        self.tab_bar = QTabWidget()
        return self.tab_bar

    def __load_dataset(self):
        if self.dm.has_data():
            msg = 'Would you like to save changes before loading a new file?'
            if user_dialog_module.msg_dlg(parent=self, msg=msg, title=self.windowTitle()):
                self.__save_as()

        file_name = self.__load_file()[0]
        _n, file_extension = os.path.splitext(file_name) #split file_name on "."
        df = None
        try:
            if file_extension.lower() == '.csv':
                df = pd.read_csv(file_name)
            elif file_extension.lower() == '.tsv':
                df = pd.read_csv(file_name, sep='\t')
            elif file_extension.lower() in ['xls', 'xlsx', 'xlsm']:
                df = pd.read_excel(file_name)
            elif file_extension.lower() == '.json':
                df = pd.read_json(file_name)
            else:
                return
        except Exception as e:
            print(e)
            df = None

        if self.df is None and df is not None: # if self.dataframe is none, this is the first load process, remove default tab
            self.df=df
            self.dm.set_data(self.df)
            self.tab_bar.removeTab(0)
            self.eda_tab = self.__create_eda_tab()
            self.tab_widget.insertTab(0, self.eda_tab, 'E.D.A')
            self.tab_widget.tabBar().setCurrentIndex(0)
            self.__connect_tool_menu_actions()
        else:
            self.df = df
            self.dm.set_data(self.df)
            self.refresh()
            self.column_manager.refresh_column_list()

    def __connect_tool_menu_actions(self):
        self.view_correlations.triggered.connect(self.column_manager.show_correlations)
        self.feature_scaling.triggered.connect(self.column_manager.scale_btn_clicked)
        self.feature_transformations.triggered.connect(self.column_manager.transform_selected)
        self.remove_duplicates.triggered.connect(self.column_manager.handle_duplicates)
        self.handle_outliers.triggered.connect(self.column_manager.handle_outliers)

    def __load_file(self):
        """Allows the user to select a file from a system location by displaying
        a file selection dialog"""
        return QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\',
                                           "Doc files (*.csv *.json *.tsv *.xlsx *.xlx)")

    def refresh(self):
        self.info_widget.refresh()
        self.viewer.refresh()

    def closeEvent(self, e):
        if self.df is not None:
            msg = 'Would you like to save your file before exiting?'
            self.exit_dlg = user_dialog_module.msg_dlg(self,msg=msg, type=None , title=self.windowTitle())
            if self.exit_dlg:
                self.__save_as()
                e.accept()
        else:
            e.accept()

    def __change_theme(self, style):
        if style == 'dark':
            with open('res/themes/orange.qss', 'r') as f:
                stylesheet = f.read()
            self.setStyleSheet(stylesheet)
        elif style == '': self.setStyleSheet('')


# if __name__=='__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())