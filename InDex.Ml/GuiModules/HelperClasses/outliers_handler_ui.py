import numpy as np
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QButtonGroup, \
    QRadioButton, QLabel, QLineEdit, QDoubleSpinBox, QPushButton, QGroupBox,\
    QComboBox, QLayout, QMessageBox, QListView, QAbstractItemView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigCanvas
#custom imports
from DataManagement.data_manager import DataManager
from MLModules.distribution_stats_module import DistStats
from MLModules.visualisations_module import Visualizations


class OutliersUi(QWidget):
    def __init__(self, parent=None, columns=None):
        super().__init__()
        self.parent = parent
        self.dm:DataManager = DataManager.get_instance()
        self.columns = columns
        self.col_cards = {str:object} #keep OutlierCard object references dict{name:str : object:OutlierCard()}
        self.current_card = None #hold ref for currently displaying OutlierCard
        self.setWindowTitle('Outlier handler')
        if columns:
            self.setup_ui()

    def setup_ui(self):
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.list_view = QListView()
        self.list_view.setMaximumWidth(250)
        self.model = QStandardItemModel()
        self.list_view.clicked[QModelIndex].connect(self.list_clicked)
        self.list_view.setModel(self.model)
        self.list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.close_btn = QPushButton('Close', self)
        self.close_btn.clicked.connect(self.quit)


        #create layouts and add widgets
        self.sub_lt = QHBoxLayout()
        main_lt = QVBoxLayout()
        self.sub_lt.addWidget(self.list_view)
        #sub_lt.addWidget()
        main_lt.addLayout(self.sub_lt)
        main_lt.addWidget(self.close_btn)
        self.setLayout(main_lt)

        self.set_feature_widgets()
        self.setMinimumWidth(850)

    def quit(self):
        self.close()

    def list_clicked(self):
        col_name = self.list_view.selectedIndexes()[0].data()
        if self.current_card is not None:
            self.current_card.setVisible(False)

        self.current_card = self.col_cards[col_name]
        self.current_card.setVisible(not self.col_cards[col_name].isVisible())

    def set_feature_widgets(self):
        filtered_cols = []  # keep names of non-object cols
        for col in self.columns:
            if self.dm.get_col_type(col) != np.object:
                filtered_cols.append(col)
                self.model.appendRow(QStandardItem(col))
                self.col_cards[col] = OutlierCard(name=col)
                self.col_cards[col].setVisible(False)
                self.columns = filtered_cols
                self.sub_lt.addWidget(self.col_cards[col])

class OutlierCard(QGroupBox, Visualizations):
    """Class that displays a UI card with information and boxplot of outliers,
     of a given Series object."""
    def __init__(self, name:str=None):
        super().__init__()
        self.col_name = name
        self.dm: DataManager = DataManager.get_instance()
        self.setup_ui()

        self.setTitle(self.col_name)
        self.has_outliers = None
        self.dt = DistStats() #Distribution Stats module
        self.option = None #Ref user option for replacing outliers
        self.update_info_lbl()

    def setup_ui(self):
        self.tol_grp = QGroupBox() #groupbox for tolerance widgets
        self.tol_grp.setMaximumHeight(100)
        self.tol_grp.setTitle('Tolerance')
        self.set_btn = QPushButton('Set')
        self.set_btn.clicked.connect(self.update_outliers)
        self.upper_spn = QDoubleSpinBox()
        self.lower_spn = QDoubleSpinBox()
        #set spinbox params
        self.upper_spn.setRange(0.5, 9)
        self.upper_spn.setSingleStep(0.1)
        self.upper_spn.setValue(1.5)
        self.upper_spn.setSuffix('*iqr')
        self.lower_spn.setRange(0.5, 9)
        self.lower_spn.setSingleStep(0.1)
        self.lower_spn.setValue(1.5)
        self.lower_spn.setSuffix('*iqr')
        self.lower_spn.setPrefix('-')
        #create boxplot:
        self.plot = self.get_boxplot()
        #widgets for handling outliers. options: drop, replace:mean, median, custom, bring to limit(winsorize)
        # options wdgt for handling outliers
        self.btn_grp = QButtonGroup() #button group for mutually exclusive radio buttons
        self.drop_btn = QRadioButton('Drop outliers')
        self.replace_btn = QRadioButton('Replace')
        #button group settings
        self.btn_grp.addButton(self.replace_btn)
        self.btn_grp.addButton(self.drop_btn)
        self.btn_grp.setExclusive(True)
        self.btn_grp.buttonToggled.connect(self.btn_changed)
        # Apply button; create, configure and connect
        self.apply_btn = QPushButton('Apply')
        self.apply_btn.setDisabled(True)
        self.apply_btn.clicked.connect(self.apply_changes)
        #Set and configure QCombo box for replace outliers options
        self.replace_options = QComboBox()
        self.replace_options.addItems(['mean','median', 'mode', 'custom', 'winsorize' ])
        self.replace_options.setDisabled(True)
        self.replace_options.setPlaceholderText('With...')
        self.replace_options.currentIndexChanged.connect(self.replace_option_changed)
        self.desc_lbl = QLabel('')
        self.desc_lbl.setWordWrap(True)
        #set and configure custom value wgt
        self.cust_val = QLineEdit()
        self.cust_val.setMaximumWidth(100)
        self.cust_val.setPlaceholderText('0')
        self.cust_val.setVisible(False)
        #info label
        self.info_lbl = QLabel('info')
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setMaximumWidth(450)
        #define and set layouts for card
        self.main_layout = QHBoxLayout() #main layout
        #tolerance layout
        self.set_tolerance_layout()
        #set plot layout
        self.plt_layout = QVBoxLayout()
        self.set_plot_layout()
        #--->MAIN LAYOUT<---
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.plt_layout)
        self.main_layout.addLayout(self.get_options_layout())
        self.setLayout(self.main_layout)

        self.setMinimumWidth(750)

    def set_tolerance_layout(self):
        """Configures the 'tolerance Groupbox' widgets by adding them to a horizontal layout, and
        setting the layout for the Groupbox."""
        tol_layout = QHBoxLayout()
        tol_layout.setAlignment(Qt.AlignLeft)
        tol_layout.addWidget(self.lower_spn)
        tol_layout.addWidget(self.upper_spn)
        tol_layout.addWidget(self.set_btn)
        self.tol_grp.setLayout(tol_layout)

    def get_options_layout(self) -> QLayout:
        opt_layout = QVBoxLayout()
        opt_layout.addWidget(self.tol_grp)
        opt_layout.addWidget(self.drop_btn)
        opt_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        replace_layout = QVBoxLayout()
        replace_layout.addWidget(self.replace_btn)
        replace_layout.addWidget(self.replace_options)
        replace_layout.addWidget(self.cust_val)
        replace_layout.addWidget(self.desc_lbl)
        replace_layout.setAlignment(Qt.AlignTop)
        replace_layout.addWidget(self.apply_btn)
        opt_layout.addLayout(replace_layout)
        opt_layout.setAlignment(Qt.AlignLeft)
        return opt_layout

    def set_plot_layout(self):
        self.plt_layout.addWidget(self.plot)
        self.plt_layout.addWidget(self.info_lbl)

    def btn_changed(self, object):
        self.apply_btn.setEnabled(True)
        if object == self.replace_btn:
            self.option = 'Replace'
            self.replace_options.setEnabled(True)
            self.replace_option_changed()
        else:
            self.desc_lbl.setText('Remove all rows that contain outliers within the specified range')
            self.option = 'Drop'
            self.replace_options.setEnabled(False)

    def replace_option_changed(self):
        description = {'mean':'Replace outliers with the mean value of the column',
                       'median':'Replace outliers with the median value of the columns',
                       'mode':'Replace with the most repeated value that is not an outlier',
                       'custom':'Replace outliers with a custom value of the same dtype',
                       'winsorize':'Reduce outliers to the min-max limits of the normal range'}
        self.desc_lbl.setText(description[self.replace_options.currentText()])
        if self.replace_options.currentText() == 'custom':
            self.cust_val.setVisible(True)
            self.option = 'custom'
        else:
            self.cust_val.setVisible(False)
            self.option = self.replace_options.currentText()

    def apply_changes(self):
        """Attempts to apply the user-defined changes, by replacing or dropping
        outliers in the original dataset.
        """
        if self.warn_dlg():
            try:
                data = self.dm.get_col_contents(self.col_name)
                ub = self.upper_spn.value()
                lb = self.lower_spn.value()

                treated_data = self.dt.handle_outliers(data=data,method=self.option,
                                                       upper_bound=ub,lower_bound=lb,
                                                       custom_value=self.cust_val.text())
                self.dm.replace_col(col_name=self.col_name, data=treated_data)
                self.update_outliers()
                self.plot.figure.update()

                QMessageBox.information(self, 'Information',
                                        'Selected outliers have been successfully replaced. Percentiles and IQR '
                                        'will be recalculated with current values.')

            except Exception as e:
                print(e)

        #self.next_btn.setDisabled(True)

    def get_boxplot(self, canvas:FigCanvas=None)->FigCanvas:
        data = self.dm.get_col_contents(self.col_name)
        low_bound = float(self.lower_spn.value())
        upper_bound = float(self.upper_spn.value())
        return self.create_boxplot(data, low_iqr_bound=low_bound, high_iqr_bound=upper_bound,
                                   canvas=canvas)
    def update_outliers(self):
        self.get_boxplot(canvas=self.plot)
        self.update_info_lbl()

    def update_info_lbl(self):
        data = self.dm.get_col_contents(self.col_name)
        high_iqr = float(self.upper_spn.value())
        low_iqr = float(self.lower_spn.value())
        self.outliers = self.dt.count_iqr_outliers(data=data,high_iqr_bound= high_iqr,low_iqr_bound= low_iqr)
        msg = (f'Total outliers:{sum(self.outliers)} ({self.outliers[0]} above, and {self.outliers[1]} below specified bounds)')
        self.info_lbl.setText(msg)

    def warn_dlg(self):
        msg = "This action will permanently replace the values within specified range. " \
              "Do you wish to continue?"
        mb = QMessageBox()
        reply = mb.question(self,'',msg, mb.Yes | mb.No)
        return reply == QMessageBox.Yes