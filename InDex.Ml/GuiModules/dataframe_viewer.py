import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QModelIndex, QAbstractTableModel
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QTableView, QHBoxLayout, QComboBox, QPushButton, QLabel, QLineEdit

from DataManagement.data_manager import DataManager


class DfViewer(QGroupBox):
    def __init__(self, data:pd.DataFrame=None):
        super().__init__()
        self.dm = DataManager.get_instance()
        self.data = self.dm.get_data() if data is None else data
        if self.data is not None:
            self.table_model = PandasModel(self.data)
            self.setup_widgets()


    def setup_widgets(self):
        self.setMinimumSize(900,300)
        self.setTitle('Dataframe viewer')
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addLayout(self.create_filter_bar())
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        #set table view
        self.table_view.resize(800, 500)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.setAlternatingRowColors(True)
        self.main_layout.addWidget(self.table_view)
        self.table_view.setModel(self.table_model)
        # self.table_view.horizontalHeader().setSortIndicatorShown(True)
        # self.table_view.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        # self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().sectionClicked.connect(self.header_click)

    def create_filter_bar(self):
        #display from head, tail, sample, type
        fb_layout = QHBoxLayout()
        lbl = QLabel('View:')
        self.cb = QComboBox()
        self.cb.addItems(['All', 'Head', 'Tail', 'Sample', 'Selected'])
        self.usr_input = QLineEdit()
        self.apply_btn = QPushButton('Apply')

        #connect signals
        self.cb.currentIndexChanged.connect(self.cb_action)
        self.apply_btn.clicked.connect(self.apply_btn_action)

        #add widgets to layout
        fb_layout.addWidget(lbl)
        fb_layout.addWidget(self.cb)
        fb_layout.addWidget(self.usr_input)
        fb_layout.addWidget(self.apply_btn)

        #Widget constraints
        self.usr_input.setVisible(False)
        lbl.setMaximumWidth(40)
        self.usr_input.setMaximumWidth(40)
        self.cb.setMaximumWidth(100)
        self.apply_btn.setMaximumWidth(100)

        #lineEdit constraints
        only_int = QIntValidator()
        only_int.setRange(0, 99)
        self.usr_input.setText('5')
        self.usr_input.setValidator(only_int)
        self.usr_input.setMaximumWidth(40)

        fb_layout.setAlignment(Qt.AlignLeft)
        return fb_layout

    def cb_action(self):
        """Controls the action of combo box, hiding and
        showing the user input LineEdit respectively"""
        if self.cb.currentText() == "All":
            self.usr_input.setVisible(False)
            self.apply_btn_action()
        else:
            self.usr_input.setHidden(False)

    def apply_btn_action(self):
        choice = self.cb.currentText()
        entries = int(self.usr_input.text())
        if choice == 'All':
            self.table_view.setModel(PandasModel(self.data))
        elif choice == 'Head':
            self.table_view.setModel(PandasModel(self.data.head(entries)))
        elif choice == 'Tail':
            self.table_view.setModel(PandasModel(self.data.tail(entries)))
        elif choice == 'Sample':
            self.table_view.setModel(PandasModel(self.data.sample(entries)))

    def refresh(self):
        self.data = self.dm.get_data()
        self.table_view.setModel(PandasModel(self.data))
        self.apply_btn_action()

    def header_click(self, i):
        try:
            self.table_view.setSortingEnabled(True)
            self.table_view.model().sort(i)
            self.table_view.setSortingEnabled(True)
        except Exception as e:
            print(e)

    def wipe(self):
        self.data = None

class PandasModel(QtCore.QAbstractTableModel):

    def __init__(self, data, parent=None):
        """

        :param data: a pandas dataframe
        :param parent:
        """
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        # self.headerdata = data.columns


    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, rowcol, orientation, role):

        #if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
        if orientation == QtCore.Qt.Horizontal and 0 <= rowcol < len(self._data.columns) and role == QtCore.Qt.DisplayRole:
            return self._data.columns[rowcol]
        #if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
        elif orientation == QtCore.Qt.Vertical and 0 <= rowcol < len(self._data.index) and role == QtCore.Qt.DisplayRole:
            return self._data.index[rowcol]
        return None

    def flags(self, index):
        flags = super(self.__class__, self).flags(index)
        flags |= QtCore.Qt.ItemIsEditable
        flags |= QtCore.Qt.ItemIsSelectable
        flags |= QtCore.Qt.ItemIsEnabled
        flags |= QtCore.Qt.ItemIsDragEnabled
        flags |= QtCore.Qt.ItemIsDropEnabled
        return flags

    def sort(self, n_col, order='Ascending'):
        """Sort table by given column number.
        """
        try:
            self.layoutAboutToBeChanged.emit()
            self._data = self._data.sort_values(self._data.columns[n_col], ascending=not order)
            self.layoutChanged.emit()
        except Exception as e:
            print(e)