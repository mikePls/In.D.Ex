from typing import List
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QWIDGETSIZE_MAX, QCheckBox


class CorrelationsWindow(QWidget):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.dataframe = dataframe
        self.setup_ui()

    def setup_ui(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Correlation App')
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.heatmap_canvas = self.create_heatmap()
        self.heatmap_canvas.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
        self.pairplot_canvas = self.create_pairplot()
        self.pairplot_canvas.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)

        #create checkboxes
        self.heatmap_checkbox = QCheckBox('Toggle heatmap', self)
        self.heatmap_checkbox.setChecked(True)
        self.heatmap_checkbox.stateChanged.connect(self.toggle_heatmap)
        self.pairplot_checkbox = QCheckBox('Toggle pairplot', self)
        self.pairplot_checkbox.setChecked(True)
        self.pairplot_checkbox.stateChanged.connect(self.toggle_pairplot)
        #add checkboxes to checkbox layout
        checkbox_lt = QHBoxLayout()
        checkbox_lt.setAlignment(Qt.AlignLeft)
        checkbox_lt.addWidget(self.heatmap_checkbox)
        checkbox_lt.addWidget(self.pairplot_checkbox)
        self.layout.addLayout(checkbox_lt)

        graph_lt = QHBoxLayout()
        graph_lt.addWidget(self.heatmap_canvas)
        graph_lt.addWidget(self.pairplot_canvas)

        self.layout.addLayout(graph_lt)

        self.show()

    def create_heatmap(self):
        fig, ax = plt.subplots()
        sns.heatmap(self.dataframe.corr(), annot=True, cmap='coolwarm', ax=ax)
        canvas = FigureCanvas(fig)
        canvas.setFixedSize(400, 400)
        return canvas

    def create_pairplot(self):
        g = sns.pairplot(self.dataframe)
        for ax in g.axes.flat:
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontsize(8)
        canvas = FigureCanvas(g.fig)
        canvas.setFixedSize(400, 400)
        return canvas

    def toggle_heatmap(self):
        if self.heatmap_canvas.isVisible():
            self.heatmap_canvas.setVisible(False)
            self.pairplot_canvas.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
        else:
            self.heatmap_canvas.setVisible(True)
            self.pairplot_canvas.setMaximumSize(400, 400)

    def toggle_pairplot(self):
        if self.pairplot_canvas.isVisible():
            self.pairplot_canvas.setVisible(False)
            self.heatmap_canvas.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
        else:
            self.pairplot_canvas.setVisible(True)
            self.heatmap_canvas.setMaximumSize(400, 400)

##test code
if __name__ == '__main__':
    # Test sample data
    data = {'A': [1, 2, 3, 4, 5], 'B': [2, 3, 1, 5, 4], 'C': [5, 4, 3, 2, 1], 'D': [4, 5, 2, 1, 3]}
    df = pd.DataFrame(data)

    app = QApplication([])
    corr_app = CorrelationsWindow(df)
    app.exec_()
