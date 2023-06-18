import io
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QHBoxLayout, QPushButton


class Canvas(FigCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(5,4), dpi=200)
        super().__init__(fig)
        self.setParent(parent)

        t = np.arange(0.0, 2.0, 0.1)
        s = np.sin(2 * np.pi * t)
        self.ax.hist(s, bins=30)

        self.ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='Mini grid performance')
        self.ax.grid()


class Demo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1000, 800)
        self.layout = QHBoxLayout()
        self.widget1 = QWidget()
        self.widget2 = QWidget()
        self.layout.addWidget(self.widget1)
        #self.layout.addWidget(self.widget2)
        chart = Canvas(self.widget1)
        #chart2 = Canvas(self.widget2)

        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.layout)
        self.setCentralWidget(self.centralWidget)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        im = Image.open(img_buf)
        im.show(title="My Image")

        #img_buf.close()


app = QApplication(sys.argv)
window = Demo()
window.show()
sys.exit(app.exec())




