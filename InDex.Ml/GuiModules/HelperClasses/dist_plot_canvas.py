import numpy as np

from PIL import Image
import PIL
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QSizePolicy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas


class DistCanvas(FigCanvas):
    def __init__(self, data=None, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(2,2))
        super().__init__(self.fig)

        #drop NaN values in object columns as they cause plt to crash
        if data is not None and data.dtype == np.object:
            data = data.dropna()

        self.ax.hist(data, bins=30)

        #self.ax.grid()
        self.draw()

       # self.fig.subplots_adjust(0.2, 0.2, 0.8, 0.8)  # left,bottom,right,top
        plt.close()

    def get_png_fig(self):
        return PIL.Image.frombytes('RGB', self.fig.canvas.get_width_height(),
                                   self.fig.canvas.tostring_rgb()).toqpixmap()