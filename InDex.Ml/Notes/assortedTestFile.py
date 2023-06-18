import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QLabel, QApplication, QMainWindow, QPushButton, QMenu

# class Window(QScrollArea):
#     def __init__(self):
#         super(Window, self).__init__()
#         widget = QWidget()
#         layout = QVBoxLayout(widget)
#         layout.setAlignment(Qt.AlignTop)
#         for index in range(100):
#             layout.addWidget(QLabel('Label %02d' % index))
#         self.setWidget(widget)
#         self.setWidgetResizable(True)
#
# app = QApplication(sys.argv)
# w = Window()
# w.show()
# sys.exit(app.exec())

from PyQt5 import QtGui, QtCore
import sys

from scipy.stats import iqr

from MLModules.visualisations_module import Visualizations

# class ButtonWithMenu(QMainWindow):
#     def __init__(self, parent=None):
#         super(Main, self).__init__(parent)
#         pushbutton = QPushButton('Popup Button')
#         menu = QMenu()
#         subm = menu.addMenu('sub menu')
#         subm.setLayoutDirection(QtCore.Qt.RightToLeft)
#         subm.addAction('This is Action 1', self.Action2)
#
#         menu.addAction('This is Action 2', self.Action2)
#         pushbutton.setMenu(menu)
#         self.setCentralWidget(pushbutton)
#
#     def Action1(self):
#         print('You selected Action 1')
#
#     def Action2(self):
#         print('You selected Action 2')


# if __name__ == '__main__':
#
#     app = QApplication(sys.argv)
#     main = Main()
#     main.show()
#     app.exec_()

# class test(QWidget):
#     def __init__(self):
#         super().__init__()
#         df=pd.read_csv('CarPrice_Assignment.csv')
#         col = df['carlength']
#         v = Visualizations()
#         plot = v.create_boxplot(col)
#
#         layout = QVBoxLayout()
#         layout.addWidget(plot)
#         self.setLayout(layout)
#
# app = QApplication(sys.argv)
# win = test()
# win.show()
# app.exec_()

# def get_outliers(data, factor=1.5):
#     limit1 = np.quantile(data, 0.25) - factor * iqr(data)
#     limit2 = np.quantile(data, 0.75) + factor * iqr(data)
#     outliers = data[(data < limit1) | (data > limit2)]
#     return outliers
#
# d = np.array([-10, 2,3,4,5, 20])
# print(get_outliers(d))


# class Check_no:
#
#     # decorator function
#     def decor(func):
#         def check(self, no):
#             func(self, no)
#             if no % 2 == 0:
#                 print('Yes, it\'s EVEN Number.')
#             else:
#                 print('No, it\'s ODD Number.')
#
#         return check
#
#     @decor
#     # instance method
#     def is_even(self, no):
#         print('User Input : ', no)
#
#
# obj = Check_no()
# obj.is_even(45)
# obj.is_even(2)
# obj.is_even(7)

# from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog
#
# class SaveDataFrameDialog(QDialog):
#     def __init__(self, data_frame):
#         super().__init__()
#         self.data_frame = data_frame
#         self.init_ui()
#
#     def init_ui(self):
#         self.setWindowTitle('Save DataFrame')
#         self.layout = QVBoxLayout()
#
#         # Add label
#         label = QLabel('Choose save format:')
#         self.layout.addWidget(label)
#
#         # Add buttons for CSV and TSV save
#         csv_button = QPushButton('Save as CSV')
#         tsv_button = QPushButton('Save as TSV')
#         self.layout.addWidget(csv_button)
#         self.layout.addWidget(tsv_button)
#
#         # Connect buttons to slots
#         csv_button.clicked.connect(self.save_csv)
#         tsv_button.clicked.connect(self.save_tsv)
#
#         self.setLayout(self.layout)
#
#     def save_csv(self):
#         file_name, _ = QFileDialog.getSaveFileName(self, 'Save as CSV', '', 'CSV files (*.csv)')
#         if file_name:
#             self.data_frame.to_csv(file_name, index=False)
#
#     def save_tsv(self):
#         file_name, _ = QFileDialog.getSaveFileName(self, 'Save as TSV', '', 'TSV files (*.tsv)')
#         if file_name:
#             self.data_frame.to_csv(file_name, sep='\t', index=False)
#
#
# df = pd.read_csv('CarPrice_Assignment.csv')
# app = QApplication(sys.argv)
# w = SaveDataFrameDialog(df)
# w.show()
# sys.exit(app.exec())
#
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QCheckBox, QPushButton, QVBoxLayout, QWidget
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.cluster import KMeans
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
#
#
# class MLModelPipeline(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle('ML Model Pipeline')
#         self.setGeometry(100, 100, 400, 400)
#         self.setup_ui()
#
#     def setup_ui(self):
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         self.layout = QVBoxLayout()
#
#         self.model_label = QLabel('Select Model:')
#         self.model_combobox = QComboBox()
#         self.model_combobox.addItems(
#             ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'KNN', 'K-means'])
#
#         self.transform_label = QLabel('Select Transformation:')
#         self.transform_combobox = QComboBox()
#         self.transform_combobox.addItems(['None', 'Polynomial Features', 'Log Transformation'])
#
#         self.scale_checkbox = QCheckBox('Standard Scaling')
#
#         self.hyperparam_label = QLabel('Select Hyperparameters:')
#         self.hyperparam_combobox = QComboBox()
#         self.hyperparam_combobox.addItems(['None', 'alpha', 'l1_ratio', 'n_neighbors', 'n_clusters'])
#
#         self.create_button = QPushButton('Create Model')
#         self.create_button.clicked.connect(self.create_model)
#
#         self.metrics_label = QLabel()
#
#         self.layout.addWidget(self.model_label)
#         self.layout.addWidget(self.model_combobox)
#         self.layout.addWidget(self.transform_label)
#         self.layout.addWidget(self.transform_combobox)
#         self.layout.addWidget(self.scale_checkbox)
#         self.layout.addWidget(self.hyperparam_label)
#         self.layout.addWidget(self.hyperparam_combobox)
#         self.layout.addWidget(self.create_button)
#         self.layout.addWidget(self.metrics_label)
#
#         self.central_widget.setLayout(self.layout)
#
#     def create_model(self):
#         model_type = self.model_combobox.currentText()
#         transform_type = self.transform_combobox.currentText()
#         scale = self.scale_checkbox.isChecked()
#         hyperparam_type = self.hyperparam_combobox.currentText()
#
#         # Create a sample dataset
#         X, y = create_sample_dataset()
#
#         # Split the dataset into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#         # Create a pipeline with selected options
#         pipeline_steps = []
#         if transform_type == 'Polynomial Features':
#             pipeline_steps.append(('polynomial_features', PolynomialFeatures()))
#         elif transform_type == 'Log Transformation':
#             pipeline_steps.append(('log_transformation', FunctionTransformer(np.log)))
#
#         if scale:
#             pipeline_steps.append(('scaling', StandardScaler()))
#
#         if hyperparam_type != 'None':
#             if model_type == 'Linear Regression':
#                 if hyperparam_type == 'alpha':
#                     pipeline_steps.append(('model', Ridge(alpha=0.5)))
#                 elif hyperparam_type == 'l1_ratio':
#                     pipeline_steps.append(('model', ElasticNet(l1_ratio=0)))
#             elif model_type == 'Ridge Regression':
#                 if hyperparam_type == 'alpha':
#                     pipeline_steps.append(('model', Ridge(alpha=0.5)))
#                 elif hyperparam_type == 'l1_ratio':
#                     pipeline_steps.append(('model', Ridge(alpha=0.5, solver='saga')))
#             elif model_type == 'Lasso Regression':
#                 if hyperparam_type == 'alpha':
#                     pipeline_steps.append(('model', Lasso(alpha=0.5)))
#                 elif hyperparam_type == 'l1_ratio':
#                     pipeline_steps.append(('model', Lasso(alpha=0.5, precompute=True, positive=True)))
#             elif model_type == 'Elastic Net':
#                 if hyperparam_type == 'alpha':
#                     pipeline_steps.append(('model', ElasticNet(alpha=0.5, l1_ratio=0.5)))
#                 elif hyperparam_type == 'l1_ratio':
#                     pipeline_steps.append(('model', ElasticNet(alpha=0.5, l1_ratio=0.5, precompute=True)))
#             elif model_type == 'KNN':
#                 if hyperparam_type == 'n_neighbors':
#                     pipeline_steps.append(('model', KNeighborsRegressor(n_neighbors=5)))
#             elif model_type == 'K-means':
#                 if hyperparam_type == 'n_clusters':
#                     pipeline_steps.append(('model', KMeans(n_clusters=3)))
#         else:
#             if model_type == 'Linear Regression':
#                 pipeline_steps.append(('model', LinearRegression()))
#             elif model_type == 'Ridge Regression':
#                 pipeline_steps.append(('model', Ridge(alpha=0.5)))
#             elif model_type == 'Lasso Regression':
#                 pipeline_steps.append(('model', Lasso(alpha=0.5)))
#             elif model_type == 'Elastic Net':
#                 pipeline_steps.append(('model', ElasticNet(alpha=0.5, l1_ratio=0.5)))
#             elif model_type == 'KNN':
#                 pipeline_steps.append(('model', KNeighborsRegressor(n_neighbors=5)))
#             elif model_type == 'K-means':
#                 pipeline_steps.append(('model', KMeans(n_clusters=3)))
#
#         pipeline = Pipeline(pipeline_steps)
#
#         # Train the model
#         pipeline.fit(X_train, y_train)
#
#         # Predict on test data
#         y_pred = pipeline.predict(X_test)
#
#         # Calculate metrics
#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
#
#         # Update metrics label
#         self.metrics_label.setText(f'Mean Squared Error: {mse:.2f}\nR2 Score: {r2:.2f}')
#
#
# def create_sample_dataset():
#     # Create a simple linear dataset for demonstration
#     import numpy as np
#     np.random.seed(42)
#     X = np.random.rand(100, 1)
#     y = 2 + 3 * X + np.random.randn(100, 1)
#     return X, y
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MLModelPipeline()
#     window.show()
#     sys.exit(app.exec_())

#######################################

# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea
# import sys
#
#
# class ScrollableWidget(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#
#         # Create a vertical layout to hold the labels
#         layout = QVBoxLayout(self)
#
#         # Add some labels to the layout
#         for i in range(20):
#             label = QLabel(f"Label {i+1}")
#             layout.addWidget(label)
#
#         # Create a scroll area and set its widget to the layout
#         scroll_area = QScrollArea(self)
#         scroll_area.setWidgetResizable(True)
#         scroll_area_widget = QWidget()
#         scroll_area_widget.setLayout(layout)
#         scroll_area.setWidget(scroll_area_widget)
#
#         # Set the main layout of the widget to the scroll area
#         main_layout = QVBoxLayout(self)
#         main_layout.addWidget(scroll_area)
#         self.setLayout(main_layout)
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = ScrollableWidget()
#     window.show()
#     sys.exit(app.exec_())

#######################################################

import sys
import time
import asyncio
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar, QLabel, QVBoxLayout

class LoadingThread(QThread):
    finished_signal = pyqtSignal()

    def run(self):
        # Simulate a long async method
        asyncio.run(self.async_method())
        self.finished_signal.emit()

    async def async_method(self):
        await asyncio.sleep(35) # Simulate a long process

class LoadingWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Loading Window')
        self.setFixedSize(250, 180)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet('background-color: white;')
        layout = QVBoxLayout(self)
        self.loading_label = QLabel()
        self.loading_label.setPixmap(QPixmap('loading.gif'))
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_bar = QProgressBar()
        self.loading_bar.setFixedSize(200, 20)
        self.loading_bar.setRange(0, 0) # Infinite progress
        self.loading_bar.setStyleSheet('QProgressBar {border: 2px solid grey; border-radius: 10px; padding: 1px;} \
                                         QProgressBar::chunk {background-color: #00BFFF; border-radius: 10px;}')
        self.time_label = QLabel('Elapsed time: 0 sec')
        self.time_label.setFont(QFont('Arial', 10))
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)
        layout.addWidget(self.loading_bar)
        layout.addWidget(self.time_label)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.start_time = time.time()
        self.thread = LoadingThread()
        self.thread.finished_signal.connect(self.close)
        self.thread.start()
        self.timer.start(1000) # Update every second

    def update_time(self):
        elapsed_time = time.time() - self.start_time
        self.time_label.setText(f'Elapsed time: {int(elapsed_time)} sec')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    loading_window = LoadingWindow()
    loading_window.show()
    sys.exit(app.exec_())

