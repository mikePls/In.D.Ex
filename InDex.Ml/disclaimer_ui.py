import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTextEdit, QPushButton, QWidget, QHBoxLayout


class DisclaimerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("In.D.Ex - Disclaimer Agreement")
        self.resize(400, 300)
        os.path.dirname(os.path.dirname(__file__))
        icon = QIcon('res/ico/appico.png')
        self.setWindowIcon(icon)

        # Create central
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create vertical layout for central widget
        layout = QVBoxLayout(central_widget)

        # Create scrollable text field
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        file = open('Disclaimer.txt', 'r')
        msg = file.read()
        file.close()
        self.text_edit.setPlainText(msg)

        # OK button
        self.ok_button = QPushButton("Accept")
        self.ok_button.setEnabled(False)  # Disable the button initially

        # Decline button
        self.decline_button = QPushButton("Decline")
        self.decline_button.clicked.connect(self.close)

        # Connect button actions
        self.text_edit.verticalScrollBar().valueChanged.connect(self.scroll_bar_val_changed)
        self.ok_button.clicked.connect(self.close)

        # Add buttons to the layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.ok_button)
        btn_layout.addWidget(self.decline_button)
        layout.addLayout(btn_layout)

        self.setMinimumSize(550, 550)
        self.show()

    def scroll_bar_val_changed(self, value):
        if value == self.text_edit.verticalScrollBar().maximum():
            self.ok_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication([])
    window = DisclaimerWindow()
    app.exec()
