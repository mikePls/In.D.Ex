import sys

import disclaimer_ui
from main_gui import MainGUIWindow
from PyQt5.QtWidgets import QApplication


if __name__ == "__main__":
    def start():
        window = MainGUIWindow()
        window.show()

    app = QApplication(sys.argv)
    disclaimer = disclaimer_ui.DisclaimerWindow()
    # if the user agrees to disclaimer, start the application
    disclaimer.ok_button.clicked.connect(start)

    sys.exit(app.exec())