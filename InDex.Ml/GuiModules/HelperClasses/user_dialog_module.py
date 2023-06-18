from PyQt5.QtWidgets import QMessageBox, QWidget


def msg_dlg(parent:QWidget, msg: str, type: str = None, title:str=None):
    mb = QMessageBox()
    title = 'Column transformer'
    if type == 'info':
        return mb.information(parent, title, msg, mb.Ok)
    if type == 'warn':
        reply = mb.warning(parent, title, msg, mb.Yes | mb.No)
    elif type == 'error':
        reply = mb.critical(parent, title, msg, mb.Ok)
    else:
        reply = mb.question(parent, title, msg, mb.Yes | mb.No)

    return reply == QMessageBox.Yes