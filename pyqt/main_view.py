import pyqtgraph as pg
import sys
import typing
import qtmodern.styles
import qtmodern.windows


from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt,QModelIndex
from pyqtgraph import PlotWidget, plot

try:
    from .data import SacredDB
    from .view import Ui_MainWindow
except:
    from data import SacredDB
    from view import Ui_MainWindow


class ExperimentListModel(QtCore.QAbstractListModel):
    def __init__(self,exp: list = None):
        super().__init__()
        self.exp = exp or []

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:

        if role == Qt.DisplayRole:
            return str(self.exp[index.row()])

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.exp)



class ExpWindow(QtWidgets.QMainWindow):
    def __init__(self,exp):
        super().__init__()

        self.setWindowTitle(str(exp))

        self.resize(800,800)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.plot([x for x in range(len(exp.loss_data))],exp.loss_data)

class MainWindow(QtWidgets.QMainWindow,Ui_MainWindow):

    def __init__(self):
        super().__init__()

        self._exp_window = None
        self._ip = "localhost"
        self._port = 27017
        self._dbname = "VikingZero"

        self._db = SacredDB(ip=self._ip,port=self._port,dbname=self._dbname)

        self.setupUi(self)
        self.model = ExperimentListModel()
        self.load()
        self.listView.setModel(self.model)
        self.listView.clicked.connect(self.exp_clicked)

    def exp_clicked(self,index):
        print(f"Exp clicked idex = {index.row()}")
        exp = self.model.exp[index.row()]

        self._exp_window = ExpWindow(exp)
        self._exp_window.show()


    def load(self):

        exps = self._db.load_all()

        self.model.exp = exps


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
qtmodern.styles.dark(app)
mw = qtmodern.windows.ModernWindow(window)
mw.show()
app.exec_()
