import pyqtgraph as pg
import sys
import typing
import qtmodern.styles
import qtmodern.windows

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt,QModelIndex
from PyQt5.QtWidgets import QWidget

try:
    from dashapp.data import SacredDB
    from .view import Ui_MainWindow
except:
    from dashapp.data import SacredDB
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


class GameListModel(QtCore.QAbstractListModel):
    def __init__(self,games: dict = None):
        super().__init__()
        self.games = games or []

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:

        if role == Qt.DisplayRole:
            return str(list(self.games.keys())[index.row()])

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.games.keys())


class ExpWindow(QtWidgets.QMainWindow):
    def __init__(self,exp):
        super().__init__()

        self.setWindowTitle(str(exp))
        self.resize(800,800)

        ########################
        # Load game model data
        ########################
        self.model = GameListModel()
        self.model.games = exp.games

        self.graphWidget = pg.PlotWidget()

        self.verticalLayout = QtWidgets.QVBoxLayout()

        widget = QWidget()
        widget.setLayout(self.verticalLayout)

        self.listView = QtWidgets.QListView(widget)
        self.listView.setModel(self.model)

        self.verticalLayout.addWidget(self.graphWidget)
        self.verticalLayout.addWidget(self.listView)

        self.graphWidget.plot([x for x in range(len(exp.loss_data))],exp.loss_data)

        self.setCentralWidget(widget)


class MainWindow(QtWidgets.QMainWindow):

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
        self.expList.setModel(self.model)
        self.expList.clicked.connect(self.set_right_panel)

    def exp_clicked(self,index):
        print(f"Exp clicked idex = {index.row()}")
        exp = self.model.exp[index.row()]

        self._exp_window = ExpWindow(exp)
        self._exp_window.show()

    def load(self):

        exps = self._db.load_all()

        self.model.exp = exps

    def set_right_panel(self,index):

        right_widget = QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(QtWidgets.QLabel(f"Index = {index}"))
        self.right_layout = right_layout
        #self.right_l


    def setupUI(self):

        self.resize(1200,1200)

        self.front_layout = QtWidgets.QHBoxLayout()

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.front_layout)

        # Left Panel
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.setSpacing(20)
        self.expList = QtWidgets.QListView(self.main_widget)
        self.left_layout.addWidget(self.expList)

        # right Panel
        #self.right_layout = QtWidgets.QVBoxLayout()
        self.set_right_panel(0)

        #self.front_layout.addWidget(self.)

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
qtmodern.styles.dark(app)
mw = qtmodern.windows.ModernWindow(window)
mw.show()
app.exec_()
