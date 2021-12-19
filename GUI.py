#!/usr/bin/env python
# coding:utf-8
import sys
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from DE.solver import *
from EED.loadmodel import *
# from mainWindow import Ui_MainWindow
from Window import Ui_MainWindow


class MainWindow(QMainWindow):
    _startThread = pyqtSignal()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = None
        self.F = 0.4
        self.CR = 0.8
        self.nPop = 100
        self.progress = 0
        self.nIter = 500
        self.demand = 2.834
        self.initial = 0
        self.mutation = 0
        self.ui.pushButton_3.clicked.connect(self.file_button)
        self.ui.pushButton_2.clicked.connect(self.solve_button)
        # self.ui.progressBar.isSignalConnected(self.progress)
        self.ui.textEdit_4.textChanged.connect(self.demand_input)
        self.ui.textEdit_5.textChanged.connect(self.text_1)
        self.ui.textEdit_6.textChanged.connect(self.text_2)
        self.ui.textEdit_7.textChanged.connect(self.text_3)
        self.ui.textEdit_8.textChanged.connect(self.text_4)
        self.ui.comboBox.currentIndexChanged.connect(self.initial_selection)
        self.ui.comboBox_2.currentIndexChanged.connect(self.mutation_selection)
        self.ui.pushButton_4.clicked.connect(self.stop)
        self.solver = None
        self.thread = QThread(self)

    def file_button(self):
        file = QFileDialog.getOpenFileName(self, "*.txt")[0]
        self.model = Model(file)
        self.ui.textEdit.setText('2')
        self.ui.textEdit_2.setText(str(self.model.nGen))
        self.ui.textEdit_3.setText('1')
        for i in range(self.model.nGen):
            self.ui.tableWidget_2.insertRow(i)
        for row in range(self.model.nGen):
            self.ui.tableWidget_2.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.ui.tableWidget_2.setItem(row, 1, QTableWidgetItem(str(self.model.P[row][0])))
            self.ui.tableWidget_2.setItem(row, 2, QTableWidgetItem(str(self.model.P[row][1])))
            for j in range(5):
                self.ui.tableWidget_2.setItem(row, 3 + j, QTableWidgetItem(str(self.model.C[row][j])))
            for j in range(5):
                self.ui.tableWidget_2.setItem(row, 8 + j, QTableWidgetItem(str(self.model.E[row][j])))
        # self.ui.tableWidget_2.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def demand_input(self):
        self.demand = float(self.ui.textEdit_4.toPlainText()) / 100

    def text_1(self):
        self.nPop = int(self.ui.textEdit_5.toPlainText())

    def text_2(self):
        self.nIter = int(self.ui.textEdit_6.toPlainText())

    def text_3(self):
        self.F = float(self.ui.textEdit_7.toPlainText())

    def text_4(self):
        self.CR = float(self.ui.textEdit_8.toPlainText())

    def initial_selection(self, i):
        self.initial = i

    def mutation_selection(self, i):
        self.mutation = i

    def updateProgess(self, progress):
        self.ui.progressBar.setValue(int(progress))

    def updateImage(self, path):
        img = cv2.imread(path)  # 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(0.48)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.show()

    def updateTable_1(self, path):
        self.ui.tableWidget_4.setRowCount(0)
        self.ui.tableWidget_4.clearContents()
        data = open(path, 'r').readlines()
        for i in range(3):
            self.ui.tableWidget_4.insertRow(i)
            row_data = data[i].split()
            self.ui.tableWidget_4.setItem(i, 0, QTableWidgetItem(row_data[0]))
            for j in range(1, 9):
                self.ui.tableWidget_4.setItem(i, j, QTableWidgetItem(row_data[j][:5]))

    def updateTable_2(self, path):
        self.ui.tableWidget_3.setRowCount(0)
        self.ui.tableWidget_3.clearContents()
        data = open(path, 'r').readlines()
        for i in range(len(data)):
            self.ui.tableWidget_3.insertRow(i)
            row_data = data[i].split()
            self.ui.tableWidget_3.setItem(i, 0, QTableWidgetItem(row_data[0]))
            for j in range(1, 9):
                self.ui.tableWidget_3.setItem(i, j, QTableWidgetItem(row_data[j][:5]))

    def start(self):
        if self.thread.isRunning():
            return
        self.solver.flag = True
        self.thread.start()
        self._startThread.emit()

    def stop(self):
        if not self.thread.isRunning():
            return
        self.solver.flag = False
        self.stop_thread()

    def stop_thread(self):
        if not self.thread.isRunning():
            return
        self.thread.quit()
        self.thread.wait()

    def solve_button(self):
        arguments = {'nIter': self.nIter, 'nPop': self.nPop, 'F': self.F, 'CR': self.CR, 'init': 0}
        # DE = MMODE(model=self.model, **arguments)
        # DE.solve(self.demand, GUI=self)
        self.solver = MMODE_Thread(self.model, self.demand, **arguments)
        self.solver.moveToThread(self.thread)
        self._startThread.connect(self.solver.run)
        self.solver.signal_1.connect(self.updateProgess)
        self.solver.signal_2.connect(self.updateImage)
        self.solver.signal_3.connect(self.updateTable_1)
        self.solver.signal_4.connect(self.updateTable_2)
        self.start()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
