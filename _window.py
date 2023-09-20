from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from _ui import Ui_MainWindow
from _logic import _logic_MainWindow

"""Класс для создания логики у интерфейса"""
class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self) #здесь привязываем к интерфейсу данный класс

        self._add_method()

    """Добавляем обработчики к событиям"""
    def _add_method(self):
        self.ui.price_btn.clicked.connect(self._price_btn)

    def _price_btn(self):
        input_data = [15.0]
        input_data.append(self.ui.year.text())
        input_data.append(self.ui.engine_capacity.toPlainText())

        if self.ui.transmission_type.currentText() == "manual":
            input_data.append(0)
        else:
            input_data.append(1)

        input_data.append(self.ui.kms_driven.toPlainText())
        input_data.append(self.ui.owner_type.toPlainText())

        if self.ui.fuel_type.currentText() == "Petrol":
            input_data.append(2)
        elif self.ui.fuel_type.currentText() == "Diesel":
            input_data.append(3)

        input_data.append(self.ui.max_power.toPlainText())
        input_data.append(self.ui.seats.currentText())
        input_data.append(15.5)

        if self.ui.body_type.currentText() == "Sedan":
            input_data.append(0)
        elif self.ui.body_type.currentText() == "SUV":
            input_data.append(1)
        elif self.ui.body_type.currentText() == "Hatchback":
            input_data.append(2)
        elif self.ui.body_type.currentText() == "MUV":
            input_data.append(3)
        else:
            input_data.append(4)

        flag = True
        for elem in input_data:
            if elem == "":
               self._info()
               flag = False
               break
        if flag:
            input_data = [float(i) for i in input_data]
            _logic_MainWindow(input_data)
        print(input_data)


    def _info(self):
        error = QMessageBox()
        error.setWindowTitle("Ошибка!")
        error.setText("Заполните все поля")
        error.setIcon(QMessageBox.Warning)
        error.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        error.exec_()








