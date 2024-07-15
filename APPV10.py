import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import subprocess

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '甲状腺结节超声图像诊断系统V1.0'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        
        layout = QVBoxLayout()

        self.button1 = QPushButton('批量图像分割分类', self)
        self.button1.clicked.connect(self.run_script1)
        layout.addWidget(self.button1)

        self.button2 = QPushButton('单一图像分割分类', self)
        self.button2.clicked.connect(self.run_script2)
        layout.addWidget(self.button2)

        self.setLayout(layout)
        self.show()

    def run_script1(self):
        subprocess.run(['python', 'UI.py'])

    def run_script2(self):
        subprocess.run(['python', 'UI_2.py'])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
