import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import inference_UI2

class WorkerThread(QtCore.QThread):
    predictionCompleted = QtCore.pyqtSignal(list, list)  # 修改这里

    def __init__(self, input_dir):
        super(WorkerThread, self).__init__()
        self.input_dir = input_dir

    def run(self):
        try:
            fenlei_instance = inference_UI2.fenlei()
            image_path, text_path = fenlei_instance.run(self.input_dir)  # 单个文件路径
            self.predictionCompleted.emit([image_path], [text_path])  # 转换为包含路径的列表
        except Exception as e:
            print(f"Error in WorkerThread: {e}")


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(708, 317)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 241, 231))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 20, 221, 201))
        self.label.setObjectName("label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(260, 0, 241, 231))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)  # 修改这里
        self.label_2.setGeometry(QtCore.QRect(10, 20, 221, 201))
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 240, 111, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 240, 101, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)  # 修改这里
        self.label_3.setGeometry(QtCore.QRect(560, 70, 81, 31))
        self.label_3.setObjectName("label_3")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(530, 110, 131, 31))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 708, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.uploadImage)
        self.pushButton_2.clicked.connect(self.startPrediction)

        self.workerThread = WorkerThread(input_dir='')
        self.workerThread.predictionCompleted.connect(self.predictionCompleted)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "甲状腺结节超声图像诊断系统V1.0"))
        self.groupBox.setTitle(_translate("MainWindow", "超声图像"))
        self.groupBox_2.setTitle(_translate("MainWindow", "分割展示"))
        self.pushButton.setText(_translate("MainWindow", "上传图像"))
        self.pushButton_2.setText(_translate("MainWindow", "开始预测"))
        self.label_3.setText(_translate("MainWindow", "分类预测结果:"))  # 修改这里

    def uploadImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "打开图片文件", "",
                                                  "图片文件 (*.png *.jpg *.bmp);;所有文件 (*)", options=options)
        if fileName:
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)
            
            # Set the input_dir for the WorkerThread
            self.workerThread.input_dir = fileName

    def startPrediction(self):
        self.workerThread.start()

    def predictionCompleted(self, image_paths, text_paths):
        if image_paths and text_paths:
            # 处理图片路径
            output_segmentation_file = image_paths[0]  # Assuming the first path is the segmentation image
            pixmap = QtGui.QPixmap(output_segmentation_file)
            pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
            self.label_2.setPixmap(pixmap)

            # 处理文本路径
            text_file_path = text_paths[0]  # Assuming the first path is the text file
            with open(text_file_path, 'r') as text_file:
                text_data = text_file.read()
                if text_data == '0':
                    self.textEdit.setPlainText(f"这是良性结节。")
                    # label_text = f"Label={text_content}，这是良性结节。"
                else:
                    self.textEdit.setPlainText(f"这是可疑恶性结节。")
                



class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
