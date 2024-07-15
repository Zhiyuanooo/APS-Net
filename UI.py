import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QFileDialog, QLabel, QScrollArea, QHBoxLayout, QTextBrowser
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QPixmap
import inference


class WorkerThread(QThread):
    result_ready = pyqtSignal(list, list)

    def __init__(self, input_dir=None, output_dir=None):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        try:
            fenlei_instance = inference.fenlei()
            image_paths, text_paths = fenlei_instance.run(self.input_dir, self.output_dir)
            self.result_ready.emit(image_paths, text_paths)
        except Exception as e:
            print(f"Error in WorkerThread: {e}")


class ImageWithLabel(QWidget):
    def __init__(self, image_path, label_text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        # 获取序号
        index = int(image_path.split('\\')[-1].split('.')[0])
        index_label = QLabel(f"{index}:", self)
        layout.addWidget(index_label)

        pixmap = QPixmap(image_path)
        image_label = QLabel(self)
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)

        label = QTextBrowser(self)
        label.setPlainText(label_text)
        layout.addWidget(label)

        self.setLayout(layout)


class FileFolderSelection(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 创建垂直布局
        layout = QVBoxLayout(self)
        self.setWindowTitle("甲状腺结节超声图像诊断系统V1.0")

        # 第一行：选择输入文件夹按钮
        self.btn_select_input_folder = QPushButton('选择输入文件夹', self)
        self.btn_select_input_folder.clicked.connect(self.select_input_folder)
        layout.addWidget(self.btn_select_input_folder)

        # 第二行：显示输入文件夹路径的编辑框
        self.input_folder_path = QLineEdit(self)
        self.input_folder_path.setReadOnly(True)
        layout.addWidget(self.input_folder_path)

        # 第三行：选择输出文件夹按钮
        self.btn_select_output_folder = QPushButton('选择输出文件夹', self)
        self.btn_select_output_folder.clicked.connect(self.select_output_folder)
        layout.addWidget(self.btn_select_output_folder)

        # 第四行：显示输出文件夹路径的编辑框
        self.output_folder_path = QLineEdit(self)
        self.output_folder_path.setReadOnly(True)
        layout.addWidget(self.output_folder_path)

        # 第五行：开始任务的按钮
        self.btn_start_task = QPushButton('开始任务', self)
        self.btn_start_task.clicked.connect(self.start_long_running_task)
        layout.addWidget(self.btn_start_task)

        # 第六行：使用滚动区域显示多个图像的 QLabel 和 标签
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.image_container = QWidget(self)
        self.image_layout = QVBoxLayout(self.image_container)
        self.scroll_area.setWidget(self.image_container)
        layout.addWidget(self.scroll_area)

        # 设置窗口布局
        self.setLayout(layout)

    @pyqtSlot()
    def select_input_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder_path:
            self.input_folder_path.setText(folder_path)

    @pyqtSlot()
    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder_path:
            self.output_folder_path.setText(folder_path)

    @pyqtSlot()
    def start_long_running_task(self):
        # 创建一个工作线程并启动它
        self.worker_thread = WorkerThread(
            input_dir=self.input_folder_path.text(),
            output_dir=self.output_folder_path.text()
        )
        self.worker_thread.result_ready.connect(self.update_ui)
        self.worker_thread.start()

    def update_ui(self, image_paths, text_paths):
        # 清空之前的结果
        for i in reversed(range(self.image_layout.count())):
            self.image_layout.itemAt(i).widget().setParent(None)

        # 显示新的结果
        for image_path, text_path in zip(image_paths, text_paths):
            with open(text_path, 'r') as f:
                text_content = f.read()
                if text_content == '0':
                    label_text = f"Label={text_content}，这是良性结节。"
                else:
                    label_text = f"Label={text_content}，这是可疑恶性结节。"
                image_widget = ImageWithLabel(image_path, label_text, self)
                self.image_layout.addWidget(image_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileFolderSelection()
    ex.show()
    sys.exit(app.exec_())
