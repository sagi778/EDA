from widgets import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window layout
        self.setWindowTitle('Data Explorer')
        self.setGeometry(100, 100, 1000, 900)

        # Create a central widget with a horizontal layout
        self.central_widget = QWidget(self)
        self.layout = QHBoxLayout(self.central_widget)

        self.file_tree = FileTree()
        DATA_TABLE['file_tree'] = self.file_tree
        self.file_explorer = FileExplorer(file_tree=self.file_tree)
        self.data_viewer = DataViewer()
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.file_explorer)
        self.splitter.addWidget(self.file_tree)
        self.splitter.addWidget(self.data_viewer)

        self.layout.addWidget(self.splitter)
        self.setCentralWidget(self.central_widget)
        self.splitter.setSizes([400,300,1300])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())