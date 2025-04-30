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

        self.file_explorer = FileExplorer()
        self.data_viewer = DataViewer()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.file_explorer)
        splitter.addWidget(self.data_viewer)
        splitter.setSizes([600,1000])

        splitter.addWidget(self.data_viewer)

        self.layout.addWidget(splitter)
        self.setCentralWidget(self.central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    # handle thread stopping on window close
    def on_close():
        try:
            ThreadManager.stop_all()     
        except:
            pass     

    app.aboutToQuit.connect(on_close)

    window.show()
    sys.exit(app.exec_())