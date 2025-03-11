from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout,QComboBox,QLineEdit,QApplication, QMainWindow,QWidget, QLabel,QPushButton,QFrame
from PyQt5.QtWidgets import QTreeView, QFileSystemModel, QSplitter, QMessageBox,QTabWidget,QScrollArea

from PyQt5.QtGui import QColor,QIcon,QPixmap
from PyQt5.QtCore import QDir,Qt,QModelIndex,QSize
import sys
from func import *

class CodeLine(QWidget):
    def __init__(self,text:str='CodeLine'):
        super().__init__()
        self.setObjectName("CodeLine")
        
        self._text = text
        self._current_args = None 

        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(1)

        # play button
        BUTTON_STYLE = f"""
            QPushButton {{
                background-color: {CONFIG["Controls"]["background-color"]}; 
                padding: 8px; 
                width: 15px;
                min-width: 15px;
                max-width: 15px;
                height: 15px;
                min-height: 15px;
                max-height: 15px;
                border-radius: 15px;
                border: 1;
            }}
            QPushButton:hover {{
                background-color: {CONFIG["Controls"]["hover-color"]};
                border-color: {CONFIG["Controls"]["hover-color"]}
            }}
        """
        self.play = QPushButton('', self)
        self.play.setIcon(QIcon(f'{CURRENT_PATH}/icons/play.png'))
        self.play.setStyleSheet(BUTTON_STYLE)
        self.play.clicked.connect(self.run_command)
        layout.addWidget(self.play)

        # code line
        self.line = QLineEdit(self)
        self.line.setText(self._text)
        #self.line.setPlaceholderText(self._text)
        self.line.setStyleSheet(f'''
            QLineEdit {{
                font:{CONFIG['CodeLine']['font']};
                color:{CONFIG['CodeLine']['color']};
                background-color:{CONFIG['CodeLine']['background-color']};
                border:{CONFIG['CodeLine']['border']};
                border-radius:{CONFIG['CodeLine']['border-radius']};
                width: 100px;
                height: 25px;
            }}   

            QLineEdit:focus {{
                color: {CONFIG['CodeLine']['focus']['color']};
                background-color: {CONFIG['CodeLine']['focus']['background-color']};
                border-color: {CONFIG['CodeLine']['focus']['border-color']};
            }} 
        ''')

        self.line.returnPressed.connect(self.run_command)

        layout.addWidget(self.line)
        self.setLayout(layout)
        self._current_args = self.get_current_parameters(self.line.text())
        #print(f"self._current_args = {self._current_args}") # monitor
    
    def set_text(self,text):
        self.line.setText(text)
        self._text =text
    def run_command(self):
        def set_marks_to_text(text):
            if not text.startswith('>> '):
                text = '>> ' + text
            return text
        def get_args(text):
            def get_splits(text):
                pattern = r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)'
                return re.split(pattern, text)

            try:
                args_string = text[text.find('(')+1:text.find(')')]
                splitted_args = get_splits(args_string)
                print(splitted_args)
                return {item.split('=')[0]:item.split('=')[1] for item in splitted_args}
            except Exception as e:
                return e     
        
        #print(f"code line run Command: {set_marks_to_text(self.line.text())[3:]}") # monitor
        self.line.setText(set_marks_to_text(self.line.text()))     
        current_cmd_block = self.parent().parent()
        current_cmd_block._cmd = set_marks_to_text(self.line.text())[3:] # update new cmd
        current_cmd_block._args = get_args(current_cmd_block._cmd) # get cmd args
        current_cmd_block._output = 'my output'
        current_cmd_block.run_command()    
    def get_current_parameters(self,text): 
        cmd_string = text[text.find('(')+1:text.find(')')]
        matches = re.findall(r"([^=,]+)=('.*?'|[^,]+)",cmd_string)   
        print(f"matches = {matches}") # monitor
    
        return {item[0]:item[1] for item in matches}
class CodeControls(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("CodeControls")

        ICON_SIZE = 32

        left_layout = QHBoxLayout()
        left_layout.setAlignment(Qt.AlignLeft)
        left_layout.setSpacing(1)

        right_layout = QHBoxLayout()
        right_layout.setAlignment(Qt.AlignRight)
        right_layout.setSpacing(1)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        BUTTON_STYLE = f"""
            QPushButton {{
                background-color: {CONFIG["Controls"]["background-color"]}; 
                padding: 6px; 
                width: 20px;
                min-width: 20px;
                max-width: 20px;
                border-radius: {CONFIG["Controls"]["border-radius"]};
                border: {CONFIG["Controls"]["border"]};
            }}
            QPushButton:hover {{
                background-color: {CONFIG["Controls"]["hover-color"]};
                border-color: {CONFIG["Controls"]["hover-color"]}
            }}
        """
        # left side buttons    
        self.up = QPushButton('', self)
        self.up.setIcon(QIcon(f'{CURRENT_PATH}/icons/arrow-up.png'))
        self.up.setStyleSheet(BUTTON_STYLE)
        self.up.clicked.connect(self.set_up)
        left_layout.addWidget(self.up)

        self.down = QPushButton('', self)
        self.down.setIcon(QIcon(f'{CURRENT_PATH}/icons/arrow-down.png'))
        self.down.setStyleSheet(BUTTON_STYLE)
        self.down.clicked.connect(self.set_down)
        left_layout.addWidget(self.down)

        self.add = QPushButton('', self)
        self.add.setIcon(QIcon(f'{CURRENT_PATH}/icons/add.png'))
        self.add.setStyleSheet(BUTTON_STYLE)
        self.add.clicked.connect(self.set_add)
        left_layout.addWidget(self.add)

        # right side buttons
        self.pin = QPushButton('', self)
        self.pin.setIcon(QIcon(f'{CURRENT_PATH}/icons/thumbtack.png'))
        self.pin.setStyleSheet(BUTTON_STYLE)
        self.pin.clicked.connect(self.set_pin)
        right_layout.addWidget(self.pin)

        self.trash = QPushButton('', self)
        self.trash.setIcon(QIcon(f'{CURRENT_PATH}/icons/trash.png'))
        self.trash.setStyleSheet(BUTTON_STYLE)
        self.trash.clicked.connect(self.set_delete)
        right_layout.addWidget(self.trash)

        self.save = QPushButton('', self)
        self.save.setIcon(QIcon(f'{CURRENT_PATH}/icons/save.png'))
        self.save.setStyleSheet(BUTTON_STYLE)
        self.save.clicked.connect(self.set_save)
        right_layout.addWidget(self.save)

        self.fold = QPushButton('', self)
        self.fold.setIcon(QIcon(f'{CURRENT_PATH}/icons/fold.png'))
        self.fold.setStyleSheet(BUTTON_STYLE)
        self.fold.clicked.connect(self.func1)
        right_layout.addWidget(self.fold)

        self.setLayout(layout)

    def set_pin(self):
        current_cmd_block = self.parent().parent()
        layout = current_cmd_block.parent().layout()
        layout.removeWidget(current_cmd_block)
        layout.insertWidget(0,current_cmd_block)
    def set_up(self):
        current_cmd_block = self.parent().parent()
        layout = current_cmd_block.parent().layout()
        current_index = layout.indexOf(current_cmd_block)
        if current_index > 0:
            layout.removeWidget(current_cmd_block)
            layout.insertWidget(current_index - 1, current_cmd_block)
    def set_down(self):
        current_cmd_block = self.parent().parent()
        layout = current_cmd_block.parent().layout()
        current_index = layout.indexOf(current_cmd_block)
        if current_index < layout.count() - 1:
            layout.removeWidget(current_cmd_block)
            layout.insertWidget(current_index + 1, current_cmd_block)
    def set_add(self):
        current_cmd_block = self.parent().parent()
        layout = current_cmd_block.parent().layout()
        current_index = layout.indexOf(current_cmd_block)
        layout.insertWidget(current_index + 1, CommandBlock(cmd=current_cmd_block._cmd))        
    def set_delete(self):
        current_cmd_block = self.parent().parent()
        layout = current_cmd_block.parent().layout()
        current_index = layout.indexOf(current_cmd_block)
        layout.removeWidget(current_cmd_block)
        current_cmd_block.deleteLater()
    def set_save(self): 
        current_cmd_block = self.parent().parent()
        layout = current_cmd_block.parent().layout()
        data_viewer = current_cmd_block.parent().parent().parent().parent().parent().parent()
        story_layout = data_viewer.story_tab_layout
        story_layout.addWidget(current_cmd_block)
        story_layout.update()
        layout.update()

    def func1(self):
        print("Function 1 executed")    
class ArgsMenu(QWidget):
    def __init__(self,args:dict):
        super().__init__()
        self.setObjectName("ArgsMenu")

        self._args = {}
        MAX_ARG_WIDTH = 220

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0,0,10,0)
        self.layout.setAlignment(Qt.AlignRight) 
        
        for arg_name,arg in args.items():
            #print(f"arg_name={arg_name}; arg={arg}") # monitor
            if arg['type'] == 'category':
                combo = QComboBox()
                combo.addItems(arg['options'])
                combo.setCurrentIndex(0)  # Set default selection
                combo.setFixedWidth(min(MAX_ARG_WIDTH,20 + max([12*len(str(item)) for item in arg['options']])))
                combo.setStyleSheet(
                    "QComboBox { "
                    f"font: {CONFIG['arguments']['font']}; "
                    "color: purple;"
                    "padding: 1px 1px; "
                    "border: 1px solid #dedede; "
                    "border-radius: 5px; "
                    f"background-color: #f9f0fa; "
                    "selection-background-color: #dedede; "
                    "}"
                    "QComboBox::down-arrow {"
                        "border: none;" 
                        "background: none;" 
                    "}"
                    "QComboBox::drop-down {"
                        "border: none;" 
                    "}"
                )
                combo.currentIndexChanged.connect(self.update_command)
                self.layout.addWidget(combo)

            elif arg['type'] == 'int':    
                int_arg = QLineEdit()
                int_arg.setText(str(arg['default']))  # Set default text
                int_arg.setFixedWidth(min(MAX_ARG_WIDTH,max([12*len(str(item)) for item in arg['options']])))
                int_arg.setStyleSheet(
                    "QLineEdit { "
                    f"font: {CONFIG['arguments']['font']}; "
                    "color: green;"
                    "padding: 1px 1px; "
                    "border: 1px solid #dedede; "
                    "border-radius: 5px; "
                    f"background-color: #f0faf7; "
                    "selection-background-color: #eafaf1; "
                    "}"
                )
                int_arg.textChanged.connect(self.update_command)
                self.layout.addWidget(int_arg)
        
        self.setLayout(self.layout)

    def update_command(self):      
        def set_new_parameters_to_cmd(cmd:str,args:ArgsMenu) -> str:
            def delete_func_parameters(cmd:str): 
                new_cmd = []
                delete_flag = False
                for char in cmd:
                    if char in ['=',',',')']:
                        delete_flag = not delete_flag
                    elif delete_flag == True:
                        new_cmd += ''
                        continue
                    new_cmd += char        

                return ''.join(new_cmd)
            def get_new_parameters(args:ArgsMenu): 
                param = []
                for item in args.children():
                    #print(item)
                    if isinstance(item, QComboBox):
                        #print(f'QComboBox = {item.currentText()}')
                        param.append(item.currentText())
                    elif isinstance(item, QLineEdit):
                        #print(f'QLineEdit = {item.text()}')   
                        param.append(item.text()) 
                return param 

            cmd_string = delete_func_parameters(cmd)
            for item in get_new_parameters(args):
                cmd_string = cmd_string.replace('=',f"*{item}",1)
                print(cmd_string)

            return cmd_string.replace('*','=')   

        # set code line
        code_line = self.parent().findChild(CodeLine)
        code_line.set_text(set_new_parameters_to_cmd(cmd=code_line._text,args=self))
        code_line._current_args = code_line.get_current_parameters(code_line.line.text())

        # set arguments fields
        #print(self.children()[1:])
        for item,current_arg in zip(self.children()[1:],code_line._current_args.values()):
            #print(f"item={item}, current_arg={current_arg}") # monitor
            if isinstance(item,QComboBox):
                item.setCurrentText(current_arg)
            elif isinstance(item,QLineEdit):
                item.setText(current_arg)    
        
class CommandBlock(QWidget):
    def __init__(self,data:pd.DataFrame=DATA_TABLE,cmd:str=''):
        super().__init__()
        self.setObjectName("CommandBlock")
        
        self._cmd = cmd
        self._args = None
        self._output = None
        self._args = None

        # frame
        self.block = QFrame(self)
        self.block.setFrameShape(QFrame.StyledPanel)
        self.block.setFrameShadow(QFrame.Raised)
        self.block.setStyleSheet(f"""
            background-color: white;
            border: 1px solid #dedede;
            border-radius: 15px;
        """)
        self.block.setFixedSize(900,120) 
        
        # Create layout
        self.layout = QVBoxLayout(self.block)
        
        # Add sub-widgets
        self.code_controls = CodeControls()
        self.code_line = CodeLine(self._cmd)
        self.layout.addWidget(self.code_controls)
        self.layout.addWidget(self.code_line)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(1)
        main_layout.addWidget(self.block)
        
        self.setLayout(main_layout)
    
    def get_attributes(self):
        return {'self._cmd':f"{self._cmd}",'self._args':f"{self._args}",'self._output':f"{self._output}"}    
    def run_command(self):
        def delete_prev_output(self):
            for item_type in [ArgsMenu,TextOutput,TableOutput,ChartOutput,PlotOutput]:
                try:
                    output_widget = self.findChild(item_type)
                    output_widget.setParent(None)
                    output_widget.deleteLater()
                except:
                    pass      

        #print("command block run command") # monitor
        #print(f"current att: {self.get_attributes()}") # monitor
        delete_prev_output(self)

        df = DATA_TABLE['df']
        output_obj = eval(self._cmd)

        self._args = ArgsMenu(output_obj['args'])
        self.layout.addWidget(self._args)

        if output_obj['output_type'] == 'text':
            self._output = TextOutput(output_obj['output'])
        elif output_obj['output_type'] == 'table':
            self._output = TableOutput(output_obj['output'])
        elif output_obj['output_type'] == 'plot':  
            self._output = PlotOutput(output_obj['output'])      
        elif output_obj['output_type'] == 'chart':  
            self._output = ChartOutput(output_obj['output'])  

        self.block.setFixedHeight(self._output._rows)
        self.layout.addWidget(self._output)
    
# data viewer
class DataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("DataViewer")

        layout = QVBoxLayout()
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ 
                border: none; 
                border-radius: 15px;
            }}
            QTabBar::tab {{ 
                background: none; 
                padding: 1px; 
                font: 13px Consolas;
                color: {CONFIG['DataViewer']['unselected-tab-color']};
                border: {CONFIG['DataViewer']['border']};
                border-radius: 10px; 
                margin: 1px; 
            }}
            QTabBar::tab:selected {{ 
                background: {CONFIG['DataViewer']['selected-tab-color']}; 
                font: 14px Consolas;
                color: {CONFIG['FileExplorer']['color']}; 
                font-weight: bold; 
            }}
            QTabBar::tab:hover {{
                color: {CONFIG['FileExplorer']['color']}; 
                background-color: {CONFIG['DataViewer']['hover-color']};
            }}
            QScrollArea {{
                border: none;
                background: none;
            }}
            QScrollBar:vertical {{
                border: none;
                background: none; 
                width: 15px; /* Scrollbar width */
                margin: 0px;
                border-radius: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: white; 
                min-height: 20px;
                border-radius: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {CONFIG['CodeLine']['color']}; /* Darker on hover */
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none; /* Hide arrows */
            }}
        """)
        self.tabs.setIconSize(QSize(45,45))  # Set tab icon size
        
        # Preview
        self.preview_tab = QWidget()
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.preview_container = QWidget()
        self.preview_tab_layout = QVBoxLayout()
        self.preview_tab_layout.setAlignment(Qt.AlignTop)
        self.preview_tab_layout.setSpacing(1)
        self.preview_container.setLayout(self.preview_tab_layout)
        self.preview_scroll.setWidget(self.preview_container)
        self.tabs.addTab(self.preview_scroll, QIcon(f'{CURRENT_PATH}/icons/preview.png'), "Preview")

        # Plots
        self.plots_tab = QWidget()
        self.plots_scroll = QScrollArea()
        self.plots_scroll.setWidgetResizable(True)
        self.plots_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.plots_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plots_container = QWidget()
        self.plots_tab_layout = QVBoxLayout()
        self.plots_tab_layout.setAlignment(Qt.AlignTop)
        self.plots_tab_layout.setSpacing(1)
        self.plots_container.setLayout(self.plots_tab_layout)
        self.plots_scroll.setWidget(self.plots_container)
        self.tabs.addTab(self.plots_scroll, QIcon(f'{CURRENT_PATH}/icons/chart.png'), "Plots")
        
        # Analysis
        self.analysis_tab = QWidget()
        self.analysis_scroll = QScrollArea()
        self.analysis_scroll.setWidgetResizable(True)
        self.analysis_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.analysis_container = QWidget()
        self.analysis_tab_layout = QVBoxLayout()
        self.analysis_tab_layout.setAlignment(Qt.AlignTop)
        self.analysis_tab_layout.setSpacing(1)
        self.analysis_tab_layout.addWidget(QLabel("Content for Analysis"))
        self.analysis_container.setLayout(self.analysis_tab_layout)
        self.analysis_scroll.setWidget(self.analysis_container)
        self.tabs.addTab(self.analysis_scroll, QIcon(f'{CURRENT_PATH}/icons/statistics.png'), "Analysis")

        # Story
        self.story_tab = QWidget()
        self.story_scroll = QScrollArea()
        self.story_scroll.setWidgetResizable(True)
        self.story_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.story_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.story_container = QWidget()
        self.story_tab_layout = QVBoxLayout()
        self.story_tab_layout.setAlignment(Qt.AlignTop)
        self.story_tab_layout.setSpacing(1)
        self.story_tab_layout.addWidget(QLabel("Story"))
        self.story_container.setLayout(self.story_tab_layout)
        self.story_scroll.setWidget(self.story_container)
        self.tabs.addTab(self.story_scroll, QIcon(f'{CURRENT_PATH}/icons/story.png'), "Story")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        #self.resize(400, 300)    

    def set_preview(self):
        print('[>] Setting up preview commands')
        for cmd_string in COMMANDS['Preview']:
            self.preview_tab_layout.addWidget(CommandBlock(cmd=cmd_string))
    def set_plots(self):
        print('[>] Setting up plots commands')
        for cmd_string in COMMANDS['Plots']:
            self.plots_tab_layout.addWidget(CommandBlock(cmd=cmd_string))          
    def set_analysis(self):
        print('[>] Setting up analysis commands')
        for cmd_string in COMMANDS['Analysis']:
            self.analysis_tab_layout.addWidget(CommandBlock(cmd=cmd_string))        

# file explorer
class FileExplorer(QWidget):
    def __init__(self):
        super().__init__()

        self._selected_file = None

        layout = QVBoxLayout(self)

        self.path = QLineEdit(self)
        self.path.setStyleSheet(f'''
            QLineEdit {{
                font:{CONFIG['FileExplorer']['font']};
                color:{CONFIG['CodeLine']['color']};
                background-color:{CONFIG['CodeLine']['background-color']};
                border:{CONFIG['CodeLine']['border']};
                border-radius:{CONFIG['CodeLine']['border-radius']};
                width: 100px;
                height: 20px;
            }}   

            QLineEdit:focus {{
                color: {CONFIG['CodeLine']['focus']['color']};
                background-color: {CONFIG['CodeLine']['focus']['background-color']};
                border-color: {CONFIG['CodeLine']['focus']['border-color']};
            }} 
        ''')
        layout.addWidget(self.path)
        #self.line.returnPressed.connect(self.run_command)
        
        self.filename = QLineEdit(self)
        self.filename.setStyleSheet(f'''
            QLineEdit {{
                font:{CONFIG['FileExplorer']['font']};
                color:{CONFIG['FileExplorer']['color']};
                background-color:{CONFIG['CodeLine']['background-color']};
                border:{CONFIG['CodeLine']['border']};
                border-radius:{CONFIG['CodeLine']['border-radius']};
                width: 100px;
                height: 20px;
            }}   

            QLineEdit:focus {{
                color: {CONFIG['CodeLine']['focus']['color']};
                background-color: {CONFIG['CodeLine']['focus']['background-color']};
                border-color: {CONFIG['CodeLine']['focus']['border-color']};
            }} 
        ''')
        layout.addWidget(self.filename)

        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        self.treeView = QTreeView(self)
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index(QDir.rootPath()))
        self.treeView.setHeaderHidden(True)
        self.treeView.setColumnWidth(0, 400)
        self.treeView.setColumnHidden(3, True) # Hide the "Date Modified" column (index 3)
        self.treeView.setColumnHidden(2, True) # Hide the "type" column (index 2)
        self.treeView.doubleClicked.connect(self.get_file)
        self.setStyleSheet(f"""
            QWidget {{
                color: {CONFIG['FileExplorer']['color']};
                background-color: white;
                border-radius: 5px;
                padding: 2px;
            }}
            QTreeView {{
                border: {CONFIG['CodeLine']['border']};
                background-color: {CONFIG['FileExplorer']['background-color']};
                font: {CONFIG['FileExplorer']['font']};
            }}
            QTreeView::item {{
                padding: 1px;
                border-radius: 0px;
            }}
            QTreeView::item:hover {{
                background-color: #e0e0e0;
            }}
            QTreeView::item:selected {{
                background-color: {CONFIG['FileExplorer']['selection-color']};
                color: white;
            }}

            /* Scrollbar Customization */
            QScrollBar:vertical {{
                border: none;
                background: white; 
                width: 15px; /* Scrollbar width */
                margin: 0px;
                border-radius: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {CONFIG['Controls']['hover-color']}; 
                min-height: 20px;
                border-radius: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {CONFIG['CodeLine']['color']}; /* Darker on hover */
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none; /* Hide arrows */
            }}

            /* Horizontal Scrollbar Customization */
            QScrollBar:horizontal {{
                border: none;
                background: white;
                height: 15px; /* Scrollbar height */
                margin: 0px;
                border-radius: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {CONFIG['Controls']['hover-color']};
                min-width: 20px;
                border-radius: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {CONFIG['CodeLine']['color']}; /* Darker on hover */
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                background: none; /* Hide arrows */
            }}
        """)

        layout.addWidget(self.treeView)

    def get_file(self, index: QModelIndex):
        def load_file_to_data_viewer(self):
            data_viewer = self.parent().parent().parent().findChild(DataViewer,None,Qt.FindChildrenRecursively) # find data_viewer
            data_viewer.set_preview()
            data_viewer.set_analysis()
            data_viewer.set_plots()

        self._selected_file = self.model.filePath(index) 
        self.path.setText(self._selected_file)
        #print(self._selected_file)       

        file_type = self._selected_file.split('.')[-1] 
        if file_type in ['csv','xlsx']:
            DATA_TABLE['path'] = self._selected_file
            DATA_TABLE['file_name'] = self._selected_file.split('/')[-1]
            DATA_TABLE['df'] = read_data_file(DATA_TABLE['path'])
            self.filename.setText(f"df = {DATA_TABLE['file_name']}")
            load_file_to_data_viewer(self)
            
       
# output widgets
class PlotOutput(QWidget):
    def __init__(self, fig=None, parent=None):
        super().__init__(parent)

        self._rows = 550 # height

        self.setStyleSheet("background-color: white; border: 2px solid #333;")
        #self.setFixedSize(400,400)
        
        self.layout = QVBoxLayout()
        self.canvas = FigureCanvas(fig if fig else plt.figure())
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def set_plot(self, fig):
        self.canvas.figure = fig
        self.canvas.draw()
class ChartOutput(QWidget):
    def __init__(self, fig=None, parent=None):
        super().__init__(parent)

        self._rows = 900 # height

        self.setStyleSheet("background-color: #f0f0f0; border: 2px solid #333;")
        self.setFixedSize(600, 400)
        
        self.layout = QVBoxLayout()
        self.canvas = FigureCanvas(fig if fig else plt.figure())
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def set_chart(self, fig):
        """Update the widget with a new Seaborn/Matplotlib chart without saving as an image."""
        self.canvas.figure = fig
        self.canvas.draw()
class TableOutput(QWidget):
    def __init__(self,df:pd.DataFrame=pd.DataFrame()):
        super().__init__()    

        def set_style(plain_text:str) -> str:

            # Extract parts
            lines = plain_text.split("\n")
            header_line = lines[0]  # Column names
            data_rows = lines[1:]   # Data rows

            # HTML formatted output
            styled_text = f'<pre style="font-family: Consolas;">'

            # Bold header row
            styled_text += f'<b>{header_line}</b><br>'

            # Black underline after headers
            styled_text += f'<span style="color: {CONFIG["Table"]["color"]}; font-weight: bold;">{"─" * len(header_line)}</span><br>'

            # Add alternating row colors
            for i, row in enumerate(data_rows):
                bg_color = CONFIG['Table']['alt-row-color'] if i % 2 == 1 else "white"  # Gray for even rows
                styled_text += f'<span style="background-color: {bg_color};">{row}</span><br>'

            styled_text += '</pre>'
            
            return styled_text

        self._rows = min(max(200+20*len(df),320),800)
        self._df = df
        self._html_df = set_style(df.to_string(index=True))
        
        #print(f"height = {self._rows}") # monitor
        LABEL_STYLE = f"""
            QLabel {{
                font: {CONFIG['Table']['font']};
                color: {CONFIG['Table']['color']};
                background-color: white; 
                white-space: pre;
                padding: 1px 10px; 
                border-radius: {CONFIG["Controls"]["border-radius"]};
                border: 0px solid white;
            }}
        """
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(1)
        
        self.outbox = QLabel(self._html_df)
        self.outbox.setWordWrap(False)
        self.outbox.setAlignment(Qt.AlignLeft)
        self.outbox.setStyleSheet(LABEL_STYLE) 
        
        scroll_area = QScrollArea(self)
        scroll_area.setWidget(self.outbox)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                color: {CONFIG['FileExplorer']['color']};
                background-color: white;
                border-radius: 1px;
                border: 0px solid white;
                width: 1500;
                padding: 2px;
            }}
            /* Scrollbar Customization */
            QScrollBar:vertical {{
                border: none;
                background: white; 
                width: 10px; /* Scrollbar width */
                margin: 0px;
                border-radius: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {CONFIG['Controls']['hover-color']}; 
                min-height: 20px;
                border-radius: 2px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none; /* Hide arrows */
            }}

            /* Horizontal Scrollbar Customization */
            QScrollBar:horizontal {{
                border: none;
                background: white;
                height: 15px; /* Scrollbar height */
                margin: 0px;
                border-radius: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {CONFIG['Controls']['hover-color']};
                min-width: 10px;
                border-radius: 2px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                background: none; /* Hide arrows */
            }}
        """)
        self.layout.addWidget(scroll_area)
        self.setLayout(self.layout)
class TextOutput(QWidget):
    def __init__(self,text:str='output text'):
        super().__init__()    

        self._text = text
        self._rows = min(max(200 + 20*len(text.split('\n')),320),800)
        self._width = min(800,100*max([len(row_str) for row_str in text.split('\n')]))
        
        #print(f"height = {self._rows}") # monitor
        LABEL_STYLE = f"""
            QLabel {{
                font: {CONFIG['Output']['font']};
                color: {CONFIG['Output']['color']};
                background-color: white; 
                padding: 1px 30px; 
                border-radius: {CONFIG["Controls"]["border-radius"]};
                border: 0px solid white;
            }}
        """
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(1)
        
        self.outbox = QLabel(self._text)
        self.outbox.setWordWrap(False)
        self.outbox.setAlignment(Qt.AlignLeft)
        self.outbox.setStyleSheet(LABEL_STYLE) 
        
        scroll_area = QScrollArea(self)
        scroll_area.setWidget(self.outbox)
        scroll_area.setWidgetResizable(True)
        #scroll_area.setFixedSize(self._width,self._rows)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                color: {CONFIG['FileExplorer']['color']};
                background-color: white;
                border-radius: 1px;
                border: 0px solid white;
                padding: 2px;
            }}
            /* Scrollbar Customization */
            QScrollBar:vertical {{
                border: none;
                background: white; 
                width: 10px; /* Scrollbar width */
                margin: 0px;
                border-radius: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {CONFIG['Controls']['hover-color']}; 
                min-height: 20px;
                border-radius: 2px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none; /* Hide arrows */
            }}

            /* Horizontal Scrollbar Customization */
            QScrollBar:horizontal {{
                border: none;
                background: white;
                height: 15px; /* Scrollbar height */
                margin: 0px;
                border-radius: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {CONFIG['Controls']['hover-color']};
                min-width: 10px;
                border-radius: 2px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                background: none; /* Hide arrows */
            }}
        """)
        self.layout.addWidget(scroll_area)

        self.setLayout(self.layout)