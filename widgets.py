from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout,QComboBox,QLineEdit,QApplication, QMainWindow,QWidget, QLabel,QPushButton,QFrame
from PyQt5.QtWidgets import QTreeView, QFileSystemModel, QSplitter, QMessageBox,QTabWidget,QScrollArea,QTextEdit

from PyQt5.QtGui import QColor,QIcon,QPixmap, QTextCursor
from PyQt5.QtCore import QDir,Qt,QModelIndex,QSize,QPropertyAnimation,QObject,pyqtSignal,QThread, QEvent
import sys
import traceback
from func import *
import func


# basic widgets
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
                border: 1px solid {CONFIG['CodeLine']['focus']['border-color']};
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
        #print(f"matches = {matches}") # monitor
    
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
        self.comment = QPushButton('', self)
        self.comment.setIcon(QIcon(f'{CURRENT_PATH}/icons/comment.png'))
        self.comment.setStyleSheet(BUTTON_STYLE)
        self.comment.clicked.connect(self.set_comment)
        right_layout.addWidget(self.comment)

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
        self.fold.clicked.connect(self.set_fold)
        right_layout.addWidget(self.fold)

        self.setLayout(layout)

    def set_comment(self):
        #print('comment') # monitor
        current_cmd_block = self.parent().parent()
        current_cmd_block.add_comment()
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
    def set_fold(self):     
        def is_folded(cmd_block):
            return len(cmd_block._folded_items) > 0

        #print("Folding") 
        current_cmd_block = self.parent().parent()   
        items = current_cmd_block.children()[0].children()

        if current_cmd_block._output != None:
            if is_folded(cmd_block=current_cmd_block):
                current_cmd_block._output.show()
                current_cmd_block._folded_items = []
                current_cmd_block.block.setFixedHeight(150 + current_cmd_block._output._height)
            else:
                current_cmd_block._output.hide()
                current_cmd_block._folded_items.append(current_cmd_block._output)
                current_cmd_block.block.setFixedHeight(170)
                #current_cmd_block.layout.addWidget(TextOutput(text='>>>'))               
class ArgsMenu(QWidget):
    def __init__(self,args:dict,cmd_block=None):
        super().__init__()
        self.setObjectName("ArgsMenu")

        self._args = {}
        self._cmd_block = cmd_block
        MAX_ARG_WIDTH = 220

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(40,0,40,0)
        self.layout.setAlignment(Qt.AlignLeft) 
        
        for arg_name,arg in args.items():
            #print(f"arg_name={arg_name}; arg={arg}") # monitor
            if arg['type'] == 'category':
                combo = QComboBox()
                combo.addItems(arg['options'])
                #print(self.get_parameter_value(arg_name)) # monitor
                combo.setCurrentText(self.get_parameter_value(arg_name))  # Set default selection
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

            elif arg['type'] == 'number':    
                int_arg = QLineEdit()
                #print(self.get_parameter_value(arg_name)) # monitor
                int_arg.setText(str(self.get_parameter_value(arg_name)))  # Set default text
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

            elif arg['type'] == 'query':    
                self.sql_arg = QTextEdit()
                self.sql_arg.setStyleSheet(f"""
                                        QTextEdit {{
                                                font:{CONFIG['CodeLine']['font']};
                                                color:{CONFIG['CodeLine']['color']};
                                                border: {CONFIG['CodeLine']['border']};
                                                background-color: {CONFIG['CodeLine']['background-color']};
                                                border-radius: 5px;
                                                padding: 15px;
                                            }}
                                            QTextEdit:focus {{
                                                border: 2px solid #3498db;
                                                background-color: white;
                                            }}
                                        """)
                self.sql_arg.setHtml(str(self.get_parameter_value(arg_name)))    
                self.sql_arg.textChanged.connect(self.set_sql)              
                self.layout.addWidget(self.sql_arg)    
        
        self.setLayout(self.layout)
    
    def get_parameter_value(self,arg_name:str):
        def get_next_end_indexes(string:str):
            quat_open_flag = False
            end_indexes = [] # end as , or )
            for i in range(len(string)):
                if string[i] in ['"',"'"]:
                    quat_open_flag = ~quat_open_flag
                elif string[i] in [',',')'] and quat_open_flag == False:
                    end_indexes.append(i)

            return end_indexes             

        cmd_string = self._cmd_block._cmd
        currrent_arg_string = cmd_string[cmd_string.find(arg_name)+len(arg_name):]
        print(f"currrent_arg_string = {currrent_arg_string}") # monitor
        return currrent_arg_string[currrent_arg_string.find('=')+1:get_next_end_indexes(string=currrent_arg_string)[0]]
    def set_sql(self):
        # print(self.sql_arg.toPlainText()) # monitor
        query_string = self.sql_arg.toPlainText().replace('\n',' ')
        cmd = self._cmd_block.code_line._text
        query_end_index = min([cmd.find(c,cmd.find('query=') + len('query=') + 1) for c in [')','"',"'"]])
        old_query = cmd[cmd.find('query=') + len('query='):query_end_index]
        #self._cmd_block.code_line.set_text(f"old_query = {old_query} >> new query = {query_string}") # monitor
        self._cmd_block.code_line.set_text(cmd.replace(old_query,query_string))
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
                    elif isinstance(item, QTextEdit):
                        #print(f'QTextEdit = {item.toPlainText()}')   
                        param.append(item.toPlainText())     

                return param 

            cmd_string = delete_func_parameters(cmd)
            for item in get_new_parameters(args):
                cmd_string = cmd_string.replace('=',f"@{item}",1)
                #print(cmd_string)

            return cmd_string.replace('@','=')   

        #print('update command') # monitor
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
            elif isinstance(item,QTextEdit):
                item.setHtml(current_arg)     
class Comment(QWidget):
    def __init__(self, expanded_height=80, duration=500):
        super().__init__()

        self._ROW_HEIGHT = 30
        self._EXPANDED_HEIGHT = expanded_height

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.text_box = QTextEdit()
        self.text_box.setStyleSheet(f"""
            QTextEdit {{
                font:{CONFIG['CodeLine']['font']};
                color: black;
                border: none;
                background-color: white;
                border-radius: 5px;
                padding: 5px;
            }}
            QTextEdit:focus {{
                border: 1px solid purple;
                background-color: white;
            }}
        """)
        self.text_box.setHtml("<p>Type Comment.</p>")
        self.text_box.setFixedHeight(self._EXPANDED_HEIGHT)
        self.text_box.installEventFilter(self)

        self.layout.addWidget(self.text_box)
        self.setLayout(self.layout)

    def showEvent(self, event):
        super().showEvent(event)
        self.expand_editor()
    def eventFilter(self, obj, event):
        if obj == self.text_box:
            if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
                self.collapse_editor()
                return True
            elif event.type() in (QEvent.FocusIn, QEvent.MouseButtonPress):
                self.expand_editor()
        return super().eventFilter(obj, event)
    def collapse_editor(self):
        self.text_box.setReadOnly(True)
        self.text_box.setFixedHeight(self._ROW_HEIGHT)
        self.text_box.clearFocus()
    def expand_editor(self):
        self.text_box.setReadOnly(False)
        self.text_box.setFixedHeight(self._EXPANDED_HEIGHT)
        self.text_box.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        cursor = self.text_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_box.setTextCursor(cursor)
        

# advanced widgets
class CommandBlock(QWidget):
    def __init__(self,data:pd.DataFrame=DATA_TABLE,cmd:str=''):
        super().__init__()
        self.setObjectName("CommandBlock")
        
        self._df = data['df']
        self._cmd = cmd
        self._args = None
        self._output = None
        self._args = None
        self._folded_items = []

        # frame
        self.block = QFrame(self)
        self.block.setFrameShape(QFrame.StyledPanel)
        self.block.setFrameShadow(QFrame.Raised)
        self.block.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid #dedede;
                border-radius: 15px;
            }}
            QFrame:hover {{
                border: 2px solid #3498db;
            }}
        """)
        self.set_block_size(w=1200,h=120)
        
        # Create layout
        self.layout = QVBoxLayout(self.block)
        
        # Add sub-widgets
        self.code_controls = CodeControls()
        self.layout.addWidget(self.code_controls)

        self.code_line = CodeLine(self._cmd)
        self.layout.addWidget(self.code_line)
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(0)
        self.main_layout.addWidget(self.block)

        self.setLayout(self.main_layout)
    
    def get_attributes(self):
        return {'self._cmd':f"{self._cmd}",'self._args':f"{self._args}",'self._output':f"{self._output}"} 
    def add_comment(self):
        self.comment = Comment()
        self.layout.addWidget(self.comment)
        self.block.setFixedSize(self.block.width(),self.block.height() + 50)
    def run_command(self):
        def delete_prev_output(self):
            for item_type in [ArgsMenu,TextOutput,TableOutput,AnalysisOutput,PlotOutput]:
                try:
                    output_widget = self.findChild(item_type)
                    output_widget.setParent(None)
                    output_widget.deleteLater()
                except:
                    pass            
            
        delete_prev_output(self)

        try:
            # no threads run
            local_vars = {'df': self._df}
            local_vars.update({k: getattr(func, k) for k in dir(func) if not k.startswith("_")})
            output_obj = eval(self._cmd, {}, local_vars) 

            self._args = ArgsMenu(args=output_obj['args'],cmd_block=self)
            self.layout.addWidget(self._args)

            if output_obj['output_type'] == 'text':
                self._output = TextOutput(output_obj['output'])
            elif output_obj['output_type'] == 'table':
                self._output = TableOutput(output_obj['output'])
            elif output_obj['output_type'] == 'plot':  
                self._output = PlotOutput(fig=output_obj['output'],height=output_obj['size'][1],width=output_obj['size'][0])      
            elif output_obj['output_type'] == 'analysis':  
                self._output = AnalysisOutput(
                    plot=output_obj['output']['plot'],
                    log=output_obj['output']['log'],
                    table=output_obj['output']['table']
                    )  
                self._output.setGeometry(0,0,1000,700)    
        
        except Exception:
            #pass # use for full exception in vscode
            error = traceback.format_exc()   
            self._output = TextOutput(text=error) 

        try: 
            OUTPUT_HEIGHT=self._output.frameGeometry().height()
        except: 
            OUTPUT_HEIGHT=0    

        self.block.setFixedHeight(min(max(OUTPUT_HEIGHT + 300,220),1500)) # 300 < dynamic height < 1500 
        self.layout.addWidget(self._output)
    def set_block_size(self,h:int=120,w:int=1200):
        self.block.setFixedSize(w,h)
        
    
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
        self.tabs.setFixedWidth(1500)
        
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

        # SQL
        self.sql_tab = QWidget()
        self.sql_scroll = QScrollArea()
        self.sql_scroll.setWidgetResizable(True)
        self.sql_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.sql_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sql_container = QWidget()
        self.sql_tab_layout = QVBoxLayout()
        self.sql_tab_layout.setAlignment(Qt.AlignTop)
        self.sql_tab_layout.setSpacing(1)
        self.sql_container.setLayout(self.sql_tab_layout)
        self.sql_scroll.setWidget(self.sql_container)
        self.tabs.addTab(self.sql_scroll, QIcon(f'{CURRENT_PATH}/icons/sql.png'), "SQL")

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
        self.story_container.setLayout(self.story_tab_layout)
        self.story_scroll.setWidget(self.story_container)
        self.tabs.addTab(self.story_scroll, QIcon(f'{CURRENT_PATH}/icons/story.png'), "Story")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        #self.resize(400, 300)    

    def set_preview(self):
        print('[>] Setting up preview commands')
        for cmd_string in COMMANDS['Preview']:
            self.preview_tab_layout.addWidget(CommandBlock(cmd=cmd_string,data=DATA_TABLE))
    def set_sql(self):
        print('[>] Setting up SQL commands')
        for cmd_string in COMMANDS['SQL']:
            self.sql_tab_layout.addWidget(CommandBlock(cmd=cmd_string,data=DATA_TABLE))        
    def set_plots(self):
        print('[>] Setting up plots commands')
        for cmd_string in COMMANDS['Plots']:
            self.plots_tab_layout.addWidget(CommandBlock(cmd=cmd_string,data=DATA_TABLE))          
    def set_analysis(self):
        print('[>] Setting up analysis commands')
        for cmd_string in COMMANDS['Analysis']:
            self.analysis_tab_layout.addWidget(CommandBlock(cmd=cmd_string,data=DATA_TABLE))        

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
        self.treeView.setColumnWidth(0, 600)
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
            data_viewer.set_sql()
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
    def __init__(self, fig=None, parent=None, height:int=400, width:int=400):
        super().__init__(parent)

        self._height = height 
        self._width = width
        LABEL_STYLE = f"""
            QLabel {{
                font: {CONFIG['Table']['font']};
                color: {CONFIG['Table']['color']};
                background-color: white; 
                white-space: pre;
                padding: 20px 20px; 
                border-radius: {CONFIG["Controls"]["border-radius"]};
                border: 0px solid white;
            }}
            QLabel:hover {{
                color: {CONFIG['FileExplorer']['color']};
            }}
        """

        self.setStyleSheet("background-color: white; border: 2px solid #333;")
        self.setFixedSize(self._height,self._width)
        
        self.layout = QVBoxLayout()

        #self.log = QLabel(text=f"df = {DATA_TABLE['file_name']}\n")
        #self.log.setStyleSheet(LABEL_STYLE) 
        #self.layout.addWidget(self.log)

        self.canvas = FigureCanvas(fig if fig else plt.figure(figsize=(self._width,self._height)))
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def set_plot(self, fig):
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

        self._df = df
        self._html_df = set_style(df.to_string(index=True)) 
        MIN_TABLE_HEIGHT,MAX_TABLE_HEIGHT = 200,1000       

        LABEL_STYLE = f"""
            QLabel {{
                font: {CONFIG['Table']['font']};
                color: {CONFIG['Table']['color']}; 
                background-color: white; 
                white-space: pre;
                padding: 1px 30px; 
                border-radius: {CONFIG["Controls"]["border-radius"]};
                border: 0px solid white;
            }}
        """
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(0)

        self.outbox = QLabel(self._html_df)
        #self.outbox.setFixedHeight(18*(3 + len(df)))
        self.outbox.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.outbox.setWordWrap(False)
        self.outbox.setAlignment(Qt.AlignLeft)
        self.outbox.setStyleSheet(LABEL_STYLE) 
        
        scroll_area = QScrollArea(self)
        #print(f'table height= {self.outbox.geometry().height()}')  # monitor
        #scroll_area.setFixedHeight(min(20 + 18*(3 + len(df)),1000)) # need to fix
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
        MAX_TEXT_HEIGHT = 600
        
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
        self.layout.setSpacing(0)
        
        self.outbox = QLabel(self._text)
        self.outbox.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.outbox.setWordWrap(False)
        self.outbox.setAlignment(Qt.AlignLeft)
        self.outbox.setStyleSheet(LABEL_STYLE) 
        
        scroll_area = QScrollArea(self)
        scroll_area.setWidget(self.outbox)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(min(20*(1 + self._text.count('\n')),MAX_TEXT_HEIGHT))
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
class AnalysisOutput(QWidget):
    def __init__(self,plot,log:str='',table:pd.DataFrame=pd.DataFrame()):
        super().__init__()

        #self._height = 1300 # height

        self.setStyleSheet("background-color: #f0f0f0; border: 2px solid #333;")
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(TextOutput(text=log))
        self.layout.addWidget(TableOutput(df=table))
        self.layout.addWidget(PlotOutput(fig=plot))
        self.setLayout(self.layout)
        self.setFixedHeight(self.layout.sizeHint().height())
     