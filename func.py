import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import duckdb
from tabulate import tabulate
import os
import json
import re
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import io
import traceback
import statsmodels.api as sm
import textwrap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import linregress,gaussian_kde,shapiro,ttest_ind
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error

# basic func
def get_dir(directory):
    try:
        contents = os.listdir(directory)
        for i in range(len(contents)):
            if os.path.isdir(os.path.join(directory, contents[i])):
                contents[i] += '/'
        return contents
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
        return []
    except Exception as e:
        print(f"Error listing directory '{directory}': {e}")
        return []
def get_file_type(file_path:str):
        if file_path.endswith('/'):
            return 'dir'
        else:
            return file_path.split('.')[-1]    
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Configuration file not found.")
        return {}
    except json.JSONDecodeError:
        print("Error parsing configuration file.")
        return {}  
def read_data_file(file_full_path:str):
    file_type = file_full_path.split('.')[-1] 
    if file_type == 'csv':
        return pd.read_csv(file_full_path)
    elif file_type == 'xlsx':
        return pd.read_excel(file_full_path)    
    else:
        print('Unsupported file type.')
        return pd.DataFrame() 
def get_darker_color(hex_color, percentage=10):
    """
    Darkens the given hex color by the specified percentage.

    :param hex_color: str, the hex color string (e.g., "#E8DAEF").
    :param percentage: int, the percentage to darken the color (default is 10%).
    :return: str, the darkened hex color string.
    """
    # Ensure the input is a valid hex color
    try:
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]

        # Convert hex to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Darken the color by the specified percentage
        factor = (100 - percentage) / 100
        r = max(0, int(r * factor))
        g = max(0, int(g * factor))
        b = max(0, int(b * factor))

        # Convert RGB back to hex
        darkened_color = f"#{r:02x}{g:02x}{b:02x}"

        return darkened_color
    except:
        return 'black'   

# load config files
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_json(f'{CURRENT_PATH}/config.json')  
COMMANDS = load_json(f'{CURRENT_PATH}/commands.json')    
#print(CURRENT_PATH)

# load data
DATA_TABLE = {'path':CURRENT_PATH,
              'file_name':None,
              'df':None
             }

# charts general parameters
sns.set_theme(style="ticks")
sns.despine()
#plt.rcParams['axes.facecolor'] = CONFIG['Chart']['background']  # background
plt.rcParams['axes.edgecolor'] = get_darker_color(CONFIG['Chart']['frame_color'],50) # frame & axes
plt.rcParams['text.color'] = CONFIG['Chart']['font_color']

# help function
def get_categorical_columns(df:pd.DataFrame,max_categories=30):
    '''return columns with < max_categories objects'''
    return [col for col in df.columns if len(df[col].unique()) < max_categories]
def get_numeric_columns(df:pd.DataFrame,min_uniques=10):
    '''return columns with > 10 unique values or numeric by type'''
    return [col for col in df.columns if len(df[col].unique()) > min_uniques or col in df.select_dtypes(include=['number'])]    

# preview
def get_shape(df:pd.DataFrame,output_type:str='table'):
    data = pd.DataFrame(data=df.shape,columns=['#'],index=['rows','columns']).T
    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql') # rounded_grid/psql
    return {
        'output':data,
        'output_type':output_type,
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
                },
            'output_type':{
                'type':'category',
                'options':[f"'table'",f"'text'"],
                'default':'table'
                }
                }
        }  
def get_preview(df:pd.DataFrame,rows=5,end='head',output_type:str='table'):
    
    if rows >= len(df):
        data = df 
    elif end == 'head':
        data = df.head(rows) 
    elif end == 'tail':
        data = df.tail(rows) 
    elif end == 'random':
        data = df.sample(rows)     
    else:
        data = pd.DataFrame() 

    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')  

    return {
        'output':data,
        'output_type':output_type,
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'rows':{
                'type':'number',
                'options':[3,5,10,25],
                'default':5
            },
            'end':{
                'type':'category',
                'options':[f"'head'",f"'tail'",f"'random'"],
                'default':'head'
            },
            'output_type':{
                'type':'category',
                'options':[f"'table'",f"'text'"],
                'default':'table'
            }
        }
        }
def get_columns_info(df:pd.DataFrame,show='all',output_type:str='table'): 
    data = {'column':[],'type':[],'dtype':[],'unique':[],'Non-Nulls':[],'Non-Nulls%':[]}
    numeric_cols = df.select_dtypes(include=['number'])
    object_cols = df.select_dtypes(include=['object'])
    for column in df.columns:
        data['column'].append(column)
        data['type'].append('number' if column in numeric_cols else 'object')
        data['dtype'].append(str(df[column].dtype))
        data['unique'].append(len(df[column].unique().tolist()))
        data['Non-Nulls'].append(len(df[~df[column].isna()]))
        data['Non-Nulls%'].append(f"{round(len(df[~df[column].isna()])*100/len(df),2)}%")

    data = pd.DataFrame(data).sort_values(by=['type','dtype','Non-Nulls']).reset_index(drop=True)  

    if show != 'all':
        data = data[data.column==show].reset_index(drop=True)

    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')
    return {
        'output':data,
        'output_type':output_type,
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'show':{
                'type':'category',
                'options':["'all'"] + [f"'{col}'" for col in list(df.columns)],
                'default':f"'all'"
                },
            'output_type':{
                'type':'category',
                'options':[f"'table'",f"'text'"],
                'default':"'table'"
            }
            }
        }      
def get_numerics_desc(df:pd.DataFrame,show='all',output_type:str='table'):
    data = df.describe().T
    numeric_columns = list(data.index)
    data['count'] = data['count'].astype(int)
    data["skewness"] = 3*(data['mean'] - data['50%'])/data['std']

    if show != 'all':
        data = data[data.index == show]

    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')
    return {
        'output':data,
        'output_type':output_type,
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'show':{
                'type':'category',
                'options':["'all'"] + [f"'{col}'" for col in list(df.columns)],
                'default':f"'all'"
                },
            'output_type':{
                'type':'category',
                'options':[f"'table'",f"'text'"],
                'default':"'table'"
            }
        }
        }    
def get_categorical_desc(df:pd.DataFrame,show='all',outliers='None',output_type:str='table'):
    categorical_columns = [col for col in df.columns if str(df[col].dtype) in ['object','category','bool']]
    df = df[categorical_columns].copy()
    data = {'column':[],'type':[],'unique':[],'mode':[],'mode_occurances':[],'mode%':[],'prob_outliers':[],'outlier_items':[],'outliers_occurance_probability':[]}
    
    for column in df.columns:
        data['column'].append(column)
        data['type'].append(str(df[column].dtype))
        data['unique'].append(len(df[column].unique().tolist()))
        data['mode'].append(df[column].mode()[0])
        data['mode_occurances'].append(len(df[df[column]==df[column].mode()[0]]))
        data['mode%'].append(f"{100*len(df[df[column]==df[column].mode()[0]])/len(df):.2f}%")

        if outliers in [None,'None','none']:
            data['prob_outliers'].append([])
            data['outlier_items'].append([])
            data['outliers_occurance_probability'].append([])
        else:    
            PERCENTAGE = float(outliers)
            df['occ_prob'] = df[column].map(df[column].value_counts(normalize=True))
            data['prob_outliers'].append(len(df[df['occ_prob'] < PERCENTAGE/100]))
            data['outlier_items'].append(str(list(df.loc[df['occ_prob'] < PERCENTAGE/100,column].unique())))
            data['outliers_occurance_probability'].append(str(list(df.loc[df['occ_prob'] < PERCENTAGE/100,'occ_prob'].unique())))
        
    data = pd.DataFrame(data).reset_index(drop=True)

    if show != 'all':
        data = data[data['column'] == show]

    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')
    return {
        'output':data,
        'output_type':output_type,
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'show':{
                'type':'category',
                'options':["'all'"] + [f"'{col}'" for col in list(df.columns)],
                'default':f"'all'"
                },
            'outliers':{
                'type':'number',
                'options':[0.3,0.5,1,3,5,10],
                'default':0.3
            },
            'output_type':{
                'type':'category',
                'options':[f"'table'",f"'text'"],
                'default':"'table'"
            }
        }
    }    

# sql
def get_data(df:pd.DataFrame,output_type:str='table',show='100',query:str='SELECT * FROM df'):
    data = duckdb.query(query).to_df()
    data = data.head(show) if show != 'all' else data
    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')

    return {
            'output':data,
            'output_type':output_type,
            'args':{
                'df':{
                    'type':'category',
                    'options':['df'],
                    'default':f"'df'"
                    },
                'output_type':{
                    'type':'category',
                    'options':[f"'table'",f"'text'"],
                    'default':'table'
                    },
                'show':{
                    'type':'category',
                    'options':['10','25','50','100','200','all'],
                    'default':'100'
                },    
                'query':{
                    'type':'query',
                    'options':[f"'SELECT * FROM df LIMIT 10'"],
                    'default':"'SELECT * FROM df LIMIT 10'"
                    },       
                }
            }  

# plots for plots
def get_box_plot(df:pd.DataFrame,y:str=None,by:str=None,orient:str='v',overall_mean=False,category_mean=True,std_lines=True):
    def set_axis_style(ax,y:str,x:str,orient=orient):
        if orient == 'v':
            ax.set_xlabel(x, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(y, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
        else:
            ax.set_xlabel(y, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(x, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])   

        ax.tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
    def set_height(df=df,by=by,orient=orient,num_of_categories:int=5):
        try:
            if orient == 'v':
                max_data_to_category = max([ len(df[df[by]==cat]) for cat in df[by].unique() ])
                return min(max(350,2*max_data_to_category),600)
            elif orient == 'h':
                return min(max(350,num_of_categories*20),600)
        except:    
            return 300        
    def set_width(df=df,by=by,orient=orient,num_of_categories:int=5):
        try:
            if orient == 'h':
                max_data_to_category = max([ len(df[df[by]==cat]) for cat in df[by].unique() ])
                return min(max(300,2*max_data_to_category),600)
            elif orient == 'v':
                return min(max(200,num_of_categories*20),600)
        except:        
            return 300 
        
    MAX_CATEGORIES = 30
    NUM_OF_CATEGORIES = 1 if by in [None,'none','None'] else min(len(df[by].unique()),MAX_CATEGORIES)
    HEIGHT, WIDTH = set_height(df=df,by=by,orient=orient,num_of_categories=NUM_OF_CATEGORIES), set_width(orient=orient,num_of_categories=NUM_OF_CATEGORIES)
    #print(f"height: {HEIGHT}, width: {WIDTH}") # monitor

    fig, ax = plt.subplots(figsize=(HEIGHT,WIDTH),dpi=75)
    set_axis_style(ax=ax,y=y,x=by,orient=orient)

    try:
        set_strip_plot(ax=ax,df=df,y=y,by=by,orient=orient,color=None)
        set_box_plot(ax=ax,df=df,y=y,by=by,orient=orient,overall_mean=overall_mean,category_mean=category_mean,std_lines=std_lines)
    except Exception as e:
        print(e)    
    
    return {
        'output':fig,
        'size':(HEIGHT,WIDTH),
        'output_type':'plot',
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'y':{
                'type':'category',
                'options':[f"'{item}'" for item in get_numeric_columns(df=df,min_uniques=int(0.1*len(df)))],
                'default':None
            },
            'by':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            },
            'orient':{
                'type':'category',
                'options':["'v'","'h'"],
                'default':"'v'"
            },
            'overall_mean':{
                'type':'category',
                'options':['True','False'],
                'default':'False'
            },
            'category_mean':{
                'type':'category',
                'options':['True','False'],
                'default':'False'
            },
            'std_lines':{
                'type':'category',
                'options':['True','False'],
                'default':'False'
            }
        }
    }
def get_count_plot(df:pd.DataFrame,y:str=None,by:str=None,orient:str='h'):
    def get_num_of_categories(df:pd.DataFrame=df,y:str=y,by:str=by):
        y_amount = 1 if y in [None,'none','None'] else len(df[y].unique())
        by_amount = 1 if by in [None,'none','None'] else len(df[by].unique())
        return y_amount*by_amount
    def set_height(df=df,by=by,orient=orient,num_of_categories:int=5):
            try:
                if orient == 'v':
                    max_data_to_category = max([ len(df[df[by]==cat]) for cat in df[by].unique() ])
                    return min(max(300,2*max_data_to_category),600)
                elif orient == 'h':
                    return min(max(200,num_of_categories*20),600)
            except:    
                return 300        
    def set_width(df=df,by=by,orient=orient,num_of_categories:int=5):
            try:
                if orient == 'h':
                    max_data_to_category = max([ len(df[df[by]==cat]) for cat in df[by].unique() ])
                    return min(max(300,2*max_data_to_category),600)
                elif orient == 'v':
                    return min(max(200,num_of_categories*20),600)
            except:        
                return 300 
        
    NUM_OF_CATEGORIES = get_num_of_categories(df,y,by)
    HEIGHT, WIDTH = set_height(df=df,by=by,orient=orient,num_of_categories=NUM_OF_CATEGORIES), set_width(orient=orient,num_of_categories=NUM_OF_CATEGORIES)

    fig, ax = plt.subplots(figsize=(HEIGHT,WIDTH),dpi=75)
      
    try:
        set_count_plot(ax=ax,df=df,y=y,by=by,orient=orient)
    except Exception as e:
        print(e) 

    return {
        'output':fig,
        'size':(HEIGHT,WIDTH),
        'output_type':'plot',
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'y':{
                'type':'category',
                'options':[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':None
            },
            'by':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            },
            'orient':{
                'type':'category',
                'options':["'h'","'v'"],
                'default':"'h'"
            }
        }
    }
def get_scatter_plot(df:pd.DataFrame,y:str=None,x:str=None,by:str=None):
    def set_axis_style(ax,y:str,x:str):
        ax.set_xlabel(x, fontsize=11, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(y, fontsize=11, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    POINT_SIZE = 5 if len(df) > 1000 else 8 if len(df) > 200 else 9
    ALPHA = 0.2 if len(df) > 1000 else 0.4 if len(df) > 200 else 0.6
    
    fig, ax = plt.subplots(figsize=(5,POINT_SIZE),dpi=75)
    set_axis_style(ax,y,x)

    try:
        set_scatter_plot(ax=ax,df=df,y=y,x=x,by=by)
    except Exception as e:
        print(e)    
    
    return {
        'output':fig,
        'size':(300,700),
        'output_type':'plot',
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'y':{
                'type':'category',
                'options':[f"'{item}'" for item in get_numeric_columns(df=df,min_uniques=int(0.1*len(df)))],
                'default':None
            },
            'x':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_numeric_columns(df=df,min_uniques=int(0.1*len(df)))],
                'default':'None'
            },
            'by':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            }
        }
    }
def get_dist_plot(df:pd.DataFrame,y:str=None,by:str=None,stat:str='count',orient='h',category_stats=True,overall_stats=False):
    def get_num_of_categories(df:pd.DataFrame=df,y:str=y,by:str=by):
        y_amount = 1 if y in [None,'none','None'] else len(df[y].unique())
        by_amount = 1 if by in [None,'none','None'] else len(df[by].unique())
        return y_amount*by_amount

    NUM_OF_CATEGORIES = get_num_of_categories(df,y,by)
    fig, ax = plt.subplots(figsize=(1,NUM_OF_CATEGORIES),dpi=75)
    
    set_dist_plot(ax=ax,df=df,y=y,by=by,stat=stat,orient=orient,category_stats=category_stats,overall_stats=overall_stats)

    return {
        'output':fig,
        'size':(300,500) if orient == 'v' else (500,700),
        'output_type':'plot',
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'y':{
                'type':'category',
                'options':[f"'{item}'" for item in get_numeric_columns(df=df,min_uniques=max(int(0.01*len(df)),1))],
                'default':None
            },
            'by':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            },
            'stat':{
                'type':'category',
                'options':['"count"','"density"','"percent"'],
                'default':'"count"'
            },
            'orient':{
                'type':'category',
                'options':['"v"','"h"'],
                'default':'"v"'
            },
            'category_stats':{
                'type':'category',
                'options':['"True"','"False"'],
                'default':'"True"'
            },
            'overall_stats':{
                'type':'category',
                'options':['"False"','"True"'],
                'default':'"False"'
            }
        }
    }

# plots for use
def set_strip_plot(ax,df:pd.DataFrame,y:str=None,by:str=None,orient='v',color:str=None,opacity=None):
    def set_axis_style(ax,y:str,x:str):
        if orient == 'v':
            ax.set_xlabel(x, fontsize=13, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(y, fontsize=13, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        elif orient == 'h':
            ax.set_xlabel(y, fontsize=13, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(x, fontsize=13, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color']) 

        ax.tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
    def set_opacity(opacity):
        if opacity == None:
            return 0.2 if len(data) > 1000 else 0.4 if len(data) > 200 else 0.6
        else:
            return opacity

    LINE_WIDTH = 0.5 if color in [None,'none','None'] else 1

    if by in [None,'none','None']:
        COLOR_INDEX = 0
        data = df[y].copy()
        POINT_SIZE = 5 if len(data) > 1000 else 8 if len(data) > 200 else 9
        ALPHA = set_opacity(opacity)
        
        if orient == 'v':
            sns.stripplot(y=data,
                ax=ax,alpha=ALPHA,size=POINT_SIZE,
                linewidth=LINE_WIDTH,
                dodge=0.4,
                color=CONFIG['Chart']['data_colors'][COLOR_INDEX] if color in [None,'none','None'] else 'white',
                edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70) if color in [None,'none','None'] else color,
                jitter=0.35,zorder=0
                ) 
        elif orient == 'h':
            sns.stripplot(x=data,
                ax=ax,alpha=ALPHA,size=POINT_SIZE,
                linewidth=LINE_WIDTH,dodge=0.4,
                color=CONFIG['Chart']['data_colors'][COLOR_INDEX] if color in [None,'none','None'] else 'white',
                edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70) if color in [None,'none','None'] else color,
                jitter=0.35,zorder=0
                )  
    else:    
        for i,cat in enumerate(df[by].unique()):
            #print(cat) # monitor
            COLOR_INDEX = i % len(CONFIG['Chart']['data_colors'])
            data = df.loc[df[by]==cat,[y,by]].copy()
            POINT_SIZE = 5 if len(data) > 1000 else 8 if len(data) > 200 else 9
            ALPHA = set_opacity(opacity)

            if orient == 'v':
                sns.stripplot(y=data[y],x=data[by],
                    ax=ax,alpha=ALPHA,size=POINT_SIZE,
                    linewidth=LINE_WIDTH,
                    color=CONFIG['Chart']['data_colors'][COLOR_INDEX] if color in [None,'none','None'] else 'white',
                    edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70) if color in [None,'none','None'] else color,
                    jitter=0.35,zorder=0
                    ) 
            else:
                sns.stripplot(x=data[y],y=data[by],
                    ax=ax,alpha=ALPHA,size=POINT_SIZE,linewidth=LINE_WIDTH,
                    color=CONFIG['Chart']['data_colors'][COLOR_INDEX] if color in [None,'none','None'] else 'white',
                    edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70) if color in [None,'none','None'] else color,
                    jitter=0.35,zorder=0
                    )        

    #set_axis_style(ax,y,by)
def set_box_plot(ax,df:pd.DataFrame,y:str=None,by:str=None,orient='v',overall_mean=True,category_mean=True,std_lines:bool=True):
    def set_axis_style(ax,y:str,x:str,orient=orient):
        if orient == 'v':
            ax.set_xlabel(x, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(y, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
        else:
            ax.set_xlabel(y, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(x, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])   

        ax.tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
    def set_overall_mean(ax=ax,df:pd.DataFrame=df,y:str=y,std_lines:bool=std_lines):
        LINE_COLOR = 'grey'
        if orient == 'h':
            ax.axvline(df[y].mean(), color=LINE_COLOR, linestyle="dashed", linewidth=1, label="Overall Mean",zorder=0)
            if std_lines in [True,'true','True']:
                ax.axvline(df[y].mean() + df[y].std(), color=LINE_COLOR, linestyle="dashed", linewidth=1, label="Overall Std",zorder=0) 
                ax.axvline(df[y].mean() - df[y].std(), color=LINE_COLOR, linestyle="dashed", linewidth=1, zorder=0) 
        elif orient == 'v':
            ax.axhline(df[y].mean(), color=LINE_COLOR, linestyle="dashed", linewidth=1, label="Overall Mean",zorder=0)
            if std_lines in [True,'true','True']:
                ax.axhline(df[y].mean() + df[y].std(), color=LINE_COLOR, linestyle="dashed", linewidth=1, label="Overall Std",zorder=0)
                ax.axhline(df[y].mean() - df[y].std(), color=LINE_COLOR, linestyle="dashed", linewidth=1, zorder=0)
    def set_category_mean(ax=ax,df:pd.DataFrame=df,y:str=y,by:str=by,std_lines:bool=std_lines):
        
        MEAN_POINT_SIZE = 80
        STAT_COLOR = 'red'
        
        if by in [None,'none','None']:
            MEAN, STD = df[y].mean(), df[y].std()
            if orient == 'h':
                ax.scatter(x=MEAN,y=y,color=STAT_COLOR,marker='x',s=MEAN_POINT_SIZE)
                if std_lines in [True,'true','True']:
                    MARKER = '|'
                    ax.scatter(x=[MEAN + STD,MEAN - STD],y=[y,y],color=STAT_COLOR,marker=MARKER,s=MEAN_POINT_SIZE)
            elif orient == 'v':
                ax.scatter(y=MEAN,x=y,color=STAT_COLOR,marker='x',s=MEAN_POINT_SIZE)
                if std_lines in [True,'true','True']:
                    MARKER = '_'
                    ax.scatter(y=[MEAN + STD,MEAN - STD],x=[y,y],color=STAT_COLOR,marker=MARKER,s=MEAN_POINT_SIZE)
        else:
            mean_dict = dict(zip(df[by].unique().tolist(),[df.loc[df[by]==cat,y].mean() for cat in df[by].unique()]))
            std_dict = dict(zip(df[by].unique().tolist(),[df.loc[df[by]==cat,y].std() for cat in df[by].unique()]))

            if orient == 'h':
                ax.scatter(x=list(mean_dict.values()),y=list(mean_dict.keys()),color=STAT_COLOR,marker='x',s=MEAN_POINT_SIZE)
                if std_lines in [True,'true','True']:  
                    MARKER = '|'
                    ax.scatter(x=np.array(list(mean_dict.values())) + np.array(list(std_dict.values())),y=list(mean_dict.keys()),color=STAT_COLOR,marker=MARKER,s=MEAN_POINT_SIZE)
                    ax.scatter(x=np.array(list(mean_dict.values())) - np.array(list(std_dict.values())),y=list(mean_dict.keys()),color=STAT_COLOR,marker=MARKER,s=MEAN_POINT_SIZE)
            elif orient == 'v':
                ax.scatter(y=list(mean_dict.values()),x=list(mean_dict.keys()),color=STAT_COLOR,marker='x',s=MEAN_POINT_SIZE)
                if std_lines in [True,'true','True']:  
                    MARKER = '_'
                    ax.scatter(y=np.array(list(mean_dict.values())) + np.array(list(std_dict.values())),x=list(mean_dict.keys()),color=STAT_COLOR,marker=MARKER,s=MEAN_POINT_SIZE)
                    ax.scatter(y=np.array(list(mean_dict.values())) - np.array(list(std_dict.values())),x=list(mean_dict.keys()),color=STAT_COLOR,marker=MARKER,s=MEAN_POINT_SIZE)
    def set_mean_distances(ax=ax,df:pd.DataFrame=df,y:str=y,by:str=by,orient=orient):
        if by not in [None,'none','None']:
            overall_mean = df[y].mean()
            for cat in df[by].unique():
                #print(f"draw: [{cat},{cat}],[{overall_mean},{df.loc[df[by]==cat,y].mean()}]") # monitor
                if orient == 'v':
                    ax.plot([cat,cat],[overall_mean,df.loc[df[by]==cat,y].mean()],color='red')
                elif orient == 'h':   
                    ax.plot([overall_mean,df.loc[df[by]==cat,y].mean()],[cat,cat],color='red') 

    #set_axis_style(ax,y,by)

    if by in [None,'none','None']:    
        if orient == 'h':
            sns.boxplot(
                x=df[y],
                linewidth=1,
                boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 2},
                showfliers=False,
                ax=ax
                )
        elif orient == 'v':    
            sns.boxplot(
                y=df[y],
                boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 2},
                showfliers=False,
                linewidth=1,
                ax=ax
                )    
    else:       
        if orient == 'h': 
            sns.boxplot(
                y=df[by],x=df[y],
                linewidth=1,
                boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 2},
                showfliers=False,
                ax=ax
                    )
        elif orient == 'v':
             sns.boxplot(
                x=df[by],y=df[y],
                linewidth=1,
                boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 2},
                showfliers=False,
                ax=ax
                    )           
    
    if overall_mean in [True,'true','True']:
        set_overall_mean(ax=ax,df=df,y=y,std_lines=std_lines)

    if category_mean in [True,'true','True']:
        set_category_mean(ax=ax,df=df,y=y,by=by,std_lines=std_lines)
    
    if category_mean in [True,'true','True'] and overall_mean in [True,'true','True']:
        set_mean_distances(ax=ax,df=df,y=y,by=by,orient=orient)    
def set_dist_plot(ax,df:pd.DataFrame,y:str=None,by:str=None,stat:str='count',orient='v',category_stats=True,overall_stats=True): #################### need fix #################
    def set_axis_style(ax,y:str,x:str,stat=stat):
        ax.set_xlabel(x, fontsize=13, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(stat, fontsize=13, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)  
    def set_stats(ax,data,color='grey',style='--',orient=orient):
        if orient == 'v':
            for stat,value in {'mean':np.mean(data),'median':np.median(data)}.items():
                ax.axvline(value, color=color, linestyle=style,linewidth=1)
                ax.text(value, 0, stat, horizontalalignment="center", verticalalignment="top", transform=ax.get_xaxis_transform(), rotation=45,color=color)  
        elif orient == 'h':   
            for stat,value in {'mean':np.mean(data),'median':np.median(data)}.items():
                ax.axhline(value, color=color, linestyle=style,linewidth=1)
                ax.text(0,value, stat, verticalalignment="center", horizontalalignment="left", transform=ax.get_yaxis_transform(), rotation=0,color=color)      
    
    set_axis_style(ax=ax,y=y,x=by,stat=stat)

    if y not in [None,'none','None']:
        if by in [None,'none','None']:
            if orient == 'v':
                sns.histplot(
                    data=df,
                    x=None if y in [None,'none','None'] else y,
                    multiple='layer',stat=stat,
                    kde=True,
                    legend=True,
                    color=get_darker_color(CONFIG['Chart']['data_colors'][0],10),
                    ax=ax
                )     
            elif orient == 'h':
                sns.histplot(
                    data=df,
                    y=None if y in [None,'none','None'] else y,
                    multiple='layer',stat=stat,
                    kde=True,
                    legend=True,
                    color=get_darker_color(CONFIG['Chart']['data_colors'][0],10),
                    ax=ax
                )     
        else:
            print(f"{by=}\n{df.head()}") # monitor
            COLOR_PALLETTE = {cat:CONFIG['Chart']['data_colors'][i % len(CONFIG['Chart']['data_colors'])] for i,cat in enumerate(df[by].unique())}
            for cat,color in COLOR_PALLETTE.items():
                if orient == 'v':
                    sns.histplot(
                        data=df[df[by]==cat],
                        x=None if y in [None,'none','None'] else y,
                        multiple='layer',stat=stat,
                        kde=True,
                        legend=True,
                        color=get_darker_color(color,10),
                        ax=ax
                    ) 
                elif orient == 'h':        
                    sns.histplot(
                        data=df[df[by]==cat],
                        y=None if y in [None,'none','None'] else y,
                        multiple='layer',stat=stat,
                        kde=True,
                        legend=True,
                        color=get_darker_color(color,10),
                        ax=ax
                    ) 

                if category_stats in [True,'true','True']:
                        set_stats(ax=ax,data=df.loc[df[by]==cat,y],color=get_darker_color(color,20),style='-')

    if overall_stats in [True,'true','True']:
        set_stats(ax=ax,data=df[y],color='grey',style='--')

    
def set_count_plot(ax,df:pd.DataFrame,y:str=None,by:str=None,orient:str='h'):
    def set_axis_style(ax,y:str,x:str):
        ax.set_xlabel(ax.get_xlabel() , fontsize=13, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(ax.get_ylabel() , fontsize=13, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    def set_anotation(ax=ax,orient=orient):
        for container in ax.containers:
            ax.bar_label(
                container, label_type='edge', fontsize=10, color='black', padding=3,
                labels=[f'{int(bar.get_height())}' if orient == 'v' else f'{int(bar.get_width())}'  for bar in container]
                )

    #print(f'set_count_plot({y},{by})') # monitor

    category_column = y if by in [None,'none','None'] else by
    COLOR_PALLETTE = {cat:CONFIG['Chart']['data_colors'][i % len(CONFIG['Chart']['data_colors'])] for i,cat in enumerate(df[category_column].unique())}
    sns.countplot(
        ax=ax,data=df,
        y=y if orient == 'h' else None,
        x=y if orient == 'v' else None,
        hue=None if by in [None,'none','None'] else by,
        #dodge=True,
        palette=COLOR_PALLETTE,
        edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][0],40)
        )

    set_axis_style(ax=ax,y=y,x=by)
    set_anotation(ax=ax,orient=orient)
def set_scatter_plot(ax,df:pd.DataFrame,y:str=None,x:str=None,by:str=None,color:str=None):
    def set_axis_style(ax,y:str,x:str):
        ax.set_xlabel(x, fontsize=13, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(y, fontsize=13, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    POINT_SIZE = 8 if len(df) > 1000 else 11 if len(df) > 200 else 15
    ALPHA = 0.2 if len(df) > 1000 else 0.4 if len(df) > 200 else 0.6

    if by in ['None','none',None]:
        ax.scatter(
            df[x],df[y],
            alpha=ALPHA,edgecolors=get_darker_color(CONFIG['Chart']['data_colors'][0],50),
            s=POINT_SIZE, c=CONFIG['Chart']['data_colors'][0]
        )
    else:
        for i,cat in enumerate(df[by].unique()):
            COLOR_INDEX = i % len(CONFIG['Chart']['data_colors'])
            data = df.loc[df[by]==cat,[y,x,by]]
            ax.scatter(
                data[x],data[y],
                alpha=ALPHA,edgecolors=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],50),
                s=POINT_SIZE, c=CONFIG['Chart']['data_colors'][COLOR_INDEX],
                label=cat
            ) 

    try:
        ax.legend()
    except:
        pass    
    set_axis_style(ax=ax,y=y,x=x)

# analysis 
def get_chi2_analysis(df:pd.DataFrame,y:str=None,by:str=None,alpha:float=0.05):
    def set_log(df,y,by,alpha,ct):
        ct = pd.crosstab(df[by], df[y])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        return textwrap.dedent(f'''\
                Chi-Squared Analysis:
                ---------------------
                df = '{DATA_TABLE["file_name"]}'
                y = '{y}' (= Response column)
                by = {by} 
                α = {alpha} (= Significance level)

                Null Hypothesis (H0): {y} and {by} are independent
                Alternative Hypothesis (H1): {y} and {by} are dependent

                Chi-Squared Test:
                ---------------------
                Chi2 = {chi2:.2f}
                p-value = {p:.4f}
                Degrees of Freedom (dof) = {dof}
                Decision: {'Reject H0 (dependent)' if p < alpha else 'Fail to reject H0 (independent)'}
                ''')
    
    ct = pd.crosstab(df[by], df[y])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    residuals = (ct - expected) / expected**0.5
    log = set_log(df,y,by,alpha,ct)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Observed Counts')

    sns.heatmap(pd.DataFrame(expected, index=ct.index, columns=ct.columns),
                annot=True, fmt='.1f', cmap='Oranges', ax=axes[1])
    axes[1].set_title('Expected Counts')

    sns.heatmap(residuals, annot=True, fmt='.2f', center=0, cmap='coolwarm', ax=axes[2])
    axes[2].set_title('Standardized Residuals')

    return {
        'output':{'log':log,'plot':fig,'table':ct},
        'output_type':'analysis',
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'y':{
                'type':'category',
                'options':[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':None
            },
            'by':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            },
            alpha:{
                'type':'number',
                'options':[0.01,0.05,0.1],
                'default':0.05
            }
        }
    }
def get_correlation_analysis(df:pd.DataFrame,y:str=None,x:str=None,by:str=None,contamination:float=0.03):
    def set_log(df,y,x,by,contamination):
        return textwrap.dedent(f"""\
        Correlation Analysis (= Numeric vs Numeric):
        --------------------------------------------
        df = '{DATA_TABLE["file_name"]}'
        y  = '{y}'  (= Response column)
        X  = '{x}'  (= Predictor column)
        by = '{by}'
        Contamination = {contamination}  (= Ignored % of data points)
    """)

    fig = plt.figure(figsize=(6,7), dpi=80,constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5], wspace=0.05, hspace=0.05)

    #set_axis_style(ax=ax,y=y,x=by)
    log = set_log(df,y,x,by,contamination)
    table = pd.DataFrame(columns=['category','count','included','ignored','equation','r2','rmse'])
    

    if y not in [None,'none','None'] and x not in [None,'none','None']:

        if by in [None,'none','None']:

            data = df[[x,y]].copy()
            data['pred'],data['inlier'] = None, None  # Initialize column
            if contamination > 0:
                clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
                data['inlier'] = clf.fit_predict(data[[y,x]])
            else:
                data['inlier'] = 1  

            reg_data = data[data.inlier == 1].copy()
            X, y_act = reg_data[[x]], reg_data[y] 
            lr = LinearRegression()
            lr.fit(X, y_act)
            reg_data['pred'] = lr.predict(X)
            r2 = r2_score(y_act, reg_data['pred'])
            rmse = mean_squared_error(y_act, reg_data['pred'], squared=False)
            table.loc[len(table)] = [None,len(data),len(reg_data),int(len(data)*contamination),f"y = {lr.intercept_:.2f} + {lr.coef_[0]:.2f}*x",r2,rmse]

            ax3 = fig.add_subplot(gs[1,0]) 
            set_scatter_plot(ax=ax3,df=reg_data,y=y,x=x,by=by) # inliers
            ax3.plot(X,reg_data['pred'], color='red',linewidth=1.5)

        else:
            data = df[[x, y, by]].copy()
            data['pred'], data['inlier'] = None, None  # Initialize column
            ax3 = fig.add_subplot(gs[1, 0]) 

            # --- Generate label_color_map before loop ---
            temp_ax = fig.add_subplot(gs[0, 0])  # temporary axis for color mapping
            set_scatter_plot(ax=temp_ax, df=data, y=y, x=x, by=by)
            handles, labels = temp_ax.get_legend_handles_labels()
            label_color_map = {label: handle.get_facecolor()[0] for label, handle in zip(labels, handles)}
            plt.delaxes(temp_ax)  # remove temp axis

            for i, cat in enumerate(data[by].unique()):
                sub_data = data[data[by] == cat].copy()
                if contamination > 0:
                    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
                    sub_data['inlier'] = clf.fit_predict(sub_data[[y, x]])
                else:
                    sub_data['inlier'] = 1

                X, y_act = sub_data[[x]], sub_data[y]
                lr = LinearRegression()
                lr.fit(X, y_act)
                preds = lr.predict(X)

                data.loc[sub_data.index, 'inlier'] = sub_data['inlier']
                data.loc[sub_data.index, 'pred'] = preds

                r2 = r2_score(y_act, preds)
                rmse = mean_squared_error(y_act, preds, squared=False)
                table.loc[len(table)] = [cat,len(sub_data),len(sub_data[sub_data['inlier'] == 1]),len(sub_data[sub_data['inlier'] == -1]),f"y = {lr.intercept_:.2f} + {lr.coef_[0]:.2f}*x",r2,rmse]

                ax3.plot(X, preds, color=label_color_map.get(cat, 'black'), linewidth=1.5)

        ax1 = fig.add_subplot(gs[0,0])  
        set_dist_plot(ax=ax1,df=data[data.inlier == 1],y=x,by=by,stat='count',orient='v',category_stats=False,overall_stats=False)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1,1])  
        set_dist_plot(ax=ax2,df=data[data.inlier == 1],y=y,by=by,stat='count',orient='h',category_stats=False,overall_stats=False)
        ax2.axis('off')

        set_scatter_plot(ax=ax3,df=data[data.inlier == 1],y=y,x=x,by=by)
        handles, labels = ax3.get_legend_handles_labels()
        label_color_map = {label: handle.get_facecolor()[0] for label, handle in zip(labels, handles)}
        #set_scatter_plot(ax=ax3,df=df[df.inlier==-1],y=y,x=x,by=by) # outliers    
            
        try:
            ax3.legend_.remove()
        except:
            pass  

        # legend
        ax_legend = fig.add_subplot(gs[0, 1])
        ax_legend.axis('off')
        handles, labels = ax3.get_legend_handles_labels()
        ax_legend.legend(handles, labels, loc='center',bbox_to_anchor=(1,1),fontsize=11,frameon=False)

    #set_axes_style(ax=axs)

    return {
        'output':{'log':log,'plot':fig,'table':table},
        'output_type':'analysis',
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'y':{
                'type':'category',
                'options':[f"'{item}'" for item in get_numeric_columns(df=df,min_uniques=1)],
                'default':None
            },
            'x':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_numeric_columns(df=df,min_uniques=1)],
                'default':'None'
            },
            'by':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            },
            'contamination':{
                'type':'number',
                'options':[0.03,0.05,0.1],
                'default':0.03
            }
        }
    }
def get_anova_analysis(df:pd.DataFrame,y:str=None,by:str=None,contamination:float=0.03):
    def set_log(df,y,by,contamination):

        try:
            f_stat, p_val = stats.f_oneway(*[df.loc[df[by]==cat,y].values for cat in df[by].unique()])
        except:
            f_stat, p_val = None,None    

        anova_decision = None if y in ['None','none',None] or by in ['None','none',None] else 'Significant' if p_val < 0.05 else 'Not significant'    
        decision_text = None if anova_decision == None else f"{by} effect on {y} is {anova_decision} (**ignoring {contamination*100}% of data)"

        return textwrap.dedent(f"""\
        Analysis of Variance:
        ---------------------
        df = '{DATA_TABLE["file_name"]}'
        y = '{y}' (Numeric tested column)
        by = '{by}' (Predictor Categorical column)
        Contamination = {contamination} (=Ignored % of data points)

        F-Statistic = {f_stat:.4f}
        P-Value = {p_val:.4f}

        Decision: {decision_text}
        1. Null Hypothesis (H0): {by} has no effect on {y}
        2. Alternative Hypothesis (H1): {by} has an effect on {y}
        """)
    def get_stats(data:pd.Series):
        return {
            'count':len(data),
            'min':data.min(),
            'max':data.max(),
            'mean':data.mean(),
            'median':data.median(),
            'skewness':(3*(data.mean() - data.median()))/data.std() if data.std() != 0 else 0,
            'std':data.std(),
            'q1':data.quantile(0.25),
            'q3':data.quantile(0.75),
            'lcl':data.quantile(0.003),
            '-3*std':data.quantile(0.003),
            'ucl':data.quantile(0.997),
            '+3*std':data.quantile(0.997),
            'iqr':data.quantile(0.75) - data.quantile(0.25),
            'IQR':f"[{data.quantile(0.25)}:{data.quantile(0.75)}]",
            'lower_whisker':max(data.min(), data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'upper_whisker':min(data.max(), data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'outliers':[]
        }
    def set_data(df:pd.DataFrame,y:str=None,by:str=None,contamination=0.0):
        
        data = df[[y]].copy() if by in [None,'None','none'] else df[[y,by]].copy()
        STATS = get_stats(data[y])

        if contamination > 0:
            iso_forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
            data['inlier'] = iso_forest.fit_predict(data[[y]])
        else:
            data['inlier'] = 1    

        outliers = data.loc[data['inlier']==-1,:].drop('inlier', axis=1).copy()
        inliers = data.loc[data['inlier']==1,:].drop('inlier', axis=1).copy()
        return data,inliers,outliers
    def set_axis_style(ax,y:str,x:str,orient:str=['v','h']):
        if orient == 'v':
            ax.set_xlabel(x, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(y, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
        else:
            ax.set_xlabel(y, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(x, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])   

        ax.tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    TTEST_ALPHA = 0.05
    OPACITY = 0.4
    try:
        log = set_log(df,y,by,contamination)
    except:
        log = 'No data to analyze'    
    HEIGHT = 1 if by in [None,'none','None'] else len(df[by].unique())
    fig, ax = plt.subplots(figsize=(5,HEIGHT),dpi=80)
    set_axis_style(ax=ax,y=y,x=by,orient='h')
    
    if y in [None,'none','None']:
        table = pd.DataFrame(data={'Missing Data':['Pick y column']})
    else:
        all_data, inliers, outliers = set_data(df=df,y=y,by=by,contamination=contamination)
        STATS = get_stats(data=inliers[y])
        table = pd.DataFrame(
                { # summary table
                    'category':['all'],
                    'count':[f"{int(STATS['count'])}"],
                    'included':[None],
                    'mean':[f"{STATS['mean']:.2f}"],
                    'median':[f"{STATS['median']:.4f}"],
                    'std':[f"{STATS['std']:.2f}"],
                    'IQR':[f"[{STATS['q1']:.2f}:{STATS['q3']:.2f}]"],
                    'skewness':[f"{STATS['skewness']:.2f}"],
                    'outliers':[len(outliers)],
                    'p_value':[None],
                    'decision':[None]
                }
            )
        
        if by in [None,'none','None']:
            set_box_plot(ax=ax,df=all_data,y=y,by=by,orient='h',overall_mean=True,category_mean=False,std_lines=True)
            set_strip_plot(ax=ax,df=inliers,y=y,by=by,orient='h',opacity=OPACITY)
            set_strip_plot(ax=ax,df=outliers,y=y,by=by,orient='h',color='red',opacity=OPACITY)
        else:    
            set_box_plot(ax=ax,df=all_data,y=y,by=by,orient='h',overall_mean=True,category_mean=True,std_lines=True)
            set_strip_plot(ax=ax,df=inliers,y=y,by=by,orient='h',opacity=OPACITY)

            _, all_cat_inliers, _ = set_data(df=df,y=y,by=by,contamination=contamination)
            for cat in df[by].unique():
                all_data, inliers, outliers = set_data(df=df.loc[df[by]==cat,[y,by]],y=y,by=by,contamination=contamination)
                STATS = get_stats(all_data[y])
                t_stat, p_val = stats.ttest_ind(all_cat_inliers.loc[all_cat_inliers[by]==cat,y].dropna(), all_cat_inliers.loc[all_cat_inliers[by]!=cat,y].dropna())
                table.loc[len(table)] = { # category stats
                    'category':cat,
                    'count':f"{int(STATS['count'])}",
                    'included':f"{len(inliers)}",
                    'mean':f"{STATS['mean']:.2f}",
                    'median':f"{STATS['median']:.4f}",
                    'std':f"{STATS['std']:.2f}",
                    'IQR':f"[{STATS['q1']:.2f}:{STATS['q3']:.2f}]",
                    'skewness':f"{STATS['skewness']:.2f}",
                    'outliers':len(outliers),
                    'p_value':f"{p_val:.4f}",
                    'decision':"Significant" if p_val < TTEST_ALPHA else "Insignificant"
                }
                set_strip_plot(ax=ax,df=outliers,y=y,by=by,orient='h',color='red',opacity=OPACITY) 
              

    return {
        'output':{'log':log,'plot':fig,'table':table},
        'output_type':'analysis',
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'y':{
                'type':'category',
                'options':[f"'{item}'" for item in get_numeric_columns(df=df,min_uniques=1)],
                'default':None
            },
            'by':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            },
            'contamination':{
                'type':'number',
                'options':[0.03,0.05,0.1],
                'default':0.03
            }
        }
    }
def get_outliers_analysis(df:pd.DataFrame,y:str=None,by:str=None,contamination=0.03):
    def set_log(y,by,contamination):
        return textwrap.dedent(f'''\
                Outliers Detection:
                -------------------
                df = '{DATA_TABLE["file_name"]}'
                y = '{y}'
                by = '{by}'
                Contamination = {contamination}
                ''')
    def set_data(df:pd.DataFrame,y:str=None,by:str=None,contamination=0.0):
        
        data = df[[y]].copy() if by in [None,'None','none'] else df[[y,by]].copy()
        STATS = get_stats(data[y])

        if contamination > 0:
            iso_forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
            data['inlier'] = iso_forest.fit_predict(data[[y]])
        else:
            data['inlier'] = 1    

        outliers = data.loc[data['inlier']==-1,:].drop('inlier', axis=1).copy()
        inliers = data.loc[data['inlier']==1,:].drop('inlier', axis=1).copy()
        return data,inliers,outliers
    def get_stats(data:pd.Series):
        return {
            'count':len(data),
            'min':data.min(),
            'max':data.max(),
            'mean':data.mean(),
            'median':data.median(),
            'skewness':(3*(data.mean() - data.median()))/data.std() if data.std() != 0 else 0,
            'std':data.std(),
            'q1':data.quantile(0.25),
            'q3':data.quantile(0.75),
            'lcl':data.quantile(0.003),
            '-3*std':data.quantile(0.003),
            'ucl':data.quantile(0.997),
            '+3*std':data.quantile(0.997),
            'iqr':data.quantile(0.75) - data.quantile(0.25),
            'IQR':f"[{data.quantile(0.25)}:{data.quantile(0.75)}]",
            'lower_whisker':max(data.min(), data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'upper_whisker':min(data.max(), data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'outliers':[]
        }
    def set_axis_style(ax,y:str,x:str):
        ax[1].set_xlabel(y, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])
        ax[1].set_ylabel(x, fontsize=12, fontfamily=CONFIG['Chart']['font'], color=CONFIG['Chart']['font_color'])   
        ax[1].tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax[1].tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].get_xaxis().set_visible(False)
        ax[0].spines['left'].set_linewidth(1)
        ax[1].spines['left'].set_linewidth(1)
        ax[1].spines['bottom'].set_linewidth(1)

    fig, ax = plt.subplots(2,1,figsize=(7,10),dpi=75,sharex='all')
    log = set_log(y,by,contamination)
    set_axis_style(ax=ax,y=y,x=by)

    if y not in [None,'none','None']:
        #print(f'>> set_data(df={df},y={y},contamination={contamination})')
        all_data, inliers, outliers = set_data(df=df,y=y,by=by,contamination=contamination)
        STATS = get_stats(all_data[y])
        table = pd.DataFrame(
                { # summary table
                    'category':['all'],
                    'count':[f"{int(STATS['count'])}"],
                    'min':[f"{STATS['min']:.2f}"],
                    'mean':[f"{STATS['mean']:.2f}"],
                    'median':[f"{STATS['median']:.4f}"],
                    'std':[f"{STATS['std']:.2f}"],
                    'max':[f"{STATS['max']:.2f}"],
                    'IQR':[f"[{STATS['q1']:.2f}:{STATS['q3']:.2f}]"],
                    'skewness':[f"{STATS['skewness']:.2f}"],
                    'outliers':[len(outliers)]
                }
            )

        if by not in [None,'none','None']:
            set_box_plot(ax=ax[0],df=all_data,y=y,by=by,orient='h',overall_mean=False,category_mean=True,std_lines=False)
            set_strip_plot(ax=ax[0],df=inliers,y=y,by=by,orient='h') 
            set_dist_plot(ax=ax[1],df=all_data,y=y,by=by,category_stats=True,overall_stats=False)

            for cat in df[by].unique():
                all_data, inliers, outliers = set_data(df=df.loc[df[by]==cat,[y,by]],y=y,by=by,contamination=contamination)
                STATS = get_stats(all_data[y])
                table.loc[len(table)] = { # category stats
                    'category':cat,
                    'count':f"{int(STATS['count'])}",
                    'min':f"{STATS['min']:.2f}",
                    'mean':f"{STATS['mean']:.2f}",
                    'median':f"{STATS['median']:.4f}",
                    'std':f"{STATS['std']:.2f}",
                    'max':f"{STATS['max']:.2f}",
                    'IQR':f"[{STATS['q1']:.2f}:{STATS['q3']:.2f}]",
                    'skewness':f"{STATS['skewness']:.2f}",
                    'outliers':len(outliers)
                }
                set_strip_plot(ax=ax[0],df=outliers,y=y,by=by,orient='h',color='red') 
        else:
            set_box_plot(ax=ax[0],df=all_data,y=y,by=by,orient='h',overall_mean=False,category_mean=True,std_lines=False)
            set_strip_plot(ax=ax[0],df=inliers,y=y,by=by,orient='h') 
            set_strip_plot(ax=ax[0],df=outliers,y=y,by=by,orient='h',color='red') 
            set_dist_plot(ax=ax[1],df=all_data,y=y,by=by,category_stats=False,overall_stats=True)

        fig.tight_layout()

    else:
        table = pd.DataFrame()    

    return {
        'output':{'log':log,'plot':fig,'table':table},
        'output_type':'analysis',
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'y':{
                'type':'category',
                'options':[f"'{item}'" for item in get_numeric_columns(df=df,min_uniques=1)],
                'default':None
            },
            'by':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            },
            'contamination':{
                'type':'number',
                'options':[0.03,0.05,0.1],
                'default':0.03
            }
        }
    }

    def get_stats(data):
        return {
            'count':len(data),
            'min':data.min(),
            'max':data.max(),
            'mean':data.mean(),
            'median':data.median(),
            'skewness':(3*(data.mean() - data.median()))/data.std() if data.std() != 0 else 0,
            'std':data.std(),
            'q1':data.quantile(0.25),
            'q3':data.quantile(0.75),
            'lcl':data.quantile(0.003),
            '-3*std':data.quantile(0.003),
            'ucl':data.quantile(0.997),
            '+3*std':data.quantile(0.997),
            'iqr':data.quantile(0.75) - data.quantile(0.25),
            'IQR':f"[{data.quantile(0.25)}:{data.quantile(0.75)}]",
            'lower_whisker':max(data.min(), data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'upper_whisker':min(data.max(), data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'outliers':[]
        }
    def set_vlines(ax,stats:{},keys:{}):
        for key,color in keys.items():
            #print(f"{key}:{stats[key]}(color={color})")
            LABEL,VALUE,COLOR = key,stats[key],color
            ax.axvline(VALUE, color=COLOR, linestyle='-',linewidth=2) #label=f'{LABEL} = {VALUE:.2f}'
            ax.text(VALUE, 0, LABEL, horizontalalignment="center", verticalalignment="top", transform=ax.get_xaxis_transform(), rotation=45,color=COLOR)           
    def set_data(data:pd.DataFrame,x:str,outliers=None):
        #inliers_data,outliers_data = pd.DataFrame(columns=data.columns), pd.DataFrame(columns=data.columns)
        STATS = get_stats(data[x])

        if any([item in outliers for item in ['iqr','IQR']]):
            #print(' >>> iqr')
            outliers_data = data.loc[(data[x] < STATS['lower_whisker'])|(data[x] > STATS['upper_whisker']),:].copy()
            inliers_data = data.loc[(data[x] >= STATS['lower_whisker'])&(data[x] <= STATS['upper_whisker']),:].copy()
        elif '%' in outliers:     
            #print(' >>> %')
            perc_string = outliers.split('_')[1]
            PERCENTAGE = float(perc_string[:perc_string.find('%')])
            #print(f" >>> PERCENTAGE={PERCENTAGE}")
            iso_forest = IsolationForest(n_estimators=200, contamination=PERCENTAGE/100, random_state=42)
            data['inlier'] = iso_forest.fit_predict(data[[x]])
            outliers_data = data.loc[data['inlier']==-1,:].drop('inlier', axis=1).copy()
            inliers_data = data.loc[data['inlier']==1,:].drop('inlier', axis=1).copy()
        else: # outliers in [None,"None",'none']
            #print(' >>> no outliers')    
            inliers_data = data.loc[:,:].copy()
            outliers_data = data.loc[0:-1,:].copy()

        #print(f" >>> inliers_data={len(inliers_data)}\n >>> outliers_data={len(outliers_data)}")
        return inliers_data,outliers_data
    def set_kde(ax,data,color='#d89fee'):
        kde_x = np.linspace(data.values.min(), data.values.max(),max(100,int((len(data))**0.5)))
        kde_y = gaussian_kde(data.values)(np.linspace(data.values.min(), data.values.max(),max(100,int((len(data))**0.5))))
        ax.twinx().plot(kde_x,kde_y, color=get_darker_color(color,30), label='Density', linewidth=2)   

    if x in [None,'none','None']:
        x = list(df.select_dtypes(include=['number']).columns)[0]  

    data= df[[x]].dropna().reset_index(drop=True) if by in ['none','None',None] else df[[x,by]].dropna().reset_index(drop=True)
    inliers_data,outliers_data = set_data(data=data,x=x,outliers=outliers)
    STATS = get_stats(data[x])    
    st = { # summary table
            'category':['all'],
            'count':[f"{STATS['count']:.2f}"],
            'min':[f"{STATS['min']:.2f}"],
            'mean':[f"{STATS['mean']:.2f}"],
            'median':[f"{STATS['median']:.4f}"],
            'std':[f"{STATS['std']:.2f}"],
            'max':[f"{STATS['max']:.2f}"],
            'IQR':[f"[{STATS['q1']:.2f}:{STATS['q3']:.2f}]"],
            'skewness':[f"{STATS['skewness']:.2f}"],
            'outliers':[len(outliers_data)]
            }

    if by in [None,'none','None']:
        POINT_SIZE = 5 if len(inliers_data) > 1000 else 8 if len(inliers_data) > 200 else 9
        ALPHA = 0.1 if len(inliers_data) > 1000 else 0.4 if len(inliers_data) > 200 else 0.6
        HEIGHT = 3
        fig, axs = plt.subplots(2,1,figsize=(6,HEIGHT),dpi=75,sharex=True,gridspec_kw={'height_ratios': [HEIGHT,3]})

        sns.stripplot(inliers_data[x],ax=axs[0],orient='h',alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=CONFIG['Chart']['data_colors'][0],edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][0],70),jitter=0.35,zorder=0) 
        if any([item in outliers for item in ['exclude','Exclude','EXCLUDE']]):
            STATS = get_stats(inliers_data[x])
            sns.boxplot(inliers_data[x],linewidth=2,orient='h',boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},showfliers=False, ax=axs[0])
            axs[1].hist(inliers_data[x],bins=min(len(inliers_data[x]),50),color=CONFIG['Chart']['data_colors'][0],edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][0],70), alpha=0.3)
            if len(inliers_data) > 100:
                set_kde(ax=axs[1],data=inliers_data[x],color=CONFIG['Chart']['data_colors'][0])
        else:
            STATS = get_stats(data[x])
            sns.stripplot(outliers_data[x],ax=axs[0],orient='h',alpha=0.4,size=POINT_SIZE,linewidth=0.5,color='red',edgecolor=get_darker_color(CONFIG['Chart']['frame_color'],70),jitter=0.3,zorder=0)   
            sns.boxplot(data[x],linewidth=2,boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},showfliers=False, ax=axs[0])    
            axs[1].hist(data[x],bins=min(len(data[x]),50),color=CONFIG['Chart']['data_colors'][0],edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][0],70), alpha=0.3)
            if len(data[x]) > 100:
                    set_kde(ax=axs[1],data=data[x],color=CONFIG['Chart']['data_colors'][0])

    else: # by categories
        POINT_SIZE = 5 if len(data) > 1000 else 8 if len(data) > 200 else 9
        HEIGHT = int(1.5*len(data[by].unique()))
        fig, axs = plt.subplots(2,1,figsize=(6,HEIGHT),dpi=75,sharex=True,gridspec_kw={'height_ratios': [HEIGHT,3]})

        if any([item in outliers for item in ['exclude','Exclude','EXCLUDE']]):
            sns.boxplot(x=inliers_data[x],y=inliers_data[by],orient='h',linewidth=2,boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},showfliers=False, ax=axs[0])
        else:
            sns.boxplot(x=data[x],y=data[by],linewidth=2,orient='h',boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},showfliers=False, ax=axs[0])

        for i,cat in enumerate(data[by].unique()):
            
            COLOR_INDEX = i % len(CONFIG['Chart']['data_colors'])
            ALPHA = 0.1 if len(data[data[by]==cat]) > 1000 else 0.4 if len(data[data[by]==cat]) > 200 else 0.6
            inliers_data,outliers_data = set_data(data=data.loc[data[by]==cat],x=x,outliers=outliers)
            #print(f" >>> category={cat}, inliers={len(inliers_data)}, outliers={len(outliers_data)}")

            # update summary table
            STATS = get_stats(data.loc[data[by]==cat,x]) 
            for key in st.keys():
                st[key].append(len(outliers_data) if key=='outliers' else cat if key=='category' else STATS[key])

            sns.stripplot(x=inliers_data.loc[inliers_data[by]==cat,x],y=[cat]*len(inliers_data.loc[inliers_data[by]==cat,x]),ax=axs[0],orient='h',alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=CONFIG['Chart']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70),jitter=0.35,zorder=0) 
            if any([item in outliers for item in ['exclude','Exclude','EXCLUDE']]):
                STATS = get_stats(inliers_data[x])
                axs[1].hist(inliers_data.loc[inliers_data[by]==cat,x],bins=min(len(inliers_data.loc[inliers_data[by]==cat,x]),50),color=CONFIG['Chart']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70), alpha=0.3)
                set_kde(ax=axs[1],data=inliers_data.loc[inliers_data[by]==cat,x],color=CONFIG['Chart']['data_colors'][COLOR_INDEX])
            else:    
                sns.stripplot(x=outliers_data[x],y=[cat]*len(outliers_data.loc[outliers_data[by]==cat,:]),ax=axs[0],orient='h',alpha=0.4,size=POINT_SIZE,linewidth=0.5,color='red',edgecolor=get_darker_color(CONFIG['Chart']['frame_color'],70),jitter=0.3,zorder=0)   
                axs[1].hist(data.loc[data[by]==cat,x],bins=min(len(data.loc[data[by]==cat,x]),50),color=CONFIG['Chart']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70), alpha=0.3)
                set_kde(ax=axs[1],data=data.loc[data[by]==cat,x],color=CONFIG['Chart']['data_colors'][COLOR_INDEX])

            #set_vlines(ax=axs[1],stats=STATS,keys={'mean':get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],60),'median':get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],30)})   
    
    for side in ['top','bottom','right','left']: 
        axs[0].spines[side].set_linewidth(1)
        axs[1].spines[side].set_linewidth(1)
    
    plt.tight_layout()
    axs[0].set(xlabel=None)
    axs[0].set(ylabel=None)
    axs[1].set_ylabel("Count")
    axs[1].twinx().set_ylabel("Density")
    axs[1].legend()

    return {
        'output':fig,
        'output_type':'chart',
        'title':f'"{x}" Values distribution:',
        'table':pd.DataFrame(st),
        'args':{
            'df':{
                'type':'category',
                'options':['df'],
                'default':f"'df'"
            },
            'x':{
                'type':'category',
                'options':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
                'default':list(df.select_dtypes(include=['number']).columns)[0]
                },
            'by':{
                'type':'category',
                'options':["None"] + [f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
                },
            'outliers':{
                'type':'category',
                'options':[f'"None"',f'"show_IQR"',f'"show_0.3%"',f'"show_0.5%"',f'"show_1%"',f'"show_5%"',f'"exclude_IQR"',f'"exclude_0.3%"',f'"exclude_0.5%"',f'"exclude_1%"',f'"exclude_5%"'],
                'default':'None'
            }
            }
        }