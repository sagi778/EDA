import pandas as pd
import numpy as np
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
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
def set_data(df:pd.DataFrame,y:str,by:str,contamination=0.0):
        data = df[[y]].dropna() if by in [None,'None','none'] else df[[y,by]].dropna()
        STATS = get_stats(data[y])

        if contamination > 0:
            iso_forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
            data['inlier'] = iso_forest.fit_predict(data[[y]])
        else:
            data['inlier'] = 1    

        outliers = data.loc[data['inlier']==-1,:].drop('inlier', axis=1).copy()
        inliers = data.loc[data['inlier']==1,:].drop('inlier', axis=1).copy()
        return data,inliers,outliers

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

# plots for plots
def get_box_plot(df:pd.DataFrame,y:str=None,by:str=None,orient:str='v',overall_mean=False,category_mean=True,std_lines=True):
    def set_axis_style(ax,y:str,x:str):
        ax.set_xlabel(x, fontsize=9, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(y, fontsize=9, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    MAX_CATEGORIES = 30
    NUM_OF_CATEGORIES = 1 if by in [None,'none','None'] else max(len(df[by].unique()),MAX_CATEGORIES)
    HEIGHT = 5 if orient == 'v' else 1*NUM_OF_CATEGORIES
    WIDTH = 1*NUM_OF_CATEGORIES if orient == 'v' else 5
    fig, ax = plt.subplots(figsize=(WIDTH,HEIGHT),dpi=85)

    try:
        set_box_plot(ax=ax,df=df,y=y,by=by,orient=orient,overall_mean=overall_mean,category_mean=category_mean,std_lines=std_lines)
        set_strip_plot(ax=ax,df=df,y=y,by=by,orient=orient)
    except Exception as e:
        print(e)    
    
    return {
        'output':fig,
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
def get_count_plot(df:pd.DataFrame,y:str=None,by:str=None):
    def get_num_of_categories(df:pd.DataFrame=df,y:str=y,by:str=by):
        y_amount = 1 if y in [None,'none','None'] else len(df[y].unique())
        by_amount = 1 if by in [None,'none','None'] else len(df[by].unique())
        return y_amount*by_amount

    NUM_OF_CATEGORIES = get_num_of_categories(df,y,by)
    fig, ax = plt.subplots(figsize=(1,NUM_OF_CATEGORIES),dpi=85)
    
    try:
        set_count_plot(ax=ax,df=df,y=y,by=by)
    except Exception as e:
        print(e) 

    return {
        'output':fig,
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
    
    fig, ax = plt.subplots(figsize=(5,5),dpi=85)
    try:
        set_scatter_plot(ax=ax,df=df,y=y,x=x,by=by)
    except Exception as e:
        print(e)    

    set_axis_style(ax,y,x)
    

    return {
        'output':fig,
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
def get_dist_plot(df:pd.DataFrame,y:str=None,by:str=None,stat:str='count'):
    def get_num_of_categories(df:pd.DataFrame=df,y:str=y,by:str=by):
        y_amount = 1 if y in [None,'none','None'] else len(df[y].unique())
        by_amount = 1 if by in [None,'none','None'] else len(df[by].unique())
        return y_amount*by_amount

    NUM_OF_CATEGORIES = get_num_of_categories(df,y,by)
    fig, ax = plt.subplots(figsize=(1,NUM_OF_CATEGORIES),dpi=85)
    
    try:
        set_dist_plot(ax=ax,df=df,y=y,by=by,stat=stat)
    except Exception as e:
        print(e) 

    return {
        'output':fig,
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
            'stat':{
                'type':'category',
                'options':['"count"'],
                'default':'"count"'
            }
        }
    }

# plots for everythig
def set_strip_plot(ax,df:pd.DataFrame,y:str=None,by:str=None,orient='v',color:str=None):
    def set_axis_style(ax,y:str,x:str):
        if orient == 'v':
            ax.set_xlabel(x, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(y, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        elif orient == 'h':
            ax.set_xlabel(y, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(x, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color']) 

        ax.tick_params(axis='x',labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=11,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    if by in [None,'none','None']:
        COLOR_INDEX = 0
        data = df.loc[:,[y]].copy()
        POINT_SIZE = 5 if len(data) > 1000 else 8 if len(data) > 200 else 9
        ALPHA = 0.2 if len(data) > 1000 else 0.4 if len(data) > 200 else 0.6
        
        if orient == 'v':
            sns.stripplot(y=data[y],
                ax=ax,alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,
                color=CONFIG['Chart']['data_colors'][COLOR_INDEX] if color in [None,'none','None'] else 'white',
                edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70) if color in [None,'none','None'] else color,
                jitter=0.35,zorder=0
                ) 
        elif orient == 'h':
            sns.stripplot(x=data[y],
                ax=ax,alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,
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
            ALPHA = 0.2 if len(data) > 1000 else 0.4 if len(data) > 200 else 0.6

            if orient == 'v':
                sns.stripplot(y=data[y],x=data[by],
                    ax=ax,alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,
                    color=CONFIG['Chart']['data_colors'][COLOR_INDEX] if color in [None,'none','None'] else 'white',
                    edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70) if color in [None,'none','None'] else color,
                    jitter=0.35,zorder=0
                    ) 
            else:
                sns.stripplot(x=data[y],y=data[by],
                    ax=ax,alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,
                    color=CONFIG['Chart']['data_colors'][COLOR_INDEX] if color in [None,'none','None'] else 'white',
                    edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70) if color in [None,'none','None'] else color,
                    jitter=0.35,zorder=0
                    )        

    #set_axis_style(ax,y,by)
def set_box_plot(ax,df:pd.DataFrame,y:str=None,by:str=None,orient='v',overall_mean=True,category_mean=True,std_lines:bool=True):
    def set_axis_style(ax,y:str,x:str,orient=orient):
        if orient == 'v':
            ax.set_xlabel(x, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(y, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        else:
            ax.set_xlabel(y, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
            ax.set_ylabel(x, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])   

        ax.tick_params(axis='x',labelsize=8,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=8,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
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
        MEAN_POINT_SIZE = 100
        mean_dict = dict(df.groupby(by)[y].mean())
        std_dict = dict(df.groupby(by)[y].std())
        if orient == 'h':
            ax.scatter(x=[value for value in mean_dict.values()],y=[cat for cat in mean_dict.keys()],color='red',marker='x',s=MEAN_POINT_SIZE)
            if std_lines in [True,'true','True']:
                MARKER = '|'
                ax.scatter(x=[mean + std for std,mean in zip(std_dict.values(),mean_dict.values())],y=[cat for cat in std_dict.keys()],color='purple',marker=MARKER,s=MEAN_POINT_SIZE)
                ax.scatter(x=[mean - std for std,mean in zip(std_dict.values(),mean_dict.values())],y=[cat for cat in std_dict.keys()],color='purple',marker=MARKER,s=MEAN_POINT_SIZE)
        elif orient == 'v':
            ax.scatter(y=[value for value in mean_dict.values()],x=[cat for cat in mean_dict.keys()],color='red',marker='x',s=MEAN_POINT_SIZE)
            if std_lines in [True,'true','True']:
                MARKER = '_'
                ax.scatter(y=[mean + std for std,mean in zip(std_dict.values(),mean_dict.values())],x=[cat for cat in std_dict.keys()],color='purple',marker=MARKER,s=MEAN_POINT_SIZE)
                ax.scatter(y=[mean - std for std,mean in zip(std_dict.values(),mean_dict.values())],x=[cat for cat in std_dict.keys()],color='purple',marker=MARKER,s=MEAN_POINT_SIZE)
    def set_mean_distances(ax=ax,df:pd.DataFrame=df,y:str=y,by:str=by):
        overall_mean = df[y].mean()
        for cat in df[by].unique():
            print(f"draw: [{cat},{cat}],[{overall_mean},{df.loc[df[by]==cat,y].mean()}]") # monitor
            ax.plot([cat,cat],[overall_mean,df.loc[df[by]==cat,y].mean()],color='red')

    if by in [None,'none','None']:    
        if orient == 'h':
            sns.boxplot(
                x=df[y],
                linewidth=1,
                boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 1},
                showfliers=False,
                ax=ax
                )
        else:    
            sns.boxplot(
                y=df[y],
                boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 1},
                showfliers=False,
                linewidth=1,
                ax=ax
                )    
    else:       
        if orient == 'h': 
            sns.boxplot(
                y=df[by],x=df[y],
                linewidth=1,
                boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 1},
                showfliers=False,
                ax=ax
                    )
        else:
             sns.boxplot(
                x=df[by],y=df[y],
                linewidth=1,
                boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 1},
                showfliers=False,
                ax=ax
                    )           
    
    if overall_mean in [True,'true','True'] :
        set_overall_mean(ax=ax,df=df,y=y,std_lines=std_lines)

    if category_mean in [True,'true','True'] and by not in [None,'none','None']:
        set_category_mean(ax=ax,df=df,y=y,by=by,std_lines=std_lines)
    
    if category_mean in [True,'true','True'] and overall_mean in [True,'true','True']:
        set_mean_distances(ax=ax,df=df,y=y,by=by)
    
    set_axis_style(ax,y,by)
def set_dist_plot(ax,df:pd.DataFrame,y:str=None,by:str=None,stat:str='count'):
    def set_axis_style(ax,y:str,x:str,stat=stat):
        ax.set_xlabel(x, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(stat, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=8,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=8,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
    def set_vlines(ax,stats:{},keys:{}):
        for key,color in keys.items():
            #print(f"{key}:{stats[key]}(color={color})")
            LABEL,VALUE,COLOR = key,stats[key],color
            ax.axvline(VALUE, color=COLOR, linestyle='-',linewidth=2) #label=f'{LABEL} = {VALUE:.2f}'
            ax.text(VALUE, 0, LABEL, horizontalalignment="center", verticalalignment="top", transform=ax.get_xaxis_transform(), rotation=45,color=COLOR)  

    #BINS_AMOUNT = int(len(df)**0.5) if by in [None,'none','None'] else [int(len(df[df[by]==cat])**0.5) for cat in df[by].unique()]
    
    category_column = 1 if by in [None,'none','None'] else by
    COLOR_PALLETTE = {cat:CONFIG['Chart']['data_colors'][i % len(CONFIG['Chart']['data_colors'])] for i,cat in enumerate(df[category_column].unique())}
    
    if by in [None,'none','None']:
        sns.histplot(
            data=df,
            x=None if y in [None,'none','None'] else y,
            multiple='layer',stat=stat,
            element='step',
            kde=True,
            legend=True,
            color=get_darker_color(CONFIG['Chart']['data_colors'][0],10),
            ax=ax
        ) 
    else:
        for cat,color in COLOR_PALLETTE.items():
            sns.histplot(
                data=df[df[by]==cat],
                x=None if y in [None,'none','None'] else y,
                multiple='layer',stat=stat,
                element='step',
                kde=True,
                legend=True,
                #palette=COLOR_PALLETTE,
                color=get_darker_color(color,10),
                ax=ax
        ) 

    set_vlines(
        ax=ax,
        stats=get_stats(df[y]),
        keys={'mean':get_darker_color(CONFIG['Chart']['data_colors'][0],60),'median':get_darker_color(CONFIG['Chart']['data_colors'][0],30)}
        )   
    set_axis_style(ax=ax,y=y,x=by,stat=stat)
def set_count_plot(ax,df:pd.DataFrame,y:str=None,by:str=None):
    def set_axis_style(ax,y:str,x:str):
        ax.set_xlabel(ax.get_xlabel() , fontsize=11, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(ax.get_ylabel() , fontsize=11, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=9,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=9,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    def set_bars_values(ax=ax):
        def is_horizontal(ax):
            """Returns True if the countplot is horizontal, else False."""
            return ax.get_xlim()[1] > ax.get_ylim()[1]  
        
        if is_horizontal(ax) == True:
            for p in ax.patches:
                ax.annotate(f'{int(p.get_width())}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            for p in ax.patches:
                if ax.get_yscale() == 'linear':  # Check if y-axis is linear
                    ax.annotate(f'{int(p.get_height())}',(p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9, fontweight='bold')

    #print(f'set_count_plot({y},{by})') # monitor

    category_column = y if by in [None,'none','None'] else by
    COLOR_PALLETTE = {cat:CONFIG['Chart']['data_colors'][i % len(CONFIG['Chart']['data_colors'])] for i,cat in enumerate(df[category_column].unique())}
    sns.countplot(
        ax=ax,data=df,
        y=y,
        hue=None if by in [None,'none','None'] else by,
        #dodge=True,
        palette=COLOR_PALLETTE,
        edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][0],40)
        )

    set_axis_style(ax=ax,y=y,x=by)
    set_bars_values(ax=ax)
def set_scatter_plot(ax,df:pd.DataFrame,y:str=None,x:str=None,by:str=None):
    def set_axis_style(ax,y:str,x:str):
        ax.set_xlabel(x, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(y, fontsize=11, fontfamily='Ubuntu', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=9,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=9,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers
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

    set_axis_style(ax=ax,y=y,x=x)

# analysis 
def get_outliers_analysis(df:pd.DataFrame,y:str=None,by:str=None,contamination=0.03):
    def set_log(y,by,contamination):
        return f'''
        Outliers Detection:
        df = '{DATA_TABLE["file_name"]}'
        y = '{y}'
        Contamination = {contamination}
        '''

    fig, ax = plt.subplots(3,1,figsize=(10,10),dpi=85)
    log = set_log(y,by,contamination)

    if y not in [None,'none','None']:
        all_data, inliers, outliers = set_data(df=df,y=y,by=by,contamination=contamination)
        STATS = get_stats(df[y])
        table = pd.DataFrame(
            { # summary table
                'category':['all'],
                'count':[f"{STATS['count']:.2f}"],
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

        set_box_plot(ax=ax[0],df=all_data,y=y,by=by,orient='h')
        set_strip_plot(ax=ax[0],df=inliers,y=y,by=by,orient='h') 
        set_strip_plot(ax=ax[0],df=outliers,y=y,by=by,orient='h',color='red') 
        set_dist_plot(ax=ax[1],df=all_data,y=y,by=by)
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