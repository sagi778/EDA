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
plt.rcParams['axes.facecolor'] = CONFIG['Chart']['background']  # background
plt.rcParams['axes.edgecolor'] = get_darker_color(CONFIG['Chart']['frame_color'],10) # frame & axes
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
                'type':'int',
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
                'default':'table'
            }
            }
        }      

# charts
def get_scatter_plot(df:pd.DataFrame,y:str=None,x:str=None,by:str=None):
    def set_axis_style(ax,y:str,x:str):
        ax.set_xlabel(x, fontsize=9, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(y, fontsize=9, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers

    POINT_SIZE = 5 if len(df) > 1000 else 8 if len(df) > 200 else 9
    ALPHA = 0.2 if len(df) > 1000 else 0.4 if len(df) > 200 else 0.6
    
    fig, ax = plt.subplots(figsize=(5,5),dpi=85)
    try:
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
def get_box_plot(df:pd.DataFrame,y:str=None,by:str=None):
    def set_axis_style(ax,y:str,x:str):
        ax.set_xlabel(x, fontsize=9, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(y, fontsize=9, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers

    NUM_OF_CATEGORIES = 1 if by in [None,'none','None'] else len(df[by].unique())
    fig, ax = plt.subplots(figsize=(5,5),dpi=85)
    try:
        sns.boxplot(
            x=df[by],y=df[y],
            linewidth=2,boxprops={"facecolor": "none", "edgecolor": 'black', "linewidth": 1.5},showfliers=False,
            ax=ax
            )
        for i,cat in enumerate(df[by].unique()):
            #print(cat) # monitor
            COLOR_INDEX = i % len(CONFIG['Chart']['data_colors'])
            data = df.loc[df[by]==cat,[y,by]]
            POINT_SIZE = 5 if len(data) > 1000 else 8 if len(data) > 200 else 9
            ALPHA = 0.2 if len(data) > 1000 else 0.4 if len(data) > 200 else 0.6
            sns.stripplot(
                y=data[y],x=data[by],
                ax=ax,alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,
                color=CONFIG['Chart']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][COLOR_INDEX],70),
                jitter=0.35,zorder=0
                ) 

    except Exception as e:
        print(e)    

    set_axis_style(ax,y,by)
    

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
            }
        }
    }
def get_count_plot(df:pd.DataFrame,y:str=None,by:str=None,sub_category:str=None,stack=False):
    def set_axis_style(ax,y:str,x:str):
        LABEL = f'Count({y})' if sub_category in [None,'None','none'] else f'Count({y}) by {sub_category}'
        ax.set_xlabel(LABEL, fontsize=11, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.set_ylabel(y, fontsize=11, fontfamily='Consolas', color=CONFIG['Chart']['font_color'])
        ax.tick_params(axis='x',labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # x-axis tick numbers
        ax.tick_params(axis='y', labelsize=7,labelcolor=CONFIG['Chart']['font_color'])  # y-axis tick numbers

    NUM_OF_CATEGORIES = 1 if by in [None,'none','None'] else len(df[by].unique())
    fig, ax = plt.subplots(figsize=(5,NUM_OF_CATEGORIES),dpi=85)
    try:
        if sub_category in [None,'None','none']:
            #COLOR_PALLETTE = {cat:CONFIG['Chart']['data_colors'][i % len(CONFIG['Chart']['data_colors'])] for i,cat in enumerate(df[by].unique())}
            sns.countplot(
                ax=ax,data=df,y=y,dodge=True,
                orient='h',
                #palette=COLOR_PALLETTE,
                edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][0],40),
                color=CONFIG['Chart']['data_colors'][0]
                )
        else:
            COLOR_PALLETTE = {cat:CONFIG['Chart']['data_colors'][i % len(CONFIG['Chart']['data_colors'])] for i,cat in enumerate(df[sub_category].unique())}
            #print(COLOR_PALLETTE) # monitor
            sns.countplot(
                ax=ax,data=df,y=y,dodge=True,
                hue=sub_category,
                orient='h',
                #palette=COLOR_PALLETTE,
                edgecolor=get_darker_color(CONFIG['Chart']['data_colors'][0],40)
                )

    except Exception as e:
        print(e)    

    set_axis_style(ax,y,by)
    

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
            'sub_category':{
                'type':'category',
                'options':['None']+[f"'{item}'" for item in get_categorical_columns(df=df,max_categories=30)],
                'default':'None'
            },
            'stack':{
                'type':'category',
                'options':['False','True'],
                'default':'False'
            }
        }
    }

# analysis 
def get_dist_plot(df:pd.DataFrame,x:str=None,by:str=None,outliers="none"):
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