
import warnings, gc
import numpy as np 
import pandas as pd
import matplotlib.colors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error,mean_absolute_error
from lightgbm import LGBMRegressor
from decimal import ROUND_HALF_UP, Decimal
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff

import sys
import pdb




def get_data():
    train = pd.read_csv("~/code/private/kaggle/jpx_stockmkt/data/stock_prices.csv", parse_dates=['Date'])
    stock_list = pd.read_csv("~/code/private/kaggle/jpx_stockmkt/data/stock_list.csv")

    pdb.set_trace()
    print("The training data begins on {} and ends on {}.\n".format(train.Date.min(),train.Date.max()))
#    display(train.describe().style.format('{:,.2f}'))



def main():
    get_data()

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
