
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
import pickle as pk
import pdb


def lgbm_algo(price_features):
    ts_fold = TimeSeriesSplit(n_splits=10, gap=10000)
    pdb.set_trace()
    prices = price_features.dropna().sort_values(['Date','SecuritiesCode'])
    y = prices['Target'].to_numpy()
    X = prices.drop(['Target'],axis=1)

    feat_importance=pd.DataFrame()
    sharpe_ratio=[]
        
    for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
        
        print("\n========================== Fold {} ==========================".format(fold+1))
        X_train, y_train = X.iloc[train_idx,:], y[train_idx]
        X_valid, y_val = X.iloc[val_idx,:], y[val_idx]
        
        print("Train Date range: {} to {}".format(X_train.Date.min(),X_train.Date.max()))
        print("Valid Date range: {} to {}".format(X_valid.Date.min(),X_valid.Date.max()))
        
        X_train.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
        X_val=X_valid[X_valid.columns[~X_valid.columns.isin(['Date','SecuritiesCode'])]]
        val_dates=X_valid.Date.unique()[1:-1]
        print("\nTrain Shape: {} {}, Valid Shape: {} {}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
        
        params = {'n_estimators': 1500,
                   'num_leaves' : 100,
                  'learning_rate': 0.1,
                  'colsample_bytree': 0.9,
                  'subsample': 0.8,
                  'reg_alpha': 0.4,
                  'metric': 'mae',
                  'random_state': 21}
        
        gbm = LGBMRegressor(**params).fit(X_train, y_train, 
                                          eval_set=[(X_train, y_train), (X_val, y_val)],
                                          verbose=300, 
                                          eval_metric=['mae','mse'])
        y_pred = gbm.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        feat_importance["Importance_Fold"+str(fold)]=gbm.feature_importances_
        feat_importance.set_index(X_train.columns, inplace=True)
        
        rank=[]
        X_val_df=X_valid[X_valid.Date.isin(val_dates)]
        for i in X_val_df.Date.unique():
            temp_df = X_val_df[X_val_df.Date == i].drop(['Date','SecuritiesCode'],axis=1)
            temp_df["pred"] = gbm.predict(temp_df)
            temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False)-1).astype(int)
            rank.append(temp_df["Rank"].values)

        stock_rank=pd.Series([x for y in rank for x in y], name="Rank")
        df=pd.concat([X_val_df.reset_index(drop=True),stock_rank,
                      prices[prices.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
        sharpe=calc_spread_return_sharpe(df)
        sharpe_ratio.append(sharpe)
        print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe,rmse,mae))
        
        del X_train, y_train,  X_val, y_val
        gc.collect()
        
        print("\nAverage cross-validation Sharpe Ratio: {:.4f}, \
                standard deviation = {:.2f}.".format(np.mean(sharpe_ratio),np.std(sharpe_ratio)))

        return

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()

    return sharpe_ratio


def plot_features(price_names,stock_list,temp):
    features = ['MovingAvg','ExpMovingAvg','Return', 'Volatility']
    names=['Average', 'Exp. Moving Average', 'Period', 'Volatility']
    buttons=[]

    fig = make_subplots(rows=2, cols=2, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=('Adjusted Close Moving Average',
                                        'Exponential Moving Average',
                                        'Stock Return', 'Stock Volatility'))

    for i, sector in enumerate(price_names.SectorName.unique()):
        
        sector_df=price_names[price_names.SectorName==sector]
        periods=[0,10,30,50]
        colors=px.colors.qualitative.Vivid
        dash=['solid','dash', 'longdash', 'dashdot', 'longdashdot']
        row,col=1,1
        
        for j, (feature, name) in enumerate(zip(features, names)):
            if j>=2:
                row,periods=2,[10,30,50]
                colors=px.colors.qualitative.Bold[1:]
            if j%2==0:
                col=1
            else:
                col=2
            
            for k, period in enumerate(periods):
                if (k==0)&(j<2):
                    plot_data=sector_df.groupby(sector_df.index)['AdjustedClose'].mean().rename('Adjusted Close')
                elif j>=2:
                    plot_data=sector_df.groupby(sector_df.index)['{}_{}Day'.format(feature,period)].mean().mul(100).rename('{}-day {}'.format(period,name))
                else:
                    plot_data=sector_df.groupby(sector_df.index)['{}_{}Day'.format(feature,period)].mean().rename('{}-day {}'.format(period,name))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data, mode='lines',
                                         name=plot_data.name, marker_color=colors[k+1],
                                         line=dict(width=2,dash=(dash[k] if j<2 else 'solid')), 
                                         showlegend=(True if (j==0) or (j==2) else False), legendgroup=row,
                                         visible=(False if i != 0 else True)), row=row, col=col)
                
        visibility=[False]*14*len(price_names.SectorName.unique())
        for l in range(i*14, i*14+14):
            visibility[l]=True
        button = dict(label = sector,
                      method = "update",
                      args=[{"visible": visibility}])
        buttons.append(button)

    fig.update_layout(title='Stock Price Moving Average, Return,<br>and Volatility by Sector',
                      template=temp, yaxis3_ticksuffix='%', yaxis4_ticksuffix='%',
                      legend_title_text='Period', legend_tracegroupgap=250,
                      updatemenus=[dict(active=0, type="dropdown",
                                        buttons=buttons, xanchor='left',
                                        yanchor='bottom', y=1.105, x=.01)], 
                      hovermode='x unified', height=800,width=1200, margin=dict(t=150))

    fig.show()

    return

def create_features(df):
    df=df.copy()
    col='AdjustedClose'
    periods=[5,10,20,30,50]
    for period in periods:
        df.loc[:,"Return_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].pct_change(period)
        df.loc[:,"MovingAvg_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).mean().values
        df.loc[:,"ExpMovingAvg_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].ewm(span=period,adjust=False).mean().values
        df.loc[:,"Volatility_{}Day".format(period)] = np.log(df[col]).groupby(df["SecuritiesCode"]).diff().rolling(period).std()
    return df

def adjusted_price(price):
    '''
    Generate the adjusted close price using the function provided by the 
    competition hosts in the Train Demo Notebook, which will adjust the close 
    price to account for any stock splits or reverse splits.
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    ''' 

    # Transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
           Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df
    
    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)

    return price

def correlation_matrix(train_df,temp):
    df_pivot=train_df.pivot_table(index='Date', columns='SectorName', values='Close').reset_index()
    corr=df_pivot.corr().round(2)
    mask=np.triu(np.ones_like(corr, dtype=bool))
    c_mask = np.where(~mask, corr, 100)
    c=[]
    for i in c_mask.tolist()[1:]:
        c.append([x for x in i if x != 100])
        
    cor=c[::-1]
    x=corr.index.tolist()[:-1]
    y=corr.columns.tolist()[1:][::-1]
    fig=ff.create_annotated_heatmap(z=cor, x=x, y=y, 
                                    hovertemplate='Correlation between %{x} and %{y} stocks = %{z}',
                                    colorscale='viridis', name='')
    fig.update_layout(template=temp, title='Stock Correlation between Sectors',
                      margin=dict(l=250,t=270),height=800,width=900,
                      yaxis=dict(showgrid=False, autorange='reversed'),
                      xaxis=dict(showgrid=False))
    fig.show()
    return

def correlation_bar(train_df,temp):
    # Correlation between `Target` and `Open`
    corr=train_df.groupby('SecuritiesCode')[['Target','Open']].corr().unstack().iloc[:,1]
    stocks=corr.nlargest(10).rename("Return").reset_index()
    stocks=stocks.merge(train_df[['Name','SecuritiesCode']], on='SecuritiesCode').drop_duplicates()
    pal=sns.color_palette("magma_r", 14).as_hex()
    rgb=['rgba'+str(matplotlib.colors.to_rgba(i,0.7)) for i in pal]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=stocks.Name, y=stocks.Return, text=stocks.Return, 
                         texttemplate='%{text:.2f}', name='', width=0.8,
                         textposition='outside',marker=dict(color=rgb, line=dict(color=pal,width=1)),
                         hovertemplate='Correlation of %{x} with target = %{y:.3f}'))
    fig.update_layout(template=temp, title='Most Correlated Stocks with Target Variable',
                      yaxis=dict(title='Correlation',showticklabels=False), 
                      xaxis=dict(title='Stock',tickangle=45), margin=dict(b=100),
                      width=800,height=500)
    fig.show()
    return 



def scatterPlot_matrix(train_df,temp):
    stocks=train_df[train_df.SecuritiesCode.isin([4169,7089,4582,2158,7036])]
    df_pivot=stocks.pivot_table(index='Date', columns='Name', values='Close').reset_index()
    pal=['rgb'+str(i) for i in sns.color_palette("coolwarm", len(df_pivot))]

    # Scatter plots on off-diagonal and histogram on the diagonal.
    fig = ff.create_scatterplotmatrix(df_pivot.iloc[:,1:], diag='histogram', name='')
    fig.update_traces(marker=dict(color=pal, opacity=0.9, line_color='white', line_width=.5))
    fig.update_layout(template=temp, title='Scatterplots of Highest Performing Stocks', 
                      height=1000, width=1000, showlegend=False)
    fig.show()
    return


def high_low_sector(train_df,temp):
    stock=train_df.groupby('Name')['Target'].mean().mul(100)
    stock_low=stock.nsmallest(7)[::-1].rename("Return")
    stock_high=stock.nlargest(7).rename("Return")
    stock=pd.concat([stock_high, stock_low], axis=0).reset_index()
    stock['Sector']='All'
    for i in train_df.SectorName.unique():
        sector=train_df[train_df.SectorName==i].groupby('Name')['Target'].mean().mul(100)
        stock_low=sector.nsmallest(7)[::-1].rename("Return")
        stock_high=sector.nlargest(7).rename("Return")
        sector_stock=pd.concat([stock_high, stock_low], axis=0).reset_index()
        sector_stock['Sector']=i
        stock=stock.append(sector_stock,ignore_index=True)
        
    fig=go.Figure()
    buttons = []
    for i, sector in enumerate(stock.Sector.unique()):
        
        x=stock[stock.Sector==sector]['Name']
        y=stock[stock.Sector==sector]['Return']
        mask=y>0
        fig.add_trace(go.Bar(x=x[mask], y=y[mask], text=y[mask], 
                             texttemplate='%{text:.2f}%',
                             textposition='auto',
                             name=sector, visible=(False if i != 0 else True),
                             hovertemplate='%{x} average return: %{y:.3f}%',
                             marker=dict(color='green', opacity=0.7)))
        fig.add_trace(go.Bar(x=x[~mask], y=y[~mask], text=y[~mask], 
                             texttemplate='%{text:.2f}%',
                             textposition='auto',
                             name=sector, visible=(False if i != 0 else True),
                             hovertemplate='%{x} average return: %{y:.3f}%',
                             marker=dict(color='red', opacity=0.7)))
        
        visibility=[False]*2*len(stock.Sector.unique())
        visibility[i*2],visibility[i*2+1]=True,True
        button = dict(label = sector,
                      method = "update",
                      args=[{"visible": visibility}])
        buttons.append(button)

    fig.update_layout(title='Stocks with Highest and Lowest Returns by Sector',
                      template=temp, yaxis=dict(title='Average Return', ticksuffix='%'),
                      updatemenus=[dict(active=0, type="dropdown",
                                        buttons=buttons, xanchor='left',
                                        yanchor='bottom', y=1.01, x=.01)], 
                      margin=dict(b=150),showlegend=False,height=700, width=900)
    fig.show()
    return 

def price_sector_dropdown(train_df,temp):
    train_date = train_df.Date.unique()
    sectors = train_df.SectorName.unique().tolist()
    sectors.insert(0, 'All')
    open_avg = train_df.groupby('Date')['Open'].mean()
    high_avg = train_df.groupby('Date')['High'].mean()
    low_avg = train_df.groupby('Date')['Low'].mean()
    close_avg = train_df.groupby('Date')['Close'].mean() 
    buttons = []

    fig = go.Figure()
    for i in range(18):
        if i != 0:
            open_avg = train_df[train_df.SectorName==sectors[i]].groupby('Date')['Open'].mean()
            high_avg = train_df[train_df.SectorName==sectors[i]].groupby('Date')['High'].mean()
            low_avg = train_df[train_df.SectorName==sectors[i]].groupby('Date')['Low'].mean()
            close_avg = train_df[train_df.SectorName==sectors[i]].groupby('Date')['Close'].mean()        
        
        fig.add_trace(go.Candlestick(x=train_date, open=open_avg, high=high_avg,
                                     low=low_avg, close=close_avg, name=sectors[i],
                                     visible=(True if i==0 else False)))
        
        visibility = [False]*len(sectors)
        visibility[i] = True
        # ---- Add buttons for 3month, 6month and All buttons ----
#        button = dict(label = sectors[i],
#                      method = "update",
#                      args=[{"visible": visibility}])
#        buttons.append(button)
        
#    fig.update_xaxes(rangeslider_visible=True,
#                     rangeselector=dict(
#                         buttons=list([
#                             dict(count=3, label="3m", step="month", stepmode="backward"),
#                             dict(count=6, label="6m", step="month", stepmode="backward"),
#                             dict(step="all")]), xanchor='left',yanchor='bottom', y=1.16, x=.01))

    fig.update_layout(template=temp,title='Stock Price Movements by Sector', 
                      hovermode='x unified', showlegend=False, width=1000,
                      updatemenus=[dict(active=0, type="dropdown",
                                        #buttons=buttons, xanchor='left',
                                        yanchor='bottom', y=1.01, x=.01)],
                      yaxis=dict(title='Stock Price'))
    fig.show()
    return  

def return_dist_sector(train_df,df,temp):
    ''' 
    Called from `plot_returns`
    '''
    pal = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 18)]
    fig = go.Figure()
    for i, sector in enumerate(df.index[::-1]):
        y_data = train_df[train_df['SectorName']==sector]['Target']
        fig.add_trace(go.Box(y=y_data*100, name=sector,
                             marker_color=pal[i], showlegend=False))
    fig.update_layout(template=temp, title='Target Distribution by Sector',
                      yaxis=dict(title='Stock Return',ticksuffix='%'),
                      margin=dict(b=150), height=750, width=900)
    fig.show()
    return


def plot_distribution(train_df,colors,temp):
    fig = go.Figure()
    x_hist=train_df['Target']
    fig.add_trace(go.Histogram(x=x_hist*100,
                               marker=dict(color=colors[0], opacity=0.7, 
                                           line=dict(width=1, color=colors[0])),
                               xbins=dict(start=-40,end=40,size=1)))
    fig.update_layout(template=temp,title='Target Distribution', 
                      xaxis=dict(title='Stock Return',ticksuffix='%'), height=450)
    fig.show()
    return

def plot_returns(train_df,colors,temp):
    years = {year: pd.DataFrame() for year in train_df.Year.unique()[::-1]}
    for key in years.keys():
        df = train_df[train_df.Year == key]
        years[key] = df.groupby('SectorName')['Target'].mean().mul(100).rename("Avg_return_{}".format(key))
    df = pd.concat((years[i].to_frame() for i in years.keys()), axis=1)
    df = df.sort_values(by="Avg_return_2021")

    fig = make_subplots(rows=1, cols=5, shared_yaxes=True)
    for i, col in enumerate(df.columns):
        x = df[col]
        mask = x<=0
        fig.add_trace(go.Bar(x=x[mask], y=df.index[mask],orientation='h', 
                             text=x[mask], texttemplate='%{text:.2f}%',textposition='auto',
                             hovertemplate='Average Return in %{y} Stocks = %{x:.4f}%',
                             marker=dict(color='red', opacity=0.7),name=col[-4:]), 
                      row=1, col=i+1)
        fig.add_trace(go.Bar(x=x[~mask], y=df.index[~mask],orientation='h', 
                             text=x[~mask], texttemplate='%{text:.2f}%', textposition='auto', 
                             hovertemplate='Average Return in %{y} Stocks = %{x:.4f}%',
                             marker=dict(color='green', opacity=0.7),name=col[-4:]), 
                      row=1, col=i+1)
        fig.update_xaxes(range=(x.min()-.15,x.max()+.15), title='{} Returns'.format(col[-4:]), 
                         showticklabels=False, row=1, col=i+1)
    fig.update_layout(template=temp,title='Yearly Average Stock Returns by Sector', 
                      hovermode='closest',margin=dict(l=250,r=50),
                      height=600, width=1000, showlegend=False)
    fig.show()

    # ===== PLOT DISTRIBUTION OF RETURNS BY SECTOR =====
    return_dist_sector(train_df,df,temp)

    return

def plot_trend(train,colors,temp):
    date = train.Date.unique()
    returns = train.groupby('Date')['Target'].mean().rename('Ave. Return')
    close_avg  = train.groupby('Date')['Close'].mean().rename('Ave. Close')
    volume_avg = train.groupby('Date')['Volume'].mean().rename('Ave. Volume')

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    for i,j in enumerate([returns,close_avg,volume_avg]):
        fig.add_trace(go.Scatter(x=date, y=j, mode='lines', marker_color=colors[i]), \
                                 row=i+1, col=1)

    fig.update_xaxes(rangeslider_visible=False,
                 rangeselector=dict(
                     buttons=list([
                         dict(count=6, label="6m", step="month",stepmode="backward"),
                         dict(count=1, label="1y", step="year", stepmode="backward"),
                         dict(count=2, label="2y", step="year", stepmode="backward"),
                         dict(step="all")])),
                 row=1,col=1)

    fig.update_layout(template=temp,title='JPX Market Average Stock Return, Closing Price, and Shares Traded', 
                  hovermode='x unified', height=700, 
                  yaxis1=dict(title='Stock Return', ticksuffix='%'), 
                  yaxis2_title='Closing Price', yaxis3_title='Shares Traded',
                  showlegend=False)

    fig.show()
    return 


def get_data():
    train = pd.read_csv("~/code/private/kaggle/jpx_stockmkt/data/stock_prices.csv", parse_dates=['Date'])
    stock_list = pd.read_csv("~/code/private/kaggle/jpx_stockmkt/data/stock_list.csv")

    print("The training data begins on {} and ends on {}.\n".format(train.Date.min(),train.Date.max()))
    #display(train.describe().style.format('{:,.2f}'))
#    print(train.describe())
    
    return train, stock_list

def main():
    train, stock_list = get_data()
    
    # === Pre-processign ===
    colors = px.colors.qualitative.Plotly
    temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), width=800))
    stock_list['SectorName'] = [i.rstrip().lower().capitalize() for i in stock_list['17SectorName']]
    stock_list['Name'] = [i.rstrip().lower().capitalize() for i in stock_list['Name']]
    train_df = train.merge(stock_list[['SecuritiesCode','Name','SectorName']], on='SecuritiesCode', how='left')
    train_df['Year'] = train_df['Date'].dt.year

    # === Plots ===
#    plot_trend(train,colors,temp)              # PLOT TREND
#    plot_returns(train_df,colors,temp)         # PLOT RETURNS AND
#    plot_distribution(train,colors,temp)       # PLOT DISTRIBUTION
#    price_sector_dropdown(train_df,temp)       # PLOT PRICE SECTOR DROPDOWN 
#    high_low_sector(train_df,temp)             # PLOT HIGH LOW BY SECTOR
#    scatterPlot_matrix(train_df, temp)         # SCATTER PLOT AND HISTOGRAM
#    correlation_bar(train_df,temp)             # CORRELATION BAR PLOT
#    correlation_matrix(train_df,temp)          # CORRELATION MATRIX

    # === Create features ===
#    train_df = adjusted_price(train_df)
#    train_df = create_features(train_df)
#    train_df.drop(['RowId','SupervisionFlag','AdjustmentFactor',\
#                   'CumulativeAdjustmentFactor','Close'],axis=1,inplace=True)
#    with open('train_df.pk','wb') as handle:
#        pk.dump(train_df,handle)

#    train_df = plot_features(train_df,stock_list,temp)
    # === LGBM Algo ===
    with open('train_df.pk', 'rb') as input_file:
        train_df = pk.load(input_file) 
    lgbm_algo(train_df)
    train = train.drop('ExpectedDividend',axis=1).fillna(0)
    train = train[train.Date>'2020-12-23']
    print("New Train Shape {}.\nMissing values in Target = {}".format(train.shape,train['Target'].isna().sum()))

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
