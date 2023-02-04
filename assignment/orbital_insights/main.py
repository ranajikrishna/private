
import pdb
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
import scipy.stats as stats
import itertools
import collections
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def arima_model(data):
    data['mon'] = np.zeros(data.shape[0])
    data['tue'] = np.zeros(data.shape[0])
    data['wed'] = np.zeros(data.shape[0])
    data['thu'] = np.zeros(data.shape[0])
    data['fri'] = np.zeros(data.shape[0])
    data['sat'] = np.zeros(data.shape[0])
#    data['sun'] = np.zeros(data.shape[0])
    data.loc[data.day_of_week=='Monday','mon']=1
    data.loc[data.day_of_week=='Tuesday','tue']=1
    data.loc[data.day_of_week=='Wednesday','wed']=1
    data.loc[data.day_of_week=='Thursday','thu']=1
    data.loc[data.day_of_week=='Friday','fri']=1
    data.loc[data.day_of_week=='Saturday','sat']=1
#    data.loc[data.day_of_week=='Sunday','sun']=1
#    day = 'Wednesday'
    data.set_index('date',inplace=True)
    data['cld_ind'] = np.zeros(data.shape[0])
    data.loc[data.cloud_indicator=='clear','cld_ind'] = 1
#    data = data.groupby(pd.Grouper(freq='W')).sum()
#    roll_sum = data[['car_count','cld_ind']].rolling(90).sum().dropna()
#    data.rename(columns={'car_count':'daily_car','cld_ind':'daily_cld'},inplace=True)
#    data = pd.merge(data,roll_sum,left_index=True,right_index=True)
#    model = SARIMAX(data[data.day_of_week==day]['car_count'],\
#           # exog=data[data.day_of_week==day][['cld_ind']],\
#            order=(1,1,1))
    model = SARIMAX(data['car_count'],\
            exog=data[['cld_ind',\
            "fri","wed","thu","tue","mon","sat"]],\
            order=(1,1,1),\
            seasonal_order = (0,0,0,0))
    result = model.fit()
    print(result.summary())
    result.plot_diagnostics()
    plt.show()
    
    #start_date = data.index[data.day_of_week==day][200]
    start_date = data.index[1500]
    pdb.set_trace()
    pred = result.get_prediction(start=pd.to_datetime(start_date), \
            dynamic=False)
    y_forecasted = pred.predicted_mean
    #y_truth = data[data.day_of_week == day][start_date:].car_count
    y_truth = data[start_date:].car_count
    sqrt_mse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
    print(np.mean(abs(y_truth-y_forecasted)))
    pred_ci = pred.conf_int(alpha = 0.05)
    ax = data.car_count.plot(label='Car counts')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', \
            alpha=.7, figsize=(14, 4))
    ax.fill_between(pred_ci.index,pred_ci.iloc[:,0],pred_ci.iloc[:,1],color='k',\
            alpha=0.2)
    ax.set(xlabel='date',ylabel='car counts',title='Plot of Car count data and its forecasted values')
    ax.legend()

    data = data.join(y_forecasted).dropna()
    data['sqr_error'] = (data.predicted_mean - data.car_count)**2 
    data['abs_error'] = abs(data.predicted_mean - data.car_count)
    print(data.groupby('cloud_indicator').mean()['abs_error'])
    print(np.sqrt(data.groupby('cloud_indicator').mean()['sqr_error']))
    pdb.set_trace()


    return 

def sarimax_selection(data):
    data['mon'] = np.zeros(data.shape[0])
    data['tue'] = np.zeros(data.shape[0])
    data['wed'] = np.zeros(data.shape[0])
    data['thu'] = np.zeros(data.shape[0])
    data['fri'] = np.zeros(data.shape[0])
    data['sat'] = np.zeros(data.shape[0])
#    data['sun'] = np.zeros(data.shape[0])
    data.loc[data.day_of_week=='Monday','mon']=1
    data.loc[data.day_of_week=='Tuesday','tue']=1
    data.loc[data.day_of_week=='Wednesday','wed']=1
    data.loc[data.day_of_week=='Thursday','thu']=1
    data.loc[data.day_of_week=='Friday','fri']=1
    data.loc[data.day_of_week=='Saturday','sat']=1
#    data.loc[data.day_of_week=='Sunday','sun']=1
    
#    day = 'Sunday'
    data['cld_ind'] = np.zeros(data.shape[0])
    data.loc[data.cloud_indicator=='clear','cld_ind'] = 1
    data.set_index('date',inplace=True)
#    data = data.groupby(pd.Grouper(freq='1D')).sum()
#    data = data.rolling(7).sum().dropna()
#    roll_sum = data[['car_count','cld_ind']].rolling(90).sum().dropna()
#    data.rename(columns={'car_count':'daily_car','cld_ind':'daily_cld'},inplace=True)
#    data = pd.merge(data,roll_sum,left_index=True,right_index=True)
#    model = auto_arima(data[(data.day_of_week==day)].car_count, \
#            X=data[data.day_of_week == day][["cld_ind"]],\
#            trace=True)                                              
    pdb.set_trace()
    model = auto_arima(data.car_count, \
            X=data[["cld_ind", \
            "fri","wed","thu","tue","mon","sat"]],\
#            seasonal=True, m = 52, \
#            with_intercept = True, \
#            trend = 'c', \
            trace=True)

    print(model.summary())
    pdb.set_trace()
    return 

def stationarity_test(data):
    cloud_type = 'cloudy'
    prd=7
    data.set_index('date',inplace=True)
#    roll_sum = data[['car_count']].rolling(7).sum().dropna()
#    data.rename(columns={'car_count':'daily_car'},inplace=True)
#    data = pd.merge(data,roll_sum,left_index=True,right_index=True)
    data_type = data#[(data.cloud_indicator==cloud_type)]# & (data.day_of_week == 'Monday')]
    #data = data_type.groupby(pd.Grouper(freq='1W')).sum()
    result = adfuller(data.car_count.dropna(),regression='ct')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    pdb.set_trace()
    plot_data(data_type, 'trend_plot')

    return 


def anova_test(data):
    cloud_type = 'clear'
    data_type = data#[data.cloud_indicator==cloud_type]
    g = globals()
    for day in data_type.day_of_week.unique():
        g['{0}'.format(day[0:3].lower())] = \
        data_type.groupby(['day_of_week']).get_group(day)['car_count'].tolist()

#    fvalue, pvalue = stats.f_oneway(mon,tue,wed,thu,fri,sat,sun)
    fvalue, pvalue = stats.kruskal(mon,tue,wed,thu,fri,sat,sun)

    pdb.set_trace()
    plot_data(data, 'day_plot')
    pdb.set_trace()
    return 

def equal_var_test(data):
    cloud_type = 'clear'
    data_type = data[data.cloud_indicator==cloud_type]
    g = globals()
    for day in data_type.day_of_week.unique():
        g['{0}'.format(day[0:3].lower())] = \
        data_type.groupby(['day_of_week']).get_group(day)['car_count'].tolist()

    pdb.set_trace()
    stat, p = stats.bartlett(mon,tue,wed,thu,fri,sat,sun)
    return 

def normality_test(data):
    shapiro = collections.defaultdict(float)
    cloud_type = 'clear'
    for day in data.day_of_week.unique():
        data_type = data[(data.day_of_week==day)]['car_count'] # \ & \
        #                        (data.cloud_indicator==cloud_type)]['car_count']
        shapiro[day] = \
        {'stat': stats.shapiro(data_type).statistic, \
        'pvalue':stats.shapiro(data_type).pvalue}

    print(pd.DataFrame(shapiro))
    plot_data(data,'qq_plot')
    pdb.set_trace()
    return 

def ttest_ind(data):

    res = stats.ttest_ind(data[data.cloud_indicator=='clear']['car_count'],\
            data[data.cloud_indicator=='cloudy']['car_count'])
    print(res)
    return

def data_preprocessing(data,win=7,demean=False):

    # Rename `car.count` to `car_count`
    data.rename(columns={"car.count": "car_count",\
            "day.of.week":"day_of_week",\
            "cloud.indicator":"cloud_indicator"},inplace=True)
    # Set `date` to type `datetime` 
    data.date = pd.to_datetime(data.date)
    
    if demean:
        grp_mean = data.groupby('cloud_indicator').rolling(win).mean()
        roll_mean = grp_mean.loc['clear'].car_count.append(grp_mean.loc['cloudy'].car_count)
        roll_mean_shf = grp_mean.loc['clear'].car_count.shift().append(grp_mean.loc['cloudy'].car_count.shift())
#        data = pd.merge(data,roll_mean,left_index=True,right_index=True)
        data = pd.merge(data,roll_mean_shf,left_index=True,right_index=True)
        data.dropna(inplace=True)
        pdb.set_trace()
        data['car_count'] = data.car_count_x - data.car_count_y
    return data

def plot_data(data, plot_type):


    if plot_type == 'time_plot':
        # Create figure and plot space
        fig, ax = plt.subplots(figsize=(10, 10))

        # Add x-axis and y-axis
        ax.plot(data.date, data.car_count)
        #ax.scatter(data.weather, data.car_count)
        #ax.plot(data.loc[data.cloud_indicator=='clear'].date, data.loc[data.cloud_indicator=='clear'].car_count)
        ax.grid()

        # Set title and labels for axes
        ax.set(xlabel="Date", ylabel="Number of cars", \
                title="Plot of number of cars against time")

        # Rotate tick marks on x-axis
        plt.setp(ax.get_xticklabels(), rotation=45)

    if plot_type == 'box_plot':
        # Create figure and plot space
        fig, ax = plt.subplots(figsize=(10, 10))

        sns.set_theme(style="whitegrid")
        ax = sns.boxplot(x=data.car_count)
        ax.set(xlabel="Car count",\
                title="Box-plot of car count")
#        ax = sns.boxplot(x=data.day_of_week, y=data.car_count)
#        ax = sns.boxplot(x=data[data.cloud_indicator=='clear'].day_of_week, y=data[data.cloud_indicator=='clear'].car_count)
#        ax = sns.boxplot(x=data.cloud_indicator, y=data.car_count)

#        ax = sns.boxplot(x="day_of_week", y="weather", hue="cloud_indicator", \
#                       data=data)
        # Set title and labels for axes
#        ax.set(xlabel="Days of the week", ylabel="Car Counts", \
#                title="Box-plot of weather readings on cloudy and clear days of the week")



    if plot_type == 'dist_plot':

        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Distribution of car counts')
        ax[0,0].hist(data.car_count)
        ax[0,0].set(xlabel='Bins',ylabel='Frequency',title='All days')
        ax[0,0].grid()
        ax[0,1].hist(data[data.cloud_indicator=='clear'].car_count)
        ax[0,1].set(xlabel='Bins',ylabel='Frequency',title='Clear days')
        ax[0,1].grid()
        ax[1,0].hist(data[data.cloud_indicator=='cloudy'].car_count)
        ax[1,0].set(xlabel='Bins',ylabel='Frequency',title='Cloudy days')
        ax[1,0].grid()
        fig.delaxes(ax[1,1])
        pdb.set_trace()

    if plot_type == 'qq_plot':
        data_type = data[data.cloud_indicator == 'cloudy']
#        data_type = data

        # Initialise the subplot function using number of rows and columns
        fig, ax = plt.subplots(3, 3, sharex=True)
        fig.suptitle('Q-Q Plots of residuals of car counts')
        comb = list(itertools.product([0,1,2],[0,1,2]))
        for i, day in enumerate(data_type.day_of_week.unique()):
            data_day = data_type[data_type.day_of_week ==  day]["car_count"]
            sm.qqplot(data_day.sub(data_day.mean()),stats.t,fit=True,line="45",
                    ax=ax[comb[i][0],comb[i][1]])
            ax[comb[i][0]][comb[i][1]].set_title(day)
            ax[comb[i][0]][comb[i][1]].grid()
        fig.delaxes(ax[2,1])
        fig.delaxes(ax[2,2])
        pdb.set_trace()

    if plot_type == 'day_plot':
        data_type = data#[data.cloud_indicator == 'clear']
        fig, ax = plt.subplots(7, 1, sharex=True)
        fig.suptitle('Plot of number of cars on days of week')
        for i, day in enumerate(data_type.day_of_week.unique()):
            data_day = data_type[data_type.day_of_week ==  day]
            ax[i].plot(data_day.date,data_day.car_count)
            ax[i].plot(data_day.date,data_day.car_count.rolling(30).mean(),label=str(30)+'-day rolling mean')
            ax[i].set_title(day)
            ax[i].grid()
            ax[i].legend()


    if plot_type == 'month_plot':
        fig, ax = plt.subplots()
        data.set_index('date',inplace=True)
        mth_data = data.groupby(pd.Grouper(freq='1M')).sum()
        pdb.set_trace()
        for mth in data.index.month_name().unique():
            ax.plot(mth_data[mth_data.index.month_name()==mth].car_count,label=str(mth)[0:3])
        ax.set(xlabel='date',ylabel='monthly car counts',title='Plot of cars on months of year')
        ax.grid()
        ax.legend()
        pdb.set_trace()


    if plot_type == 'p_acf_plot':

        cloud_type = 'clear'
        prd = 0
        L = 100
        data_type = data#[(data.cloud_indicator == cloud_type)]# & (data.day_of_week == 'Monday')]
        # Original Series
        fig, axes = plt.subplots(3, 2)
        axes[0, 0].plot(data_type.car_count)
        axes[0,0].grid()
        axes[0, 0].set_title('Original Series')
        plot_acf(data_type.car_count, ax=axes[0, 1], lags=L)

        # 1st Differencing
        axes[1, 0].plot(data_type.car_count.diff().dropna())
        axes[1, 0].set_title('1st Order Differencing')
        plot_acf(data_type.car_count.diff().dropna(), ax=axes[1, 1], lags=L)
        axes[1,0].grid()

        # 2nd Differencing
        axes[2, 0].plot(data_type.car_count.diff().diff().dropna())
        axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(data_type.car_count.diff().diff().dropna(),ax=axes[2, 1], lags=L)
        axes[2,0].grid()

        # Original Series
        fig, axes = plt.subplots(3, 2)
        axes[0, 0].plot(data_type.car_count)
        axes[0, 0].set_title('Original Series')
        plot_pacf(data_type.car_count, ax=axes[0, 1], lags=L)
        axes[0,0].grid()

        # 1st Differencing
        axes[1, 0].plot(data_type.car_count.diff().dropna())
        axes[1, 0].set_title('1st Order Differencing')
        plot_pacf(data_type.car_count.diff().dropna(), ax=axes[1, 1], lags=L)
        axes[1,0].grid()

        # 2nd Differencing
        axes[2, 0].plot(data_type.car_count.diff().diff().dropna())
        axes[2, 0].set_title('2nd Order Differencing')
        plot_pacf(data_type.car_count.diff().diff().dropna(),ax=axes[2, 1], lags=L)
        axes[2,0].grid()

        plt.show()
        pdb.set_trace()

    if plot_type == 'timescale_plot':

        # Original Series
        fig, axes = plt.subplots(3, 2)
        data_type = data.groupby(pd.Grouper(freq='1D')).sum()
        plot_acf(data_type.car_count, ax=axes[0, 0], lags=120)
        axes[0, 0].set_title('ACF: Daily Series')
        plot_pacf(data_type.car_count, ax=axes[0, 1], lags=120)
        axes[0, 1].set_title('PACF: Daily Series')

        # 1st Differencing
        data_type = data.groupby(pd.Grouper(freq='1W')).sum()
        plot_acf(data_type.car_count, ax=axes[1, 0], lags=60)
        axes[1, 0].set_title('ACF: Weekly Series')
        plot_pacf(data_type.car_count, ax=axes[1, 1], lags=60)
        axes[1, 1].set_title('PACF: Weekly Series')

        # 2nd Differencing
        data_type = data.groupby(pd.Grouper(freq='1M')).sum()
        plot_acf(data_type.car_count, ax=axes[2, 0], lags=30)
        axes[2, 0].set_title('ACF: Monthly Series')
        plot_pacf(data_type.car_count, ax=axes[2, 1], lags=30)
        axes[2, 1].set_title('PACF: Monthly Series')
        pdb.set_trace()


    if plot_type == 'trend_plot':
        series = data.groupby(pd.Grouper(freq='1D')).sum()
        fig, ax = plt.subplots()
        series.car_count.plot(label='Daily data')
        for m in [7,14,30,60,90,180]:
            series.car_count.rolling(m).mean().plot(label=str(m)+'-day rolling mean')
            ax.set(xlabel='Date',ylabel='Daily car counts',\
                    title='Plot of rolling means on daily car counts')
            ax.legend()
            ax.grid()
        #result = STL(series.car_count).fit()
       # result.plot()
        #plot_acf(result.resid.dropna(),lags=30)
        pdb.set_trace()


    return 





def get_data():
    data = pd.read_csv('/Users/vashishtha/myGitCode/private/orbital_insights/data/data.csv')
    return data


def main():
    
    data = get_data()
    data = data_preprocessing(data,7,False)
#    plot_data(data, 'month_plot')  # time_plot, box_plot, qq_plot, dist_plot, trend_plot
#    normality_test(data)
#    equal_var_test(data)
#    anova_test(data)
#    stationarity_test(data)
#    sarimax_selection(data)
    arima_model(data)

    return


if __name__ == '__main__':
    status = main()
    sys.exit()
