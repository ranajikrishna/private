
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

def arima_model(data):
    model = SARIMAX(data['car_count'],\
            exog=data[['cld_ind']],\
            order=(1,1,1),\
            seasonal_order = (0,0,0,0))
    result = model.fit()
    print(result.summary())
    result.plot_diagnostics()
    plt.show()
    
    start_date = data.index[1500]   # Start date of forecasting.
    #start_date = data.index[data.day_of_week==day][200]

    # Out-of-sample cross validation using one-step-ahead technique. This is
    # achieved by setting `dyamic=False`
    pred = result.get_prediction(start=pd.to_datetime(start_date), \
            dynamic=False)
    y_forecasted = pred.predicted_mean
    y_truth = data[start_date:].car_count
    data = data.join(y_forecasted).dropna()
    data['sqr_error'] = (data.predicted_mean - data.car_count)**2 
    data['abs_error'] = abs(data.predicted_mean - data.car_count)
    print('RMSE: %f', np.sqrt(data['sqr_error'].mean()))
    print('MAE: %f', np.mean(data['abs_error']))
    print(data.groupby('cloud_indicator').mean()['abs_error'])
    print(np.sqrt(data.groupby('cloud_indicator').mean()['sqr_error']))

    # ----- Plot forecasted values and conf. interval ----
    pred_ci = pred.conf_int(alpha = 0.05)
    ax = data.car_count.plot(label='Car counts')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', \
            alpha=.7, figsize=(14, 4))
    ax.fill_between(pred_ci.index,pred_ci.iloc[:,0],pred_ci.iloc[:,1],color='k',\
            alpha=0.2)
    ax.set(xlabel='date',ylabel='car counts', \
            title='Plot of Car count data and its forecasted values')
    ax.legend()

    return 

def sarimax_selection(data):
    data = data.groupby(pd.Grouper(freq='1D')).sum()
    model = auto_arima(data.car_count, \
            X=data[["cld_ind"]], \
            trace=True)

    print(model.summary())
    return 

def ttest_ind(data):
    res = stats.ttest_ind(data[data.cloud_indicator=='clear']['car_count'],\
            data[data.cloud_indicator=='cloudy']['car_count'])
    print(res)
    return

def data_preprocessing(data):

    # Rename `car.count` to `car_count`
    data.rename(columns={"car.count": "car_count",\
            "day.of.week":"day_of_week",\
            "cloud.indicator":"cloud_indicator"},inplace=True)
    # Set `date` to type `datetime` 
    data.date = pd.to_datetime(data.date)
    # Set `date` as index
    data.set_index('date', inplace=True)
    # Create a dummy variable called `cld_ind`
    data['cld_ind'] = np.zeros(data.shape[0])
    data.loc[data.cloud_indicator=='clear','cld_ind'] = 1
    return data

def plot_data(data, plot_type):
    '''
    Functions to plot the data.

    time_plot: plots data aginst the date
    box_plot: generates box plots of variables
    dist_plot: plots distributions
    dayofweek_plot: plots data across days of weeks in subplots
    monthofyear_plot: plots data acress montho of year on one plot
    diff_plot: plots ACF and PACF at different differences
    timescale_plot: plots ACF and PACF of different timescales
    trend_plt: plots trends using rolling means
    '''


    if plot_type == 'time_plot':
        # Create figure and plot space
        fig, ax = plt.subplots(figsize=(10, 10))

        # Add x-axis and y-axis
        ax.plot(data.index, data.car_count)
        #ax.scatter(data.weather, data.car_count)
        #ax.plot(data.loc[data.cloud_indicator=='clear'].date, \
        #            data.loc[data.cloud_indicator=='clear'].car_count)
        #ax.grid()

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
#        ax.set(xlabel="Car count",\
#                title="Box-plot of car count")
#        ax = sns.boxplot(x=data.day_of_week, y=data.car_count)
#        ax = sns.boxplot(x=data[data.cloud_indicator=='clear'].day_of_week,\
#                                y=data[data.cloud_indicator=='clear'].car_count)
#        ax = sns.boxplot(x=data.cloud_indicator, y=data.car_count)
#        ax = sns.boxplot(x="day_of_week", y="weather", hue="cloud_indicator", \
#                       data=data)

         # Set title and labels for axes
        ax.set(xlabel="Days of the week", ylabel="Car Counts", \
        title="Box-plot of weather readings on cloudy and clear days of the week")

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

    if plot_type == 'dayofweek_plot':
        data_type = data#[data.cloud_indicator == 'clear']
        fig, ax = plt.subplots(7, 1, sharex=True)
        fig.suptitle('Plot of number of cars on days of week')
        for i, day in enumerate(data_type.day_of_week.unique()):
            data_day = data_type[data_type.day_of_week ==  day]
            ax[i].plot(data_day.date,data_day.car_count)
            ax[i].plot(data_day.date,data_day.car_count.rolling(30).mean(), \
                    label=str(30)+'-day rolling mean')
            ax[i].set_title(day)
            ax[i].grid()
            ax[i].legend()

    if plot_type == 'monthofyear_plot':
        fig, ax = plt.subplots()
        mth_data = data.groupby(pd.Grouper(freq='1M')).sum()
        for mth in data.index.month_name().unique():
            ax.plot(mth_data[mth_data.index.month_name()==mth].car_count,\
                                                        label=str(mth)[0:3])
        ax.set(xlabel='date',ylabel='monthly car counts',\
                                    title='Plot of cars on months of year')
        ax.grid()
        ax.legend()


    if plot_type == 'diff_plot':
        L = 100
        data_type = data#[(data.cloud_indicator == cloud_type)]# & (data.day_of_week == 'Monday')]

        # ----- ACF Plots ----
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

        # ----- PACF Plots ----
        fig, axes = plt.subplots(3, 2)
        # Original Series
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


    if plot_type == 'trend_plot':
        series = data[data.cloud_indicator=='clear'].groupby(pd.Grouper(freq='1D')).sum()
        series = series[series.car_count!=0]
        fig, ax = plt.subplots()
        series.car_count.plot(label='Daily data')
        for m in [7,14,30,60,90,180]:
            series.car_count.rolling(m).mean().plot(label=str(m)+'-day rolling mean')
            ax.set(xlabel='Date',ylabel='Daily car counts',\
                    title='Plot of rolling means on daily car counts on clear days')
            ax.legend()
            ax.grid()

    plt.show()
    return 

def get_data():
    data = pd.read_csv('/Users/vashishtha/myGitCode/private/orbital_insights/data/data.csv')
    return data


def main():
    data = get_data()
    data = data_preprocessing(data)
    plot_type = 'trend_plot' # time_plot, box_plot, dist_plot, trend_plot
    plot_data(data, plot_type) 
    sarimax_selection(data)
    arima_model(data)
    return


if __name__ == '__main__':
    status = main()
