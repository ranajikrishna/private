
import sys
import pdb
import pandas as pd
import numpy as np

import plot as plt
import interp_seasonal as intrp_sea
import data_aggregation as da
import decomposition as de
import forecast_fft as fcst_fft
import performance_simulation as per_sim 
import roc_train_test as roc_tr
import correlation_analysis as ca 
import other_tech as ot

def missing_data(all_data):

    data_col = ['air_temp','average_wave_period','dominant_wave_period','wave_height']
    for key, data in all_data.items():
        data['no_data'] = data[data_col].sum(1)==0
        print('Percentage of rows with no data at all = ', data.no_data.sum()/data.shape[0])

    pdb.set_trace()
    return


def get_data():

    buoy = pd.read_csv('data/buoy-data.csv').set_index('date')
    wide = pd.read_csv('data/wide.csv').set_index('date')

    # Set `date` to `datetime`.
    buoy.index = pd.to_datetime(buoy.index)
    wide.index = pd.to_datetime(wide.index)
    return buoy, wide

def main():

    buoy, wide = get_data()

    station_data = {}
    station_data['21418t'] = buoy.loc[buoy['station_id']=='21418t']
    station_data['46402t'] = buoy.loc[buoy['station_id']=='46402t']
    station_data['51000h'] = buoy.loc[buoy['station_id']=='51000h']
    station_data['51101h'] = buoy.loc[buoy['station_id']=='51101h']
    station_data['51201h'] = buoy.loc[buoy['station_id']=='51201h']

    # === Examine correlations ===
#    ot.examine_correl(wide)


    # === Plots ===
#    plt.plot(wide, 'wide', col)
#    plt.plot(station_data)

    # === Analyse missing data ===
#    missing_data(station_data)

    # === Interpolate ===
    print(wide.columns)
    fcst_data = intrp_sea.interpolate_seasonal(wide,['wave_height_51201h'])
    
    # === Data aggregation ===
    data_agg = da.data_agg(fcst_data,['wave_height_51201h'])

    # === Trend and Season decomposition === 
    res = de.data_decompose(data_agg)#,model='multiplicative')
    
    # === Compute FFT ===
#    data_trd = pd.DataFrame(res.trend).rename(columns={'trend':'wave_height_51201h'})
#    data_sea = pd.DataFrame(res.seasonal).rename(columns={'season':'wave_height_51201h'})
#    data_red = pd.DataFrame(res.resid).rename(columns={'resid':'wave_height_51201h'})
#    fcst_data = fcst_fft.compute_fft(data_red)

    # === Forecast ===
    k, up = 10, 0.1
    roc_tr.roc_sim([data_agg,res],k,up,component=True,plot=False)

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
