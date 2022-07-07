
import sys
import pdb
import pandas as pd
import numpy as np

import plot as plt
#import interp_spline as intrp_spl
#import interp_stl as intrp_stl
#import interp_mean as intrp_mean
import interp_seasonal as intrp_sea
import forecast_fft as fcst_fft
import performance_simulation as per_sim 


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

    # === Plots ===
#    plt.plot(wide, 'wide', col)
#    plt.plot(station_data)

    # === Analyse missing data ===
#    missing_data(station_data)

    # === Interpolate ===
#    intrp_spl.interpolate_spline(wide,['air_temp_51101h'])
#    intrp_stl.interpolate_stl(wide,['air_temp_51101h'])
#    intrp_mean.interpolate_rollmean(wide,['air_temp_51101h'])
    print(wide.columns)
    fcst_data = intrp_sea.interpolate_seasonal(wide,['wave_height_51201h'])
    
    # === Forecast ===
    fcst_fft.compute_fft(fcst_data)
    k, up = 10, 0.1
    per_sim.simulate(fcst_data,k,up)
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
