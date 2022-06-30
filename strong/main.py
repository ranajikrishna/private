
import sys
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def plot(all_data, type='buoy'):

    if type=='wide':
        plt.figure(figsize=(10,6), tight_layout=True)
        plt.plot(data['date'], data['air_temp_51000h'],label='air_temp_51000h')
        pdb.set_trace()

    if type=='buoy':
        plt.style.use('seaborn')
        pdf = PdfPages('./buoy_1.pdf')
        for key,data in all_data.items():
            print(key)
            fig, axs = plt.subplots(4, sharex=True, tight_layout=True)

            for i in range(4):
                axs[i].xaxis.set_major_locator(mdates.YearLocator())
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
                axs[i].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[i].tick_params(axis="y", labelsize=5)
#                axs[i].xaxis.set_minor_formatter(mdates.DateFormatter('%B'))
                axs[i].grid(which='major',color='white',linewidth=0.5)

            axs[0].plot(data.index, data['air_temp'], label='air_temp')
            axs[0].set_ylabel('Temp.', fontsize=6)
            axs[0].set_title('Air temperature', fontsize=8)

            axs[1].plot(data.index, data['average_wave_period'], label='av_wav_prd')
            axs[1].set_ylabel('Wave prd.', fontsize=6)
            axs[1].set_title('Av. wave prd.', fontsize=8)

            axs[2].plot(data.index, data['dominant_wave_period'], label='dom_wav_prd')
            axs[2].set_ylabel('Wave prd.', fontsize=6)
            axs[2].set_title('Dom. wave prd.', fontsize=8)

            axs[3].plot(data.index, data['wave_height'], label='wav_hgt')
            axs[3].set_xlabel('Date',fontsize=7)
            axs[3].set_ylabel('Wave hgt.', fontsize=6)
            axs[3].set_title('Wave height', fontsize=8)
            axs[3].tick_params(axis="x", rotation=45,labelsize=6)
            plt.rcParams['font.size'] = '5'
            fig.suptitle('Station: ' + key, fontsize=10)
            
            pdf.savefig(fig)

        pdf.close()
        
        pdb.set_trace()


    return 


def get_data():
    buoy = pd.read_csv('data/buoy-data.csv').set_index('date')
    wide = pd.read_csv('data/wide.csv').set_index('date')

    return buoy, wide



def main():

    buoy, wide = get_data()

    station_data = {}
    station_data['21418t'] = buoy.loc[buoy['station_id']=='21418t']
    station_data['46402t'] = buoy.loc[buoy['station_id']=='46402t']
    station_data['51000h'] = buoy.loc[buoy['station_id']=='51000h']
    station_data['51101h'] = buoy.loc[buoy['station_id']=='51101h']
    station_data['51201h'] = buoy.loc[buoy['station_id']=='51201h']
    plot(station_data)

    return 



if __name__ == '__main__':
    status = main()
    sys.exit()
