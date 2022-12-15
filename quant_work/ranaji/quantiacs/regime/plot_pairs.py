
import sys
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def plot_pairs(pairs, data):
    plt.style.use('seaborn')
    pdf = PdfPages('./pair_plot_log.pdf')
    itr = 0
    for _ in range(pairs.shape[0]):
        fig, axs = plt.subplots(nrows=2,ncols=2,tight_layout=True)
        ax0, ax1, ax2, ax3 = axs.flatten()
        fig.suptitle(str(pairs.iloc[itr].asset1) + ":" + str(pairs.iloc[itr].asset2))
        ax0.set_title('Price')
        ax0.plot(data[pairs.iloc[itr].asset1])
        ax0.plot(data[pairs.iloc[itr].asset2])
        ratio = data[pairs.iloc[itr].asset1]/data[pairs.iloc[itr].asset2]
        ax1.set_title('Ratio')
        ax1.plot(ratio)
        ax2.set_title('Ratio 2016 onwards')
        ax2.plot(ratio['2016-01-01':])
        z_score = (ratio['2016-01-01':]-np.mean(ratio['2016-01-01':]))/np.std(ratio['2016-01-01':])
        ax3.set_title('Z-Score')
        ax3.plot(z_score)
#        sm.qqplot(np.array(ratio.loc['2016-01-01':]),ax=ax2)
#        ratio.loc['2016-01-01':].hist(bins=[50])
        itr += 1
        fig.tight_layout()
        pdf.savefig(fig)
    
    pdf.close()
    return 
