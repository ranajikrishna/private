
import get_data as gd 
import handle_missing_data as hmd
import prep_model_data as pmd
import gradient_boosted_regression as gbr
import gradient_boosted_classification as gbc
import neural_net as nn
import quantile_regression as qr
import gradient_boosted_hyperopt as gbh

import sys
import pandas as pd
import numpy as np

def main():
    #  === Process and Examine data === 
#    gd.get_data()

#    universe = pd.read_csv('universe_all.csv')
#    universe.drop('Unnamed: 0',axis=1,inplace=True)
    universe = pd.read_csv('universe_all_time.csv')
#    universe.drop('Unnamed: 0.1',axis=1,inplace=True)
    universe.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1,inplace=True)
    # ======

    # Examine columns with empty string. 
    universe_clean = hmd.handle_missing_data(universe)

    # Prepare data for model.
    data = pmd.prep_model_data(universe_clean)

    # === Gradient Boosted CLASSIFICATION ===
    gbc.boosted_clf(data, False)
    
    # === Gradient Boosted REGRESSION ===
#    gbr.boosted_reg(data)
#    gbr.boosted_reg_cv(data)
    # ======

    # === Neural Network ===
#    nn.neural_net(data)
#    data.set_index('loan_seq',inplace=True,drop=True)

    # === Quantile Regression Tree ===
#    qr.rfq_reg(data)
#    qr.rfq_reg_cv(data)
    # ======

    # === Gradient Booted Decision Tree Hyperopt ===
#    hyp = gbh.hyperOpt(data)
#    hyp.hyper_opt_gbm()
#    hyp.hyper_opt_cat()
#    plot(data)
    # ======

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()


