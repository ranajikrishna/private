
import sys
import pandas as pd
import numpy as np
import scipy.stats

import pdb

def cohorted_analysis(form6, form5):

    form6['checkout_status'] = 0
    form5['checkout_status'] = 0
    
    form6.loc[form6.checkout_id.notnull(),'checkout_status'] =1
    form5.loc[form5.checkout_id.notnull(),'checkout_status'] =1

    form6_date = form6.groupby(['prequal_date']).sum()
    form5_date = form5.groupby(['prequal_date']).sum()

    form6_total= form6.groupby('prequal_date').count()['prequal_id']
    form5_total = form5.groupby('prequal_date').count()['prequal_id']
    
    # Get total customers by date
    form6_date = form6_date.merge(form6_total,left_index=True, right_index=True, how='left')
    form5_date = form5_date.merge(form5_total,left_index=True, right_index=True, how='left')
    form6_date.rename(columns={"prequal_id": "total"}, inplace=True)
    form5_date.rename(columns={"prequal_id": "total"}, inplace=True)
    
    # `Form6` datafarme with daily rates.
    form6_date['comp_rate'] = form6_date.completed_prequal/form6_date.total
    form6_date['app_rate'] = form6_date.approved/form6_date.completed_prequal
    form6_date['chk_rate'] = form6_date.checkout_status/form6_date.approved
    form6_date['ovr_rate'] = form6_date.checkout_status/form6_date.total

    # `Form5` datafarme with daily rates
    form5_date['comp_rate'] = form5_date.completed_prequal/form5_date.total
    form5_date['app_rate'] = form5_date.approved/form5_date.completed_prequal
    form5_date['chk_rate'] = form5_date.checkout_status/form5_date.approved
    form5_date['ovr_rate'] = form5_date.checkout_status/form5_date.total

    # Both `Form6` and `Form5` dataframe with daily rates.
    universe_date = form6_date.merge(form5_date, left_index=True, right_index=True, how='left', suffixes=('_form6','_form5'))

    # AB-Test dataframe
    ab = universe_date[-15:]

    # ==== Analysis of Completion rates of `form6` and `form5` ===
    comp_rate_form5 = ab.completed_prequal_form5.sum()/ab.total_form5.sum() 
    comp_rate_form6 = ab.completed_prequal_form6.sum()/ab.total_form6.sum()
    diff_comp = comp_rate_form5 - comp_rate_form6
    # Compute z-score
    z_comp = two_prop_test(comp_rate_form5,ab.total_form5.sum(),comp_rate_form6,ab.total_form6.sum())
    # Compute p-value.
    p_comp = scipy.stats.norm.sf(abs(z_comp))

    # ==== Analysis of Approval rates of `form6` and `form5` ====
    app_rate_form5 = ab.approved_form5.sum()/ab.completed_prequal_form5.sum()  
    app_rate_form6 = ab.approved_form6.sum()/ab.completed_prequal_form6.sum()
    diff_app = app_rate_form6 - app_rate_form5
    # Compute z-score
    z_app = two_prop_test(app_rate_form5,ab.completed_prequal_form5.sum(),app_rate_form6,ab.completed_prequal_form6.sum())
    p_app = scipy.stats.norm.sf(abs(z_app))

    # ==== Analysis of Checkout rates of `form6` and `form5` ===
    chk_rate_form5 = ab.checkout_status_form5.sum()/ab.approved_form5.sum()  
    chk_rate_form6 = ab.checkout_status_form6.sum()/ab.approved_form6.sum()
    diff_chk = chk_rate_form6 - chk_rate_form5
    # Compute z-score
    z_chk = two_prop_test(chk_rate_form5,ab.approved_form5.sum(),chk_rate_form6,ab.approved_form6.sum())
    # Compute p-value.
    p_chk = scipy.stats.norm.sf(abs(z_chk))

    # ==== Analysis of Overall Checkout rates of `form6` and `form5`==== 
    overall_chkout_rate_form5 = ab.checkout_status_form5.sum()/ab.total_form5.sum() 
    overall_chkout_rate_form6 = ab.checkout_status_form6.sum()/ab.total_form6.sum()
    diff_rate = overall_chkout_rate_form5 - overall_chkout_rate_form6
    # Compute z-score
    z_rate = two_prop_test(overall_chkout_rate_form5,ab.total_form5.sum(),overall_chkout_rate_form6,ab.total_form6.sum())
    # Compute p-value.
    p_rate = scipy.stats.norm.sf(abs(z_rate))

    # ==== Analysis of Checkout/Completed rates of `form6` and `form5` ====
    chk_comp_form5 = ab.checkout_status_form5.sum()/ab.completed_prequal_form5.sum()  
    chk_comp_form6 = ab.checkout_status_form6.sum()/ab.completed_prequal_form6.sum()
    diff_chk_comp = chk_comp_form6 - chk_comp_form5
    # Compute z-score
    z_chk_comp = two_prop_test(chk_comp_form5,ab.completed_prequal_form5.sum(),chk_comp_form6,ab.completed_prequal_form6.sum())
    p_chk_comp = scipy.stats.norm.sf(abs(z_chk_comp))
    pdb.set_trace()
    return 


def two_prop_test(p1,n1,p2,n2):
    Y1 = p1 * n1
    Y2 = p2 * n2
    p = (Y1 + Y2)/(n1 + n2)
    z = (p1 - p2)/np.sqrt(p * (1-p) * (1/n1 + 1/n2))
    return z

def process_data(prequal,intell):
    universe = prequal.merge(intell,on='prequal_id',how='left')
    data_intel = universe[pd.notna(universe.assignment_date)]
    data_preq = universe[pd.isna(universe.assignment_date)]
    return universe, data_preq, data_intel

def get_data():
    prequal = pd.read_csv('prequal.csv')
    intell = pd.read_csv('intellicron_prequals.csv')
    return prequal,intell 


def main():
        
    prequal,intell = get_data()  
    universe,preq,intell = process_data(prequal,intell)

    cohorted_analysis(preq,intell)

    # NOTE: No. rows in `prequal` == `universe`
    # No. rows in `intell` == `data_preq`
    # Hence no data leakage
    print (len(np.unique(universe.prequal_id)))

    return 

if __name__ == '__main__':
    status = main()
    sys.exit()
