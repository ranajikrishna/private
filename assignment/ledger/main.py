
import sys
import pdb
import pandas as pd

def arl(qs,x):
    '''
    Compute Annual Cumulative Reinsured Loss for the scenario.
    Input: 
        qs: QS Structure
        x: scenarios
    Return: 
        Cumulative Annual Reinsured Loss. 
    '''
    cml = x[2:].cumsum() # Cummulative Annual Gross Paid Loss.
    add = 0              # Full Layer term.
    res = []
    for keys,values in qs[x['Programs']].items():
        if keys[1]!='Unlimited':
            res.append((cml[(keys[0]<cml) & (cml<keys[1])] - keys[0]) * values + add)
            add += (keys[1]- keys[0]) * values
        else:
            res.append((cml[(keys[0]<cml)] - keys[0]) * values + add)
    return pd.concat(res)

def get_data():
    qs_structure = pd.read_excel('./Ledger_exercise_20221207_v3_sent.xlsx', \
            sheet_name='QS Structure', header=1,usecols="B:N")
    
    agp_loss = pd.read_excel('./Ledger_exercise_20221207_v3_sent.xlsx', \
            sheet_name='Annual Gross Paid Loss')
    return qs_structure, agp_loss 


def main():
    qs,agp = get_data()
    qs.set_index('Programs',inplace=True)
    # Store Layer rates as a dictionary. 
    rate_dict = {}      
    for idx in qs.index:
        rates = qs.loc[idx]
        keys = [(rates['Attachment LR'],rates['Detachment LR']),\
                (rates['Attachment LR.1'],rates['Detachment LR.1']),\
                (rates['Attachment LR.2'],rates['Detachment LR.2']),\
                (rates['Attachment LR.3'],rates['Detachment LR.3'])]
        values = [rates['Reinsured %'],rates['Reinsured %.1'],\
                rates['Reinsured %.2'],rates['Reinsured %.3']]

        # Dictionary of dictonaries.
        rate_dict[idx] = dict(zip(keys,values))

    # Annual Cummulative Reinsured Loss.
    ann_crl = agp.apply(lambda x: arl(rate_dict,x),axis=1)
    # Annual Incremental Reinsured Loss.
    ann_irl = ann_crl.diff(periods=1,axis=1)
    ann_irl[1.0] = ann_crl[1.0]
    ann_irl['Program'] = ['A'] * 250 + ['B'] * 250  # Append Program.
    ann_irl['Scenrio'] = [*range(1,251)] * 2        # Append Scenario.
    cols = ann_irl.columns[-2:].tolist() + ann_irl.columns[:-2].tolist()
    ann_irl = ann_irl[cols]     # Rearrange columns.

    print(ann_irl)

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()
