
import sys
import pandas as pd
import numpy as np
import h2o
import pickle as pk
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score




import pdb

def hist_fxn(df,name,case_rec,decision):
    tmp = df.loc[df['employer_name']==name]
    tmp ['case_received_date'] = pd.to_datetime(tmp['case_received_date'])
    tmp ['decision_date'] = pd.to_datetime(tmp['decision_date'])
    tmp.loc[(tmp['case_received_date'] > case_rec) & (tmp['decision_date'] < decision)]

    tmp1=(tmp['decision_date'] - tmp['case_received_date']).dt.days
    tmp1.plot.hist(bins=50)
#    plt.show()

    return


def variable_importance (df):
 
    y = df['wage_offer']
    X = df.drop('wage_offer', axis=1)

    X['emp_cat'] =  pd.Categorical(X['employer_name']) 
    #X['job_edu_cat'] =  pd.Categorical(X['job_education']) 
    X['job_sta_cat'] =  pd.Categorical(X['job_state']) 
    X['lang_cat'] =  pd.Categorical(X['job_foreign_lang_req']) 
    X['wage_unit_cat'] =  pd.Categorical(X['wage_unit']) 
    X['cit_cat'] =  pd.Categorical(X['employee_citizenship']) 

    X['emp_cat'] = X['emp_cat'].cat.codes 
    #X['job_edu_cat'] = X['job_edu_cat'].cat.codes 
    X['job_sta_cat'] = X['job_sta_cat'].cat.codes
    X['lang_cat'] = X['lang_cat'].cat.codes
    X['wage_unit_cat'] = X['wage_unit_cat'].cat.codes
    X['cit_cat'] = X['cit_cat'].cat.codes
    label = LabelEncoder()
    int_data = label.fit_transform(df.job_education)
    onehot_data = OneHotEncoder(sparse=False)
    job_edu = onehot_data.fit_transform(int_data.reshape(len(int_data), 1))
    X.reset_index(inplace=True,drop=True)
    X = X.join(pd.DataFrame(job_edu).add_prefix('job_edu_'))
    X.drop(['employer_name','job_education','job_state','job_foreign_lang_req',\
            'employee_citizenship','wage_unit','job_edu_0','job_edu_1','job_edu_2','job_edu_3','job_edu_4',
            'dur','wage_unit_cat'],axis=1, inplace=True)

    model = RandomForestRegressor()
    model.fit(X, y)
    importance = model.feature_importances_

    print([*zip(importance,X.columns)])

    X.drop(['employer_yr_established','cit_cat','lang_cat'],axis=1,inplace=True)
    regressor = DecisionTreeRegressor(random_state=0)
    cv = cross_val_score(regressor, X, y, cv=10,scoring="neg_mean_absolute_error")

    print (cv)
    pdb.set_trace()
    return 



def examine_data(df):
    '''
    Examine data for missing values and handle them
    '''

    cert_df = df[df['case_status']=="Certified"]
    is_emp = ([]) # Columns and the number of missing values
    for col in cert_df.columns:
        is_emp.append((col,cert_df[col].isna().sum()))

    cert_df['case_received_date'] = pd.to_datetime(cert_df['case_received_date'])
    cert_df['decision_date'] = pd.to_datetime(cert_df['decision_date'])
    cert_df['dur'] = (cert_df['decision_date'] - cert_df['case_received_date']).dt.days 

    cert_df.drop(['case_number','case_received_date','decision_date','case_status','job_experience_num_months'],axis=1, inplace=True)
    cert_df.dropna(inplace=True)


    cert_df.loc[cert_df.employer_num_employees == 1400000,'employer_num_employees'] = 700000
    return cert_df 

def get_data():

    universe = pd.read_csv('data_perm_take_home.csv',encoding='ISO-8859-1')

    return universe 


def main():


    universe = get_data()
    uni_clean = examine_data(universe)
    hist_fxn(universe,'GOOGLE INC.', '2012-10-22','2014-10-06')
    variable_importance(uni_clean)
    pdb.set_trace()

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()




