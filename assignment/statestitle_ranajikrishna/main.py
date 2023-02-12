
# ******************************
# Name: insight.py
# Auth: Ranaji Krishna
# Date: Apr 28 2019	
# Desc: 
# ***********************
# Change History
# ***********************
# PR   Date	    Author   Description	
# --   --------   -------   ----------------------------
# 1    04/28/2019  Ranaji   Original code.
#		
# ************************************************/


from my_quant_library import *


def auc_rev(train, ind_col, dep_col, train_data):

    best_model=H2OGradientBoostingEstimator(col_sample_rate=1, \
                learn_rate=0.01, max_depth=5, sample_rate=0.8, ntrees=100, \
                keep_cross_validation_predictions=True, nfolds=4, seed=1)
    
    best_model.train(x=ind_col, y=dep_col, training_frame=train)

    predictions = best_model.cross_validation_holdout_predictions()
    risk = h2o.as_list(predictions)

    train_data['_risk']=risk.p0.values

    # Confusion matrix stats.
    fpr, tpr, thresholds = roc_curve(train_data._lien, train_data._risk)
   
    # Revenue stats.
    income=[sum(1-train_data._lien[train_data._risk >= thr])*800 for thr in thresholds]
    loss=[sum(train_data.lien_amount[train_data._risk>thr]) for thr in thresholds]
    profit=[income[i] - loss[i] for i in range(0,len(thresholds))]

    # Cut-off for maximum profit.
    cut_off = thresholds[np.argmax(profit)]

    # TPR and FPR at maximum profitability.
    true_pos = 1 - tpr[thresholds==cut_off]
    flase_pos = 1 - fpr[thresholds==cut_off]

    # Plot revenue curves.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(loss, income, 'g-')
    ax2.plot(loss, profit, 'b-')
    ax1.set_xlabel('loss')
    ax1.set_ylabel('income', color='g')
    ax2.set_ylabel('profit', color='b')



    return

def gbm_model(train, valid, test, ind_col, dep_col):

    # GBM hyperparameters
    gbm_params = {'learn_rate': [0.01, 0.1],\
                  'max_depth': [3, 5, 9], \
                  'sample_rate': [0.8, 1.0],\
                  'col_sample_rate': [0.2, 0.5, 1.0]}

    # Train and validate a cartesian grid of GBMs
    gbm_grid = H2OGridSearch(model=H2OGradientBoostingEstimator, \
                                grid_id='gbm_grid', hyper_params=gbm_params)

    gbm_grid.train(x=ind_col, y=dep_col, training_frame=train, nfolds=3,\
                    ntrees=100, keep_cross_validation_predictions=True, seed=1)

    #gbm_grid.train(x=ind_col, y=dep_col, training_frame=train, \
    #                                validation_frame=valid, ntrees=100, seed=1)

    #Get the grid results, sorted by validation AUC
    gbm_gridperf = gbm_grid.get_grid(sort_by='auc', decreasing=True)
    gbm_gridperf

    # Grab the top GBM model, chosen by validation AUC
    best_gbm = gbm_gridperf.models[0]

    # Evaluate the model performance on the test set
    test_hf = h2o.H2OFrame(test)
    test_predict = best_gbm.predict(test_hf)
    test_risk = h2o.as_list(test_predict)
    test.ix[:,'_risk'] = test_risk.p0
    
    pdb.set_trace()

    return



def model_explore(train_data, test_data):

    train_data['_lien'] = 0
    train_data._lien[train_data.lien_amount>0] = 1

    h2o.init()
    train_hf = h2o.H2OFrame(train_data)
    train = train_hf
    #train, valid = train_hf.split_frame([0.6], seed=1234)
    train['_lien'] = train['_lien'].asfactor()
    #valid['_lien'] = valid['_lien'].asfactor()
    valid=0

    ind_col = ['state','zipcode','county_fips','total_bath_count','year_built',\
    'building_area_sq_ft','property_type','exterior_walls','water','sewer',\
    'heating','heating_fuel_type','fireplace','style','garage_type_parking',\
    'event_type_sum','event_type_def']
    dep_col = '_lien'

    # Hyper-parameter tunning for GBM model.
    gbm_model(train, valid, test_data, ind_col, dep_col)

    pdb.set_trace()
    # ROC, AUC, Revenue performance.
    auc_rev(train, ind_col, dep_col, train_data)

    return


def feat_eng(data, default_data):

    # Set appropriate index
    data.set_index('house_id', inplace=True) 

    # Left join defaults to data.
    data = data.join(default_data)

    # Only include instances where record date is before the title check date.
    data = data[data.record_date <= data.title_check_date]
    data.drop(['record_date'], axis=1, inplace=True)

    # Design variable `event_type_sum` to show outstanding position of NOD
    data.event_type.replace('default_notice',1,inplace=True)
    data.event_type.replace('default_rescind',-1,inplace=True)
    data.event_type.fillna(0)

    event_sum = data.groupby(data.index)['event_type'].agg('sum')
    data = data.join(event_sum, rsuffix='_sum')

    # Design variable `event_type_def` to show no. NOD
    data.event_type.replace(-1,0,inplace=True)
    def_sum = data.groupby(data.index)['event_type'].agg('sum')
    data = data.join(def_sum, rsuffix='_def')
    
    data.drop(['event_type'], axis=1, inplace=True)

    return data.drop_duplicates()


def get_data():

    # ----- Convert data to storable HDFS objects -----
   # file_location = '/Users/ranajikrishna/invoice2go/git_code/state/default_notices.csv'
   # default_notices = pd.read_csv(file_location)
   # universe = pd.HDFStore('universe.h5')
   # universe['default_notices'] = default_notices
   # 
   # file_location = '/Users/ranajikrishna/invoice2go/git_code/state/train_property_data.csv'
   # train_property_data = pd.read_csv(file_location)
   # universe['train_property_data'] = train_property_data
   # 
   # file_location = '/Users/ranajikrishna/invoice2go/git_code/state/test_property_data.csv'
   # test_property_data = pd.read_csv(file_location)
   # universe['test_property_data'] = test_property_data
    # -----
    
    universe = pd.HDFStore('universe.h5')  
    return universe['default_notices'], universe['train_property_data'], universe['test_property_data']

def main(argv=None):

    default_data, train_data, test_data = get_data()	# Fetch data

    default_data.set_index('house_id', inplace=True) 
    train_data = feat_eng(train_data, default_data)
    test_data = feat_eng(test_data, default_data)

    # Random Forests model for Vaiable importance
    model_explore(train_data, test_data)	
    
    # Multinomial model
    multinom(data)
    return

if __name__ == "__main__":
	status = main()
	sys.exit(status)
