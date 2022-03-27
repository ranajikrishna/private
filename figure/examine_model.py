
import sys
import pandas as pd
import numpy as np

import pdb

def examine_model(X_te,y_te,X_tr,y_tr,pred,thr):

    pred[pred>thr] = 1
    pred[pred<thr] = 0

    df = pd.DataFrame({'true':y_te,'pred':np.array(pred)})
    X_te = X_te.merge(df,how='left',left_index=True,right_index=True) 
    errors = X_te[X_te.true!=X_te.pred]
    # Remove unused categories from `post_code`, otherwise `groupby` will have
    # `count` of 0.
    errors.post_code=errors.post_code.cat.remove_unused_categories()
    
    print(errors.groupby(['seller_name']).count().sort_values)
    print(errors.groupby(['post_code']).count().sort_values)

    
    pdb.set_trace()


    return 

