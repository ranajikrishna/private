

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import pyspark.sql.functions as func 
from pyspark.sql.functions import *
from pyspark.sql import types as T
from pyspark import SparkContext

import pickle as pk

import sys
import pdb


def examine_loan_age(df):
    '''
    Function to identify columns in "monthly performance data", after retaining
    only the rows with `loan age = '000'`, have no data. These columns are 
    excluded from the final data set
    '''
    df = df.replace('', 'null')
    for col in df.columns:
        df.agg(countDistinct(col)).show()
    return

def get_del_def_status(x):
    del_status = set(['3'])
    return 1 - int(set(x).intersection(del_status)==set())

def get_zero_def_status(x):
    del_status = set(['03','06','09'])
    return 1 - int(set(x).intersection(del_status)==set())

def get_data():

    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)
    # ---- Load 2009 Q1 historical data ----
    filepath = '/Users/vashishtha/myGitCode/private/figure/historical_data_2009Q1/historical_data_2009Q1.txt'
    data_rdd = sc.textFile(filepath)
    columns = ['credit_score','first_pmt_date','first_time_buyer','mat_date',\
            'msa','mi','units','ocp_status','cltv','dti','upb','ltv',\
            'ori_int_rate','channel','ppm','amtr_type','prp_state','prp_type',\
            'post_code','loan_seq','loan_purpose', 'ori_loan_term',\
            'num_borrower','seller_name','serv_name','super','pre_harp',\
            'prog_ind','hard_ind','prp_val_method','io_ind']
    # Data: 2009 Q1 historical data
    data = data_rdd.map(lambda x: x.split('|')).toDF(columns)

    # ---- Load 2009 Q1 monthly performance data ----
    filepath = '/Users/vashishtha/myGitCode/private/figure/historical_data_2009Q1/historical_data_time_2009Q1.txt'
    time_rdd = sc.textFile(filepath)
    columns = ['loan_seq','mth_rep_prd','act_upb','loan_del_status','loan_age',\
            'rem_mth_mat','def_setl','mod_flag','zero_bal_cde','zero_bal_dte',\
            'int_rate','def_upb','ddlpi','mi_rec','net_sale','non_mi_rec','exp',\
            'lgl_cost','map_costs','tax_insr','misc_exp','act_loss','mod_cost',\
            'step_mod_flag','def_pmt_plan','eltv','zero_bal_upb','del_acd_int',\
            'del_dis', 'brw_asst_cde','cmm_cost','int_upb']
    # Data: 2009 Q1 monthly performance data
    time = time_rdd.map(lambda x: x.split('|')).toDF(columns)

    # Two ways of handling of missing `loan age = '000'`: 
    # [1] Remove loans that have missing `loan age = '000'`
    # [2] Get the next minimum loan age, which is `loan age = '001'`
    # Method [1]
#    time_origin = time.filter(time.loan_age=='000')
    # Method [2]
    time_min_loan_age = time.groupBy('loan_seq').agg(func.min('loan_age').\
            alias('min_loan_age'))
    time_origin = time.join(time_min_loan_age, (time.loan_seq == \
        time_min_loan_age.loan_seq) & \
        (time.loan_age==time_min_loan_age.min_loan_age),\
                                                     "inner").select(time["*"])
    # Examine columns of dataframe that have no data in them.
#    examine_loan_age(time_origin)

    # Compute default status by using `zero balance code`
    zero_bal_def = func.udf(get_zero_def_status,T.IntegerType())
    def_bal_cde = time.groupBy('loan_seq').agg(zero_bal_def(func.collect_list(\
            'zero_bal_cde')).alias('zero_def_bin'))
    # Compute default status by using `Current Loan Delinquency Status`
    del_stat_def = func.udf(get_del_def_status,T.IntegerType())
    def_del_stat = time.groupBy('loan_seq').agg(del_stat_def(func.collect_list(\
            'loan_del_status')).alias('del_def_bin'))

    # Get time to default
    time_len = time.groupBy('loan_seq').count().withColumnRenamed("count","def_time")
    def_len = def_del_stat.join(time_len,time_len.loan_seq==def_del_stat.loan_seq,
            how='inner').select(def_del_stat["loan_seq"],time_len["def_time"])
    def_len_pd = def_len.toPandas()
    pdb.set_trace() 
    # All data without time to default
    universe = \
            data.join(time_origin,\
            data.loan_seq==time_origin.loan_seq,how='left').join(def_bal_cde, \
            data.loan_seq==def_bal_cde.loan_seq, how='left').join(def_del_stat, \
            data.loan_seq==def_del_stat.loan_seq, how='left').select(data["*"], \
            time_origin["act_upb"],time_origin["rem_mth_mat"],\
            time_origin["int_upb"], time_origin["zero_bal_upb"],\
            def_bal_cde["zero_def_bin"], def_del_stat["del_def_bin"])

    pdb.set_trace() 
    # Save data in pickle format.
    universe_pd = universe.toPandas()
#    universe_pd.to_csv('universe_all.csv')
    # Join default time to universe data set
    universe_time_pd=pd.merge(universe_pd, def_len_pd, how="left", on=["loan_seq"])
    # All data with time to default.
#    universe_time_pd.to_csv('universe_all_time.csv')
    return 

