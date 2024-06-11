#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent
from cassandra.query import ordered_dict_factory
import uuid
import eventsClustering
import datetime

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle

def read_data(entid, partition, ts1, ts2, ph):

    # define lists of events descriptors, one for each phase
    maphase = {'A': ['pwrA']}


    # partition = 1609459200000
    # ts1 = 1610024400000  # for testing purposes
    # ts2 = 1610028000000

    # Connect to keyspace, start session, configure results
    cassIP = ['10.102.10.197']
    cluster = Cluster()
    session = cluster.connect()
    session.row_factory = ordered_dict_factory

    # prepare statement
    stmt = session.prepare(
        "SELECT key,ts,dbl_v,long_v FROM thingsboard.ts_kv_cf where entity_type='DEVICE' and entity_id=? and key=? and "
        "partition=? and ts>? AND ts<?")

    df = pd.DataFrame([])

    # create statements&params
    statements_and_params = []
    for desc in maphase[ph]:
        params = (uuid.UUID(entid), desc, partition, ts1, ts2)
        statements_and_params.append((stmt, params))

    # execute statement
    results = execute_concurrent(session, statements_and_params, raise_on_first_error=False)
    

    # convert ResultSet to dataframe
    for (_, res) in results:
        df = pd.concat([df, pd.DataFrame(dict_convertion(res))])
    df = df.set_index('ts', drop=True)
   

    # transfer int values stored in long_v, to dbl_v column
    df['dbl_v'][df['dbl_v'].isnull()] = df['long_v']
    df = df.drop('long_v', axis=1)
    df = df.pivot_table(values='dbl_v', index=df.index, columns='key', aggfunc='first')
    df = df.apply(pd.to_numeric).round(3)
    df = df[maphase[ph]]
    

    cluster.shutdown()
    return df



# connect to cassandra and retrieve raw events' data
def fetch_meas(entid, partition, ts1, ts2, ph):

    # define lists of events descriptors, one for each phase
    maphase = {'A': ['avgSP_A', 'sign_A',
                     'avgPA_1', 'minPA_1', 'maxPA_1', 'stdPA_1', 'avgRA_1', 'minRA_1', 'maxRA_1', 'stdRA_1',
                     'avgPA_2', 'minPA_2', 'maxPA_2', 'stdPA_2', 'avgRA_2', 'minRA_2', 'maxRA_2', 'stdRA_2',
                     'avgPA_3', 'minPA_3', 'maxPA_3', 'stdPA_3', 'avgRA_3', 'minRA_3', 'maxRA_3', 'stdRA_3',
                     'avgPA_4', 'minPA_4', 'maxPA_4', 'stdPA_4', 'avgRA_4', 'minRA_4', 'maxRA_4', 'stdRA_4'],
               'B': ['avgSP_B', 'sign_B',
                     'avgPB_1', 'minPB_1', 'maxPB_1', 'stdPB_1', 'avgRB_1', 'minRB_1', 'maxRB_1', 'stdRB_1',
                     'avgPB_2', 'minPB_2', 'maxPB_2', 'stdPB_2', 'avgRB_2', 'minRB_2', 'maxRB_2', 'stdRB_2',
                     'avgPB_3', 'minPB_3', 'maxPB_3', 'stdPB_3', 'avgRB_3', 'minRB_3', 'maxRB_3', 'stdRB_3',
                     'avgPB_4', 'minPB_4', 'maxPB_4', 'stdPB_4', 'avgRB_4', 'minRB_4', 'maxRB_4', 'stdRB_4'],
               'C': ['avgSP_C', 'sign_C',
                     'avgPC_1', 'minPC_1', 'maxPC_1', 'stdPC_1', 'avgRC_1', 'minRC_1', 'maxRC_1', 'stdRC_1',
                     'avgPC_2', 'minPC_2', 'maxPC_2', 'stdPC_2', 'avgRC_2', 'minRC_2', 'maxRC_2', 'stdRC_2',
                     'avgPC_3', 'minPC_3', 'maxPC_3', 'stdPC_3', 'avgRC_3', 'minRC_3', 'maxRC_3', 'stdRC_3',
                     'avgPC_4', 'minPC_4', 'maxPC_4', 'stdPC_4', 'avgRC_4', 'minRC_4', 'maxRC_4', 'stdRC_4']}


    # partition = 1609459200000
    # ts1 = 1610024400000  # for testing purposes
    # ts2 = 1610028000000

    # Connect to keyspace, start session, configure results
    cassIP = ['10.102.10.197']
    cluster = Cluster()
    session = cluster.connect()
    session.row_factory = ordered_dict_factory

    # prepare statement
    stmt = session.prepare(
        "SELECT key,ts,dbl_v,long_v FROM thingsboard.ts_kv_cf where entity_type='DEVICE' and entity_id=? and key=? and "
        "partition=? and ts>? AND ts<?")

    df = pd.DataFrame([])

    # create statements&params
    statements_and_params = []
    for desc in maphase[ph]:
        params = (uuid.UUID(entid), desc, partition, ts1, ts2)
        statements_and_params.append((stmt, params))

    # execute statement
    results = execute_concurrent(session, statements_and_params, raise_on_first_error=False)
    
    
    # convert ResultSet to dataframe
    for (_, res) in results:
        df = pd.concat([df, pd.DataFrame(dict_convertion(res))])
    
    df = df.set_index('ts', drop=True)
    df['ts'] = pd.to_datetime(df.index,unit='ms')
    df = df.set_index('ts', drop=True)
     

    # transfer int values stored in long_v, to dbl_v column
    df['dbl_v'][df['dbl_v'].isnull()] = df['long_v']
    df = df.drop('long_v', axis=1)
    df = df.pivot_table(values='dbl_v', index=df.index, columns='key', aggfunc='first')
    df = df.apply(pd.to_numeric).round(3)
    df = df[maphase[ph]]
    

    cluster.shutdown()
    return df
    
    
# convert ResultSet to compact dictionary
def dict_convertion(dicts):
    keys = dicts[0].keys()
    values = zip(*(d.values() for d in dicts))
    mydict = dict(zip(keys, values))
    return mydict
   
   
def train_apps(dfs, devserial, mdlpath):
    
    # create directory with device serial
    mdlpath = mdlpath + str(devserial) + '/A'
    if not os.path.exists(mdlpath):
        os.makedirs(mdlpath)

    # calculate average active power of each appliance
    avpwr = dict()
    for i in (dfs['label'].unique()):
        # rearrange order so that the ith dataframe is first.
        dfcurr = dfs.copy()
        
        dfcurr.loc[dfcurr['label']!=i, 'label']=0
        dfcurr.loc[dfcurr['label']!=0, 'label']=1
        
        print(dfcurr['label'].value_counts())
        
        # compute average active power of each appliance
        avpwr['appl_%s' % i] = np.mean(dfcurr.loc[dfcurr['label'] == 1, 'avgP_4'])
        print('average power of %i is %f' % (i, avpwr['appl_%s' % i]))

        y = dfcurr['label'].values
        y = np.repeat(y,4)
        X = dfcurr.drop('label', axis=1)
        X = np.reshape(X.to_numpy(), (-1, 8))
        print('X shape:',X.shape)
        print('y shape:',y.shape)
        # Perform grid search to tune hyperparameters
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        parameters = {'min_samples_split': range(2, 20), 'max_depth': np.arange(5, 30)}
        gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                          param_grid=parameters,
                          scoring=scoring, refit='AUC', return_train_score=True, n_jobs=-1)
        gs.fit(X, y)
        print('Best score from grid search:', gs.best_score_)
        # after grid search, train entire dataset with best parameters
        mdl = DecisionTreeClassifier(random_state=42, max_depth=gs.best_params_['max_depth'],
                                     min_samples_leaf=gs.best_params_['min_samples_split'])
        mdl = mdl.fit(X, y)

        filename = mdlpath + '/appl_%s' % i + '.sav'
        pickle.dump(mdl, open(filename, 'wb'))

    # save average power to csv
    avpwr = pd.DataFrame(avpwr.items())
    avpwr.columns = ['app', 'pwr']
    avpwr.to_csv(mdlpath + '/avg_pwr.csv', index=False)

    del avpwr, dfcurr, mdl, gs, X, y

    return
    
    
def main():

    mdlpath='/home/thingsuser/disaggregation_modules/models/'


    dt = datetime.datetime.utcnow()
    partition = datetime.datetime(year=dt.year, month=dt.month, day=1, hour=0, minute=0, second=0)
    partition = int((partition - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    ts1 = dt - datetime.timedelta(hours=1, minutes=dt.minute, seconds = dt.second, microseconds = dt.microsecond)
    ts2 = ts1 +datetime.timedelta(hours=5)
    ts1new = ts1 - datetime.timedelta(days=7)
    
    partitionNew = datetime.datetime(year=ts1new.year, month=ts1new.month, day=1, hour=0, minute=0, second=0)
    partitionNew = int((partitionNew - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    

    
    ts1 = int((ts1 - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    ts2 = int((ts2 - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    ts1new = int((ts1new - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    
    
    entid = '8a65b210-609b-11eb-bf57-2df896770934'
    phases = ['A','B','C']
    devserial = '102.402.000045'
    
    # appliances are: 1.radiator 2.water heater 3.dishwasher 4.coffeemaker 5.fridge 6.microwave 7.aerothermo 8.Other
    
    i = 0
    dfs = pd.DataFrame([])
    dftotal = pd.DataFrame([])
    for ph in phases:
        meas = fetch_meas(entid, partition, ts1new, ts2, ph)
        
        meas = meas[meas.columns[-32:]]
        meas = meas.resample('5S').max()
                
        
        meas.columns=meas.columns.str.replace('A','')
        meas.columns=meas.columns.str.replace('B','')
        meas.columns=meas.columns.str.replace('C','')
        
        if ph=='A':
            dftotal = pd.concat([dftotal,meas],axis=1)
            dftotal = dftotal.dropna()
        else:
            meas = meas.dropna()
            meas['label'] = i
            
            
            dfs = pd.concat([dfs,meas[['label']]])
            
       
        i += 1
    
    # Tavoularis plugs are: 2=dishwasher, 4=coffeemaker, 14=fridge, 13=microwave, 8 = aerothermo
    entities = ['5d2eb3e0-6d5d-11eb-bf57-2df896770934','df1bd470-633d-11eb-bf57-2df896770934','7a8d0260-6133-11eb-bf57-2df896770934', '5d41c6b0-6d5d-11eb-bf57-2df896770934',
    'e5ee01f0-609b-11eb-bf57-2df896770934']
    #serials = ['101.111.000002','101.111.000004','101.111.000014','101.111.000013']    ,'101.111.000008']
    
    for j in range(0,len(entities)):
        entid = entities[j]
        ph = 'A'
        meas = fetch_meas(entid, partition, ts1new, ts2, ph)
        
        meas = meas[meas.columns[-32:]]
        meas = meas.resample('5S').max()

        meas = meas.dropna()

        meas['label'] = i
        
        meas.columns=meas.columns.str.replace('A','')
        
        dfs = pd.concat([dfs,meas[['label']]])
        

        
        i += 1
    dfs = dfs[~dfs.index.duplicated(keep='first')]
    
    
    dftotal = pd.concat([dftotal,dfs],axis=1)
    print(dftotal.loc[dftotal['label'].isnull()])
    dftotal = dftotal.dropna(subset=['avgP_1'])
    dftotal.loc[dftotal['label'].isnull(),'label'] = i
    
    print(dftotal['label'].value_counts())
    # train models
    #train_apps(dftotal, devserial, mdlpath)
        

if __name__ == "__main__":
    # sys.exit(main(sys.argv))
    sys.exit(main())     
    