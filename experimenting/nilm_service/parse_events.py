#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent
from sklearn.cluster import KMeans
from cassandra.query import ordered_dict_factory
import uuid
import eventsClustering
import datetime
import pickle
pd.set_option("display.max_rows", None)


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
     

    # transfer int values stored in long_v, to dbl_v column
    df['dbl_v'][df['dbl_v'].isnull()] = df['long_v']
    df = df.drop('long_v', axis=1)
    df = df.pivot_table(values='dbl_v', index=df.index, columns='key', aggfunc='first')
    df = df.apply(pd.to_numeric).round(3)
    df = df[maphase[ph]]
    

    cluster.shutdown()
    return df


def run_models(df, phase, mdlpath):

    fileslist = [i for i in os.listdir(mdlpath) if i.endswith('.sav')]
    appliances = [os.path.splitext(x)[0] for x in fileslist]
    
    apps=['radiator','aerothermo','dishwasher','Other','fridge','water heater','coffeemaker','microwave']
    #apps = ['radiator','water heater','dishwasher','coffeemaker','fridge','microwave','aerothermo','Other']

    
    avgpr = pd.read_csv(mdlpath+'/avg_pwr.csv')
    mapping={'radiator':'appl_1.0','aerothermo':'appl_7.0','dishwasher':'appl_3.0','Other':'appl_8.0','fridge':'appl_5.0','water heater':'appl_2.0','coffeemaker':'appl_4.0','microwave':'appl_7.0'}
    
    df.columns=df.columns.str.replace('A','')
    df.columns=df.columns.str.replace('B','')
    df.columns=df.columns.str.replace('C','')


    events = []
    nums = []
    state = []
    ev_ts = []
    dpwr = []  # delta active power --> |previous power - current power|
    conflicts = {}
    i = 0
    
    df['ts'] = pd.to_datetime(df.index, unit='ms')
    df = df.set_index('ts',drop=True)
    df_info = df[['sign_','avgSP_']]
    df = df.drop(['sign_','avgSP_'],axis=1)
    
    #print(df)
    
    for k in range(0, df.shape[0]):

        #print(df.index[k])
        mdlsum = 0
        assigned = False

        # run all models for this phase
        for j in range(0, len(appliances)):
            
            filename = mdlpath + '/' + str(appliances[j]) + '.sav'
            mdl = pickle.load(open(filename, 'rb'))
            steady = (df['avgP_3'].iloc[k]+df['avgP_4'].iloc[k])/2
            X = np.reshape(df.iloc[k].to_numpy(),(-1,8))
            
            y_pred = mdl.predict(X)
            
            #                 print(y_pred)
            if np.sum(y_pred) >= 0.75 * len(y_pred):
                
                #print(df.index[k],apps[j])
                #                 if np.sum(y_pred[1:])>=0.75*len(y_pred):

                if not assigned:
                    #                         dpwr.append(np.abs(np.mean(change['pwr'].iloc[-20:-10])-np.mean(steady['pwr'])))
                    dpwr.append(steady)
                    nums.append(np.sum(y_pred))
                    mdlsum = np.sum(y_pred)
                    #                         print(mdlphase[phase][j])
                    events.append(apps[j])
                    #                         print(mdlphase[phase][j], tsm)
                    ev_ts.append(df.index[k])
                    #                         state.append(st)
                    assigned = True

                    i = i + 1
                else:
                    if np.sum(y_pred) > mdlsum:  # nums[i-1]:
                        # print('previous sum %d, current sum %d:' % (mdlsum,np.sum(y_pred)))
                        dpwr[-1] = steady
                        events[-1] = apps[j]
                        #  events.append(mdlphase[phase][j])
                        mdlsum = np.sum(y_pred)
                    elif np.sum(y_pred) == mdlsum:  # nums[i-1]:
                        conflicts[ev_ts[i - 1]] = [apps[j]]
                        print('New conflict at time %s, prev app is %s new app is %s' % (
                        ev_ts[i - 1], events[-1], apps[j]))
        #print('*******************************************')
     
    
    
    ev = confl_postproc(events, state, ev_ts, conflicts, dpwr)

    #ev = ev.merge(df_info,how='left')
    ev = pd.concat([ev,df_info],axis=1,join='inner')
    ev.rename(columns={'dpwr':'steady','avgSP_':'prev_steady','sign_':'state'},inplace=True)
    
    #outliers(ev,apps)
    
    ev = postproc(ev)
    
    #print(ev.loc[ev['appl']=='fridge'])
    #print(ev.loc[ev['appl']=='aerothermo'])
    #print(ev.loc[ev['appl']=='coffeemaker'])
    #print(ev.loc[ev['appl']=='dishwasher'])
    #print(ev.loc[ev['appl']=='radiator'])
    #print(ev.loc[ev['appl']=='microwave'])
    #print(ev.loc[ev['appl']=='water heater'])
    
    
    ev = reduce_onoffs(ev)
    
    ev = postproc(ev)
    
    ev = ev.dropna(subset=['appl'])
    ev['avgpwr']=0
    
    all_appls = pd.DataFrame([])
    
    for app in ev['appl'].unique():
        
        print('Consumption for appliance:',app)
        #print(ev.loc[ev['appl']==app])    
        
        # assign average operation power extracted from file
        ev.loc[ev['appl']==app,'avgpwr'] = avgpr.loc[avgpr['app']==mapping[app],'pwr'].values[0]
        
        # keep only onoff rows and compute consumption as average power*seconds/3600
        tmp = ev.loc[ev['appl']==app].copy()
        
        #*********************************
        tmp.loc[((tmp['state']>0) & (tmp['state'].shift(-1)>0) & (tmp['dif'].shift(-1)>90)),'onoff']=1
        #*********************************
        
        tmp['end'] = 0
        tmp.loc[(tmp['onoff']==1) & (tmp['dif'].shift(-1)>90),'end']=1
        
        lastind = tmp[tmp['onoff'] == 1].index[-1]
        tmp.loc[tmp.index==lastind,'end']=1 # last onoff must be set to 1
        
        #print('BEFORE:',tmp)
        tmp.drop(tmp[(tmp['onoff']==0) & (tmp['onoff'].shift(-1)==0)].index,inplace=True)
        tmp.loc[(tmp['onoff']==1) & (tmp['dif'].shift(-1)>90),'end']=1
        
        #print('AFTER reduction:',tmp)
        tmp.drop(tmp[(tmp['end']==0) & (tmp['end'].shift()==0)].index,inplace=True)
        
        tmp['ts'] = tmp.index
        tmp['dif'] = tmp['ts'].values-tmp['ts'].shift().values
        tmp['dif'] = tmp['dif'].dt.seconds.fillna(0)
        tmp.drop('ts',axis=1,inplace=True)
        
        if app != 'Other':
            #print(tmp)
            all_appls = pd.concat([all_appls,tmp[['appl','state','end','dif']]])
        tmp = tmp.loc[tmp['end']==1]
        tmp['Consumed'] = (tmp['avgpwr']*tmp['dif'])/3600
        
        print('Total consumed energy:',tmp['Consumed'].sum())
    
    all_appls = all_appls.sort_index()
    print(all_appls) 
    
    return ev
   
    #print(exp)
    #print(exp[exp['tdif'] > exp['tdif'].quantile(.98)])
    #print(exp[exp['tdif'] > thres])
    
    
    #####################################
    
def outliers(ev,apps):
    for app in apps:
        
         ###################################
        exp = ev.loc[ev['appl']==app]
        print(exp.head())
        if not exp.empty:
            exp['ts'] = pd.to_datetime(exp.index)
            exp['tdif'] = exp['ts'].values - exp['ts'].shift().values
            exp['tdif'] = exp['tdif'].dt.seconds
            exp['tdif'].iloc[0] = 0
            exp = exp[['tdif','state']]
            #avg = exp['tdif'].mean()
            #std = exp['tdif'].std()
            #thres = avg+(2*std)
            #print(exp)
            #print(app)
            if np.array(exp['tdif']).reshape(-1, 1).shape[0]>1:
            
                kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(exp['tdif']).reshape(-1, 1))
                exp['labels'] = kmeans.labels_
                print('Appliance',app)
                print(exp)
            print('Quantile 0.94:',exp[exp['tdif'] > exp['tdif'].quantile(.95)])
                #print(exp[exp['tdif'] > thres])
    
    
def reduce_onoffs(events):
    
    a = events['appl'].unique()
    a = a[a==a]
   
    for j in a:
        
        #events.loc[events['appl']==j] = events.loc[events['appl']==j][events.loc[events['appl']==j]['state']!= events.loc[events['appl']==j]['state'].shift()]
        
        
        tmp = events.loc[events['appl']==j].copy()
        
#         # if 1st event is 0 then drop
#         if tmp['state'].iloc[0]<1:
#             tmp = tmp.iloc[1:]
#         # if last event is 1 then drop
#         if tmp['state'].iloc[-1]>0:
#             tmp = tmp.iloc[:-1]
        
        # calculate time difference between on/off
        tmp = tmp.loc[tmp['state'].isnull()==False]
        tmp['ts'] =  tmp.index
        #tmp['dif'] = tmp['ts'].values-tmp['ts'].shift().values
        #tmp['dif'] = tmp['dif'].dt.seconds.fillna(0)
        
        # drop on/offs with duration less than 1 min
#         tmp = tmp[(tmp['dif']>60) | (tmp['dif'].shift(-1)>60) | (tmp['dif']==0)]
        
        
        # keep only first instance of each state
        
        tmp['onoff'] = 0
        tmp.loc[(tmp['state']<0) & (tmp['state'].shift()>0),'onoff'] = 1
        #tmp.loc[(tmp['state']>1) & (tmp['state'].shift()<0) & (tmp['onoff'].shift()>0),'onoff'] = 1

        if tmp['state'].iloc[0]<0:
            tmp['onoff'].iloc[0] = 1
#         tmp = tmp.loc[(tmp['state']==0.0) & (tmp['state'].shift()==1.0)]
        
        #calculate time difference again
        tmp['dif'] = tmp['ts'].values-tmp['ts'].shift().values
        tmp['dif'] = tmp['dif'].dt.seconds.fillna(0)
        tmp.drop('ts',axis=1, inplace=True)  
        
#         tmp = tmp.loc[tmp['onoff']==1]
        
        events = events.loc[events['appl']!=j]
        events = pd.concat([events,tmp[['appl','state','steady','prev_steady','conflict','dif','onoff']]])
        events.sort_index(inplace=True)
    return events


def confl_postproc(events, state, ev_ts, conflicts, dpwr):
    ev = pd.DataFrame([])
    ev['appl'] = events
    #     ev['state'] = state
    ev['ts'] = ev_ts
    ev['dpwr'] = dpwr
    ev = ev.dropna()
    ev.set_index('ts', inplace=True)


    if len(conflicts) > 0:  # if there are conflicts
        confl = pd.DataFrame(conflicts).T
        confl.columns = ['conflict']
        ev = pd.concat([ev, confl], axis=1)

        for i in range(5, ev.shape[0] - 5):
            if pd.isna(ev['conflict'].iloc[i]) == False:
                #         print(ev['conflict'].iloc[i],ev['appl'].iloc[i],ev['appl'].iloc[i-1])

                # check neighborhood -- 5 previous and 5 next points-- to decide if conflict will replace value
                if ev['conflict'].iloc[i] == ev['appl'].iloc[i - 5:i + 5].value_counts()[:1].index.tolist()[0]:
                    print('appliance before conflict:',ev.iloc[i])
                    ev['appl'].iloc[i] = ev['conflict'].iloc[i]
                    ev['conflict'].iloc[i] = np.nan
                    print('appliance after conflict:',ev.iloc[i])

    else:
        ev['conflict'] = np.nan
    #         ev.drop('conflict',axis=1,inplace=True)

    return ev



def postproc(events):
    # drop events corresponding to only one appearance of an appliance
    
    singlapp = events['appl'].value_counts()
    print('singlapp',singlapp[singlapp == 1])
    while singlapp[singlapp == 1].shape[0] > 0:
    #if singlapp[singlapp == 1].shape[0] > 0:
        events = events[events['appl'] != singlapp[singlapp == 1].index.values[0]]
        print('appliance with one appearance:',singlapp[singlapp==1])
        singlapp = events['appl'].value_counts()
           
    return events



# convert ResultSet to compact dictionary
def dict_convertion(dicts):
    keys = dicts[0].keys()
    values = zip(*(d.values() for d in dicts))
    mydict = dict(zip(keys, values))
    return mydict
   


def main():

    mdlpath='/home/thingsuser/disaggregation_modules/models/'


    dt = datetime.datetime.utcnow()
    partition = datetime.datetime(year=dt.year, month=dt.month, day=1, hour=0, minute=0, second=0)
    partition = int((partition - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    ts1 = dt - datetime.timedelta(hours=1, minutes=dt.minute, seconds = dt.second, microseconds = dt.microsecond)
    
    ts2 = ts1 +datetime.timedelta(minutes=59, seconds=59)
  
    ts1 = ts1 + datetime.timedelta(hours=-12)
    
    
    print('ts1 %s ts2%s' % (ts1,ts2))
    ts1new = ts1 - datetime.timedelta(days=7)
    
    partitionNew = datetime.datetime(year=ts1new.year, month=ts1new.month, day=1, hour=0, minute=0, second=0)
    partitionNew = int((partitionNew - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    

    
    ts1 = int((ts1 - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    ts2 = int((ts2 - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    ts1new = int((ts1new - datetime.datetime(1970, 1, 1)).total_seconds())*1000
    
    # for testing purposes of specific dates
    ts1 = 1614722400000
    ts2 = 1614808800000
    

    # open devices.txt to read entity IDs and phase information of old & new devices
    with open('/home/thingsuser/disaggregation_modules/devices_info/devices.txt') as f1, open('/home/thingsuser/disaggregation_modules/devices_info/new_devices.txt') as f2:
        devs = [line.rstrip() for line in f1]
        ndevs = [line.rstrip() for line in f2]
        print('New devices found:',ndevs)
        f1.close()
        f2.close()

    # transfer new devices to old
    #with open('/home/thingsuser/disaggregation_modules/devices_info/new_devices.txt', 'r+') as f1, open('/home/thingsuser/disaggregation_modules/devices_info/devices.txt', 'a+') as f2:
    #    if os.path.getsize("devices_info/devices.txt") != 0:
    #        f2.write('\n')
    #    f2.write(f1.read())
    #    f1.truncate(0)
    #    f1.close()
    #    f2.close()


    # iterate over new devices
    for dev in ndevs:
        entid, phases, devserial = dev.split(',')
        print('Processing device:', devserial)
        if phases=='1':
            phases=['A']
            print('Single-phase house')
        elif phases=='3':
            phases=['A','B','C']
            print('Three-phase house')
 
        for ph in phases:
            tmp = read_data(entid, partition, ts1new, ts2, ph)
            tmp['dt'] = pd.to_datetime(tmp.index, unit='ms')
            tmp.set_index('dt', inplace=True, drop=True)
            tmp.to_excel('full_data.xlsx')
            
            meas = fetch_meas(entid, partition, ts1new, ts2, ph)
            # if no models have een created for this device perform clustering
            print(meas.head())
            eventsClustering.cluster_events(ph, meas, devserial, mdlpath)

    # iterate over old devices
    for dev in devs:
        entid, phases, devserial = dev.split(',')
        if phases=='1':
            phases=['A']
        elif phases=='3':
            phases=['A','B','C']

        for ph in phases:
            meas = fetch_meas(entid, partition, ts1, ts2, ph)
            run_models(meas,ph,mdlpath + str(devserial) + '/' + str(ph))

if __name__ == "__main__":
    # sys.exit(main(sys.argv))
    sys.exit(main())

