import thingsio
import sys
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
import datetime
import time
import json
import requests
import pytz
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")



def send_data(devtoken, ts, desc, val):
    
    my_json = json.dumps({'ts': ts, 'values': {desc:val}})
    
    address = 'http://localhost:8080'
    r = requests.post(address + "/api/auth/login",
                      json={'username': 'tenant@thingsboard.org', 'password': 'tenant'}).json()

    # acc_token is the token to be used in the next request
    acc_token = 'Bearer' + ' ' + r['token']
    
    # write telemetry to TB
    r = requests.post(url=address + "/api/v1/" + devtoken + "/telemetry",data=my_json, headers={'Content-Type': 'application/json', 'Accept': '*/*',
                                           'X-Authorization': acc_token})
    return
    
    

def identify_events(df, interval):
    # Iterave over dataframe row by row to identify start of each event and assign unique index

    ind=0
    df['ind'] = np.nan # this column contains a unique index for each event
    event = 0
    thres = 3*interval+1 # time-difference threshold to detect consequtive changes on power
    for i in range(0, int(df.shape[0]-2)):
        if  event==0:
            # start of event
            if (df['dif'].iloc[i]>thres and df['dif'].iloc[i+1]<=thres):
                # if current row has big time difference with previous instance, but next row is below threshol,
                # an event is detecte and the row is marked as the beginning of the event with a unique index
                event=1 
                df['ind'].iloc[i] = int(ind)
                df['ind'].iloc[i-1] = int(ind)

        elif event==1:
            # end of even
            if (df['dif'].iloc[i+1]>thres and df['dif'].iloc[i]<thres):
                df['ind'].iloc[i] = int(ind)
                #df['ind'].iloc[i+1] = int(ind)
                event = 0
                ind += 1
            else:
                # event is ongoing, assing the same event index to row
                df['ind'].iloc[i] = int(ind)
    df = df.dropna()
    return df


def main(argv):
    print('*************************************************************')
    device = argv[1]
    interval = int(50) # interval in milliseconds
    descriptors = 'pwrA,rpwrA'
    address = 'http://localhost:8080' 
    
    # define star/end dates
    local_tz = pytz.timezone('Europe/Athens')
    dt = datetime.datetime.utcnow() # current datetime
    dt = dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    dt = dt - datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second,microseconds=dt.microsecond)
    end_ts = dt 
    start_ts = end_ts + relativedelta(days= -1)
    
    # store actual previous day
    start_day = start_ts
    end_day = end_ts
    #print('start day, end day:', start_day, end_day)
    
    # take a few hours before and after day of interest, to include possible extension of washing machine cycle
    start_ts = start_ts + relativedelta(hours= 3)
    end_ts = end_ts + relativedelta(hours = 3)


    
    ##### testing 
    #start_day = datetime.datetime(2023,11,15)
    #start_day = local_tz.localize(start_day)
    #end_day = start_day+ relativedelta(days= 1)
    #start_ts = start_day + relativedelta(hours= 3)
    #end_ts = end_day + relativedelta(hours = 3)
    ################
    
    # convert to unix timestamps
    start_time = str(int(start_ts.timestamp()*1e3))
    end_time = str(int(end_ts.timestamp()*1e3))
    

    
    nrg_ts = int(start_day.timestamp()*1e3) # timestamp for washing machine energy
    print('start day end day:', start_day.day, end_day)
    print('nrg_ts', nrg_ts )

   

    # Get device info and access token
    [devid, devtoken, acc_token] = thingsio.get_dev_info(device, address)
    
    ####################
    # slice time range in slots of 4 hours, to reduce data volume in http requests
    timethres = 4*3600000
    svec = np.arange(int(start_time),int(end_time),timethres)
    df = pd.DataFrame([])
    
    for st in svec:
        print('looping')
        en = st+timethres-1
        
        if int(end_time)-en<=0: en = int(end_time)
        tmp = thingsio.read_data(acc_token, devid, address,  str(st), str(en), descriptors)
        tmp = tmp.resample(str(interval)+'ms').max()
        tmp = tmp.dropna()
        df = pd.concat([df,tmp])
        time.sleep(0.1)
    df.sort_index(inplace=True)
    
    ####################
    

    #df = thingsio.read_data(acc_token, devid, address, start_time, end_time, descriptors)
    #df = df.resample(str(interval)+'ms').max()
    #df = df.dropna()
    #df.sort_index(inplace=True)

    

    # dif column contains the time difference between every couple of consequtive rows
    df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')

    # identify events
    df = identify_events(df,interval)

    
    #Value before the start of the event 
    prev_df = df.groupby('ind').nth(0)
    prev_df.rename(columns = {'pwrA':'prev_pwrA', 'rpwrA':'prev_rpwrA'}, inplace = True)
    prev_df = prev_df[['prev_pwrA', 'prev_rpwrA']]
    print('first event before:', df.loc[df['ind']==0])
    #Mean pwrA per event
    df_new = df.groupby('ind').apply(lambda group: group.iloc[1:,:3])
    print('df new:', df_new.head())
    now_df = df_new.groupby('ind').mean('pwrA')
    print('npw_df:', now_df.head())

    #Df with previous values and averages of event
    merged = pd.concat([prev_df, now_df], axis=1)
    merged = merged.drop('dif', axis=1)

    #Find activations or deactivation and assign value 'Act/Deact'. 
    #Then find ON events and assign value 'ON'
    merged['Event'] = np.where(((merged['pwrA'] > merged['prev_pwrA']) & (merged['pwrA']>=(merged['prev_pwrA'] + 30)) & (merged['pwrA']<=(merged['prev_pwrA'] + 500))) | ((merged['pwrA'] < merged['prev_pwrA']) & (merged['pwrA']<=(merged['prev_pwrA'] - 30)) & (merged['pwrA']>=(merged['prev_pwrA'] - 500))), 'Act/Deact', 'None')
    merged['ON_OFF'] = np.where(((merged['pwrA'] > merged['prev_pwrA']) & (merged['pwrA']>=(merged['prev_pwrA'] + 30)) & (merged['pwrA']<=(merged['prev_pwrA'] + 500))), 'ON', 'None')



    #Keep only ON events (using it later for dbscan)
    #on_events = merged[merged['ON_OFF']=='ON']
    #on_events = on_events['ON_OFF']
    
    #Keep first row of each event 
    merged = merged['Event']
    merged1 = df.join(merged, on='ind', how='left')
    merged1['ts'] = merged1.index

    merged2 = merged1.groupby('ind').nth(1).dropna()
    merged2['ind'] = merged2.index
    merged2['Ev'] = np.where(merged2['Event']=='Act/Deact', 1, 0)
    merged2 = merged2.set_index('ts')

    #Resampling with 50 ms
    resampling = merged2.resample(str(interval)+'ms').max()
    resampling = resampling['Ev'].fillna(0)

    #Rolling window to count Activations/Deactivation in a sliding window of 20 minutes centered on sample k
    wash = pd.DataFrame(resampling.rolling(24000, center=True).sum())
    
    #Keep samples with >= 35 activations/deactivations
    wash_ev = wash[wash['Ev']>=35]
    

    df['ts'] = df.index
    
    ##
    # keep instances of wm and concatenate to df. Then check for multiple cycles.
    df['wm'] = wash_ev['Ev']
    df.loc[df['wm']>1, 'wm']=1
    df.loc[df['wm']<1, 'wm']=0
    
    ##################################################
    
    # Test for clusters
    tmp = df[['wm']].copy()
    tmp = tmp.loc[tmp['wm']==1]
    
    # initialize wm energy
    resnrg = 0
    
    if not tmp.empty:
        print('not empty')
        difcluster = 9e5 # timedif in milliseconds
        tmp['dif'] = tmp.index.to_series().diff().astype('timedelta64[ms]')
        
        tmp['dif'].iloc[0] = 0
        tmp.loc[tmp['dif']>difcluster].shape[0]
        if tmp.loc[tmp['dif']>difcluster].shape[0]>0:
            print('found cluster')
            tmp['cluster'] = np.nan
            tmp['cluster'].iloc[0] = 1 # assign value 1 to the 1st cluster 
            tmp.loc[tmp['dif']>difcluster,'cluster'] = np.arange(2,tmp.loc[tmp['dif']>difcluster].shape[0]+2)# assign ascending index to cluster 
            tmp['cluster'] = tmp['cluster'].ffill()
        
            tmp.drop('wm', axis=1, inplace=True)
            tmp.rename(columns={'cluster':'wm'}, inplace=True)
        
            df.drop('wm', axis=1, inplace=True)
            df['wm'] = tmp['wm']
            
            
        # group by washing machine cycles 
        grouped = df.groupby('wm')
        for _,group in grouped:
            
            tstart = group.index[0]
            tend = group.index[-1]
            dur = (tend-tstart).total_seconds() / 60
            print('duration:', dur)
            print('tstart tend of cluster:', tstart, tend)
            durm = 15
            if dur>durm: # at least durm minutes duration
                tmp = df[tstart:tend].copy()
                #################################################
                # group events to check for resistive loads while heating
                resDict = {}
                gpd_wm = tmp.groupby('ind')
                motornrg = 0
                
                testdict = {}
                for _, group in gpd_wm:
                    delta = group['pwrA'].iloc[-3:].mean() - group['pwrA'].iloc[0]
                    #print(group.index[0],delta)
                    
                    #if np.abs(delta)>1000:
                    #    testdict[group.index[0]] = delta
                        
                        
                    if ((np.abs(delta)>1200) & (np.abs(delta)<3200)):
                        resDict[group.index[0]] = delta
                        print('Resistive load')
                    elif ((np.abs(delta)>30) & (np.abs(delta)<500)):
                        motornrg += 3*np.abs(delta)
                motornrg = motornrg/3600
                
                # for testing
                #test = pd.DataFrame.from_dict(testdict, orient='index')
                #test.rename(columns={0:'pwrdif'}, inplace=True)
                #print('testdf', test)
                #test.to_json(start_time+'.json')
                                
                
                resdf = pd.DataFrame.from_dict(resDict, orient='index')
                print('resdf:', resdf)
                
                if not resdf.empty:
                    resdf.rename(columns={0:'pwrdif'}, inplace=True)
                    #print('resdf',resdf)
                    resdf['pwrdif'] = resdf['pwrdif'].astype('float64')
                    
                    
                    
                    resdf['dif'] = resdf.index.to_series().diff().astype('timedelta64[s]')
                    resdf['resOnOff'] = np.nan
                    resdf['avgP'] = np.nan
                    resdf['resnrg'] = np.nan
                    resdf.loc[(resdf['pwrdif']<0) & (resdf['pwrdif'].shift()>0), 'resOnOff'] = 1 # this is a cycle of resistive load on/off 
                    
                    
                    resdf.loc[resdf['resOnOff']==1, 'avgP'] = (np.abs(resdf['pwrdif'])+resdf['pwrdif'].shift())/2 
                    resdf.loc[resdf['resOnOff']==1, 'resnrg'] = (resdf['avgP']*resdf['dif'])/3600
                    resdf = resdf.loc[resdf['resOnOff']==1]
                    
                    
                    #print('First great resistive load minutes:', (resdf.index[0]-tstart).total_seconds()/60)
                    
                    #if (   ((resdf.index[0]-tstart).total_seconds()/60<=durm) &  ((tend-tstart).total_seconds() / 60>durm) ):
                    if not resdf.empty:
                        if (tend-tstart).total_seconds() / 60>durm:
                            print('First great resistive load minutes:', (resdf.index[0]-tstart).total_seconds()/60)
                            resnrg += resdf['resnrg'].sum()+motornrg
                            print('estimated motor energy in Wh:', motornrg)
                            print('estimated total wm energy in Wh:', resnrg)
                            
                            
                            print('Washing machine started at ',tstart,' and finished at ',tend)
                            
                            if tstart.day!= start_day.day:
                                nrg_ts = nrg_ts+86400000
                                print('new ts:', nrg_ts)
                            tstart = int(tstart.timestamp()) * 1000
                            send_data(devtoken,tstart, 'wm_on', '1')
                                              
                                                       
                            tend = int(tend.timestamp()) * 1000
                            send_data(devtoken,tend, 'wm_off', '0')
                            
        
    # write wm energy even if it's zero
    print('total energy:', resnrg)
    send_data(devtoken,nrg_ts, 'wmnrg',str(np.round(resnrg, 2)))    



if __name__ == '__main__':
    sys.exit(main(sys.argv))