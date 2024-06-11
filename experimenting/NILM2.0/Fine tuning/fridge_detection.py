#from scipy.signal import argrelextrema
import eventDetection as ed
import thingsio
import requests
import json
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
import pytz
import datetime
from dateutil.tz import gettz
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")
import time
import sys



def transform_dicts(offdeltas, offre, deltas, deltasre, d):
    # convert dictionaries to dataframes
    off = pd.DataFrame.from_dict(offdeltas, orient='index')
    off.rename(columns={0:'pwr'}, inplace=True)
    off_re = pd.DataFrame.from_dict(offre, orient='index')
    off_re.rename(columns={0:'rpwr'}, inplace=True)
    off = pd.concat([off,off_re], axis=1)

    dl = pd.DataFrame.from_dict(deltas, orient='index')
    dl.rename(columns={0:'pwr'}, inplace=True)
    dl_re = pd.DataFrame.from_dict(deltasre, orient='index')
    dl_re.rename(columns={0:'rpwr'}, inplace=True)
    dl = pd.concat([dl,dl_re], axis=1)


    fr = pd.DataFrame.from_dict(d, orient='index')
    del offdeltas,offre,deltas,deltasre
    return off, dl, fr, d


def eliminate_offs(dl,off):

    # define threshold for delta OFFs active power, based on average delta ONs
    up_thres = dl['pwr'].mean()
    down_thres = 0.7*up_thres
    print('up_thres, down_thres',up_thres, down_thres)

    # define threshold for delta OFFs reactive power, based on average delta ONs
    up_thres_re = dl['rpwr'].mean()
    down_thres_re = 0.8*up_thres_re
    up_thres_re = up_thres_re + 0.3*up_thres_re 
    print('up_thres_re, down_thres_re',up_thres_re, down_thres_re)

    # keep only OFF events that lie between thresholds
    print('OFFS BEFORE', off)
    off = off.loc[(off['pwr']>-up_thres) & (off['pwr']<-down_thres) & (off['rpwr']>-up_thres_re) & (off['rpwr']<-down_thres_re)]
    off['pwr'] = off['pwr']*(-1)
    off['rpwr'] = off['rpwr']*(-1)
    print('OFFS after', off)
    return off, up_thres, down_thres, up_thres_re, down_thres_re


def clear_offs(off, up_thres, down_thres, up_thres_re, down_thres_re):
    off = off.loc[(off['pwr']>-up_thres) & (off['pwr']<-down_thres) & (off['rpwr']>-up_thres_re) & (off['rpwr']<-down_thres_re)]
    off['pwr'] = off['pwr']*(-1)
    off['rpwr'] = off['rpwr']*(-1)

    off['timedif'] = off.index.to_series().diff().astype('timedelta64[m]')
    #off['timedif'] = off.index.to_series().diff()/np.timedelta64(1, 'm')


    return off


def estimate_duration(df, fr, off):


    # concatenate all information to a df to plot results 

    df2 = pd.concat([df,fr], axis=1)
    df2.rename(columns={0:'fridge'}, inplace=True)
    df2 = pd.concat([df2,off['pwr']], axis=1)
    df2.rename(columns={'pwr':'deltaoff'}, inplace=True)

    df2.loc[df2['fridge']!=1, 'fridge'] = 0

    df2 = df2[['pwrA','fridge', 'deltaoff']]
    df2['pwrA'].fillna(method='bfill', inplace=True)
    df2['deltaoff'].fillna(value=0, inplace=True)

    # create onoff df to calculate duration between on/off
    onoff = pd.DataFrame([])
    onoff = pd.concat([onoff,df2[['pwrA','deltaoff','fridge']]], axis=1)
    onoff['rule'] = np.nan
    onoff.loc[onoff['fridge']>0, 'rule'] = 1
    onoff.loc[onoff['deltaoff']>0, 'rule'] = 0
    onoff = onoff.dropna()
    print(onoff)

    # declare duration of ON and time between off and on
    onoff['timedif'] = onoff.index.to_series().diff().astype('timedelta64[m]')
    #onoff['timedif'] = onoff.index.to_series().diff()/np.timedelta64(1, 'm')
    onoff['timedifON'] = np.nan
    onoff['timedifOFF'] = np.nan
    onoff.loc[(onoff['rule']==0) & (onoff['rule'].shift()==1), 'timedifON'] =onoff['timedif']
    onoff.loc[(onoff['rule']==1) & (onoff['rule'].shift()==0), 'timedifOFF'] =onoff['timedif']
    onoff.drop('timedif',axis=1, inplace=True)
    print(onoff)
    T_on = onoff['timedifON'].quantile(.5)
    T_off = onoff['timedifOFF'].quantile(.5)
    print('Average duration of fridge ON-->OFF:',T_on)
    print('Average duration of fridge OFF-->ON:',T_off)
    
    return df2, onoff, T_on, T_off
 

def obj_fun(T_on, P_up, Q_up, events):
    
    events['dist'] = (events['deltaoff']-P_up)**2+(events['deltaoffR']-Q_up)**2 + (events['timedifON']-T_on)#**2
        
    return events
    

def test_duration(df, fr, off):
    # concatenate all information to a df to plot results 
    
    # reverse sign
    off['pwr'] = off['pwr']*(-1)
    off['rpwr'] = off['rpwr']*(-1)
    
    
    df2 = pd.concat([df,fr], axis=1)
    df2.rename(columns={0:'fridge'}, inplace=True)
    df2 = pd.concat([df2,off[['pwr','rpwr']]], axis=1)
    df2.rename(columns={'pwr':'deltaoff','rpwr':'deltaoffR'}, inplace=True)

    df2.loc[df2['fridge']!=1, 'fridge'] = 0
    
    df2 = df2[['pwrA','fridge', 'deltaoff','deltaoffR']]
    df2['pwrA'].fillna(method='bfill', inplace=True)
    df2['deltaoff'].fillna(value=0, inplace=True)
    df2['deltaoffR'].fillna(value=0, inplace=True)

    # create onoff df to calculate duration between on/off
    onoff = pd.DataFrame([])
    onoff = pd.concat([onoff,df2[['pwrA','deltaoff','fridge','deltaoffR']]], axis=1)
    onoff['rule'] = np.nan
    onoff.loc[onoff['fridge']>0, 'rule'] = 1
    onoff.loc[onoff['deltaoff']>0, 'rule'] = 0
    onoff = onoff.dropna()

    # declare duration of ON and time between off and on
    onoff['timedif'] = onoff.index.to_series().diff().astype('timedelta64[m]')
    onoff['timedifON'] = np.nan
    onoff['timedifOFF'] = np.nan
    onoff.loc[(onoff['rule']==0) & (onoff['rule'].shift()==1), 'timedifON'] =onoff['timedif']
    onoff.loc[(onoff['rule']==1) & (onoff['rule'].shift()==0), 'timedifOFF'] =onoff['timedif']
    
    if (onoff['rule'].iloc[0]==0 and np.isnan(onoff['timedifON'].iloc[0])):
        onoff['timedifON'].iloc[0]=0

    while not onoff.loc[(onoff['rule']==0) & onoff['timedifON'].isna()].empty: 
       onoff.loc[(onoff['rule']==0) & (onoff['rule'].shift()==0), 'timedifON'] =onoff['timedifON'].shift()+onoff['timedif']
    
    onoff.drop('timedif',axis=1, inplace=True)
    
    return onoff



def fix_missing_events(df, params, start_ts, end_ts):
    
    cycle = params['T_on']+params['T_off']
    df['div'] = round(df['timedif']/cycle)
    if df['div'].iloc[0]>1:
        firstel=1
        print('first element 1')
    else:
        firstel=0
    # count how many cycles are between events
    tmp = df.loc[df['div']>1].copy()

    for i in range(0,tmp.shape[0]):
        if ((firstel==1) & (i==0)):
            indic = pd.date_range( end = tmp.index[i], periods=tmp['div'].iloc[i]+1, freq=str(cycle)+'T', inclusive='left', tz='Europe/Athens')
        else:
            ncycle = int(tmp['timedif'].iloc[i]/tmp['div'].iloc[i])
            # time difference division
            indic = pd.date_range( end = tmp.index[i], periods=tmp['div'].iloc[i], freq=str(ncycle)+'T', inclusive='left', tz='Europe/Athens')

        df = df.append(pd.DataFrame(index=indic))

    df = df.sort_index()
    # check last elements
    lastdiv = round(((end_ts - df.index[-1]).total_seconds()/60)/cycle)
    if lastdiv>1:
        indices = pd.date_range( start = df.index[-1], periods=lastdiv, freq=str(cycle)+'T', inclusive='right', tz='Europe/Athens')
        df = df.append(pd.DataFrame(index=indices))
        df = df.sort_index()

    df = df.drop('div', axis=1)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    #df['timedif'] = df.index.to_series().diff().astype('timedelta64[m]')
    #df['timedif'].iloc[0] = (df.index[0]-start_ts).total_seconds()/60
    
    return df

def fix_exchanged(df, cycle, start_ts, avgToff, avgTon):
    df['divonoff'] = round(df['timedif']/avgTon)
    df['divoffon'] = round(df['timedif']/avgToff)
    # fix 1=>0
    tmp = df.loc[(df['fridge']==0) & (df['fridge'].shift()==1) & (df['divonoff']>1)]    
    for i in range(0,tmp.shape[0]):
        indic = tmp.index[i]-pd.Timedelta(minutes=avgTon)
        df = df.append(pd.DataFrame(index=[indic]))
    df.loc[df['fridge'].isna(), 'fridge'] = 1
    df = df.sort_index()
    df['timedif'] = df.index.to_series().diff().astype('timedelta64[m]')
    df['timedif'].iloc[0] = (df.index[0]-start_ts).total_seconds()/60

    df['div'] = round(df['timedif']/cycle)

    # fix 0=>1
    tmp = df.loc[(df['fridge']==1) & (df['fridge'].shift()==0) & (df['divoffon']>1)]    
    for i in range(0,tmp.shape[0]):
        indic = tmp.index[i]-pd.Timedelta(minutes=avgToff)
        df = df.append(pd.DataFrame(index=[indic]))
    df.loc[df['fridge'].isna(), 'fridge'] = 0
    df = df.sort_index()
    df['timedif'] = df.index.to_series().diff().astype('timedelta64[m]')
    df['timedif'].iloc[0] = (df.index[0]-start_ts).total_seconds()/60

    df['div'] = round(df['timedif']/cycle)
    df.drop(['divonoff','divoffon'], axis=1, inplace=True)
    return df


def fix_consequtive(df,val, cycle, avgTon):
    length = df.shape[0]-2
    # fix consequtive 0
   
    for i in range(0, length):
        if ((df['fridge'].iloc[i]==val) & (df['fridge'].iloc[i+1]==val)): #if consequtive 
            if i==0: # check first element
                div = round(df['timedif'].iloc[i+1]/cycle)+1
                df['timedif'].iloc[i+1] = div*cycle
            else:
                div = round(df['timedif'].iloc[i+1]/cycle)
            
            #print(df.iloc[i:i+2])
            
            if div>0:
                toff = (df['timedif'].iloc[i+1])/div - avgTon
               
                
                ncycle = int(toff+avgTon) # new cycle based on time difference
                # add ONs
                
                if div>1:
                    if val==0:
                        end = df.index[i+1]-pd.Timedelta(minutes=avgTon)
                    else:
                        end = df.index[i+1]-pd.Timedelta(minutes=toff)
                    indic = pd.date_range( end = end, periods=div, freq=str(ncycle)+'T', inclusive='right', tz='Europe/Athens')
                else:
                    if val==0:
                        indic = [df.index[i+1]-pd.Timedelta(minutes=avgTon)]
                    else:
                        indic = [df.index[i+1]-pd.Timedelta(minutes=toff)]
                    
                df = df.append(pd.DataFrame(index=indic))
                if val==0:
                    df.loc[df['fridge'].isna(), 'fridge'] = 1
                else:
                    df.loc[df['fridge'].isna(), 'fridge'] = 0
    
                # add OFFs
                indic = pd.date_range(end = df.index[i+1], periods=div, freq=str(ncycle)+'T', inclusive='left', tz='Europe/Athens')
                df = df.append(pd.DataFrame(index=indic))
                if val==0:
                    df.loc[df['fridge'].isna(), 'fridge'] = 0
                else:
                    df.loc[df['fridge'].isna(), 'fridge'] = 1
            # very close events, less than a cycle, so drop row
            else:
                df = df.drop(df.index[i])
            
    df = df.sort_index()
    return df


def fix_updated(df, params, start_ts, end_ts):
    avgToff = df['timedif'][df['fridge']==1].quantile(.5)
    avgTon = df['timedif'][df['fridge']==0].quantile(.5)
    print('avg time Off, On:', avgToff, avgTon)

    cycle = avgToff+avgTon
    print('average cycle:', cycle)
    df['div'] = round(df['timedif']/cycle)
    # check 1st element
    res1 = np.abs(df['div'].iloc[1] - (df['timedif'].iloc[1]/cycle))
    res2 = np.abs(df['div'].iloc[1] - (df['timedif'].iloc[1]/cycle)+avgToff)
    res3 = np.abs(df['div'].iloc[1] - (df['timedif'].iloc[1]/cycle)+avgTon)

    if min(res1, res2, res3)==res1:
        if (df['timedif'].iloc[1]/cycle)<df['div'].iloc[1]:
            df['fridge'].iloc[0] = 1 - df['fridge'].iloc[1]    
        else:
            df['fridge'].iloc[0] = df['fridge'].iloc[1]
    else:
        df['fridge'].iloc[0] = 1-df['fridge'].iloc[0]
    

    
    df = fix_exchanged(df, cycle, start_ts, avgToff, avgTon)


    df = fix_consequtive(df, 0, cycle, avgTon)
    
            
    #df = df.sort_index()
    df['timedif'] = df.index.to_series().diff().astype('timedelta64[m]')
    df['timedif'].iloc[0] = (df.index[0]-start_ts).total_seconds()/60
    #len = df.shape[0]-1

    df = fix_consequtive(df, 1, cycle, avgTon)

    # fill active power
    avgON = df.loc[df['fridge']==1, 'pwr'].mean()
    avgOFF = df.loc[df['fridge']==0, 'pwr'].mean()

    df.loc[(df['fridge']==1) & (df['pwr'].isna()), 'pwr'] = avgON
    df.loc[(df['fridge']==0) & (df['pwr'].isna()), 'pwr'] = avgOFF
    

    
    return df
    

    


def main(argv):

        
    interval = int(50) # interval in milliseconds
    descriptors = 'pwrA,rpwrA'
    local_tz = pytz.timezone('Europe/Athens')


    #address = 'http://devmeazonthings.westeurope.cloudapp.azure.com:8080'
    address = 'http://localhost:8080' #
    
    
    device = argv[1]
    print('device:', device)
    [devid, devtoken, acc_token] = thingsio.get_dev_info(device, address)
    with open('params/'+device+'_params.json', 'r') as f:
            parameters = json.load(f)
    
    #check if it's Monday, to run the parameters optimization
    
    if datetime.datetime.now().weekday() == 0:
    #if 1:
        print('Fixing parameters based on previous week')
        dt = datetime.datetime.utcnow() # current datetime
        dt = dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
        dt = dt - datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second,microseconds=dt.microsecond)
        end_ts = dt + relativedelta(days = -1)
        start_ts = end_ts + relativedelta(days= -7)
        start_time = str(int(start_ts.timestamp()*1e3))
        end_time = str(int(end_ts.timestamp()*1e3))


        
    
        df = thingsio.read_data(acc_token, devid, address, start_time, end_time, descriptors)
        df = df.resample(str(interval)+'ms').max()
        df = df.dropna()
        df.sort_index(inplace=True)
        df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')
        #df['dif'] = df.index.to_series().diff()/np.timedelta64(1, 'ms')
        
    
        df = df.loc[(df['dif'].shift(-2)<=(3*interval)) | (df['dif'].shift()<=(3*interval))]
        
        df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')
        #df['dif'] = df.index.to_series().diff()/np.timedelta64(1, 'ms')
        
        
        # identify events
        
        df = ed.identify_events(df,interval)
        print('Finished identify events')
        
        # detect fridge on off
        [offdeltas, offre, deltas, deltasre, d] = ed.detect_appliance(df, interval, parameters)
        print('Finished detect_appliance')
        
        # transform dictionaries
        off, dl, fr, d = transform_dicts(offdeltas, offre, deltas, deltasre, d)
        print('Finished transform dicts')
        
        # eliminate off events based on thresholds
        [off, P_up, P_down, Q_up, Q_down] = eliminate_offs(dl,off)
        print('Expected Active up-down:', P_up, P_down)
        print('Expected Reactive up-down:', Q_up, Q_down)
        
        # on off duration
        [_, onoff, T_on, T_off] = estimate_duration(df, fr, off)
        param_dict = {}
        param_dict['P_up'] = np.round(P_up,2)
        param_dict['P_down'] = np.round(P_down,2)
        param_dict['Q_up'] = np.round(Q_up,2)
        param_dict['Q_down'] = np.round(Q_down,2)
        param_dict['T_on'] = T_on
        param_dict['T_off'] = T_off
        
        attdict = {}
        att_ts = int(dt.timestamp()) * 1000
        attdict[att_ts] = param_dict
        thingsio.send_data(address, devtoken, attdict)
        
        myjson = json.dumps(param_dict)
        print(myjson)
        with open('params/'+device+"_profile.json", "w") as outfile:
            outfile.write(myjson)
            
    #Run everyday section to disaggregate fridge
    
    dt = datetime.datetime.utcnow() # current datetime
    dt = dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    dt = dt - datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second,microseconds=dt.microsecond)
    end_ts = dt 
    #start_ts = end_ts + relativedelta(days= -2)
    
    start_ts = end_ts + relativedelta(days= -4)
    #end_ts = start_ts + relativedelta(days= 3)
    
    # store actual previous day
    start_day = start_ts
    end_day = end_ts
    print('start day, end day:', start_day, end_day)
    
    # take a few hours before and after day of interest
    start_ts = start_ts + relativedelta(hours= -4)
    end_ts = end_ts + relativedelta(hours = 4)

    start_time = str(int(start_ts.timestamp()*1e3))
    end_time = str(int(end_ts.timestamp()*1e3))

    
    print(start_ts, end_ts)


    with open('params/'+device+"_profile.json", 'r') as f:
        param_dict = json.load(f)

    with open('params/'+device+'_params.json', 'r') as f:
        parameters = json.load(f)
     
    df = thingsio.read_data(acc_token, devid, address, start_time, end_time, descriptors)
    df = df.resample(str(interval)+'ms').max()
    df = df.dropna()
    df.sort_index(inplace=True)
    df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')
    df = df.loc[(df['dif'].shift(-2)<=(3*interval)) | (df['dif'].shift()<=(3*interval))]
    
    df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')
    
    
    # identify events
    
    df = ed.identify_events(df,interval)
    # detect fridge on off
    [offdeltas, offre, deltas, deltasre, d] = ed.detect_appliance(df, interval, parameters)
    # transform dictionaries
    off, dl, fr, d = transform_dicts(offdeltas, offre, deltas, deltasre, d)
    
    off = clear_offs(off, param_dict['P_up'], param_dict['P_down'], param_dict['Q_up'], param_dict['Q_down'])

    dl = dl.loc[dl['pwr']>0.5*param_dict['P_up']]
    
    dl['timedif'] = dl.index.to_series().diff().astype('timedelta64[m]')
    
    
    # set first row of time dif
    dl['timedif'].iloc[0] = (dl.index[0]-start_ts).total_seconds()/60
    off['timedif'].iloc[0] = (off.index[0]-start_ts).total_seconds() /60

    # fill missing events
    #dl = fix_missing_events(dl, param_dict, start_ts, end_ts)
    #off = fix_missing_events(off, param_dict, start_ts, end_ts)
    
    # merge into one df
    dl['fridge']=1
    dl = pd.concat([dl,off])
    dl.loc[dl['fridge']!=1, 'fridge'] = 0
    dl = dl.sort_index()
    dl = dl[['pwr','fridge']]

    #dl = pd.concat([dl, pd.DataFrame(index=[start_ts])], ignore_index=True)
    #dl = pd.concat([dl, pd.DataFrame(index=[end_ts])], ignore_index=True)
    dl = dl.append(pd.DataFrame(index=[start_ts]))
    dl = dl.append(pd.DataFrame(index=[end_ts]))
    dl = dl.sort_index()
    
    
    
    #if dl['fridge'].iloc[1]==1:
    #    dl['fridge'].iloc[0]=0
    #else:
    #    dl['fridge'].iloc[0]=1
    dl['pwr'].iloc[0] = dl['pwr'].iloc[2]
    
    dl['timedif'] = dl.index.to_series().diff().astype('timedelta64[m]')
    dl = dl.drop(dl[dl['timedif']<1].index)
    dl['timedif'].iloc[0] = (dl.index[0]-start_ts).total_seconds()/60
    

    
    dl = fix_updated(dl, param_dict, start_ts, end_ts)
    dl['timedif'] = dl.index.to_series().diff().astype('timedelta64[m]')
    #print(dl)
    

     # calculate energy consumption
    dl = dl.resample('1S').max()
    dl['fridge'] = dl['fridge'].ffill()
    dl.loc[((dl['fridge']<1) & (dl['fridge'].shift()<1)), 'pwr']=0
    dl['pwr'].interpolate(inplace=True)
    
    dl['energy'] = dl['pwr']/3600
    dl = dl.loc[(dl.index>=start_day) & (dl.index<end_day)]
    energy = pd.DataFrame(dl).resample('1D').sum()
    print('Energy consumption:',energy)
    
    # retrieve estimated energy of previous 7 days and correct energy if exceeding thresholds
    
    time.sleep(60)

    end_ts = end_ts + relativedelta(days= -2)
    start_ts = end_ts + relativedelta(days= -7)
    start_time = str(int(start_ts.timestamp()*1e3))
    end_time = str(int(end_ts.timestamp()*1e3))
    en = thingsio.read_data(acc_token, devid, address, start_time, end_time, 'frnrg')
    
    upthresnrg = 1.3*(en['frnrg'].quantile(0.6))
    downthresnrg = 0.75*(en['frnrg'].quantile(0.6))
    
    avg_nrg = en.loc[(en['frnrg']<upthresnrg) & (en['frnrg']>downthresnrg),'frnrg'].mean()
    energy.loc[(energy['energy']>upthresnrg) | (energy['energy']<downthresnrg), 'energy'] = avg_nrg
    energy['energy'] = 1.1*energy['energy'] 
    print('Updated energy:',energy)


    # write energy telemetry
    energy = energy[['energy']]
    energy.rename(columns={'energy':'frnrg'}, inplace=True)
    energy['ts'] = energy.index
    energy['ts'] = energy.apply(lambda row: int(row['ts'].timestamp()) * 1000, axis=1)
    energy.set_index('ts', inplace=True, drop=True)
    energy = energy.round({'frnrg':2})
    mydict = energy.to_dict('index')

    thingsio.send_data(address, devtoken, mydict)


    dl = dl.resample('1T').max()
    # send telemetry
    dl = dl[['pwr']]
    dl.rename(columns={'pwr':'fridge'}, inplace=True)
    dl['ts'] = dl.index
    dl['ts'] = dl.apply(lambda row: int(row['ts'].timestamp()) * 1000, axis=1)
    dl.set_index('ts', inplace=True, drop=True)
    dl = dl.round({'fridge':2})
    mydict = dl.to_dict('index')
    #print(dl)
    #thingsio.send_data(address, devtoken, mydict)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

