#from scipy.signal import argrelextrema
import eventDetection as ed
import thingsio
import requests
import json
import pandas as pd
import numpy as np
import pytz
import datetime
from dateutil.tz import gettz
from dateutil.relativedelta import relativedelta
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings("ignore")
import time
import sys


def fine_tune(df, interval):
    
    def objective(parameters):
        d = {}
        d['half'] = parameters[0]['half']
        d['coef3'] = parameters[1]['coef3']
        d['coef4'] = parameters[2]['coef4']
        [_, _, deltas, _, _] = ed.detect_appliance(df, interval, d)
                
        # transform dictionaries
        #off, dl, fr, d = transform_dicts(offdeltas, offre, deltas, deltasre, d)
        fridge_on = pd.DataFrame.from_dict(deltas, orient='index')
        fridge_on.rename(columns={0:'pwr'}, inplace=True)
        fridge_on['dur'] = fridge_on.index.to_series().diff().astype('timedelta64[m]')
        fridge_on = fridge_on.iloc[1:]
        
        if fridge_on.shape[0]>1:
            loss = fridge_on['dur'].std() + fridge_on['dur'].mean()
        else:
            loss=1e10
        print('Loss:', loss)
        return {'loss': loss, 'parameters': parameters, 'status': STATUS_OK}

    
    space = [{'half':hp.choice('half',[3,5,7,10])},
            {'coef3':hp.choice('coef3',[2,3,4.5, 5, 5.5])},
            {'coef4':hp.choice('coef4',[0.8,0.9,1,1.1,1.2, 1.3])}
            ]
    
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=25)
    best_trial = sorted(trials.results, key=lambda x: x['loss'])
    
    best_space = best_trial[0]['parameters']
    best_loss = best_trial[0]['loss']
 
    return best_space

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


  


def main(argv):
    firstrun=0
    interval = int(50) # interval in milliseconds
    descriptors = 'pwrA,rpwrA'
    local_tz = pytz.timezone('Europe/Athens')

    '''DinRails to take into consideration for testing:
    102.402.000751, # moutsios
    102.402.000447, 
    102.402.000045, # tavoularis
    102.402.000201 # stelios '''
    

    #device = '102.402.000045'
    device = argv[1]
    #address = 'http://devmeazonthings.westeurope.cloudapp.azure.com:8080'
    #address = 'http://51.103.232.223:8080' # stelios address
    address = 'http://localhost:8080'
    [devid, devtoken, acc_token] = thingsio.get_dev_info(device, address)
    
       

    dt = datetime.datetime.utcnow() # current datetime
    dt = dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    dt = dt - datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second,microseconds=dt.microsecond)
    end_ts = dt + relativedelta(days = -1)
    start_ts = end_ts + relativedelta(days= -7)
    start_time = str(int(start_ts.timestamp()*1e3))
    end_time = str(int(end_ts.timestamp()*1e3))

    #start_time = '1673359200000'
    #end_time = '1673429494000'
    print(start_time, end_time)


    # slice time range in slots of 12 hours, to reduce data volume in http requests
    timethres = 12*3600000
    svec = np.arange(int(start_time),int(end_time),timethres)
    df = pd.DataFrame([])
    
    for st in svec:
        en = st+timethres-1
        
        if int(end_time)-en<=0: en = int(end_time)
        tmp = thingsio.read_data(acc_token, devid, address,  str(st), str(en), descriptors)
        tmp = tmp.resample(str(interval)+'ms').max()
        tmp = tmp.dropna()
        df = pd.concat([df,tmp])
    df.sort_index(inplace=True)
        
    #df = thingsio.read_data(acc_token, devid, address, start_time, end_time, descriptors)
    #df = df.resample(str(interval)+'ms').max()
    #df = df.dropna()
    #df.sort_index(inplace=True)
    df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')
    df = df.loc[(df['dif'].shift(-2)<=(3*interval)) | (df['dif'].shift()<=(3*interval))]
    
    df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')
    
    
    # identify events
    
    df = ed.identify_events(df,interval)
    print('Finished identify events')
    
    # detect fridge on off
    best_space = fine_tune(df, interval)
    print('Best space:',best_space)

    d = {}
    d['half'] = best_space[0]['half']
    d['coef3'] = best_space[1]['coef3']
    d['coef4'] = best_space[2]['coef4']
    myjson = json.dumps(d)
    
    with open('params/'+device+"_params.json", "w") as outfile:
        outfile.write(myjson)

    # TEST
    [offdeltas, offre, deltas, deltasre, d] = ed.detect_appliance(df, interval, d)

    fridge_on = pd.DataFrame.from_dict(deltas, orient='index')
    fridge_on.rename(columns={0:'pwr'}, inplace=True)
    fridge_on['dur'] = fridge_on.index.to_series().diff().astype('timedelta64[m]')
    avg_pwr = fridge_on['pwr'].quantile(.5)
    avg_dur = fridge_on['dur'].quantile(.5)
    print(fridge_on)
    print('Average power when turning ON:', avg_pwr)
    print('Average duration when turning ON:', avg_dur)
    
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))