
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import warnings
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
import datetime as dt
import pytz
from dateutil.relativedelta import relativedelta
from scipy import stats
from sklearn.metrics import silhouette_score
import json
import sys
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)


def get_dev_info(device, address):

    r = requests.post(address + "/api/auth/login",
                    json={'username': 'tenant@thingsboard.org', 'password': 'tenant'}).json()

    # acc_token is the token to be used in the next request
    acc_token = 'Bearer' + ' ' + r['token']
    # get devid by serial name
    r1 = requests.get(
        url=address + "/api/tenant/devices?deviceName=" + device,
        headers={'Content-Type': 'application/json', 'Accept': '*/*', 'X-Authorization': acc_token}).json()

    devid = r1['id']['id']
    r1 = requests.get(
        url=address + "/api/device/" + devid + "/credentials",
        headers={'Content-Type': 'application/json', 'Accept': '*/*', 'X-Authorization': acc_token}).json()
    devtoken = r1['credentialsId']

    return devid,devtoken,acc_token


def read_data(acc_token, devid, address, start_time, end_time, descriptors):

    r2 = requests.get(
        url=address + "/api/plugins/telemetry/DEVICE/" + devid + "/values/timeseries?keys=" + descriptors + "&startTs=" + start_time + "&endTs=" + end_time + "&agg=NONE&limit=1000000",
        headers={'Content-Type': 'application/json', 'Accept': '*/*', 'X-Authorization': acc_token}).json()

    if r2:
        df = pd.DataFrame([])

        for desc in r2.keys():
            df1 = pd.DataFrame(r2[desc])
            df1.set_index('ts', inplace=True)
            df1.columns = [str(desc)]

            df1.reset_index(drop=False, inplace=True)
            df1['ts'] = pd.to_datetime(df1['ts'], unit='ms')
            df1['ts'] = df1['ts'].dt.tz_localize('utc').dt.tz_convert('Europe/Athens')
            df1 = df1.sort_values(by=['ts'])
            df1.reset_index(drop=True, inplace=True)
            df1.set_index('ts', inplace=True, drop=True)

            df = pd.concat([df, df1], axis=1)

        if df.empty:
            df = pd.DataFrame([])
        else:
            for col in df.columns:
                df[col] = df[col].astype('float64')
    else:
        df = pd.DataFrame([])
        print('Empty json!')

    return df


def identify_events(df, interval):
# Run dataframe row by row to identify start of each event
    ind=0
    df['ind'] = np.nan
    event = 0
    thres = 3*interval+1
    for i in range(0, int(df.shape[0]-2)):
        if  event==0:
            # start of event
            if (df['dif'].iloc[i]>thres and df['dif'].iloc[i+2]<=thres): #and df['dif'].iloc[i+1]>thres):
                event=1
                #df['ind'].iloc[i] = int(ind)

        elif event==1:
            # end of event
            if (df['dif'].iloc[i]>thres and df['dif'].iloc[i-1]<thres):
                #df['ind'].iloc[i] = int(ind)
                event = 0
                ind += 1
                if (df['dif'].iloc[i]>thres and df['dif'].iloc[i+2]<=thres):
                    #df['ind'].iloc[i] = int(ind)
                    event=1
            else:
                df['ind'].iloc[i] = int(ind)
    #remove last element of group here
    #df = df.dropna()
    return df#"""


def joinFreqEvents(df, interval_ms):
    df2 = df.copy()
    df = df.dropna()
    for i in df.dropna(subset = ['ind']).ind.unique():
        if i == 0:
            lastTs = (df.loc[df['ind'] == i].iloc[-1]).name.to_pydatetime()
            lastVal = df.loc[df['ind'] == i].pwrA.mean()
            prevInd = i
            continue

        firstTs = (df.loc[df['ind'] == i].iloc[0]).name.to_pydatetime()

        firstVal = df.loc[df['ind'] == i].pwrA.mean()

        if ((firstTs - lastTs).total_seconds() * 1000 < interval_ms) and ((abs(firstVal - lastVal) < 1000)):
            df2['ind'].loc[df2['ind'] == i] = prevInd
        else:
            prevInd = i

        lastTs = (df.loc[df['ind'] == i].iloc[-1]).name.to_pydatetime()
        lastVal = df.loc[df['ind'] == i].pwrA.mean()

    return df2


def dropShortEvents(df, interval_ms):
    df2 = pd.DataFrame()
    for i in df.dropna(subset = ['ind']).ind.unique():
        firstTs = (df.loc[df['ind'] == i].iloc[0]).name.to_pydatetime()
        lastTs = (df.loc[df['ind'] == i].iloc[-1]).name.to_pydatetime()
        #print((lastTs - firstTs).total_seconds() * 1000 > interval_ms)
        if (lastTs - firstTs).total_seconds() * 1000 > interval_ms:
            df2 = pd.concat([df2, df.loc[df['ind'] == i]])
    return df2

def dropFluctuatingEvents(df, minAvgOfCanonEvent, stdDeviationLim):
    df2 = pd.DataFrame()
    for i in df.dropna(subset = ['ind']).ind.unique():
        if df['pwrA'].loc[df['ind'] == i].mean() >= minAvgOfCanonEvent:
            df['deviation'] = abs(df['pwrA'].loc[df['ind'] == i] - df['pwrA'].loc[df['ind'] == i].mean())
            if (df['deviation'].loc[df['ind'] == i].mean() < stdDeviationLim):# or df.loc[df['ind'] == i].pwrA.mean() > 4000):
                df2 = pd.concat([df2, df.loc[df['ind'] == i]], axis=0)
    return df2

def joinFinalEvents(df, interval_ms):
    df2 = df.copy()
    df = df.dropna()
    try:    allInds = df.dropna(subset = ['ind']).ind.unique()
    except: allInds = []
    for i in allInds:
        if i == allInds[0]:
            lastTs = (df.loc[df['ind'] == i].iloc[-1]).name.to_pydatetime()
            lastAvg = df.loc[df['ind'] == i].pwrA.mean()
            prevInd = i
            continue
        firstTs = (df.loc[df['ind'] == i].iloc[0]).name.to_pydatetime()
        offset = 1000
        if ((firstTs - lastTs).total_seconds() * 1000 < interval_ms and \
            (df.loc[df['ind'] == i].pwrA.mean() > (lastAvg - offset) and (df.loc[df['ind'] == i].pwrA.mean() < (lastAvg + offset)))):
            df2['ind'].loc[df2['ind'] == i] = prevInd
        else:
            prevInd = i

        lastTs = (df.loc[df['ind'] == i].iloc[-1]).name.to_pydatetime()
        lastAvg = df.loc[df['ind'] == i].pwrA.mean()
    return df2

def fillNANInEvent(df):
    df2 = pd.DataFrame()
    df3 = df.reset_index()
    for i in df.dropna(subset = ['ind']).ind.unique():
        startDate = df.loc[df['ind'] == i].iloc[0].name.to_pydatetime()
        if i > 0:
            df2 = pd.concat([df2, df3.loc[(df3['ts'] < startDate) & (df3['ts'] > endDate) & df3['ind'].isna()]])
            df2['ind'].loc[df2['ind'].isna()] = prevInd
        endDate = df.loc[df['ind'] == i].iloc[-1].name.to_pydatetime()
        tempDf = df3.loc[(df3['ts'] >= startDate) & (df3['ts'] <= endDate)]
        tempDf['ind'] = i
        prevInd = i
        df2 = pd.concat([df2, tempDf], axis=0)
    return df2.set_index('ts')

def untangleOverlappingEvents(df):
    for i in df.itertuples():
        if i.Index == 0:
            endDt = i.event_end.to_pydatetime()
            prevValue = i.avg_value
            continue
        startDt = i.event_start.to_pydatetime()
        #print((startDt - endDt).total_seconds())
        if (startDt - endDt).total_seconds() < 30 and i.avg_value > 2 * prevValue:
            df['avg_value'].iloc[i.Index] -= prevValue
            print('index', i.Index)
        endDt = i.event_end.to_pydatetime()
        prevValue = i.avg_value
    return df

def classify(events):
    n_samples = len(events)
    if n_samples == 0:
        return 0
    if n_samples == 1:
        events['cluster'] = 0
        return events
    if n_samples == 2:
        events['cluster'] = 0
        if abs(events['avg_value'].iloc[0] - events['avg_value'].iloc[1]) > 600:
            events.iloc[1, events.columns.get_loc('cluster')] = 1
        return events.reset_index()

    numClusters = list(range(2, n_samples))
    maxSilhouette = 0
    for i in numClusters:
        hac = AgglomerativeClustering(n_clusters = i, affinity='euclidean', linkage='ward')
        hac.fit(events[events.columns[0:2]])
        membership = hac.labels_
        tempSilhouette = silhouette_score(events[events.columns[0:2]], membership)
        if tempSilhouette > maxSilhouette:
            maxSilhouette = tempSilhouette
            events['cluster'] = membership
            print('clusters:', i, ', silhouette:', maxSilhouette)
    return events.reset_index(drop = True)


def getIdxOfShortLargeEvents(df, largePwrA, maxMinutesOfEvent): #largePwrA should be around 1.4 * highV
    df2 = df.copy()
    df2 = df2.loc[df2['avg_value'] > largePwrA]
    df2['diff'] = (df2['event_end'] - df2['event_start']).astype('timedelta64[s]') / 60
    df2 = df2.loc[df2['diff'] < maxMinutesOfEvent]
    return df2.index


def mergeEventsSplitByLargerOnes(df2, idxes, interval):
    for i in idxes:
        try:
            if abs(df2.iloc[i-1].avg_value - df2.iloc[i+1].avg_value) < 100 and \
                (df2.iloc[i+1].event_start - df2.iloc[i-1].event_end).seconds / 60 <= interval:
                df2.loc[i-1, 'event_end'] = df2.iloc[i+1].event_end
                df2.loc[i-1, 'avg_value'] = (df2.iloc[i+1].avg_value + df2.iloc[i-1].avg_value) / 2
                df2.loc[i, 'avg_value'] = df2.iloc[i].avg_value - (df2.iloc[i+1].avg_value + df2.iloc[i-1].avg_value) / 2
                df2 = pd.concat([df2.iloc[:i+1], df2.iloc[i+2:]], axis=0)
        except:
            pass
    return df2.reset_index(drop = True)


address = 'http://devmeazonthings.westeurope.cloudapp.azure.com:8080'
#devId = 'b6fabb90-48b1-11ed-bdcf-11f9ad7246f5'

devs = ['102.402.000045', '102.402.000751']
local_tz = pytz.timezone('Europe/Athens')


dt = datetime.utcnow() # current datetime
dt = dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
dt = dt - timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
end_ts = dt
start_ts = end_ts + relativedelta(days = -1)

# store actual previous day
start_day = start_ts
end_day = end_ts

firstTime = False
try:
    if sys.argv[1]:
         firstTime = True
except:
    pass

if firstTime:
    start_ts = end_ts + relativedelta(days = -6)
    devs = [devs[0]]

#get one more hour before start of day and one more after end of day
st = int(start_ts.timestamp()*1e3) - 3600000 #60*60*1000
en = int(end_ts.timestamp()*1e3) + 3600000 #60*60*1000

#delete this******************
#st = 1703455200000
#en = 1704712860000
#devs = [devs[0]]
#*****************************

print(st)
print(en)
#sys.exit()
descriptors = 'pwrA,rpwrA,pwrB,rpwrB,pwrC,rpwrC'

for device in devs:
    key = 'pwrB'
    if device == '102.402.000045':
        key = 'pwrC'

    [devid, devtoken, acc_token] = get_dev_info(device, address)
    df = pd.DataFrame()
    for i in range(st, en, 86400000): #24 * 60 * 60000 = 86400000
        df = pd.concat([df, read_data(acc_token, devid, address, str(i), str(i + 86399999), descriptors)], axis=0)
    print('Data collected')
    startingDf = df.reset_index()
    df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')

    df = identify_events(df, 100)
    #df = eventToLargeCloseValues(df)
    df = joinFreqEvents(df, 5*60000)
    df = fillNANInEvent(df)
    df = dropShortEvents(df, 2*60000)
    df = dropFluctuatingEvents(df, 2000, 1000)
    df = joinFinalEvents(df, 2*60000)
    events = {}
    try:
        j = 0
        clusteredDf = pd.DataFrame()
        if df.empty:
            print('empty')
            continue

        for i in df.ind.unique():
            df2 = df.loc[df['ind'] == i]
            eventStart = df2.iloc[0].name.to_pydatetime()
            eventEnd = df2.iloc[-1].name.to_pydatetime()

            """hoursOffset = 2
            startingDf = startingDf.loc[(startingDf['ts'] <= eventEnd+timedelta(hours=hoursOffset)) & (startingDf['ts'] >= eventStart-timedelta(hours=hoursOffset))]

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            plt.plot(startingDf['ts'], startingDf['pwrA'], label='pwrA')
            plt.plot(startingDf['ts'], startingDf[key], color='green', label=f'{key}: Water Heater', alpha=.6)
            plt.gcf().autofmt_xdate()
            vlinesAlpha = .4
            vlinesWidth = 3.5
            plt.axvline(x = eventStart, color = 'orange', label = 'Event start', linestyle='--', alpha=vlinesAlpha, linewidth=vlinesWidth)
            plt.axvline(x = eventEnd, color = 'red', label = 'Event end', linestyle='--', alpha=vlinesAlpha, linewidth=vlinesWidth)

            j += 1
            plt.legend()
            plt.grid()
            plt.show()#"""

            df2 = df2.reset_index()
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(5), list(df2['pwrA'].iloc[0:5]))
            clusteredDf = pd.concat([clusteredDf, pd.DataFrame({'avg_value': df2.pwrA.mean(), 'slope': slope, 'event_start': eventStart, 'event_end': eventEnd, 'rpwrA': df2.rpwrA.abs().mean()}, index=[j])], axis = 0)
            clusteredDf = clusteredDf.reset_index(drop = True)
            #clusteredDf['slope'] = 0

    except Exception as e:
        print(e)

    if clusteredDf.empty:
        #clusteredDf = pd.DataFrame({'avg_value': [0], 'event_start': [st], 'event_end': [en]})
        #clusteredDf.to_json(f'{device}_{str(start_ts.date())}-{str(end_ts.date())}_noEv.json')
        #continue
        sys.exit()

    clusteredDf = mergeEventsSplitByLargerOnes(clusteredDf, getIdxOfShortLargeEvents(clusteredDf, 6200, 4), 7)

    if firstTime:

        clusteredDf['slope'].loc[clusteredDf['slope'].isna()] = clusteredDf['slope'].mean() #some events have NaN value for slope. Here it is replaced by the mean of the rest
        #clusteredDf = untangleOverlappingEvents(clusteredDf)
        clusteredDf = classify(clusteredDf).reset_index(drop = True)
        print(clusteredDf)
        waterHeaterCluster = int(input('Which cluster is the water heater assigned to? (integer): '))
        meanPwrAWHeater = clusteredDf['avg_value'].mean()
        clusteredDf = clusteredDf.loc[clusteredDf['cluster'] == waterHeaterCluster].drop(['cluster'], axis = 1).reset_index(drop = True)
        percOfAcceptableDeviation = .1 #10%
        wHeaterValueRange = {'lowV': meanPwrAWHeater * (1-percOfAcceptableDeviation), 'highV': meanPwrAWHeater * (1+percOfAcceptableDeviation)}
    else:
        with open(f'/home/azureuser/gen_models/waterHeater/{device}_valueRange.json', 'r') as rangeFile:
            readFile = json.loads(rangeFile.read())
            clusteredDf = clusteredDf.loc[(clusteredDf['avg_value'] <= float(readFile['highV'])) & (clusteredDf['avg_value'] >= float(readFile['lowV']))].reset_index(drop = True)

    clusteredDf = joinFinalEvents(clusteredDf, 5*60000)
    clusteredDf = clusteredDf.drop(['slope', 'rpwrA'], axis = 1).reset_index(drop = True)
    clusteredJson = clusteredDf.to_json(orient = "index")
    clusteredDf.to_json(f'{device}_{str(datetime.fromtimestamp(st//1000, local_tz).date())}-{str((datetime.fromtimestamp(en//1000, local_tz)).date())}.json')

    if firstTime:
        with open(f'/home/azureuser/gen_models/waterHeater/{device}_valueRange.json', 'w') as f:
            json.dump(wHeaterValueRange, f)
    print(f'{device}_{str(datetime.fromtimestamp(st//1000, local_tz).date())}-{str((datetime.fromtimestamp(en//1000, local_tz)).date())}')
    """with open(f'/home/azureuser/gen_models/waterHeater/{device}_{str(start_ts.date())}-{str(end_ts.date())}.json', 'w') as f:
        json.dump(clusteredJson, f)#"""

