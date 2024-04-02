import sys
import pandas as pd
import requests
import numpy as np
import pytz
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)



tz_local = pytz.timezone('Europe/Athens')

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

def identify_events_vals(df, stepDiffInPwrA):
# Run dataframe row by row to identify start of each event
    df2 = df.copy()
    df2['ind'] = np.nan

    ind = 0
    event = 0
    perc = .2
    dist = 1
    for i in range(0, int(df2.shape[0]-dist)):
        if event == 0:
            event = 1
            df2['ind'].iloc[i] = ind

        elif event == 1:
            if df2['pwrA'].iloc[i] - stepDiffInPwrA >= df2['pwrA'].iloc[i+dist] or  df['pwrA'].iloc[i] + stepDiffInPwrA < df2['pwrA'].iloc[i+dist]:
                event = 0
                df2['ind'].iloc[i] = ind
                ind += 1
            else:
                df2['ind'].iloc[i] = ind
    df2['ind'].iloc[-1] = df2['ind'].iloc[-2]
    return df2

def identifyONOFFs(df, onOffDiff):
    idx = np.unique(df.ind.values, return_index=1)[1] #indexes of first elements per ind group

    df['event'] = np.nan
    for i in idx[1:]:
        if (df['pwrA'].iloc[i]+onOffDiff < df['pwrA'].iloc[i-2] and df['pwrA'].iloc[i]+onOffDiff < df['pwrA'].iloc[i-4] and df['pwrA'].iloc[i+5] < df['pwrA'].iloc[i-2]):
            df['event'].iloc[i] = 'off'
        elif (df['pwrA'].iloc[i]+onOffDiff < df['pwrA'].iloc[i+2] and df['pwrA'].iloc[i]+onOffDiff < df['pwrA'].iloc[i+5] or df['pwrA'].iloc[i-5] < df['pwrA'].iloc[i+2]):
            df['event'].iloc[i] = 'on'
    return df

def isClose(v1, v2, perc):
    if (v1 <= v2*(1+perc)) and (v1 >= v2*(1-perc)):
        return True
    else:
        return False

def isOk(prevEvType, df, on_s, off_s, acceptableDurErr):
    duration = (df.ts.iloc[-1] - df.ts.iloc[0]).total_seconds()

    if not (tempDf.event.iloc[0] == prevEvType):
        if prevEvType == 'on' and isClose(duration, off_s, acceptableDurErr):
                return True
        elif prevEvType == 'off' and isClose(duration, on_s, acceptableDurErr):
                return True
        else:
            return False
    else:
        return True

def delEventsSmallerThan(df, numLines, colName):
    temp = df.groupby([colName])[colName].count().reset_index(name='counts')
    temp = temp[colName].loc[temp['counts'] < numLines].to_list()
    df = df.loc[~df[colName].isin(temp)].reset_index(drop = True)
    return df

def getAlternateOnOffs(df):
    prev = None
    currGroupNum = 0
    timeDiff = False
    for i in df.itertuples():
        try:
            timeDiff = (i.ts - prev_ts).total_seconds()
        except:
            pass
        if i.event == prev or timeDiff >= 60:
            currGroupNum += 1
        df.loc[i.Index, 'altEv_Group'] = currGroupNum
        prev = i.event
        prev_ts = i.ts

def mergeEvents(df, diff_minutes):
    prevLastTs = df.groupby("altEv_Group")['ts'].last().to_list()[0].to_pydatetime()
    lastGroup = df.iloc[0].altEv_Group
    for i in df.altEv_Group.unique()[1:]:
        tempDf = df.loc[df['altEv_Group'] == i]
        firstTs = tempDf.ts.iloc[0].to_pydatetime()
        if (firstTs - prevLastTs).total_seconds() <= 60*diff_minutes:
            df['altEv_Group'].iloc[tempDf.index] = lastGroup
        else:
            lastGroup = tempDf['altEv_Group'].iloc[0]
        prevLastTs = tempDf.ts.iloc[-1]
    return df

def delShortEvents(df, evDuration_s):

    for i in df.itertuples():
        currDuration = (i.endTs.to_pydatetime() - i.startTs.to_pydatetime()).total_seconds()
        #print(currDuration)
        if currDuration < evDuration_s:
            #df = pd.concat([df.iloc[:i.Index+1], df.iloc[i.Index+2:]], axis=0)
            df.drop(labels=None, axis=0, index=i.Index, columns=None, level=None, inplace=True, errors='raise')
    return df

def keepReqData(df):
    df2 = pd.DataFrame()
    for i in df.altEv_Group.unique():
        tempDf = df.loc[df['altEv_Group'] == i]
        data = {'startTs': [tempDf.ts.iloc[0].to_pydatetime()], 'endTs': [tempDf.ts.iloc[-1].to_pydatetime()], 'avgPwrA': [tempDf.avgPwrA.mean()]}
        df2 = pd.concat([df2, pd.DataFrame(data)], axis=0)
    return df2


devId = '102.402.000045' #'102.402.000751'

descriptors = 'pwrA'
address = 'http://devmeazonthings.westeurope.cloudapp.azure.com:8080'


local_tz = pytz.timezone('Europe/Athens')


dt = datetime.datetime.utcnow() # current datetime
dt = dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
dt = dt - timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
end_ts = dt
start_ts = end_ts + relativedelta(days = -1)

# store actual previous day
start_day = start_ts
end_day = end_ts

#get one more hour before start of day and one more after end of day
st = int(start_ts.timestamp()*1e3) - 3600000 #60*60*1000
en = int(end_ts.timestamp()*1e3) + 3600000 #60*60*1000


[devid, devtoken, acc_token] = get_dev_info(devId, address)

#______________Add fixed timestamps here_________________
"""st = 1698571800000 #October 29, 2023 11:30:00 PM
en = 1698575100000 #October 29, 2023 12:25:00 PM"""

"""st = 1700058815000 #November 15, 2023 4:33:35 PM
en = 1700060621000 #November 15, 2023 5:03:41 PM"""

"""st = 1700925000000 #November 25, 2023 5:10:00 PM
en = 1700931600000 #November 25, 2023 7:00:00 PM #"""

"""st = 1703421000000 #December 24, 2023 2:30:00 PM
en = 1703428200000 #December 24, 2023 4:30:00 PM #"""

"""st = 1703701800000 #December 27, 2023 8:30:00 PM
en = 1703707200000 #December 27, 2023 10:00:00 PM #"""

"""st = 1707999827000 #February 15, 2024 2:23:47 PM
en = 1708125827000 #February 17, 2024 1:23:47 AM #"""

############for 045##################################
"""st = 1709125200000 #February 28, 2024 3:00:00 PM
en = 1709128800000 #February 28, 2024 4:00:00 PM v """

"""st = 1709157540000 #February 28, 2024 11:59:00 PM
en = 1709159400000 #February 29, 2024 12:30:00 AM v """

"""st = 1708891200000 #February 25, 2024 10:00:00 PM
en = 1708894800000 #February 25, 2024 11:00:00 PM v """

"""st = 1709280600000 #March 1, 2024 10:10:00 AM
en = 1709281200000 #March 1, 2024 10:20:00 AM """

"""st = 1710756010000 #March 18, 2024 12:00:10 PM
en = 1710763210000 #March 18, 2024 2:00:10 PM FALSE POSITIVE"""
#________________________________________________________

df = pd.DataFrame()

#collecting data
df = read_data(acc_token, devid, address, str(st), str(en), descriptors)
if df.empty:
    sys.exit()

df = df.loc[df['pwrA'] > 0]
df = identify_events_vals(df, 300) #adds column 'ind' which categorizes rows into groups representing events
df = identifyONOFFs(df, 0).reset_index() #events are devided into on or off
df = delEventsSmallerThan(df, 3, 'ind') #groups that consist of 3 rows or fewer are deleted
#df's columns are: | ts | pwrA | ind |


eventsDf = pd.DataFrame()
prevEvType = 'off'

#This for loop will create a new DataFrame based on df, which will have events deemed as corresponding to the rules we've set or not
for i in df.ind.unique():
    tempDf = df.loc[df['ind'] == i]
    lst = [[tempDf['ts'].iloc[0], isOk(prevEvType, tempDf, 9, 6, .99), tempDf['event'].iloc[0], tempDf.pwrA.mean()]]
    #isOk() is a function that deems events as ok or not based on their duration and their type of event (on/off) given the previous event's type
    tempDf = pd.DataFrame(lst, columns =['ts', 'ok', 'event', 'avgPwrA'])
    eventsDf = pd.concat([eventsDf, tempDf], axis=0)
    prevEvType = tempDf.event.iloc[0]#"""

# eventDf's columns: | ts | ok | event |
eventsDf = eventsDf.loc[eventsDf['ok'] == True].reset_index(drop=True)
getAlternateOnOffs(eventsDf) #adds column named altEv_Group where alternating on/offs are grouped together

eventsDf = delEventsSmallerThan(eventsDf, 2, 'altEv_Group') #delete all alternating groups smaller than 3 rows

if eventsDf.empty:
    sys.exit()

eventsDf = mergeEvents(eventsDf, 2.5) #if two or more groups are as close as 1min or less, they're considered one group
#eventsDf = delEventsSmallerThan(eventsDf, 20, 'altEv_Group') #delete all alternating groups smaller than 20 rows
eventsDf = eventsDf.drop(['ok'], axis=1)
# eventDf's columns: | ts | event | altEv_Group | pwrA |
#print(eventsDf)
#keeping the data we need: start ts of event, end ts, and AVG pwrA. So its one row for each time the microave was turned on
eventsDf = keepReqData(eventsDf) #eventDf's columns: | startTs | endTs | avgPwrA
print(eventsDf)
eventsDf = delShortEvents(eventsDf.reset_index(drop=True), 60)

print(st, "---", en)
eventsDf = eventsDf.reset_index(drop = True)
print(eventsDf)
if not eventsDf.empty:
    eventsDf.to_json(f'/home/azureuser/gen_models/microwave/{devId}_{str(datetime.datetime.fromtimestamp(st//1000, local_tz).date())}-{str((datetime.datetime.fromtimestamp(en//1000, local_tz)).date())}.json')

#sys.exit()
