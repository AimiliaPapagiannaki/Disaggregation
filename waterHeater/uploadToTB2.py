import os
import sys
import pandas as pd
import json
import requests
import time
from datetime import date, timedelta, timezone
import datetime
import pytz
pd.set_option('display.max_rows', None)


df1 = df2 = pd.DataFrame()
device = ['102.402.000045', '102.402.000751']
#device = ["", '102.402.000751']
address = 'http://localhost:8080'


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


dfs = []
local_tz = pytz.timezone('Europe/Athens')
for file_path in os.listdir():
    try:
        fDate = file_path.split('_')[1][0:10]
        lDate = file_path.split('_')[1][11:21]
        fDate = datetime.datetime.strptime(fDate, '%Y-%m-%d')
        lDate = datetime.datetime.strptime(lDate, '%Y-%m-%d')
    except:
        continue
    if len(file_path.split('-')) == 6 and (lDate-fDate).days <= 4: #and (file_path == '102.402.000045_2023-11-03-2023-11-12.json' or file_path == '102.402.000751_2023-11-03-2023-11-12.json'):
        if file_path.split('_')[0] == device[0]:
            df1 = pd.concat([df1, pd.read_json(file_path)], axis=0)
        elif file_path.split('_')[0] == device[1]:
            df2 = pd.concat([df2, pd.read_json(file_path)], axis=0)

dfs.extend([df1, df2])
stDatetime = 0
enDatetime = 0
d = 0

for i in dfs:
    if not i.empty:
        print('To upload: ')
        print(i)
        tempDf = i.copy()
        for row in i.itertuples():
            tempDf = pd.concat([tempDf, pd.DataFrame({'avg_value':[0], 'event_start':[int(row.event_start-1000)], 'event_end': [int(row.event_end+1000)]})],ignore_index = True)
        tempDf = tempDf.drop_duplicates()
        tempDf = pd.melt(tempDf, id_vars=['avg_value'], value_vars=['event_start', 'event_end']).drop(['variable'], axis=1)
        tempDf = tempDf.sort_values('value').reset_index(drop=True).rename(columns = {'value': 'ts', 'avg_value': 'values'})[['ts', 'values']]

        i['dur_ms'] = i['event_end'] - i['event_start']
        i['whnrg'] = i['avg_value'] * (i['dur_ms'] / 3600000) # /10000->sec * 60->mins * 60->hours
        i['event_end'] = i['event_end'].astype('int64')
        i['event_start'] = i['event_start'].astype('int64')
        print(i)
        for x in i.itertuples():
            try:
                i.loc[x.Index, 'event_start'] = datetime.datetime.fromtimestamp(int(i.loc[x.Index, 'event_start']) // 1000, local_tz)
                i.loc[x.Index, 'event_end'] = datetime.datetime.fromtimestamp(int(i.loc[x.Index, 'event_end']) // 1000, local_tz)
            except:
                print(x)
        #sys.exit()
        stDatetime = i['event_start'].iloc[0] #.to_pydatetime()
        enDatetime = i['event_end'].iloc[-1] #.to_pydatetime()

        for row in i.itertuples():
            if not row.event_start.date() == row.event_end.date(): #if event is split by the change of a day
                """print(i.iloc[row.Index])
                indx = row.Index
                i.loc[indx, 'event_end'] = row.event_end - timedelta(hours=row.event_end.hour, minutes=row.event_end.minute, seconds=row.event_end.second, microseconds=row.event_end.microsecond)
                i = pd.concat([i.iloc[:indx+1], pd.DataFrame({"avg_value":[row.avg_value],"event_start":[row.event_end - timedelta(hours=row.event_end.hour, minutes=row.event_end.minute, seconds=row.event_end.second, microseconds=row.event_end.microsecond)], "event_end":[row.event_end]}), i.iloc[indx+1:]]).reset_index(drop=True)
                i.loc[indx+1, 'dur_ms'] = (i.loc[indx+1, 'event_end'] - i.loc[indx+1, 'event_start']).total_seconds() * 1000
                i.loc[indx+1, 'whnrg'] = i.loc[indx+1, 'avg_value'] * (i.loc[indx+1, 'dur_ms'] / 3600000)
                i.loc[indx, 'whnrg'] = i.loc[indx, 'whnrg'] - i.loc[indx+1, 'whnrg']
                #print(indx)#"""
            i.loc[row.Index,'event_start'] = i.loc[row.Index, 'event_start'] - timedelta(hours=i.loc[row.Index, 'event_start'].hour, minutes=i.loc[row.Index, 'event_start'].minute, seconds=i.loc[row.Index, 'event_start'].second, microseconds=i.loc[row.Index, 'event_start'].microsecond)
            #i.loc[row.Index, 'event_start'] = i.loc[row.Index, 'event_start'].py_datetime()
        i = i.drop(['avg_value', 'event_end', 'dur_ms'], axis=1)
        i = i.rename({'event_start': 'dates'}, axis=1)
        i['dates'] = i['dates'].astype('str')

        dtsList = list(stDatetime.date() + timedelta(n) for n in range((enDatetime - stDatetime).days + 1))
        allDates = pd.DataFrame({"dates": dtsList})
        #print(allDates)
        for k in allDates.itertuples():
            allDates.loc[k.Index, 'dates'] = datetime.datetime.combine(allDates.loc[k.Index, 'dates'], datetime.datetime.min.time()) #.replace(tzinfo=local_tz)
            #allDates.loc[k.Index, 'dates'] = allDates.loc[k.Index, 'dates'].replace(tzinfo=pytz.utc)
        allDates['dates'] = allDates['dates'].astype('datetime64').dt.tz_localize(local_tz)
        #for k in allDates.itertuples():
        #    allDates.loc[k.Index, 'dates'] = allDates.iloc[k.Index].loc['dates'].to_pydatetime()
        allDates['dates'] = allDates['dates'].astype('str')
        allDates = allDates.merge(i, on='dates', how='outer').fillna(0)
        allDates['dates'] = pd.to_datetime(allDates['dates']) #.astype('datetime64')
        #print(i)
        #print(allDates)
        for k in allDates.itertuples():
            allDates.loc[k.Index, 'dates'] = int(allDates.loc[k.Index, 'dates'].timestamp() * 1e3)

    else:
        print('No data')
        today = datetime.datetime.now(pytz.timezone('Europe/Athens'))
        today = today - timedelta(days=1, hours = today.hour, minutes = today.minute, seconds = today.second, microseconds = today.microsecond)
        allDates = pd.DataFrame({'dates':[today], 'whnrg': [0]})
        tempDf = pd.DataFrame()

    allDates = allDates.groupby('dates').sum()
    allDates = allDates.reset_index()
    allDates = allDates.rename({'dates':'ts','whnrg': 'values'}, axis=1)
    tempJson = json.loads(tempDf.to_json(orient="records"))
    whnrgJson = json.loads(allDates.to_json(orient="records"))
    #print(whnrgJson)
    #print(allDates)
    for l in tempJson:
       l['values'] = {'event_value': l['values']}

    for l in whnrgJson:
       l['values'] = {'whnrg': l['values']}
    #print(allDates)
    jsonToUpload = whnrgJson + tempJson
    [devId, devToken, jwt] = get_dev_info(device[d], address)

    headers = {
        # Already added when you pass json= but not when you pass data=
        #'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Authorization': jwt
    }#"""

    print(jsonToUpload)
    print()
    res = requests.post(address + f'/api/plugins/telemetry/DEVICE/{devId}/timeseries/ANY?scope=ANY', data=json.dumps(jsonToUpload), headers=headers)
    d += 1
