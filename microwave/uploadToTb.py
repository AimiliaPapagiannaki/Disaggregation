import pandas as pd
import os
import json
import datetime
##import tzlocal
import requests
import datetime
import pytz
from datetime import timedelta, timezone
import time
import sys

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


#print('kostas')
address = 'http://devmeazonthings.westeurope.cloudapp.azure.com:8080'
devId = '102.402.000045'

[devId, devToken, jwt] = get_dev_info(devId, address)
headers = {
    # Already added when you pass json= but not when you pass data=
    #'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Authorization': jwt
}

list2 = []
finalList = []

dir = "/home/azureuser/gen_models/microwave/"

for item in os.listdir(dir):

    if len(item.split("-")) == 6:

         x = item.split("_")[1].split(".")[0].split("-")
         x = [int(y) for y in x]
         startDate = datetime.datetime(*x[:3])
         endDate = datetime.datetime(*x[3:])
         if (endDate - startDate).total_seconds()/60 <= 2*24*60: #Duration of event < 2 days. <---Remember to change this
              with open(dir+"/"+item, "r") as f:
                  jsonData = json.loads(f.read())
              dataDf = pd.DataFrame.from_dict(jsonData, orient='columns')
              tempJson = json.loads(dataDf.to_json(orient="records"))
              #print(tempJson)
              #[{'ts': 1700777742426, 'values': {'event_value': 0.0}},  ...
              for j in tempJson:
                  finalList.append({"ts": j["startTs"] - 10, 'values': {"mi_event_value": 0}})
                  finalList.append({"ts": j["startTs"], 'values': {"mi_event_value": j["avgPwrA"]}})
                  finalList.append({"ts": j["endTs"], 'values': {"mi_event_value": j["avgPwrA"]}})
                  finalList.append({"ts": j["endTs"] + 10, 'values': {"mi_event_value": 0}})
              #print(finalList)
              #sys.exit()
              #dataDf.loc['1', "startTs"] -= (24*60*60000)
              for i in dataDf.itertuples():
                  #dataDf.loc[i.Index, "startTs"] -= (24*60*60000)
                  stDt = datetime.datetime.fromtimestamp(i.startTs/1000, pytz.timezone('Europe/Athens'))
                  enDt = datetime.datetime.fromtimestamp(i.endTs/1000, pytz.timezone('Europe/Athens'))
                  stDt = stDt - timedelta(hours=stDt.hour, minutes=stDt.minute, seconds=stDt.second, microseconds=stDt.microsecond)
                  enDt = enDt - timedelta(hours=enDt.hour, minutes=enDt.minute, seconds=enDt.second, microseconds=enDt.microsecond)
                  offset1 = int(stDt.astimezone(pytz.timezone('Europe/Athens')).utcoffset().total_seconds()/3600)
                  offset2 = int(enDt.astimezone(pytz.timezone('Europe/Athens')).utcoffset().total_seconds()/3600)
                  dataDf.loc[i.Index, "startDay"] = stDt.replace(tzinfo=timezone(timedelta(hours=offset1))).timestamp() * 1000
                  dataDf.loc[i.Index, "endDay"] = enDt.replace(tzinfo=timezone(timedelta(hours=offset2))).timestamp() * 1000
                  #print(dataDf.loc[i.Index, "startDay"].date()- timedelta(hours=0, minutes=0, seconds=0, microseconds=0))
                  #dataDf.loc[i.Index, "startDay"] = datetime.datetime.timestamp(stDt) * 1000
                  #dataDf.loc[i.Index, "endDay"] = int(((datetime.datetime.timestamp(enDt) * 1000)))
              #print(dataDf.head())
              #sys.exit()
              dataDf['startDay'] = dataDf['startDay'].astype('int64')
              dataDf['endDay'] = dataDf['endDay'].astype('int64')
              dataDf['diff'] = (dataDf['endTs'] - dataDf['startTs'])/1000

              for i in dataDf.itertuples():
                  if i.startDay != i.endDay:
                      dataDf.loc[i.Index, "diff"] = (i.endDay - i.startTs)#.total_seconds()
                      dataDf = pd.concat([dataDf, pd.DataFrame({"startTs": [i.endTs], "endTs": [i.endTs], "avgPwrA": [i.avgPwrA], "startDay": [i.endDay], "endDay": [i.endDay], 'diff': [i.endTs - i.endDay]})], axis=0)
              print(dataDf)

              dataDf = dataDf.sort_values(by = ['startTs']).reset_index(drop = True).drop(["startTs", "endTs", "endDay"], axis=1).rename({"startDay":"day"}, axis=1)
              dataDf["value"] = (dataDf["diff"]) * dataDf["avgPwrA"]
              dataDf = dataDf.drop(["avgPwrA", "diff"], axis=1)
              dataDf = dataDf.reset_index(drop=True)

              for i in dataDf.itertuples():
                  list2.append({"ts": i.day, "values": {"minrg": i.value}}) #"""
if len(list2) == 0:
    now = datetime.datetime.now(pytz.timezone('Europe/Athens'))
    yesterday = now - timedelta(days = 1, hours=now.hour, minutes=now.minute, seconds=now.second, microseconds=now.microsecond)
    offset = int(yesterday.astimezone(pytz.timezone('Europe/Athens')).utcoffset().total_seconds()/3600)
    print(offset)
    list2.append({"ts": int(yesterday.replace(tzinfo=timezone(timedelta(hours=offset))).timestamp() * 1000), "values": {"minrg": 0}})
    #list2.append({"ts": int(yesterday.astimezone(pytz.timezone('Europe/Athens')).timestamp() * 1000), "values": {"minrg": 0}})
list2 = list2 + finalList
print(list2)
res = requests.post(address + f'/api/plugins/telemetry/DEVICE/{devId}/timeseries/ANY?scope=ANY', data=json.dumps(list2), headers=headers)


with open("/home/azureuser/gen_models/microwave/test_log.txt", "w") as f:
    f.write(str(list2)) #"""
