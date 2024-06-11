import requests
import json
import pandas as pd


def get_dev_info(devname):
    address = "http://51.103.232.223:8080"
    r = requests.post(address + "/api/auth/login",
                      json={'username': 'tenant@thingsboard.org', 'password': 'tenant'}).json()

    # acc_token is the token to be used in the next request
    acc_token = 'Bearer' + ' ' + r['token']
    # get devid by serial name
    r1 = requests.get(
        url=address + "/api/tenant/devices?deviceName=" + devname,
        headers={'Content-Type': 'application/json', 'Accept': '*/*', 'X-Authorization': acc_token}).json()

    devid = r1['id']['id']
    r1 = requests.get(
        url=address + "/api/device/" + devid + "/credentials",
        headers={'Content-Type': 'application/json', 'Accept': '*/*', 'X-Authorization': acc_token}).json()
    devtoken = r1['credentialsId']

    return devid,devtoken,acc_token,address


def send_data(df,devtoken,address,acc_token):
    # print(df)
    df['ts'] = pd.to_datetime(df.index).tz_localize('Europe/Athens')
    df['ts'] = df.apply(lambda row: int(row['ts'].timestamp()) * 1000+86400000, axis=1)

    df.set_index('ts', inplace=True, drop=True)

    mydict = df.to_dict('index')

    for key, value in mydict.items():
        my_json = json.dumps({'ts': key, 'values': value})
        print(my_json)
        r = requests.post(url=address + "/api/v1/" + devtoken + "/telemetry",
                          data=my_json, headers={'Content-Type': 'application/json', 'Accept': '*/*','X-Authorization': acc_token})
                          
                          

[devid,devtoken,acc_token,address] = get_dev_info(devname)