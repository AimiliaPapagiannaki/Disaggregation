import paho.mqtt.client as mqtt
import time
import datetime
import ssl
import mqtt_cfg
import requests
import json
import decode_binary_v2 as decoder
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


buffsize = 60  # number of samples = 20 samples per second (50msec interval) x 3 seconds
buff = {}
prev_ts = 0
event = 0
prev_deltaTs = 0


def get_measurement(msg, show_extra_vals):
    data = None
    try:
        if mqtt_cfg.test_crc and decoder.has_crc:
            if not decoder.test_crc(msg):
                print("CRC error")
                return None, None
        ts, real_payload = decoder.get_ts(msg)
        if ts is None:
            return None, None
        data = decoder.on_measurement(real_payload, show_extra_vals)
    except Exception as e:
        print('unable to decode')

    if data is None:
        print('no data')
        return None, None

    if ts == 0:
        ts = int(round(time.time() * 1000))

    return ts, data


def send_data(mydict):
    address = 'http://devmeazonthings.westeurope.cloudapp.azure.com:8080'
    r = requests.post(address + "/api/auth/login",
                      json={'username': 'tenant@thingsboard.org', 'password': 'tenant'}).json()
    acc_token = 'Bearer' + ' ' + r['token']
    devtoken = 'wWmltLl6mZbn5PIc1EeR'

    for key, value in mydict.items():
        my_json = json.dumps({'ts': key, 'values': {'candfridge':value}})
        r = requests.post(url=address + "/api/v1/" + devtoken + "/telemetry",data=my_json, headers={'Content-Type': 'application/json', 'Accept': '*/*',
                                             'X-Authorization': acc_token})
    
    return


def detect_fridge():
    global buff
    half = 10  # sliding length
    interval = int(50) #fast pace frequency, every 50msec
    sec = 1000 / interval  # number of samples per second
    events = []
    deltas = {}  # delta Power ON
    deltasre = {}  # delta reactive Power ON
    offdeltas = {}  # delta Power OFF
    offre = {}  # delta reactive Power OFF
    d = {}

    # off variables
    p_off_thres = -101.785
    q_off_thres = 141.98

    # convert dictionary to df and set DateTimeIndex
    df = pd.DataFrame.from_dict(buff, orient='index', columns=['pwrA', 'rpwrA'])
    df['ts'] = pd.to_datetime(df.index, unit='ms', utc=True)
    df.set_index('ts', inplace=True, drop=True)

    # manage start/end of window
    df['dif'] = df.index.to_series().diff().astype('timedelta64[ms]')
    df['dif'].iloc[0] = 0

    # print(df.head())
    winlen = df.shape[0]
    startPoint = int(df['dif'].iloc[1] / interval - sec)
    endPoint = int(df['dif'].iloc[-1] / interval - sec)

    # if the right edge of the event is too far, make it smaller
    if df['dif'].iloc[-1] > 120000:
        endPoint = int(df['dif'].iloc[-1] / interval + 2 * sec)
    df = df.resample('50ms').mean()

    #print('event is:', df)
    # forward fill
    df['pwrA'] = df['pwrA'].fillna(method='ffill')
    df['rpwrA'] = df['rpwrA'].fillna(method='ffill')

    # check if there is a delta OFF based on head and tail of group
    deltaO = df['pwrA'].iloc[-10:].mean() - df['pwrA'].iloc[:10].mean()
    rdeltaO = df['rpwrA'].iloc[-10:].mean() - df['rpwrA'].iloc[:10].mean()
    
    if ((deltaO < -30) & (deltaO > -200)):
        #print('power off, reactive off:', deltaO, rdeltaO)
        print('just an off')
    
    if ((deltaO <0.9*p_off_thres) & (deltaO>=1.3*p_off_thres) & (np.abs(rdeltaO) >0.9*q_off_thres) & (np.abs(rdeltaO)<=1.3*q_off_thres)):
        dt = int(datetime.datetime.timestamp(df.index[0])*1000)
        #offdeltas[df.index[0]] = deltaO
        #offre[df.index[0]] = rdeltaO  # reactive delta
        offdeltas[dt] = df['pwrA'].iloc[-1]
        offre[dt] = rdeltaO  # reactive delta
        #print('OFF DETECTION', offdeltas)
        send_data(offdeltas)
        

    df = df.iloc[startPoint:]
    df = df.iloc[:-endPoint]

    # iterate within window to ensure the algorithm catches the event
    for k in range(0, int(df.shape[0] / half) - int(winlen / half)):
        tmp = pd.DataFrame([])
        tmp = df.iloc[k * half:k * half + winlen].copy()
        k += 1

        # calculate derivative and take 3 top local maxima
        tmp['extrema'] = 0
        tmp['der1'] = np.abs(tmp['pwrA'].shift(-1) - tmp['pwrA'].shift())

        tmp['extrema'] = tmp.iloc[argrelextrema(tmp['der1'].values, np.greater_equal, order=1)[0]]['der1']
        tmp.sort_values(by='extrema', ascending=False, inplace=True)
        tmp['extrema'][:3] = 1
        tmp['extrema'][3:] = 0

        tmp.sort_index(inplace=True)

        # split event in 4 parts and calculate avg power
        s = 0
        n = 0
        parts = []
        for j in range(0, tmp.shape[0]):
            if tmp['extrema'].iloc[j] < 1:
                n += 1
                s += tmp['pwrA'].iloc[j]
            else:
                n += 1
                s += tmp['pwrA'].iloc[j]
                parts.append(s / n)
                n = 0
                s = 0

        # check rules
        if n != 0:
            parts.append(s / n)
            if len(parts) > 3:
                pmin = 50
                pmax = 700
                rule1 = parts[3] - parts[0] > pmin
                rule2 = parts[3] - parts[0] < pmax
                rule3 = parts[2] - parts[0] > 5 * (parts[3] - parts[0])
                rule4 = parts[1] - parts[0] > 1.1 * (parts[2] - parts[0])
                if (rule1 and rule2 and rule3 and rule4):
                    print('DETECTED FRIDGE')
                    hour = str(tmp.index.hour[0]) + ':' + str(tmp.index.minute[0])
                    if not hour in events:
                        #deltaP = tmp['pwrA'].iloc[-10:].mean() - tmp['pwrA'].iloc[:10].mean()
                        #deltaR = tmp['rpwrA'].iloc[-10:].mean() - tmp['rpwrA'].iloc[:10].mean()

                        deltaP = tmp['pwrA'].iloc[-1]
                        dt = int(datetime.datetime.timestamp(tmp.index[0])*1000)
                        #deltas[tmp.index[0]] = deltaP
                        #deltasre[tmp.index[0]] = deltaR  # reactive
                        deltas[dt] = deltaP
                        #deltasre[dt] = deltaR  # reactive
                        print(deltas)
                        send_data(deltas)

                        events.append(hour)
                        d[tmp.index[0]] = 1
        buff = {}
    #print(offdeltas, offre, deltas, deltasre, d)


def ingest_meas(ts, data):
    global buff
    global prev_ts
    global prev_deltaTs
    global buffsize
    global event

    # try:
    # print('event state:', event)
    pwr = float(data['pwrA'])
    rpwr = float(data['rpwrA'])
    ts = int(ts)
    # print('pwr, ts:', ts, pwr)
    buff[ts] = [pwr,rpwr]
    deltaTs = ts - prev_ts
    old_ts = prev_ts-prev_deltaTs
    #print('length, deltaTs:', len(buff), deltaTs)


    # check for event
    if deltaTs < 150:
        #print('START ')
        event = 1
    if (event == 1) & (deltaTs > 150):
        #print('END')
        event = 0
        detect_fridge()
        buff = {}

    elif (event == 0) & (len(buff)>2):
        # print('remove element')
        if old_ts in buff.keys():
            buff.pop(old_ts)

    prev_deltaTs = deltaTs
    prev_ts = ts
    # except:
    #     print('No data available', data)


def on_message(client, userdata, message):
    # print('message check')
    ts, data = get_measurement(message.payload, True)
    ingest_meas(ts, data)


def connect():
    client = mqtt.Client(mqtt_cfg.client_id, clean_session=True)
    if len(mqtt_cfg.user) > 0:
        client.username_pw_set(username=mqtt_cfg.user, password=mqtt_cfg.password)
    client.on_message = on_message

    if mqtt_cfg.root_ca is not None and len(mqtt_cfg.root_ca) > 0:
        print(mqtt_cfg.root_ca)
        client.tls_set(ca_certs=mqtt_cfg.root_ca, tls_version=ssl.PROTOCOL_TLS)

    client.connect(mqtt_cfg.broker_address, port=mqtt_cfg.port)  # connect to broker


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)

    topics = []
    topic = "bin/102.402.000045/measurements/v2"
    topics.append((topic, mqtt_cfg.qos))

    client.subscribe(topics)


# def main():

client = mqtt.Client(mqtt_cfg.client_id, clean_session=True)
if len(mqtt_cfg.user) > 0:
    client.username_pw_set(username=mqtt_cfg.user, password=mqtt_cfg.password)

client.on_connect = on_connect
client.on_message = on_message

if mqtt_cfg.root_ca is not None and len(mqtt_cfg.root_ca) > 0:
    # print(mqtt_cfg.root_ca)
    client.tls_set(ca_certs=mqtt_cfg.root_ca, tls_version=ssl.PROTOCOL_TLS)

client.connect(mqtt_cfg.broker_address, port=mqtt_cfg.port)  # connect to broker
client.loop_forever()
