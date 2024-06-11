import paho.mqtt.client as mqtt
import time
import ssl
import mqtt_cfg
import decode_binary_v2 as decoder
import pandas as pd

buffsize = 60  # number of samples = 20 samples per second (50msec interval) x 3 seconds
buff = {}
prev_ts = 0
event = 0


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


def detect_fridge():
    global buff
    df = pd.DataFrame.from_dict(buff, orient='index', columns=['pwrA'])
    df = df.resample('50ms').mean()
    print('event is:', df)


def ingest_meas(ts, data):
    global buff
    global prev_ts
    global buffsize
    global event

    try:
        # print('event state:', event)
        pwr = float(data['pwrA'])
        ts = int(ts)
        print('pwr, ts:', ts, pwr)
        buff[ts] = pwr
        deltaTs = ts - prev_ts
        print('length, deltaTs:', len(buff), deltaTs)

        # check for event
        if deltaTs < 150:
            print('START ')
            event = 1
        if (event == 1) & (deltaTs > 150):
            print('END')
            event = 0
            detect_fridge()
        elif event == 0:
            if prev_ts in buff.keys():
                buff.pop(prev_ts)

        prev_ts = ts
    except:
        print('No data available', data)


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
