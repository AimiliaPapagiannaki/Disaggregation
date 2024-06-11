import numpy as np
import serial
from serial.tools import list_ports
from datetime import time
from datetime import datetime
import time
import sys
import requests
import json
global ts
global address


def events(dct):
    global ts

    # attach the package's number to event descriptors
    pck = dct['S']
    evmap = {}
    evmap['avgP'] = 'avgP%s' % pck
    evmap['avgR'] = 'avgR%s' % pck
    evmap['minP'] = 'minP%s' % pck
    evmap['minR'] = 'minR%s' % pck
    evmap['maxP'] = 'maxP%s' % pck
    evmap['maxR'] = 'maxR%s' % pck
    evmap['stdP'] = 'stdP%s' % pck
    evmap['stdR'] = 'stdR%s' % pck

    # rename keys according to evmap
    for key in list(evmap):
        dct[evmap.get(key)] = dct.pop(key, None)

    # add timestamp to packages S1-S4
    dct['utc'] = ts
    return dct


def create_dicts(t, phase):
    global ts

    # map meter's variables to TB descriptors' names
    mapping = {}
    mapping['p'] = 'pwr%s' % phase
    mapping['q'] = 'rpwr%s' % phase
    mapping['v'] = 'vlt%s' % phase
    mapping['i'] = 'cur%s' % phase
    mapping['s'] = 'apwr%s' % phase
    mapping['f'] = 'freq%s' % phase
    mapping['pF'] = 'cos%s' % phase
    mapping['cF'] = 'scre%s' % phase
    mapping['an'] = 'angle%s' % phase
    mapping['cE'] = 'cnrg%s' % phase
    mapping['pE'] = 'pnrg%s' % phase
    mapping['cR'] = 'crnrg%s' % phase
    mapping['pR'] = 'prnrg%s' % phase

    # create dictionary from list after mapping the descriptors
    line = dict(zip((mapping.get(item, item) for item in (v.split('=')[0] for v in t)), ((v.split('=')[1]) for v in t)))
    line[list(line)[-1]] = line[list(line)[-1]][:-2] # remove \r\n from the end

    # convert 1/12000 utc to 1/1/1970 utc timestamp
    if 'utc' in line.keys():
        line['utc'] = int((datetime(2000, 1, 1) - datetime(1970, 1, 1)).total_seconds()) + int(line['utc'])

    if 'S' in line.keys(): # if event has occurred keep timestamp of the first package
        if line['S'] == '0':
            ts = line['utc']
        else:
            line = events(line)

    return line


def read_line(s, acc_token):
    phmap = {}
    phmap['1'] = 'A'
    phmap['2'] = 'B'
    phmap['3'] = 'C'

    asc = s.readline().decode('ascii').replace(',', '.')
    if asc:
        if asc[0] != 's':
            return 0
        else:
            t = asc.split('&')
            if t[2][0] == 'L':
                t[2] = 'L=' + t[2][1]
            else:
                t[2] = 'P=' + t[2][1]
                t[3] = 'S=' + t[3][1]
            phase = phmap.get(t[2][2])

            line = create_dicts(t, phase)

            # drop descriptors that aren't necessary
            line.pop('L', None)
            line.pop('sN', None)
            line.pop('ms', None)
            line.pop('P', None)
            line.pop('S', None)

            # convert dictionary to json-friendly formatting
            line = {'ts': int(line['utc']) * 1000, 'values': {x: line[x] for x in line if x not in 'utc'}}

            # send data to cloud via http
            send_data(line, acc_token)
    return


def send_data(payload, acc_token):
    global address
    dev_token = 'fpgUMFyhpCS6i2fsRax5'

    # convert dictionary to json and send data with http
    my_json = json.dumps(payload)
    r2 = requests.post(url=address + "api/v1/" + dev_token + "/telemetry",
                       data=my_json,
                       headers={'Content-Type': 'application/json', 'Accept': '*/*', 'X-Authorization': acc_token})
    print(r2.status_code)
    return

def main():
    # request access token
    global address
    address = "http://52.77.235.183:8080/"
    r = requests.post(address + "api/auth/login",
                      json={'username': 'tenant@thingsboard.org', 'password': 'tenant'}).json()
    acc_token = 'Bearer' + ' ' + r['token']
    zerotime = datetime.now() # get start time

    # read ports and try to recognize the specific usb port
    port = dict((x[0], x[2]) for x in serial.tools.list_ports.comports())  #
    com = None
    for x in port:
        a = port[x].split(' ')
        for i in a:
            if i == "VID:PID=0451:16C8":
                com = x

    nsamples = 40  # number of samples at each window, given that sampling rate is at 10msec
    thres = 2  # threshold in watts
    interv = 60000  # interval for all variables except event vars in msec

    # initialize serial port and send message to close previous connections or whatever
    s = serial.Serial(com, baudrate=115200, xonxoff=True)
    s.write('-mz ade disag stat off\r\n'.encode())
    s.close()
    time.sleep(1)
    s = serial.Serial(com, baudrate=115200, timeout=1.2, xonxoff=True)
    s.write(b'-mz ade disag mode ascii\r\n')
    s.readline().decode('ascii')

    s.write(b'-mz ade disag stat pA\r\n')  # set status to send power, not harmonics
    s.readline().decode('ascii')

    s.write(b'-mz ade rta cfg %d,%d\r\n' % (nsamples, thres))  # set number of samples per window and threshold
    s.readline().decode('ascii')

    s.write(b'-mz ade disag int %d\r\n' % interv)  # set reading interval
    s.readline().decode('ascii')

    s.write(b'-mz ade disag lines L1\r\n')  # set lines L1,L2,L3
    s.readline().decode('ascii')

    s.write(b'-mz relay stat 1\r\n') # set relay status
    s.readline().decode('ascii')


    print('Start monitoring...')

    time.sleep(1)

    # m3_thingsboard_mqtt.setup(None)  # --->setup
    # m3_thingsboard_mqtt.start()  # --->start

    # time.sleep(1)
    while (True):
        # if 60min have passed since the last access token request, refresh it
        if (datetime.now() - zerotime).total_seconds() > 3600:

            diftime = np.abs((datetime.utcnow() - datetime.strptime(str(s.readline().decode('ascii'))[:-2],
                                                                    '%y/%m/%d,%H:%M:%S')).total_seconds())
            if diftime>3:
                # dtnow = datetime.utcnow().strftime('%y/%m/%d,%H:%M:%S').encode('ascii')
                s.write(b'-mz stat utc %s\r\n' % datetime.utcnow().strftime('%y/%m/%d,%H:%M:%S').encode('ascii'))
                print('RTC drifting corrected')

            r = requests.post(address + "api/auth/login",
                              json={'username': 'tenant@thingsboard.org', 'password': 'tenant'}).json()
            acc_token = 'Bearer' + ' ' + r['token']
            zerotime = datetime.now()

        read_line(s, acc_token)

if __name__ == "__main__":
    # sys.exit(main(sys.argv))
    sys.exit(main())