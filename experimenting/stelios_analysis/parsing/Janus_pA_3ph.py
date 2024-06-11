
import serial
from serial.tools import list_ports
import time
from datetime import datetime
import os
import binascii
import thingsboard_json_formatter
import common_tb_mqtt
import m3_thingsboard_mqtt
from six import print_

# Giving permission to read and write to serial port
# os.system("echo temppwd | sudo -S  chmod 666 /dev/ttyACM0")
# set tty port baud-rate to match ours
portBBB = '/dev/ttyACM0'
baudrate = 115200
NDATA = 1000


def connect_forevah():
    while True:
        try:
            s = serial.Serial(
                port=portBBB,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE
            )
            break
        except Exception as e:
            print_("Error connecting to serial: %s" % e)
            time.sleep(2)
    s.timeout = 1.2
    s.xonoff = True

    if s.isOpen():
        s.close()
    s.open()
    return s

import struct
import io


def create_log():
    flag = 0
    # no need to do now - 1970/1/1 ... time.time() gives exactly what we need, but in seconds.
    timestamp = time.time() * 1000
    # millis = 0
    old_millis = 0
    m3_thingsboard_mqtt.setup(None)  # --->setup
    m3_thingsboard_mqtt.start()  # --->start
    time.sleep(1)

    # In python 3, received data is of bytes form, so lets stay compatible to both 2 & 3
    raw_bytes = b""
    start_of_packet = b'\xee\xee\x9c'  # this is, of course, 3 bytes
    raw_bytes = s.read(160)
    while (True):
        try:
            i = 4
            xor = hex(raw_bytes[3])
            while i <= 157:
                xor = hex(int(xor, 16) ^ int(hex(raw_bytes[i]), 16))
                i = i + 1
            # don't encode it to hex just yet
            start_pos = raw_bytes.find(start_of_packet)
            if start_pos > 0:  # we expect to find it on 0, -1 is not found, > 0 is problem
                print_("Discarding bytes: %s" % binascii.hexlify(raw_bytes[:start_pos]))
                raw_bytes = raw_bytes[start_pos:] + s.read(160 - len(raw_bytes[start_pos:]))

            if start_pos != 0 or len(raw_bytes) < 160:
                # no full packet yet, wait a bit
                time.sleep(0.01)
                continue

            if xor != hex(raw_bytes[159]):
                print_("Wrong CRC8 byte: %s" % binascii.hexlify(raw_bytes))
                raw_bytes = s.read(160)
                continue

            # if we have reached so far the packet is alright

            if flag == 0:
                millis = 0
                flag = 1
            else:
                millis = struct.unpack("I", raw_bytes[8:12])[0] - old_millis

            # reversing by pairs, after having converted to hex? imaginative, but too complicated.
            # old_millis, millis = millis, struct.unpack("I", raw_bytes[8:12])[0]
            timestamp = int(timestamp + millis)
            old_millis = struct.unpack("I", raw_bytes[8:12])[0]
            hex_packet = binascii.hexlify(raw_bytes).decode()
            # trim the received buffer, removing the bytes that were sent, regardless of whether there are more
            measurement_dict = {'packet': hex_packet}  # ok, just showing off now
            # as we said, time.time() * 1000 would be just fine
            # In a threaded use, m3_thingsboard_mqtt is of thread class,
            # and a function like enqueue(('usb', payload)) is used to put things in its queue
            # also, json-formatting packet can/should become responsibility of target mqtt thread, to save us time in this one here
            payload = thingsboard_json_formatter.thingsboard_json('103.110.000105', timestamp, measurement_dict)
            # payload being a hex string instead of bytes is wrong, but it's not your fault.
            # print_(payload)
            m3_thingsboard_mqtt.publish('103.110.000105', payload)  # --->publish
            time.sleep(0.01)  # you don't need to sleep here, since publishing in same thread took quite some time.
            raw_bytes = s.read(160)
        except Exception as e:
            print_("ERROR: %s" % e)
            break

    m3_thingsboard_mqtt.stop('103.110.000105')  # --->stop
    time.sleep(1)


if __name__ == '__main__':

    print_("Setting up port params")
    os.system("stty -F %s %s" % (portBBB, baudrate))
    # serial.Serial lets you set up the port baudrate too,
    # so that you can avoid a system call and lets it work in windows, too, where stty is not easy to have.

    # Initialize serial connection with meter
    s = connect_forevah()

    time.sleep(5)

    print_("Setting up meter")
    # Setup meter
    for i in range(5):
        s.write('-mz ade disag stat off\r\n'.encode())
    s.close()
    time.sleep(1)
    s = connect_forevah()

    time.sleep(5)

    print_("Setting up meter. Again.")
    s.write('-mz ade disag lines L1,L2,L3\r\n'.encode())
    s.read(6)
    s.write('-mz ade disag mode raw\r\n'.encode())
    s.read(6)
    s.write('-mz ade disag stat pA\r\n'.encode())
    s.read(6)

    print_("Sampling measurements.")
    # while(True):
    create_log()
    print_("End")
