import tkinter as tk  # todo import Libraries
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
import serial
from serial.tools import list_ports
from datetime import time
from datetime import datetime
import time
import thingsboard_json_formatter
import m3_thingsboard_mqtt
from six import print_
global L
global params

# ------------------- Tkinter ---------------------------
MyRoot = tk.Tk()
MyRoot.title("Meazon.AE")
MyRoot.geometry('700x700')

L2 = tk.Label(MyRoot, text=" Project Energy-Disaggregation", font=('bold', 20))
L2.place(x=100, y=50)

# --------------- L ------------------------------

Lk = tk.Label(MyRoot, text="Phase ", font=20)
Lk.place(x=20, y=110)

var1 = IntVar(value=1)
cb1 = Checkbutton(MyRoot, text="L1", variable=var1)
cb1.place(x=250, y=110)

var2 = IntVar(value=1)
cb2 = Checkbutton(MyRoot, text="L2", variable=var2)
cb2.place(x=300, y=110)

var3 = IntVar(value=1)
cb3 = Checkbutton(MyRoot, text="L3", variable=var3)
cb3.place(x=350, y=110)

# ------------------ Total -------------------------------

L9 = tk.Label(MyRoot, text="Total Time (s)", font=20)
L9.place(x=20, y=170)

entry1 = Entry(MyRoot)
entry1.place(x=250, y=170)

# --------------------- Toff -------------------------------

L4 = tk.Label(MyRoot, text="Time for the state off", font=20)
L4.place(x=20, y=220)


entry2 = Entry(MyRoot)
entry2.place(x=250, y=220)

# --------------------- Number of changes ---------------------------

L5 = tk.Label(MyRoot, text=" Number of changes(on-to-off)", font=20)
L5.place(x=20, y=260)


w = Scale(MyRoot, from_=0, to=20, orient=HORIZONTAL)
w.place(x=250, y=240)


# ----------------------- Mode ---------------------------------

L6 = tk.Label(MyRoot, text="Mode(pA/hA)", font=20)
L6.place(x=20, y=300)

var4 = IntVar()
cb4 = Checkbutton(MyRoot, text="pA", variable=var4)
cb4.place(x=250, y=300)

var5 = IntVar()
cb5 = Checkbutton(MyRoot, text="hA", variable=var5)
cb5.place(x=300, y=300)

# --------------------- Device -------------------------------

L10 = tk.Label(MyRoot, text="Device", font=20)
L10.place(x=20, y=330)

entry7 = Entry(MyRoot)
entry7.place(x=250, y=330)

# --------------------- Brand -------------------------------

L8 = tk.Label(MyRoot, text="Brand", font=20)
L8.place(x=20, y=360)

entry5 = Entry(MyRoot)
entry5.place(x=250, y=360)

# --------------------- Device name -------------------------------

L7 = tk.Label(MyRoot, text="Serial number", font=20)
L7.place(x=20, y=390)

entry4 = Entry(MyRoot)
entry4.place(x=250, y=390)

# --------------------- Description -------------------------------

L9 = tk.Label(MyRoot, text="Description", font=20)
L9.place(x=20, y=410)

entry6 = Text(MyRoot, width=100, height=3)
entry6.place(x=250, y=410)


def importimage():
    global img
    path = filedialog.askopenfilename(filetypes=[('all files', '.*'), ('text files', '.txt'), ('image files', '.png'), ('image files', '.jpg'), ])
    img = Image.open(path)
    img.save(str(entry4.get()) + "1.jpg")
    messagebox.showinfo("uploading a photo ", "successful upload")


button1 = Button(MyRoot, text="Import an image", width=20, command=importimage)
button1.place(x=370, y=500)

# ----------------- WINDOW --------------------------

def close_window():
        from six import print_
        from serial.tools import list_ports
        import serial
        from datetime import time
        from datetime import datetime
        import time
        import Thingsboard_relations
        import columns
        import Ascii_funct
        global L
        global params
        global fn
        global brnd
        global detail
        global measure

        input1 = entry1.get()
        input2 = entry2.get()
        input3 = w.get()
        input4 = entry4.get()
        input5 = entry5.get()
        input6 = entry6.get("1.0", 'end-1c')
        input7 = entry7.get()

        l1 = var1.get()  # lx = 1 : if x phase is active, else: lx = 0
        l2 = var2.get()
        l3 = var3.get()
        pA = var4.get()
        hA = var5.get()

        port = dict((x[0], x[2]) for x in serial.tools.list_ports.comports())  # todo find the port of usb
        com = None
        for x in port:
            a = port[x].split(' ')
            for i in a:
                if i == "VID:PID=0451:16C8":
                    com = x

        if (input1 == '') or (input2 == '') or (input3 == '') or (input4 == '') or \
                (input5 == '') or ((l1 == 0) and (l2 == 0) and (l3 == 0)) or (pA == 0 and hA == 0):
            messagebox.showerror("ERROR", "Please add all the data")
        elif com is None:
            messagebox.showerror("ERROR", "Failed to connect with device")
        else:
            L = [l1, l2, l3]
            mode = [pA, hA]
            params = [input1, input2, input3, input4, input5, input6, input7]

# -------------------------------------------------- Relations --------------------------------------------------

            device_name = params[6]
            serial_number = params[3]
            brand_name = params[4]
            Info = params[5]
            # acc_token = Thingsboard_relations.access_token()
            # headers = {
            #     'Content-Type': 'application/json',
            #     'X-Authorization': acc_token,
            # }
            # time.sleep(0.5)
            # id_ass_dev = Thingsboard_relations.dev_asset(device_name, headers)
            # time.sleep(0.5)
            # id_ass_srl = Thingsboard_relations.srl_asset(serial_number, headers)
            # time.sleep(0.5)
            # id_ass_bnd = Thingsboard_relations.brd_asset(brand_name, headers)
            # time.sleep(0.5)
            # Thingsboard_relations.relation_asset(id_ass_bnd, id_ass_dev, headers)
            # time.sleep(0.5)
            # Thingsboard_relations.relation_asset(id_ass_srl, id_ass_bnd, headers)
            # time.sleep(0.5)
            # times = str(datetime.utcnow())
            # id_dev = Thingsboard_relations.device(times, Info, headers)
            # time.sleep(0.5)

# -------------------------------------------------- Collect data --------------------------------------------------

            T = params[0]
            if T[-1:] == 'm':
                T = float(T[:-1]) * 60
            elif T[-1:] == 'h':
                T = float(T[:-1]) * 1200
            elif T[-1:] == 's':
                T = float(T[:-1])
            else:
                T = float(T)

            n = int(params[2])
            toff = float(params[1])

            s = serial.Serial(com, baudrate=115200, xonxoff=True)
            s.write('-mz ade disag stat off\r\n'.encode())
            s.close()
            time.sleep(1)
            s = serial.Serial(com, baudrate=115200, timeout=1.2, xonxoff=True)
            s.write('-mz ade disag mode ascii\r\n'.encode())
            s.readline().decode('ascii')
            s.write('-mz ade disag int 20\r\n'.encode())
            s.readline().decode('ascii')

            if L[0] == 1 and L[1] == 1 and L[2] == 1:
                s.write('-mz ade disag lines L1,L2,L3\r\n'.encode())
                s.readline().decode('ascii')
                if mode[0] == 1:
                    s.write('-mz ade disag stat pA\r\n'.encode())
                    s.readline().decode('ascii')
                else:
                    s.write('-mz ade disag stat hA\r\n'.encode())
                    s.readline().decode('ascii')
            elif L[0] == 1 and L[1] == 1:
                s.write('-mz ade disag lines L1,L2\r\n'.encode())
                s.readline().decode('ascii')
                if mode[0] == 1:
                    s.write('-mz ade disag stat pA\r\n'.encode())
                    s.readline().decode('ascii')
                else:
                    s.write('-mz ade disag stat hA\r\n'.encode())
                    s.readline().decode('ascii')
            elif L[0] == 1 and L[2] == 1:
                s.write('-mz ade disag lines L1,L3\r\n'.encode())
                s.readline().decode('ascii')
                if mode[0] == 1:
                    s.write('-mz ade disag stat pA\r\n'.encode())
                    s.readline().decode('ascii')
                else:
                    s.write('-mz ade disag stat hA\r\n'.encode())
                    s.readline().decode('ascii')
            elif L[1] == 1 and L[2] == 1:
                s.write('-mz ade disag lines L2,L3\r\n'.encode())
                s.readline().decode('ascii')
                if mode[0] == 1:
                    s.write('-mz ade disag stat pA\r\n'.encode())
                    s.readline().decode('ascii')
                else:
                    s.write('-mz ade disag stat hA\r\n'.encode())
                    s.readline().decode('ascii')
            elif L[0] == 1:
                s.write('-mz ade disag lines L1\r\n'.encode())
                s.readline().decode('ascii')
                if mode[0] == 1:
                    s.write('-mz ade disag stat pA\r\n'.encode())
                    s.readline().decode('ascii')
                else:
                    s.write('-mz ade disag stat hA\r\n'.encode())
                    s.readline().decode('ascii')
            elif L[1] == 1:
                s.write('-mz ade disag lines L2\r\n'.encode())
                s.readline().decode('ascii')
                if mode[0] == 1:
                    s.write('-mz ade disag stat pA\r\n'.encode())
                    s.readline().decode('ascii')
                else:
                    s.write('-mz ade disag stat hA\r\n'.encode())
                    s.readline().decode('ascii')
            elif L[2] == 1:
                s.write('-mz ade disag lines: L3\r\n'.encode())
                s.readline().decode('ascii')
                if mode[0] == 1:
                    s.write('-mz ade disag stat pA\r\n'.encode())
                    s.readline().decode('ascii')
                else:
                    s.write('-mz ade disag stat hA\r\n'.encode())
                    s.readline().decode('ascii')

            time.sleep(1)

            m3_thingsboard_mqtt.setup(None)  # --->setup
            m3_thingsboard_mqtt.start()  # --->start

            time.sleep(1)

            ton = (T - n * toff) / (n + 1)  # set the times that relay change state
            endtime = []
            endtime.append(time.time() + ton)

            for i in range(1, 2 * n + 1):
                if i % 2 == 0:
                    endtime.append(endtime[i - 1] + ton)
                else:
                    endtime.append(endtime[i - 1] + toff)

            endtime[2 * n] += 1
            endtime.append(1.0)
            i = 0
            state = 1
            a = 0
            ts = str(datetime.utcnow())
            flag = 0
            delta = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') - datetime.utcfromtimestamp(0)
            tstart = int(delta.total_seconds() * 1000)
            endtime1 = time.time() + T
            while int(time.time()) < int(endtime1):
                    if int(time.time()) == int(endtime[i]):
                        if state == 1:
                            s.write('-mz relay stat 0\r\n'.encode())
                            s.readline().decode('ascii')
                            state = 0
                        else:
                            s.write('-mz relay stat 1\r\n'.encode())
                            s.readline().decode('ascii')
                            state = 1
                        i = i + 1
                    if L[0] == 1 and L[1] == 1 and L[2] == 1:
                        line1 = Ascii_funct.read_line(s)
                        line2 = Ascii_funct.read_line(s)
                        line3 = Ascii_funct.read_line(s)
                        if line1 == 0 or line2 == 0 or line3 == 0: continue
                        line = line1 + line2[2:] + line3[2:]
                        if flag == 0:
                            ms = 0
                            flag = 1
                        else:
                            ms = float(line[1]) - a
                        tstart = tstart + int(ms)
                        a = float(line[1])
                        cols = columns.cols_hA(3, 3, '3', '5', '7')
                    elif (L[0] == 1 and L[1] == 1) or (L[0] == 1 and L[2] == 1) or (L[1] == 1 and L[2] == 1):
                        line1 = Ascii_funct.read_line(s)
                        line2 = Ascii_funct.read_line(s)
                        if line1 == 0 or line2 == 0: continue
                        line = line1 + line2[2:]
                        if flag == 0:
                            ms = 0
                            flag = 1
                        else:
                            ms = float(line[1]) - a
                        tstart = tstart + int(ms)
                        a = float(line[1])
                        cols = columns.cols_pA(line, 2)
                    else:
                        line = Ascii_funct.read_line(s)
                        if line == 0: continue
                        if flag == 0:
                            ms = 0
                            flag = 1
                        else:
                            ms = float(line[1]) - a
                        tstart = tstart + int(ms)
                        a = float(line[1])
                        cols = columns.cols_hA(1, 1, '3', '5', '7')
                    measurement_dict = {}
                    for j in range(len(cols)):
                            measurement_dict[f"{cols[j]}"] = f"{line[j]}"
                    payload = thingsboard_json_formatter.thingsboard_json('test', tstart, measurement_dict)
                    print(payload)
                    m3_thingsboard_mqtt.publish('test', payload)  # --->publish
                    time.sleep(0.01)
                    if int(time.time()) < int(endtime1 - 3 * T / 4):
                        progress['value'] = 25
                        MyRoot.update_idletasks()
                    elif int(time.time()) < int(endtime1 - T / 2):
                        progress['value'] = 50
                        MyRoot.update_idletasks()
                    else:
                        progress['value'] = 75
                        MyRoot.update_idletasks()

            m3_thingsboard_mqtt.stop('test')  # --->stop
            time.sleep(1)

            # Thingsboard_relations.relation_device(id_ass_srl, id_dev, headers)

            progress['value'] = 100
            MyRoot.update_idletasks()
            time.sleep(0.06)
            s.write('-mz ade disag stat off\r\n'.encode())
            s.close()
            time.sleep(1)
            messagebox.showinfo("Meazon", "successful Process")
            MyRoot.destroy()
            L = [l1, l2, l3]
            params = [input1, input2, input3, input4, input5, input6, input7]


progress = ttk.Progressbar(MyRoot, orient=HORIZONTAL, length=200, mode='determinate')
progress.place(x=200, y=550)
button = tk.Button(MyRoot, text='START', width=20, command=close_window)
button.place(x=150, y=500)

MyRoot.mainloop()
#  ------------------------------  Functions for main.py  ------------------------------------------


def file():
    return fn


def brands():
    return brnd


def details():
    return detail
