def read_line(s):
    ascii = s.readline().decode('ascii').replace(',', '.')
    if ascii[0] != 's': return 0
    else:
        print(ascii)
        t = ascii.split('&')
        t[2] = 'L=' + t[2][1]
        line = [float(v.split('=')[1]) for v in t]
        return line