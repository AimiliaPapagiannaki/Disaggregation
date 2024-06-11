import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

    
    
def identify_events(df, interval):
# Run dataframe row by row to identify start of each event

    ind=0
    df['ind'] = np.nan
    event = 0
    thres = 3*interval+1
    for i in range(0, int(df.shape[0]-2)):
        if  event==0:
            # start of event
            if (df['dif'].iloc[i]>thres and df['dif'].iloc[i+1]>thres and df['dif'].iloc[i+2]<=thres):
                event=1
                df['ind'].iloc[i] = int(ind)

        elif event==1:
            # end of event
            if (df['dif'].iloc[i]>thres and df['dif'].iloc[i-1]<thres):
                df['ind'].iloc[i] = int(ind)
                event = 0
                ind += 1
            else:
                df['ind'].iloc[i] = int(ind)
    df = df.dropna()
    return df

def detect_appliance(df, interval, parameters):

    half = parameters['half'] # sliding length
    coef3 = parameters['coef3']
    coef4 = parameters['coef4']

    #half = parameters[0]['half']
    #coef3 = parameters[1]['coef3']
    #coef4 = parameters[2]['coef4']
    print('half, coef3, coef4', half, coef3, coef4)

    sec = 1000/interval # number of samples per second
    events = []
    deltas = {} #delta Power ON
    deltasre = {} #delta reactive Power ON
    offdeltas = {} #delta Power OFF
    offre = {} #delta reactive Power OFF
    d={}

    # iterate over groups to detect events
    for name, tmp1 in df.groupby('ind'):
        winlen = tmp1.shape[0]
        startPoint = int(tmp1['dif'].iloc[1]/interval-sec)
        endPoint = int(tmp1['dif'].iloc[-1]/interval-sec)
        
        # if the right edge of the event is too far, make it smaller
        if tmp1['dif'].iloc[-1]>120000:
            endPoint = int(tmp1['dif'].iloc[-1]/interval+2*sec)
        tmp1 = tmp1.resample(str(interval)+'ms').mean()
        
        # forward fill
        tmp1['pwrA'] = tmp1['pwrA'].fillna(method='ffill')
        tmp1['rpwrA'] = tmp1['rpwrA'].fillna(method='ffill')

        #check if there is a delta OFF based on head and tail of group
        deltaO = tmp1['pwrA'].iloc[-5:].mean() - tmp1['pwrA'].iloc[:5].mean()
        rdeltaO = tmp1['rpwrA'].iloc[-5:].mean() - tmp1['rpwrA'].iloc[:5].mean()
        if ((deltaO<-30) & (deltaO>-200)):
            offdeltas[tmp1.index[0]] = deltaO
            offre[tmp1.index[0]] = rdeltaO # reactive delta

        tmp1 = tmp1.iloc[startPoint:]
        tmp1 = tmp1.iloc[:-endPoint]


        # iterate within window to ensure the algorithm catches the event
        for k in range(0, int(tmp1.shape[0]/half)-int(winlen/half)):
            
            tmp = pd.DataFrame([])
            tmp = tmp1.iloc[k*half:k*half+winlen].copy()
            k+=1
            
            # calculate derivative and take 3 top local maxima
            tmp['extrema'] = 0
            tmp['der1'] = np.abs(tmp['pwrA'].shift(-1)-tmp['pwrA'].shift())

            tmp['extrema'] = tmp.iloc[argrelextrema(tmp['der1'].values, np.greater_equal,order=1)[0]]['der1']
            tmp.sort_values(by='extrema', ascending=False, inplace=True)
            tmp['extrema'][:3]=1
            tmp['extrema'][3:]=0

            tmp.sort_index(inplace=True)

            
            # split event in 4 parts and calculate avg power
            s=0
            n=0
            parts = []
            for j in range(0,tmp.shape[0]):
                if tmp['extrema'].iloc[j]<1:
                    n += 1
                    s += tmp['pwrA'].iloc[j]
                else:
                    n += 1
                    s += tmp['pwrA'].iloc[j]
                    parts.append(s/n)
                    n = 0
                    s = 0

            # check rules
            if n!=0:
                parts.append(s/n)
                if len(parts)>3:
                    pmin = 30
                    pmax = 700
                    rule1 = parts[3]-parts[0]>pmin
                    rule2 = parts[3]-parts[0]<pmax
                    rule3 = (parts[2]-parts[0])>coef3*(parts[3]-parts[0])
                    rule4 = (parts[1]-parts[0])>coef4*(parts[2]-parts[0])

                    
                    if (rule1 and rule2 and rule3 and rule4):
                        hour = str(tmp.index.hour[0])+':'+str(tmp.index.minute[0])
                        if not hour in events:
                            deltaP = tmp['pwrA'].iloc[-5:].mean() - tmp['pwrA'].iloc[:5].mean()
                            deltaR = tmp['rpwrA'].iloc[-5:].mean() - tmp['rpwrA'].iloc[:5].mean()
                            deltas[tmp.index[0]] = deltaP
                            deltasre[tmp.index[0]] = deltaR # reactive

                            events.append(hour)
                            d[tmp.index[0]] = 1
#                             print('fridge ', hour)
        
    return offdeltas, offre, deltas, deltasre, d

