#!/usr/bin/env python

from datetime import datetime
import math
import numpy as np
import pandas as pd


def sim_event_detection(df):
    interval = 250 # smart meter's internal interval
    repint = 60000/interval # report interal in milliseconds
    winlen = 4 # length of window (in number of samples)

    buffer = [] # buffer aka window
    thres = 50 #Watts
    duration = 2*(1000/interval) # 2000 msec ~ aka 2seconds
    report=np.nan
    event=0
    simulated = {}

    counter = 0
    for i in range(df.shape[0]):
        counter += 1
        if event==0:
            # if the buffer is not a full window, keep appending new samples. Else pop first item and append new item
            if len(buffer)<winlen:
                buffer.append(df['pwrA'].iloc[i])
            else:
                buffer.pop(0)
                buffer.append(df['pwrA'].iloc[i])

                # if there is a reported value to compare with, check absolute difference with current window
                if not math.isnan(report): 
                    if np.abs(np.mean(buffer)-report)>thres:
                        # start transmitting at high frequency
                        df.at[df.index[i],'event']=1
                        simulated[df.index[i]] = [report, df['rpwrA'].iloc[i],df['event'].iloc[i]]
                        event=1
                        counter=0

            # when report interval is reached, send average of window to cloud
            if counter == repint:
                report = np.mean(buffer)
                simulated[df.index[i]] = [report,df['rpwrA'].iloc[i],df['event'].iloc[i]]
                counter=0

        # on event mode, continuous streaming
        else:
            buffer.pop(0)
            buffer.append(df['pwrA'].iloc[i])

            if counter == duration: # if 2 seconds have passed return to low sampling rate
                event = 0
                counter=0
                report = np.mean(buffer)
                simulated[df.index[i]] = [report,df['rpwrA'].iloc[i],df['event'].iloc[i]]
                buffer = []   
            else:
                simulated[df.index[i]] = [df['pwrA'].iloc[i],df['rpwrA'].iloc[i],df['event'].iloc[i]]
    
    return simulated
        


def event_detection(file):
    
    # Read file
    df = pd.read_csv(file, sep=';')
    if 'Date' in df.columns:
        df['ts'] = df['Date']+' '+df['Time (GMT +3)']
        df.drop(['Date','Time (GMT +3)'], axis=1, inplace=True)
        df['ts'] = pd.to_datetime(df['ts'], format='%d-%m-%Y %H:%M:%S:%f')
    
    df.set_index('ts', inplace=True, drop=True)
    df = df.resample('250ms').max()
    
    
    # Copy only power of phase A and fill Nans with forward fill
    df = df[['Active Power L1 (W)','Reactive Power L1 (Var)']]
    df.rename(columns={'Active Power L1 (W)':'pwrA','Reactive Power L1 (Var)':'rpwrA'}, inplace=True)
    for col in df.columns:
        df[col] = df[col].fillna(method='ffill')
    df['event']=0

    # Estimate events and simulate hybrid sampled data
    simulated = sim_event_detection(df)
    sim_df = pd.DataFrame.from_dict(simulated, orient='index')
    sim_df.rename(columns={0:'Active power A', 1:'Reactive power A', 2:'event'}, inplace=True)
    
    return sim_df
    
