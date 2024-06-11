import sys
import pandas as pd
import datetime
import requests
import numpy as np
from dateutil.relativedelta import relativedelta
import os
import glob
import pytz
from dateutil.tz import gettz
import timeit
from time import sleep
# from datetime import datetime
from datetime import timedelta
import time
import pickle


def read_data(devid, acc_token, address, start_time, end_time, descriptors):
    r2 = requests.get(
        url=address + "/api/plugins/telemetry/DEVICE/" + devid + "/values/timeseries?keys=" + descriptors + "&startTs=" + start_time + "&endTs=" + end_time + "&agg=NONE&limit=1000000",
        headers={'Content-Type': 'application/json', 'Accept': '*/*', 'X-Authorization': acc_token}).json()
    if r2:
        print('request completed')
        df = pd.DataFrame([])

        for desc in r2.keys():
            df1 = pd.DataFrame(r2[desc])
            df1.set_index('ts', inplace=True)
            df1.columns = [str(desc)]
            df = pd.concat([df, df1], axis=1)

        if df.empty == False:

            df.reset_index(drop=False, inplace=True)
            df = df.sort_values(by=['ts'])
            df.reset_index(drop=True, inplace=True)
            df.set_index('ts', inplace=True, drop=True)
            for col in df.columns:
                df[col] = df[col].astype('float')

            df = df.groupby(df.index).max()

        else:
            df = pd.DataFrame([])
    else:
        df = pd.DataFrame([])
    #         print('Empty json!')
    return df


def request_data(start_time, end_time, devid, acc_token, address, descriptors):
    df = pd.DataFrame([])
    svec = np.arange(int(start_time[0]), int(end_time[0]), 3600000)
    hour = 1
    for st in svec:
        print(hour)
        hour = hour + 1
        en = st + 3600000 - 1

        if int(end_time[0]) - en <= 0: en = int(end_time[0])
        #         print('start and end of iteration:',st,en)

        tmp = read_data(devid, acc_token, address, str(st), str(en), descriptors)
        if not tmp.empty:
            df = pd.concat([df, tmp])
        sleep(1)

    df['ts'] = pd.to_datetime(df.index, utc=True, unit='ms')
    df['ts'] = df['ts'].dt.tz_convert('Europe/Athens')

    df.set_index('ts', inplace=True, drop=True)
    return df


def run_models(df, maphase, phase, mdlphase, mdlpath):
    step = 20
    events = []
    nums = []
    state = []
    ev_ts = []
    dpwr = []  # delta active power --> |previous power - current power|
    conflicts = {}
    ln = 0
    i = 0

    while ln <= df.shape[0] - 141:
        dt1 = df.index[ln]
        dt2 = df.index[ln] + datetime.timedelta(milliseconds=20 * 140)

        ln = ln + df[dt1:dt2].shape[0]

        slot = df[maphase[phase]].copy()
        slot = slot[dt1:dt2]
        if slot.shape[0] > 100:
            #             print('slot:',dt1,dt2)
            tsm = slot.index[-1]
            slot.columns = ['pwr', 'rpwr']
            slot.reset_index(inplace=True, drop=True)
            steady = slot.iloc[:10].copy()
            change = slot.iloc[-80:].copy()
            change.reset_index(inplace=True, drop=True)

            # discover if appliance has been added or removed
            #             if (np.mean(change['pwr'].iloc[-20:-10])-np.mean(steady['pwr']))>0:
            if (np.mean(slot['pwr'].iloc[20:40]) - np.mean(steady['pwr'])) > 0:
                st = 1
            else:
                st = 0

            # subtract active & reactive power of steady state
            change['pwr'] = np.abs(change['pwr'] - np.mean(steady['pwr']))
            change['rpwr'] = np.abs(change['rpwr'] - np.mean(steady['rpwr']))
            change.dropna(inplace=True)

            cols = ['pwr', 'rpwr']

            df_pr = pd.DataFrame([])
            for col in cols:
                df_pr[col + '_mean'] = change[col].groupby(np.arange(len(change)) // step).mean()
                df_pr[col + '_std'] = change[col].groupby(np.arange(len(change)) // step).std()
                df_pr[col + '_min'] = change[col].groupby(np.arange(len(change)) // step).min()
                df_pr[col + '_max'] = change[col].groupby(np.arange(len(change)) // step).max()
            #             df_pr[col+'_skew'] = change[col].groupby(np.arange(len(change))//step).skew()
            #             df_pr[col+'_kurt'] = change[col].groupby(np.arange(len(change))//step).apply(pd.Series.kurt)
            df_pr.dropna(inplace=True)

            assigned = False

            # run all models for this phase
            for j in range(0, len(mdlphase[phase])):
                filename = mdlpath + str(mdlphase[phase][j]) + '.sav'
                mdl = pickle.load(open(filename, 'rb'))
                y_pred = mdl.predict(df_pr)
                # print(mdlphase[phase][j],y_pred)
                if np.sum(y_pred) >= 0.75 * len(y_pred):

                    if assigned == False:
                        dpwr.append(np.abs(np.mean(change['pwr'].iloc[-20:-10]) - np.mean(steady['pwr'])))
                        nums.append(np.sum(y_pred))
                        events.append(mdlphase[phase][j])
                        #                         print(mdlphase[phase][j], tsm)
                        ev_ts.append(tsm)
                        state.append(st)
                        assigned = True

                        i = i + 1
                    else:
                        if np.sum(y_pred) >= nums[i - 1]:
                            conflicts[ev_ts[i - 1]] = [mdlphase[phase][j]]
    # end of while

    ev = confl_postproc(events, state, ev_ts, conflicts, dpwr)
    return ev


def events_clearing(ev, events, mappings):
    # convert categorical variables to numeric

    if not ev.empty:
        ev.replace({'appl': {v: k for k, v in mappings.items()}}, inplace=True)

        ev = ev.resample('1S').max()
        globals()['ev%s' % phase] = ev.copy()

        # append events dataframes to dictionary
        events.append(globals()['ev%s' % phase])
    #     mappings.append(globals()['d%s' % phase])

    return events


def confl_postproc(events, state, ev_ts, conflicts, dpwr):
    ev = pd.DataFrame([])
    ev['appl'] = events
    ev['state'] = state
    ev['ts'] = ev_ts
    ev['dpwr'] = dpwr
    ev = ev.dropna()
    ev.set_index('ts', inplace=True)

    if len(conflicts) > 0:  # if there are conflicts
        confl = pd.DataFrame(conflicts).T
        confl.columns = ['conflict']
        ev = pd.concat([ev, confl], axis=1)

        for i in range(5, ev.shape[0] - 5):
            if pd.isna(ev['conflict'].iloc[i]) == False:
                #         print(ev['conflict'].iloc[i],ev['appl'].iloc[i],ev['appl'].iloc[i-1])

                # check neighborhood -- 5 previous and 5 next points-- to decide if conflict will replace value
                if ev['conflict'].iloc[i] == ev['appl'].iloc[i - 5:i + 5].value_counts()[:1].index.tolist()[0]:
                    # print('appliance before conflict:',ev.iloc[i])
                    ev['appl'].iloc[i] = ev['conflict'].iloc[i]
                    # print('appliance after conflict:',ev.iloc[i])

    else:
        ev['conflict'] = np.nan
    #         ev.drop('conflict',axis=1,inplace=True)

    return ev


# heatpump post process
def hp_postproc(events, mappings):
    phases = ['A', 'B', 'C']

    if len(events) == 3:
        # encode conflicts column with corresponding dictionary values
        for i in range(0, 3):
            if events[i]['conflict'].notnull().sum() > 0:
                events[i].replace({'conflict': {v: k for k, v in mappings[i].items()}}, inplace=True)

        # indexes of rows where heatpump was found at each phase
        indA = events[0].loc[events[0]['appl'] == max(mappings[0], key=lambda k: mappings[0][k] == 'heatpumpA')].index
        indB = events[1].loc[events[1]['appl'] == max(mappings[1], key=lambda k: mappings[1][k] == 'heatpumpB')].index
        indC = events[2].loc[events[2]['appl'] == max(mappings[2], key=lambda k: mappings[2][k] == 'heatpumpC')].index

        # if there is an intersection between at least two phases' indexes, assign appliance=heatpump on the ohter phase
        events[0].loc[indB.intersection(indC), 'appl'] = max(mappings[0], key=lambda k: mappings[0][k] == 'heatpumpA')
        events[1].loc[indA.intersection(indC), 'appl'] = max(mappings[1], key=lambda k: mappings[1][k] == 'heatpumpB')
        events[2].loc[indA.intersection(indB), 'appl'] = max(mappings[2], key=lambda k: mappings[2][k] == 'heatpumpC')

        # if heatpump is found at only one phase, this is a false positive. Replace with conflict or leave empty
        events[0].loc[indA.difference(indB), 'appl'] = events[0].loc[indA.difference(indB), 'conflict']
        events[1].loc[indB.difference(indC), 'appl'] = events[1].loc[indB.difference(indC), 'conflict']
        events[2].loc[indC.difference(indA), 'appl'] = events[2].loc[indC.difference(indA), 'conflict']

    return events


def postproc(events):
    # drop events corresponding to only one appearance of an appliance
    for i in range(0, len(events)):
        singlapp = events[i]['appl'].value_counts()
        if singlapp[singlapp == 1].shape[0] > 0:
            events[i] = events[i][events[i]['appl'] != singlapp[singlapp == 1].index.values[0]]
            print('appliance with one appearance:', mappings[i][singlapp[singlapp == 1].index.values[0]])
    return events


def main():
    # set phase and appliances' mappings to dictionaries
    mdlpath = 'models/'
    maphase = {'A': ['pwrA', 'rpwrA'], 'B': ['pwrB', 'rpwrB'], 'C': ['pwrC', 'rpwrC']}
    mdlphase = {'A': ['entilator', 'fridge', 'heatpumpA', 'oven', 'stove', 'vacuum', 'wash'],
                'B': ['coffee', 'dish', 'freezer', 'heatpumpB'], 'C': ['iron', 'ironpress', 'heatpumpC', 'PC']}

    # Download events for  device testdisag
    devid = '5163caf0-0d63-11eb-97dd-c792ed4b3104'

    end_ = (datetime.datetime.utcnow())
    end_ = end_ - datetime.timedelta(seconds=end_.second % 60, microseconds=end_.microsecond)
    start_ = end_ + relativedelta(minutes=-15)

    end_time = int(end_.replace(tzinfo=pytz.utc).timestamp()) * 1000
    start_time = int(start_.replace(tzinfo=pytz.utc).timestamp()) * 1000

if __name__ == "__main__":
    # sys.exit(main(sys.argv))
    sys.exit(main())