import os
import pandas as pd
from datetime import datetime

def fixOverlapping(df):
    df['dur'] = df['event_end'] - df['event_start']
    df = df.sort_values(by = ['dur'], ascending = False)
    df = df.drop_duplicates(subset=['event_start'], keep='first')
    df = df.drop_duplicates(subset=['event_end'], keep='first')
    return df.sort_index().drop(['dur'], axis=1).reset_index(drop=True)

dir_path = '/home/azureuser/gen_models/waterHeater'
df1 = pd.DataFrame()
df2 = pd.DataFrame()

device = ['102.402.000045', '102.402.000751']
dfs = []
for file_path in os.listdir(dir_path):
    if len(file_path.split('-')) == 6:
        if file_path.split('_')[0] == device[0]:
            df1 = pd.concat([df1, pd.read_json(file_path)], axis=0)
            os.system(f'sudo rm {file_path}')
        elif file_path.split('_')[0] == device[1]:
            df2 = pd.concat([df2, pd.read_json(file_path)], axis=0)
            os.system(f'sudo rm {file_path}')

dfs.extend([df1, df2])

for i in range(len(dfs)):
    dfs[i] = dfs[i].drop_duplicates().reset_index(drop=True)
    dfs[i] = fixOverlapping(dfs[i]).sort_values(by=['event_start', 'event_end'], ascending=True).reset_index(drop=True)
    start_ts = dfs[i].iloc[0].event_start
    end_ts = dfs[i].iloc[-1].event_end
    start_ts = datetime.fromtimestamp(start_ts / 1e3).date()
    end_ts = datetime.fromtimestamp(end_ts / 1e3).date()
    dfs[i].to_json(f'{device[i]}_{str(start_ts)}-{str(end_ts)}.json')
