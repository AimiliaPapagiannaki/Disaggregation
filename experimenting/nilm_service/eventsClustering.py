import numpy as np
import sys

import pytz
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import pickle

from pandas.plotting import parallel_coordinates
from sklearn import preprocessing
import matplotlib.pyplot as plt


def coupled_clusters(mdlON, mdlOFF, dataOn, dataOff):
    cnterON = mdlON.cluster_centers_
    cnterOFF = mdlOFF.cluster_centers_

    couples_ind = [np.argmin([euclidean_distances(cON.reshape(1, -1), cOFF.reshape(1, -1)) for cOFF in cnterOFF]) for
                   cON in cnterON]
    print('Coupled clusters:', couples_ind)

    # Create a separate training set for each appliance
    dfs = []
    for i in range(0, len(couples_ind)):
        globals()['appl{}'.format(i)] = pd.concat(
            [dataOn.loc[dataOn['label'] == i], dataOff.loc[dataOff['label'] == couples_ind[i]]])
        dfs.append(globals()['appl{}'.format(i)])

    return dfs


def train_coupled_clusters(dfs, phase, devserial, mdlpath):
    """This function is used to train the models of a house's appliances.
    A one-versus-rest binary technique is deployed, where every model
    assumes that the label is 1 during the appliance's operation,
    and 0 during any other appliance's operation. So there is a mutual exclusion."""

    # create directory with device serial
    mdlpath = mdlpath + str(devserial) + '/' + str(phase)
    if not os.path.exists(mdlpath):
        os.makedirs(mdlpath)

    # calculate average active power of each appliance
    avpwr = dict()
    for i in range(len(dfs)):
        # rearrange order so that the ith dataframe is first.
        dfcurr = pd.concat(sorted(dfs.copy(), key=lambda df: df is dfs[i], reverse=True))

        # label is 1 for ith df and 0 everywhere else
        dfcurr['label'].iloc[:dfs[i].shape[0]] = 1
        dfcurr['label'].iloc[dfs[i].shape[0] + 1:] = 0

        # compute average active power of each appliance
        avpwr['appl_%s' % i] = np.mean(dfcurr.loc[dfcurr['label'] == 1, 'avgP' + str(phase) + '_4'])
        print('average power of %i is %f' % (i, avpwr['appl_%s' % i]))

        y = dfcurr['label'].values
        X = dfcurr.drop('label', axis=1)

        # Perform grid search to tune hyperparameters
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        parameters = {'min_samples_split': range(2, 20), 'max_depth': np.arange(5, 30)}
        gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                          param_grid=parameters,
                          scoring=scoring, refit='AUC', return_train_score=True, n_jobs=-1)
        gs.fit(X, y)
        print('Best score from grid search:', gs.best_score_)
        # after grid search, train entire dataset with best parameters
        mdl = DecisionTreeClassifier(random_state=42, max_depth=gs.best_params_['max_depth'],
                                     min_samples_leaf=gs.best_params_['min_samples_split'])
        mdl = mdl.fit(X, y)

        filename = mdlpath + '/appl_%s' % i + '.sav'
        pickle.dump(mdl, open(filename, 'wb'))

    # save average power to csv
    avpwr = pd.DataFrame(avpwr.items())
    avpwr.columns = ['app', 'pwr']
    avpwr.to_csv(mdlpath + '/avg_pwr.csv', index=False)

    del avpwr, dfcurr, mdl, gs, X, y

    return


# optimum number of clusters
def find_nclusters(df):
    range_clusters = list(range(4, 11))
    sil_ind = {}
    for ncl in range_clusters:
        mdlON = AgglomerativeClustering(n_clusters=ncl)
        labels = mdlON.fit_predict(df)
        sil_ind[ncl] = silhouette_score(df, labels)
        print('index %i and silhouette score %f' % (ncl, sil_ind[ncl]))

    # optimum number of clusers is the one with the highest score
    nclusters = max(sil_ind, key=sil_ind.get)

    return nclusters


def cluster_events(phase, df, devserial, mdlpath):
    """This module separates events to discrete clusters (aka appliances). Events are split to ON and OFF categories,
    and then a matching is performed to detect which cluster is the OFF of each ON cluster respectively.
    Last, training is performed for each ON-OFF coupled cluster and models are stored for future use."""

    # Split On/Off events in two sets
    dataOn = df.copy() # test without splittin  to on/off
    #dataOn = df.loc[df['sign_' + str(phase)] == 1].copy()
    dataOn = dataOn[dataOn.columns[-32:]]
    #dataOn['tdif'] = dataOn.index
    #dataOn['tdif'] = (dataOn['tdif']-dataOn['tdif'].shift())/1000
    #dataOn = dataOn.iloc[1:]
    #dataOn = dataOn.drop(['stdPA_1','stdPA_2','stdPA_3','stdPA_4','stdRA_1','stdRA_2','stdRA_3','stdRA_4'], axis=1)
  
    
    dataOff = df.loc[df['sign_' + str(phase)] == -1].copy()
    dataOff = dataOff[dataOff.columns[-32:]]
    #dataOff['tdif'] = dataOff.index
    #dataOff['tdif'] = (dataOff['tdif']-dataOff['tdif'].shift())/1000
    #dataOff = dataOff.iloc[1:]
    #dataOff = dataOff.drop(['stdPA_1','stdPA_2','stdPA_3','stdPA_4','stdRA_1','stdRA_2','stdRA_3','stdRA_4'], axis=1)

    # # store sign and stable power to separate variables
    # sign = df['sign_' + str(phase)].copy()
    # sign = sign.to_frame()
    # stable_pwr = df['avgSP_' + str(phase)].copy()
    # stable_pwr = stable_pwr.to_frame()

    # df = df.drop(['sign_' + str(phase)], axis=1)
    # df = df.drop(['avgSP_' + str(phase)], axis=1)

    # find optimum number of clusters aka appliances
    dataOn = dataOn.loc[(dataOn['avgP'+str(phase)+'_4']>40) & (dataOn['avgP'+str(phase)+'_3']>40)]
    dataOff = dataOff.loc[(dataOff['avgP'+str(phase)+'_4']>40) & (dataOff['avgP'+str(phase)+'_3']>40)]
    
    nclusters = find_nclusters(dataOn)
    #nclusters=5

    print('Optimum number of clusters appears to be:', nclusters)
    # model and labels of events "ON"
    #mdlON = MiniBatchKMeans(n_clusters=nclusters, batch_size=20, random_state=0)
    mdlON = AgglomerativeClustering(n_clusters=nclusters)
    mdlON = mdlON.fit(dataOn)
    labelsON = mdlON.fit_predict(dataOn)
    dataOn['label'] = labelsON
    
    
    tmp=dataOn.copy()
    #tmp = tmp[['avgRA_1','avgRA_2','avgRA_3','avgRA_4','label']]
    tmp = tmp[['maxPA_4','minPA_4','avgPA_4','label']]
    plt.figure(figsize=(14, 10))
    parallel_coordinates(tmp, 'label',colormap='rainbow')
    #plt.scatter(dataOn[dataOn.columns[16]], dataOn[dataOn.columns[24]], c=labelsON, cmap='rainbow')
    plt.savefig("myfig.png")

    # model and labels of events "OFF"
    # mdlOFF = MiniBatchKMeans(n_clusters=nclusters, batch_size=20, random_state=0)
    mdlOFF = AgglomerativeClustering(n_clusters=nclusters)
    mdlOFF = mdlOFF.fit(dataOff)
    labelsOFF = mdlOFF.fit_predict(dataOff)
    dataOff['label'] = labelsOFF

    tmp = pd.DataFrame([])
    tmp = pd.concat([tmp, dataOn[['label']]], axis=1)
    tmp = pd.concat([tmp, dataOff[['label']]], axis=1)
    tmp['dt'] = pd.to_datetime(tmp.index, unit='ms')
    tmp['dt'] = tmp['dt'].dt.tz_localize('utc').dt.tz_convert('Europe/Athens')
    tmp['dt'] = tmp['dt'].dt.tz_localize(None)
    tmp.set_index('dt', inplace=True, drop=True)
    tmp.columns = ['On', 'Off']
    tmp.to_excel('data.xlsx')

    # detect couples of ON-OFF clusters and then proceed to training models
    dfs = coupled_clusters(mdlON, mdlOFF, dataOn, dataOff)
    train_coupled_clusters(dfs, phase, devserial, mdlpath)

    # sign = sign.loc[sign['signA'] > 0]
    # sign['labels'] = labels
    return