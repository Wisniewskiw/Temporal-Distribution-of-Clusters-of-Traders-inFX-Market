import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.stats import chisquare
import numpy as np
from scipy.optimize import fsolve
from scipy import stats
from numpy import arange, poly1d, random
from math import sqrt
from sklearn import preprocessing
from pathlib import Path
import os
import glob
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from majortrack import MajorTrack
from tqdm import tqdm_notebook
import warnings
import seaborn as sns

def indexbigger(l, x):
    out = -1
    for i in range(len(l)):
        if l[len(l) - 1 - i] >= 5:
            return len(l) - 1 - i, np.sum(l[len(l) - i:])
    return out, 0


def lambda_n_theta(n, theta):
    if n == 0:
        return 1
    elif n == 1:
        return 0
    else:
        table = np.zeros(n + 1, dtype=np.longdouble)
        table[0] = 1

        for i in np.arange(2, n + 1):
            table[i] = (i - 1) / (i - 1 + theta) * (table[i - 1] + table[i - 2] * (theta) / (theta + i - 2))

        return table[n], table


def logPochhammer(theta, n):
    if n < 0:
        return 0
    return np.sum([np.log(theta + i) for i in range(n)])


def Ewens_Expected_a_j(theta, n, j):
    if j < 1:
        return 0
    num = np.log(theta) + np.sum(np.log([np.arange(1, n + 1)])) + logPochhammer(theta, n - j)
    den = np.log(j) + np.sum(np.log([np.arange(1, n - j + 1)])) + logPochhammer(theta, n)
    return np.exp(num - den)





def f(n, k):
    def func(x):
        denominator = np.sum([x / (x + i) for i in range(n)])
        return denominator - k

    tau_initial_guess = 1
    tau_solution = fsolve(func, tau_initial_guess)

    return tau_solution



def proportion_vector_evans_estimation(name, minutes, cut, window="6months_2weeks",stratname="infomap"):
    eurusdcommunities = get_community_files(name, minutes, cut, window,stratname)


    eurusdcommunities.sort(key=lambda x: int(x.split('_')[10]))
    result = []
    M = 0

    for name in eurusdcommunities:
        d = pd.read_csv('output\\dynamic_clusters\\' + name)

        try:
            unique_elements, counts_elements = np.unique(
                d.groupby('cluster').count().sort_values('trader').values.ravel(),
                return_counts=True)
            M = max(M, unique_elements[-1])

        except:
            M = max(M, 0)

    for name in eurusdcommunities:
        d = pd.read_csv('output\\dynamic_clusters\\' + name)
        try:
            unique_elements, counts_elements = np.unique(d.groupby('cluster').count().sort_values('trader').values.ravel(),
                                                         return_counts=True)
        except:
            unique_elements, counts_elements=[0],[0]
        dictionary = dict(zip(unique_elements, counts_elements))
        partitionvector = [dictionary[i] if i in unique_elements else 0 for i in range(M + 1)]
        result.append(partitionvector)
         
    dataframe = pd.DataFrame(result)
    
    dataframe.iloc[:, 1] = 0

    K = dataframe.sum(axis=1).values
    N = dataframe.dot(np.arange(dataframe.shape[1])).values

    theta = np.array([f(N[i], K[i]) for i in range(len(K))]).ravel()

    return dataframe, K, N, theta




def get_data_for_chitest_fit_ewens(n, d, start=5):
    tetas = []
    testpass = []

    for i in tqdm_notebook(np.arange(start, len(d))):

        def ff(tt):

            ile = n[i]

            M = np.zeros((1, len(d) - start + 1))
            last, L = lambda_n_theta(ile, tt)
            L = (L / last)[::-1]

            for j in range(min(len(d) - start + 1, ile)):
                # print(j,tt,ile,Ewens_Expected_a_j(tt,ile,j),L)
                M[0, 1] = 0
                M[0, j] = Ewens_Expected_a_j(tt, ile, j) * L[j]

            index, suma = indexbigger(M[0, :], 5)

            if index == -1:
                return -1
            bc = 1

            add1 = np.zeros(index + 1)
            add1[-1] += np.sum(d.iloc[i, index + 1:].values)
            add2 = np.zeros(index + 1)
            add2[-1] += suma
            xx = d.iloc[i, 1:index + 2].values + add1
            yy = M[0, 1:index + 2] + add2
            xx = xx[1:]
            yy = yy[1:]
            # print((chisquare(xx,yy,1)[1]))
            return (chisquare(xx, yy, 1)[1])

        out = 0
        maxi = 0
        for j in (np.arange(1, 500)):
            new = ff(j)
            if new > maxi:
                maxi = new
                out = j

        tt = out

        M = np.zeros((1, len(d) - start + 1))
        last, L = lambda_n_theta(n[i], tt)
        L = (L / last)[::-1]
        tetas.append(tt)

        for j in range(min(len(d) - start + 1, n[i])):
            M[0, 1] = 0
            M[0, j] = Ewens_Expected_a_j(tt, n[i], j) * L[j]

        index, suma = indexbigger(M[0, :], 5)
        if index == -1:
            testpass.append(1)
            continue
        bc = 1

        add1 = np.zeros(index + 1)
        add1[-1] += np.sum(d.iloc[i, index + 1:].values)
        add2 = np.zeros(index + 1)
        add2[-1] += suma
        xx = d.iloc[i, 1:index + 2].values + add1
        yy = M[0, 1:index + 2] + add2
        xx = xx[1:]
        yy = yy[1:]

        # tetas.append(tt)
        testpass.append(chisquare(xx, yy, 1)[1])

    return tetas, testpass, np.sum([np.array(testpass) > 0.05]) / len(testpass)


def chitest_fit_ewens(symbol, cut):
    warnings.filterwarnings('ignore')
    theta_deltas = {}
    p_value_deltas = {}
    passtest_deltas = []
    deltas = [10, 15, 30, 60, 120, 180, 360, 1440]

    for delta in tqdm_notebook(deltas):
        d, k, n, theta = proportion_vector_evans_estimation(symbol, delta, cut)
        x, y, z = get_data_for_chitest_fit_ewens(n, d)
        # print(x,y,z)
        theta_deltas["symbol_" + symbol + "_delta" + str(delta) + "_cut" + str(cut)] = x
        p_value_deltas["symbol_" + symbol + "_delta" + str(delta) + "_cut" + str(cut)] = y
        passtest_deltas.append(z)

    fig, ax2 = plt.subplots()
    pd.DataFrame(data=theta_deltas).plot(ax=ax2)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    fig, ax3 = plt.subplots()
    pd.DataFrame(data=p_value_deltas).plot(ax=ax3)

    ax3.axhline(y=0.05, color='r', linestyle='-')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    fig, ax4 = plt.subplots()
    ax4.plot(deltas, passtest_deltas)
    ax4.set_xlabel("deltas")
    ax4.set_ylabel("p pass rate")
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    return pd.DataFrame(data=theta_deltas), pd.DataFrame(data=p_value_deltas), passtest_deltas


def passtestplot():
   
    a, b, c = chitest_fit_ewens("EURUSD",100)
    d, e, f = chitest_fit_ewens("EURUSD", 500)
    g, h, i = chitest_fit_ewens("EURUSD", 1000)

    SMALL_SIZE = 24
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    y = list(np.array(c) * 100)
    z = list(np.array(f) * 100)
    zz = list(np.array(i) * 100)
    x = list(np.arange(1, len(z) + 1))
    dff = pd.DataFrame(
        zip(x * 3, ['100 cut  pass rate'] * len(x) + ['500 cut  pass rate'] * len(x) + ['1000 cut  pass rate'] * len(x),
            y + z + zz), columns=["delta type", "kind", "data"])

    fig, ax4 = plt.subplots(figsize=(25, 8))
    ax4 = sns.barplot(x="delta type", hue="kind", y="data", data=dff)
    ax4.set_xticklabels([10, 15, 30, 60, 120, 180, 360, 1440])
    ax4.legend(bbox_to_anchor=(.9, -0.25), fancybox=False, shadow=False, ncol=3)
    ax4.set_title('Goodness of fit pass test ratio for every delta for all cutoffs')

    plt.show()

def passtestplot2(a,b,c,d,e,f,g,h,i):

    SMALL_SIZE = 24
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    y = list(np.array(c) * 100)
    z = list(np.array(f) * 100)
    zz = list(np.array(i) * 100)
    x = list(np.arange(1, len(z) + 1))
    dff = pd.DataFrame(
        zip(x * 3, ['100 cut  pass rate'] * len(x) + ['500 cut  pass rate'] * len(x) + ['1000 cut  pass rate'] * len(x),
            y + z + zz), columns=["delta type", "kind", "data"])

    fig, ax4 = plt.subplots(figsize=(25, 8))
    ax4 = sns.barplot(x="delta type", hue="kind", y="data", data=dff)
    ax4.set_xticklabels([10, 15, 30, 60, 120, 180, 360, 1440])
    ax4.legend(bbox_to_anchor=(.9, -0.25), fancybox=False, shadow=False, ncol=3)
    ax4.set_title('Goodness of fit pass test ratio for every delta for all cutoffs')

    plt.show()



def get_community_files(name, minutes, cut, window, stratname):


    mypath =  "output\\dynamic_clusters\\"

  

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


    result = [f for f in onlyfiles if ((str(cut) + "cut" in f) & ("community" in f) &(stratname in f)& (name in f) & (window in f) & (
                "_" + str(minutes) + "min" in f))]
    return result


def plot_stats(name, cut  ):

    #plt.ioff()

    toplot10 = pd.read_csv("output\\dynamic\\"+  name + "_10min_stats_rollcut" + str(cut) + "cut.csv",names=['modularity', 'active_traders',"groups",'#links','<size>','allingroups']).replace(-np.inf, 0 , regex=True)
    toplot15 = pd.read_csv("output\\dynamic\\"+ name + "_15min_stats_rollcut" + str(cut) + "cut.csv",names=['modularity', 'active_traders',"groups",'#links','<size>','allingroups']).replace(-np.inf, 0 , regex=True)
    toplot30 = pd.read_csv("output\\dynamic\\"+ name +  "_30min_stats_rollcut" + str(cut) + "cut.csv",names=['modularity', 'active_traders',"groups",'#links','<size>','allingroups']).replace(-np.inf, 0 , regex=True)
    toplot60 = pd.read_csv("output\\dynamic\\"+ name + "_60min_stats_rollcut" + str(cut) + "cut.csv",names=['modularity', 'active_traders',"groups",'#links','<size>','allingroups']).replace(-np.inf, 0 , regex=True)
    toplot120 = pd.read_csv("output\\dynamic\\"+ name +  "_120min_stats_rollcut" + str(cut) + "cut.csv",names=['modularity', 'active_traders',"groups",'#links','<size>','allingroups']).replace(-np.inf, 0 , regex=True)
    toplot180 = pd.read_csv("output\\dynamic\\"+ name +  "_180min_stats_rollcut" + str(cut) + "cut.csv",names=['modularity', 'active_traders',"groups",'#links','<size>','allingroups']).replace(-np.inf, 0 , regex=True)
    toplot360 = pd.read_csv("output\\dynamic\\"+name +  "_360min_stats_rollcut" + str(cut) + "cut.csv",names=['modularity', 'active_traders',"groups",'#links','<size>','allingroups']).replace(-np.inf, 0 , regex=True)
    toplot1440 = pd.read_csv("output\\dynamic\\"+ name +  "_1440min_stats_rollcut" + str(cut) + "cut.csv",names=['modularity', 'active_traders',"groups",'#links','<size>','allingroups']).replace(-np.inf, 0 , regex=True)


    cols= list(toplot1440.columns)+["traders/groups"]+["traders/tradersingroups"]


    for i,df in enumerate([toplot10, toplot15, toplot30, toplot60, toplot120, toplot180, toplot360, toplot1440]):


        try:
            df["traders/groups"] = df["groups"]/ df["active_traders"]
            df["traders/groups"] = df["allingroups"] / df["active_traders"]
        except:
            df["groups"][0]='0'
            df["groups"] = pd.to_numeric(df["groups"].values)

            try:


                df["traders/groups"] = df["groups"] / df["active_traders"]
                df["traders/groups"] = df["allingroups"] / df["active_traders"]
            except:
                df["traders/groups"] = df["groups"] / int(name.split('_')[1])
                df["traders/groups"] = df["allingroups"] / int(name.split('_')[1])
        df["traders/tradersingroups"] = df["groups"]/ df["allingroups"]
        df.columns = cols


    dfgroups = pd.DataFrame()
    dflinks = pd.DataFrame()
    dfmodularity = pd.DataFrame()
    dfmeansize = pd.DataFrame()
    dfallingroups = pd.DataFrame()
    dftradersgroupsratio = pd.DataFrame()
    dftraderstradersingroupsratio = pd.DataFrame()
    deltat = [10, 15, 30, 60, 120, 180, 360, 1440]

    for dt, df in zip(deltat, [toplot10, toplot15, toplot30, toplot60, toplot120, toplot180, toplot360, toplot1440]):

        dfgroups["#cluster_delta_t_" + str(dt)] = df["groups"]
    for dt, df in zip(deltat, [toplot10, toplot15, toplot30, toplot60, toplot120, toplot180, toplot360, toplot1440]):
        dflinks["#links_delta_t_" + str(dt)] = df["#links"]
    for dt, df in zip(deltat, [toplot10, toplot15, toplot30, toplot60, toplot120, toplot180, toplot360, toplot1440]):
        dfmeansize["<cluster_size>_delta_t_" + str(dt)] = df["<size>"]
    for dt, df in zip(deltat, [toplot10, toplot15, toplot30, toplot60, toplot120, toplot180, toplot360, toplot1440]):
        dfallingroups["traders_in_groups_delta_t_" + str(dt)] = df["allingroups"]
    for dt, df in zip(deltat, [toplot10, toplot15, toplot30, toplot60, toplot120, toplot180, toplot360, toplot1440]):
        dfmodularity["modularity_delta_t_" + str(dt)] = df['modularity']
    for dt, df in zip(deltat, [toplot10, toplot15, toplot30, toplot60, toplot120, toplot180, toplot360, toplot1440]):
        dftradersgroupsratio["traders_in_clusters_to_active_ratio_delta_t_" + str(dt)] = df["traders/groups"]
    for dt, df in zip(deltat, [toplot10, toplot15, toplot30, toplot60, toplot120, toplot180, toplot360, toplot1440]):
        dftraderstradersingroupsratio["#clusters_to_traders_in_clusters_ratio_delta_t_" + str(dt)] = df["traders/tradersingroups"]
    fig3 = plt.figure(constrained_layout=True, figsize=(36, 12))
    f3_ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=1, colspan=1)
    dfgroups.plot(ax=f3_ax1).legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    f3_ax2 = plt.subplot2grid((2, 4), (0, 1), rowspan=1, colspan=1)
    dflinks.plot(ax=f3_ax2).legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    f3_ax3 = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=1)
    dfmeansize.plot(ax=f3_ax3).legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    f3_ax4 = plt.subplot2grid((2, 4), (0, 3), rowspan=1, colspan=1)
    dfallingroups.plot(ax=f3_ax4).legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    f3_ax5 = plt.subplot2grid((2, 4), (1, 0), rowspan=1, colspan=1)
    dftradersgroupsratio.plot(ax=f3_ax5).legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    f3_ax6 = plt.subplot2grid((2, 4), (1, 1), rowspan=1, colspan=1)
    dftraderstradersingroupsratio.plot(ax=f3_ax6).legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    f3_ax7 = plt.subplot2grid((2, 4), (1, 2), rowspan=1, colspan=1)
    dfmodularity.plot(ax=f3_ax7).legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    plt.tight_layout()

    plt.savefig('output\\plots\\network_stats_'+name+'_cut_'+str(cut)+'.png')
    #plt.close(fig3)
    plt.show()





def proportionvectorplot(name2, minutes, cut, window, stratname ):
    eurusdcommunities = get_community_files(name2, minutes, cut, window, stratname)
    eurusdcommunities.sort(key=lambda x: int(x.split('_')[-3]))

    result = []
    M = 0
    plt.ioff()
    mypath ="output\\dynamic_clusters\\"


    for name in eurusdcommunities:

        d = pd.read_csv(mypath + name)

        le = preprocessing.LabelEncoder()
        if (len(d)==0): continue

        le.fit(d['cluster'].values)
        d['cluster'] = le.transform(d.cluster.values)
        d['trader'] = [int(x[1:]) for x in d.trader.values]

        unique_elements, counts_elements = np.unique(d.groupby('cluster').count().sort_values('trader').values.ravel(),
                                                     return_counts=True)

        try:
            M = max(M, unique_elements[-1])
        except:
            M = max(M, 0)

    for name in eurusdcommunities:
        d = pd.read_csv(mypath + name)

        if (len(d) == 0): continue
        le = preprocessing.LabelEncoder()
        le.fit(d.cluster.values)
        d['cluster'] = le.transform(d.cluster.values)
        d['trader'] = [int(x[1:]) for x in d.trader.values]

        unique_elements, counts_elements = np.unique(d.groupby('cluster').count().sort_values('trader').values.ravel(),
                                                     return_counts=True)

        dictionary = dict(zip(unique_elements, counts_elements))
        partitionvector = [dictionary[i] if i in unique_elements else 0 for i in range(M + 1)]
        result.append(partitionvector)

    try:


        #plt.ioff()
        dataframe = pd.DataFrame(result)
        fig, ax = plt.subplots()
        dataframe.plot(kind='area', stacked=True, figsize=(35, 5), ax=ax)
        ax.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.25),
            fancybox=True, shadow=True, markerscale=20, ncol=10)
        ax.set_xlabel("sliding window")
        ax.set_ylabel("K_n")
        saveto = 'stratname_' + str(stratname) + '_' + str(name2) + '_delta_' + str(minutes) + '_cut_' + str(
            cut) + '_window_' + str(window)
        savepath =  'output\\plots\\'

        plt.savefig(savepath + 'proportion_vector_' + saveto + '.png')
        ig, ax2 = plt.subplots()
        dataframe.div(dataframe.sum(axis=1), axis=0).plot(kind='area', stacked=True, figsize=(35, 5), ax=ax2)
        ax2.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.25),
            fancybox=True, shadow=True, markerscale=20, ncol=10)
        ax2.set_xlabel("sliding window")
        ax2.set_ylabel("proportion")
        plt.show()
        plt.savefig(savepath + 'proportion_vector_normalised_' + saveto + '.png')


        #plt.close(fig)
    except:
        print('no data')



def cluster_dynamics(name2, minutes, cut, window, stratname, truncate=1):
    eurusdcommunities = get_community_files(name2, minutes, cut, window, stratname)
    eurusdcommunities.sort(key=lambda x: int(x.split('_')[-3]))

    result = []
    M = 0
    plt.ioff()
    mypath ="output\\dynamic_clusters\\"
    output=[]
    alltraders=[]

    for name in eurusdcommunities:


        d = pd.read_csv(mypath + name)

        le = preprocessing.LabelEncoder()

        if (len(d) == 0): continue

        le.fit(d['cluster'].values)
        d['cluster'] = le.transform(d.cluster.values)
        d['trader'] = [int(x[1:]) for x in d.trader.values]
        alltraders.append(d['trader'].values)
    import itertools

    alltraders=list(itertools.chain(*alltraders))
    le2 = preprocessing.LabelEncoder()
    le2.fit(np.unique(np.array(alltraders)))

    for name in eurusdcommunities:

        d = pd.read_csv(mypath + name)

        le = preprocessing.LabelEncoder()
        if (len(d) == 0): continue

        le.fit(d['cluster'].values)
        d['cluster'] = le.transform(d.cluster.values)
        d['trader'] = [int(x[1:]) for x in d.trader.values]
        d['trader'] = le2.transform(d.trader.values)
        output.append(list(d.groupby('cluster')['trader'].apply(list).values))
    if truncate:
        return [[x for x in output[i] if len(x) >= truncate] for i in range(len(output))]
    return output


def alluvial(name2, minutes, cut, window, stratname,  truncate=1):

    for i in range(50):
        plt.clf()
        plt.cla()
        plt.close()
        try:

            alluvial_try(name2, minutes, cut, window, stratname,i,truncate)
            break
        except:
            pass
    print("ommited steps because of errors: ",i)


def alluvial_try(name2, minutes, cut, window, stratname, num,truncate=1):
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 20, 40, 80
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    # #############################################################################
    # Define the data time-sequence data
    o=cluster_dynamics(name2, minutes, cut, window, stratname, truncate)
    o=o[num:]
    _time_windows = [
            [-0.5+k, 0.5+k] for k in range(len(o))
            ]
    time_windows = [[10*el for el in tw] for tw in _time_windows]
    # 1. The grouping
    # # at t=0:


    individuals = [set(itertools.chain(*x)) for x in o]

    groupings = o.copy()

    groupings = [[set(grp) for grp in groups] for groups in groupings]

    # #############################################################################
    # Initiate the algorithm
    mt = MajorTrack(
            clusterings=groupings,
            individuals=individuals,
            history=0,
            timepoints=[tw[0] for tw in time_windows]
        )
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 20, 40, 80
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # #############################################################################
    # Define the data time-sequence data

    _time_windows = [
            [-0.5+k, 0.5+k] for k in range(len(o))
            ]
    time_windows = [[10*el for el in tw] for tw in _time_windows]
    # 1. The grouping
    # # at t=0:


    individuals = [set(itertools.chain(*x)) for x in o]

    groupings = o.copy()

    groupings = [[set(grp) for grp in groups] for groups in groupings]

    # #############################################################################
    # Initiate the algorithm
    mt = MajorTrack(
            clusterings=groupings,
            individuals=individuals,
            history=0,
            timepoints=[tw[0] for tw in time_windows]
        )
    mt.get_group_matchup('fraction')
    # create the different instances with different history parameters
    mt1 = deepcopy(mt)
    mt5 = deepcopy(mt)
    mt5.history = len(o)
    mt.get_dcs()
    mt.get_community_group_membership()
    mt.get_community_membership()
    mt.get_community_coloring()
    # use same colours for other visualizations
    comm_colours = list(mt.comm_colours)
    sp_commm_color_idx = dict(mt.sp_community_colour_idx)
    mt1.comm_colours = list(comm_colours)
    mt1.sp_community_colour_idx = dict(sp_commm_color_idx)
    mt5.comm_colours = list(comm_colours)
    mt5.sp_community_colour_idx = dict(sp_commm_color_idx)
    # 1 step memory
    mt1.history = 1
    mt1.get_dcs()
    mt1.get_community_group_membership()
    mt1.get_community_membership()
    # 5 step memory
    mt5.history = len(o)
    mt5.get_dcs()
    mt5.get_community_group_membership()
    mt5.get_community_membership()

    plot_params = {
            'cluster_width': 2,
            'flux_kwargs': {'alpha': 0.2, 'lw': 0.0, 'facecolor': 'cluster'},
            'cluster_kwargs': {'alpha': 1.0, 'lw': 0.0},
            'label_kwargs': {'fontweight': 'heavy'},
            'with_cluster_labels': False,
            'cluster_label': 'group_index',
            'cluster_label_margin': (-1.6, 0.1),
            'x_axis_offset': 0.07,
            'redistribute_vertically': 1,
            'cluster_location': 'center',
            'y_fix': {
                20.0: [('4', '7'), ('0', '1'), ('4', '3')],
                30.0: [('0', '3')]
                }
            }

    rawmt = deepcopy(mt1)

    # Single
    # #############################################################################
    # The trace back (memory) part
    # the merging illustration
    sankey_plot_params = dict(plot_params)
    sankey_plot_params.update({
            'merged_edgecolor': 'none',  # 'xkcd:gray',
            'merged_linewidth': 10,
            'cluster_facecolor': 'community',
            'cluster_edgecolor': 'community',
            'flux_facecolor': 'cluster',
            'flux_edgecolor': 'cluster'
            })
    # raw image
    spp_raw = deepcopy(sankey_plot_params)
    spp_raw['l_size'] = 7
    spp_raw['cluster_facecolor'] = 'xkcd:gray'
    spp_raw['default_cluster_facecolor'] = 'xkcd:gray'

    # 1 step
    spp_1step = deepcopy(sankey_plot_params)
    spp_1step['l_size'] = 9
    # 5 step
    spp_5step = deepcopy(sankey_plot_params)
    spp_5step['l_size'] = 9


    def _set_axis(axes, mt, spp, with_xaxis=True):
        axes.axis('equal')
        l_size = spp.pop('l_size', 9)
        mt.get_alluvialdiagram(
                axes,
                invisible_x=not with_xaxis,
                **spp,
                )
        if with_xaxis:
            tp = [
                    (t + .5*(mt.slice_widths[i])) + .5*plot_params['cluster_width']
                    for i, t in enumerate(mt.timepoints)
                    ]
            axes.set_xticks(tp, minor=False)
            # ax_tb3.xaxis.tick_top()
            axes.set_xticklabels(
                    [
                        r'$\mathbf{{t{0}}}$'.format(idx)
                        for idx in range(len(mt.timepoints))
                        ],
                    minor=False,
                    size=l_size
                    )
            axes.tick_params(axis=u'x', which=u'both', length=0)
            plt.setp(axes.get_xticklabels(), visible=True)
        return axes


    def set_raw_axes(axes, mt=rawmt, spp=spp_raw, with_xaxis=True):
        return _set_axis(axes, mt, spp, with_xaxis)


    def set_one_axes(axes, mt=mt1, spp=spp_1step, with_xaxis=True):
        return _set_axis(axes, mt, spp, with_xaxis)


    def set_five_axes(axes, mt=mt5, spp=spp_5step, with_xaxis=True):
        return _set_axis(axes, mt, spp, with_xaxis)




    if True:
        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 20, 40, 80
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig1 = plt.figure(figsize=(24, 12.0))
        gsIllust = gridspec.GridSpec( 22, 20, left=0.03, wspace=0.0,
                hspace=0.0, top=0.98, bottom=0.07, right=0.97)
        ax_illust_raw = fig1.add_subplot(gsIllust[1:7, 7:13])
        ax_illust_raw = set_raw_axes(ax_illust_raw)
        ax_illust_raw.xaxis.set_ticks_position('top')
        ax_illust_raw.annotate(
                'sequence\nof\nclusterings', (0.51, -0.2),
                xycoords='axes fraction', size=8, ha='center', va='center',
                fontweight='heavy',
                )

        ax_illust_one = fig1.add_subplot(gsIllust[10:25, :9])
        ax_illust_one = set_one_axes(ax_illust_one)
        ax_illust_one.patch.set_visible(False)
        ax_illust_one.set_title(
                '1-step history', fontdict={'fontweight': 'heavy'})
        ax_illust_five = fig1.add_subplot(gsIllust[10:22, 11:])
        ax_illust_five = set_five_axes(ax_illust_five)
        ax_illust_five.patch.set_visible(False)
        ax_illust_five.set_title(
                'whole-step history', fontdict={'fontweight': 'heavy'})
        #plt.tight_layout()
        plt.show()
        fig1.savefig('history.png')
        fig1.show()


def get_dynamic_clusters_start_end_dates(window, stratname="infomap"):
    for cut in tqdm_notebook([100, 500, 1000]):
        df1440 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_1440mins' + str(cut) + 'cut.csv',
                             parse_dates=['QdfTime'], infer_datetime_format=True)
        cutday = df1440.QdfTime[1]
        cutdaystop = df1440.QdfTime[len(df1440) - 2]
        df1440 = df1440[(df1440.QdfTime < cutdaystop) & (df1440.QdfTime >= cutday)]

        files = get_community_files("EURUSD", 1440, cut, window, stratname)
        files.sort(key=lambda x: int(x.split('_')[-3]))
        out = []

        for x in files:
            start = int(x.split("_")[10])
            stop = int(x.split("_")[11])
            out.append((df1440.QdfTime[start], df1440.QdfTime[stop]))

        df = pd.DataFrame.from_records(out, columns=['start', 'end'])
        df['shift_start'] = df.start.shift(-1)
        df.to_csv(
            "output\\dynamic_clusters\\clusters_start_end_dates_dynamic_cut_" + str(cut) + '_' + window + '_cut.csv',
            index=False)