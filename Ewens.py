import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.stats import chisquare
import numpy as np
from scipy.optimize import fsolve



def plot_stats(name ,cut ,range_):

    toplot10 =pd.read_csv("output\\dynamic\\" +name +"_10min_stats_rollcut" +str(cut ) +"cut.csv",names=['modularity', 'active_traders','#links','<size>','allingroups'])
    toplot15 =pd.read_csv("output\\dynamic\\" +name +"_15min_stats_rollcut" +str(cut ) +"cut.csv",names=['modularity', 'active_traders','#links','<size>','allingroups'])
    toplot30 =pd.read_csv("output\\dynamic\\" +name +"_30min_stats_rollcut" +str(cut ) +"cut.csv",names=['modularity', 'active_traders','#links','<size>','allingroups'])
    toplot60 =pd.read_csv("output\\dynamic\\" +name +"_60min_stats_rollcut" +str(cut ) +"cut.csv",names=['modularity', 'active_traders','#links','<size>','allingroups'])
    toplot120 =pd.read_csv("output\\dynamic\\" +name +"_120min_stats_rollcut" +str(cut ) +"cut.csv",names=['modularity', 'active_traders','#links','<size>','allingroups'])
    toplot180 =pd.read_csv("output\\dynamic\\" +name +"_180min_stats_rollcut" +str(cut ) +"cut.csv",names=['modularity', 'active_traders','#links','<size>','allingroups'])
    toplot360 =pd.read_csv("output\\dynamic\\" +name +"_360min_stats_rollcut" +str(cut ) +"cut.csv",names=['modularity', 'active_traders','#links','<size>','allingroups'])
    toplot1440 =pd.read_csv("output\\dynamic\\" +name +"_1440min_stats_rollcut" +str(cut ) +"cut.csv",names=['modularity', 'active_traders','#links','<size>','allingroups'])

    for df in [toplot10 ,toplot15 ,toplot30 ,toplot60 ,toplot120 ,toplot180 ,toplot360 ,toplot1440]:

        df["traders/groups" ] =df.iloc[: ,1 ] /df.iloc[: ,2]
        df["traders/tradersingroups" ] =df.iloc[: ,1 ] /df.iloc[: ,-2]

    dfgroups =pd.DataFrame()
    dflinks =pd.DataFrame()
    dfmeansize =pd.DataFrame()
    dfallingroups =pd.DataFrame()
    dftradersgroupsratio =pd.DataFrame()
    dftraderstradersingroupsratio =pd.DataFrame()
    deltat =[10 ,15 ,30 ,60 ,120 ,180 ,360 ,1440]
    for dt ,df in zip(deltat ,[toplot10 ,toplot15 ,toplot30 ,toplot60 ,toplot120 ,toplot180 ,toplot360 ,toplot1440]):
        dfgroups["#groups_delta_t_" +str(dt) ] =df.iloc[: ,2]
    for dt ,df in zip(deltat ,[toplot10 ,toplot15 ,toplot30 ,toplot60 ,toplot120 ,toplot180 ,toplot360 ,toplot1440]):
        dflinks["#links_delta_t_" +str(dt) ] =df.iloc[: ,3]
    for dt ,df in zip(deltat ,[toplot10 ,toplot15 ,toplot30 ,toplot60 ,toplot120 ,toplot180 ,toplot360 ,toplot1440]):
        dfmeansize["<groups_size>_delta_t_" +str(dt) ] =df.iloc[: ,4]
    for dt ,df in zip(deltat ,[toplot10 ,toplot15 ,toplot30 ,toplot60 ,toplot120 ,toplot180 ,toplot360 ,toplot1440]):
        dfallingroups["traders_in_groups_delta_t_" +str(dt) ] =df.iloc[: ,5]
    for dt ,df in zip(deltat ,[toplot10 ,toplot15 ,toplot30 ,toplot60 ,toplot120 ,toplot180 ,toplot360 ,toplot1440]):
        dftradersgroupsratio["traders_groups_ratio_delta_t_" +str(dt) ] =df.iloc[: ,6]
    for dt ,df in zip(deltat ,[toplot10 ,toplot15 ,toplot30 ,toplot60 ,toplot120 ,toplot180 ,toplot360 ,toplot1440]):
        dftraderstradersingroupsratio["traders_to_traders_in_groups_ratio_delta_t_" +str(dt) ] =df.iloc[: ,7]
    fig3 = plt.figure(constrained_layout=True ,figsize=(16 ,30))
    f3_ax1 = plt.subplot2grid((3 ,2 ), (0, 0), rowspan=1 ,colspan=1)
    dfgroups.plot(  ax=f3_ax1)
    f3_ax2 = plt.subplot2grid((3 ,2), (0, 1), rowspan=1 ,colspan=1)
    dflinks.plot(  ax=f3_ax2)
    f3_ax3 = plt.subplot2grid((3 ,2), (1, 0), rowspan=1 ,colspan=1)
    dfmeansize.plot(  ax=f3_ax3)
    f3_ax4 = plt.subplot2grid((3 ,2), (1, 1), rowspan=1 ,colspan=1)
    dfallingroups.plot(  ax=f3_ax4)
    f3_ax5 = plt.subplot2grid((3 ,2), (2, 0), rowspan=1 ,colspan=1)
    dftradersgroupsratio.plot(  ax=f3_ax5)
    f3_ax6 = plt.subplot2grid((3 ,2), (2, 1), rowspan=1 ,colspan=1)
    dftraderstradersingroupsratio.plot(  ax=f3_ax6)
    #plot_stats2("year_month_EURUSD" ,100 ,35)


def proportionvectorplot(name, minutes, cut, window="6months_2weeks"):
    eurusdcommunities = get_community_files(name, minutes, cut, window)
    eurusdcommunities.sort(key=lambda x: int(x.split('_')[8]))
    result = []
    M = 0

    for name in eurusdcommunities:
        d = pd.read_csv('output\\dynamic_clusters\\' + name)
        unique_elements, counts_elements = np.unique(d.groupby('cluster').count().sort_values('trader').values.ravel(),
                                                     return_counts=True)
        try:
            M = max(M, unique_elements[-1])
        except:
            M = max(M, 0)

    for name in eurusdcommunities:
        d = pd.read_csv('output\\dynamic_clusters\\' + name)
        unique_elements, counts_elements = np.unique(d.groupby('cluster').count().sort_values('trader').values.ravel(),
                                                     return_counts=True)

        dictionary = dict(zip(unique_elements, counts_elements))
        partitionvector = [dictionary[i] if i in unique_elements else 0 for i in range(M + 1)]
        result.append(partitionvector)
    dataframe = pd.DataFrame(result)
    fig, ax = plt.subplots()
    dataframe.plot(kind='area', stacked=True, figsize=(35, 5), ax=ax)
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.25),
        fancybox=True, shadow=True, markerscale=20, ncol=10)
    ax.set_xlabel("sliding window")
    ax.set_ylabel("K_n")
    plt.show()

    fig, ax2 = plt.subplots()
    dataframe.div(dataframe.sum(axis=1), axis=0).plot(kind='area', stacked=True, figsize=(35, 5), ax=ax2)
    ax2.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.25),
        fancybox=True, shadow=True, markerscale=20, ncol=10)
    ax2.set_xlabel("sliding window")
    ax2.set_ylabel("proportion")
    plt.show()

    def f(s):
        try:
            return s[s != 0].index[-1]
        except:
            return 0

    fig, ax3 = plt.subplots()
    dataframe.apply(f, axis=1).plot(figsize=(35, 5), ax=ax3)
    ax3.set_xlabel("sliding window")
    ax3.set_ylabel("$a_*$")
    plt.show()

    fig, ax4 = plt.subplots()
    dataframe.sum(axis=1).plot(figsize=(35, 5), ax=ax4)
    ax4.set_xlabel("sliding window")
    ax4.set_ylabel("K_n")
    plt.show()
    fig, ax5 = plt.subplots()
    dataframe.sum(axis=1).diff().plot(figsize=(35, 5), ax=ax5)
    ax5.set_xlabel("sliding window")
    ax5.set_ylabel("K_n diff")
    plt.show()

    fig, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(35, 5))
    dataframe.apply(f, axis=1).plot(figsize=(35, 5), ax=ax3)
    ax3.set_xlabel("sliding window")
    ax3.set_ylabel("$a_*$")

    dataframe.sum(axis=1).plot(figsize=(35, 5), ax=ax4)
    ax4.set_xlabel("sliding window")
    ax4.set_ylabel("K_n")

    dataframe.sum(axis=1).diff().plot(figsize=(35, 5), ax=ax5)
    ax5.set_xlabel("sliding window")
    ax5.set_ylabel("K_n diff")
    plt.show()


    #proportionvectorplot("EURUSD", 10, 100)


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





def get_community_files(name, minutes, cut, window="year_month"):
    mypath = "output\\dynamic_clusters\\"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    result = [f for f in onlyfiles if ((str(cut) + "cut" in f) & ("community" in f) & (name in f) & (window in f) & (
                "_" + str(minutes) + "min" in f))]
    return result


#eurusdcommunities = get_community_files("EURUSD", 10, 100)


def f(n, k):
    def func(x):
        denominator = np.sum([x / (x + i) for i in range(n)])
        return denominator - k

    tau_initial_guess = 1
    tau_solution = fsolve(func, tau_initial_guess)

    return tau_solution


def proportion_vector_evans_estimation(name, minutes, cut, window="year_month"):
    eurusdcommunities = get_community_files(name, minutes, cut, window)
    eurusdcommunities.sort(key=lambda x: int(x.split('_')[8]))
    result = []
    M = 0

    for name in eurusdcommunities:
        d = pd.read_csv('output\\dynamic_clusters\\' + name)
        unique_elements, counts_elements = np.unique(d.groupby('cluster').count().sort_values('trader').values.ravel(),
                                                     return_counts=True)
        try:
            M = max(M, unique_elements[-1])
        except:
            M = max(M, 0)

    for name in eurusdcommunities:
        d = pd.read_csv('output\\dynamic_clusters\\' + name)
        unique_elements, counts_elements = np.unique(d.groupby('cluster').count().sort_values('trader').values.ravel(),
                                                     return_counts=True)

        dictionary = dict(zip(unique_elements, counts_elements))
        partitionvector = [dictionary[i] if i in unique_elements else 0 for i in range(M + 1)]
        result.append(partitionvector)
    dataframe = pd.DataFrame(result)
    dataframe.iloc[:, 1] = 0

    K = dataframe.sum(axis=1).values
    N = dataframe.dot(np.arange(dataframe.shape[1])).values

    theta = np.array([f(N[i], K[i]) for i in range(len(K))]).ravel()

    return dataframe, K, N, theta


#d, k, n, theta = proportion_vector_evans_estimation("EURUSD", 10, 100)


def get_data_for_chitest_fit_ewens(n, d, start=5):
    tetas = []
    testpass = []

    for i in tqdm(np.arange(start, len(d))):

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
    theta_deltas = {}
    p_value_deltas = {}
    passtest_deltas = []
    deltas = [10, 15, 30, 60, 120, 180, 360, 1440]

    for delta in tqdm(deltas):
        d, k, n, theta = proportion_vector_evans_estimation(symbol, delta, cut)
        x, y, z = get_data_for_chitest_fit_ewens(n, d)
        # print(x,y,z)
        theta_deltas["symbol_" + symbol + "_delta" + str(delta) + "_cut" + str(cut)] = x
        p_value_deltas["symbol_" + symbol + "_delta" + str(delta) + "_cut" + str(cut)] = y
        passtest_deltas.append(z)
    pd.DataFrame(data=theta_deltas).plot()

    fig, ax3 = plt.subplots()
    pd.DataFrame(data=p_value_deltas).plot(ax=ax3)

    ax3.axhline(y=0.05, color='r', linestyle='-')
    plt.show()

    fig, ax4 = plt.subplots()
    ax4.plot(deltas, passtest_deltas)
    ax4.set_xlabel("deltas")
    ax4.set_ylabel("p pass rate")

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



