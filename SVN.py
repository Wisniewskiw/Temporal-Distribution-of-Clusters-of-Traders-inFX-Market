import pandas as pd
import numpy  as  np
from py_files  import Wrapper as w
import json
import itertools
import warnings
from tqdm import tqdm_notebook
from os import listdir
from os.path import isfile, join
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from karateclub.community_detection.overlapping import EgoNetSplitter, NNSED, DANMF, MNMF, BigClam, SymmNMF
from karateclub.community_detection.non_overlapping import EdMot, LabelPropagation, SCD, GEMSEC

class PrepareData(object):
    '''
    Mimic strat
    '''

    def __init__(self, parameters_dict):
        '''

        :param parameters_dict: dict with many parameters

        '''

        if parameters_dict is None:
            parameters_dict = w.MyWrapper({})
        else:
            parameters_dict = w.MyWrapper(parameters_dict)

        self.df_path = parameters_dict.df_path
        self.openTimeColumnName = parameters_dict.openTimeColumnName
        self.closeTimeColumnName = parameters_dict.closeTimeColumnName
        self.lower_time_cutoff = parameters_dict.lower_time_cutoff
        self.upper_time_cutoff = parameters_dict.upper_time_cutoff
        self.time_resolution = parameters_dict.time_resolution
        self.min_number_trades_cutoff = parameters_dict.min_number_trades_cutoff
        self.symbol = parameters_dict.symbol
        self.imbalance_ratio_threshold = parameters_dict.imbalance_ratio_threshold
        self.all_partition = parameters_dict.all_partition

        self.lower_volume_quantile_cutoff = parameters_dict.lower_volume_quantile_cutoff
        self.upper_volume_quantile_cutoff = parameters_dict.upper_volume_quantile_cutoff
        # self.save_name = parameters_dict.save_name

    def construct_states_vs_t_matrix(self):
        #print("LOADING DATASET...")
        self.load_df()
        #print("DONE")
        #print("PREPARING DATASET...")
        self.prepare_df()
        #print("DONE")
        #print("SELECTING SUBSET WITH RESPECT TO CONDITIONS...")
        self.select_sub_df()
        #print("DONE")
        #print("PREPARING STATESvsTIME MATRIX FOR SVN...")

        return self.SVN_prepared_df()

    def load_df(self):
        '''

        :return:
        '''

        self.df = pd.read_csv(self.df_path, parse_dates=[self.openTimeColumnName, self.closeTimeColumnName, 'QdfTime'],
                              infer_datetime_format=True)
        self.df['Symbol']="EUR/USD"
        self.df['Side']= self.df['Type'].map({'buy': 1, 'sell': -1})

    def prepare_df(self):
        self.df = self.df[self.df.ClosePrice >0 ]
        self.df = self.df[self.df[self.openTimeColumnName] != self.df[self.closeTimeColumnName]]
        self.df = self.df[self.df.Amount != 0]
        df1 = self.df[[  'QdfOpenTime', 'QdfCloseTime',
                         'TraderId', 'OrderId',
                        'Amount', 'Symbol',   'Side',
                        ]]
        df1['QdfTime'] = df1['QdfOpenTime']

        df2 = self.df[[  'QdfOpenTime', 'QdfCloseTime',
                         'TraderId', 'OrderId',
                        'Amount', 'Symbol',   'Side',
                        ]]
        df2['QdfTime'] = df2['QdfCloseTime']

        self.df = df1.append(df2, ignore_index=True)
        del df1, df2

    def select_sub_df(self):
        self.df = self.df[(self.df.Symbol == self.symbol)]
        most_activ_clients = self.df.groupby('TraderId').TraderId.count()
        most_activ_clients = most_activ_clients[most_activ_clients >= self.min_number_trades_cutoff]
        self.df = self.df[self.df.TraderId.isin(most_activ_clients.index)]
        #if self.lower_time_cutoff != self.upper_time_cutoff:
        #    self.df['hour'] = self.df.QdfTime.dt.hour
        #    self.df = self.df[(self.df.hour >= self.lower_time_cutoff) & (self.df.hour <= self.upper_time_cutoff)]

    def SVN_prepared_df(self):
        self.df['transaction_volume'] = self.df.Amount * self.df.Side * (
                self.df.QdfTime == self.df[self.openTimeColumnName]) - self.df.Amount * self.df.Side * (
                                                self.df.QdfTime == self.df[self.closeTimeColumnName])
        self.df['abstransaction_volume'] = self.df.Amount

        if self.all_partition:
            x = self.df.set_index('QdfTime').groupby('TraderId').resample(str(self.time_resolution) + 'min')[
                ['transaction_volume', 'abstransaction_volume']].sum()
        else:
            x = self.df.groupby(['TraderId', pd.Grouper(freq=str(self.time_resolution) + 'T', key='QdfTime')])[
                ['transaction_volume', 'abstransaction_volume']].sum()

        x['imbalance_ratio'] = x['transaction_volume'] / x['abstransaction_volume']
        x = x['imbalance_ratio'].unstack(level=0)  # .fillna(2)#inactive state
        x[(x > self.imbalance_ratio_threshold) & (x < 2)] = 1
        x[x < -self.imbalance_ratio_threshold] = -1
        x[(abs(x) <= self.imbalance_ratio_threshold)] = 0

        if self.lower_time_cutoff != self.upper_time_cutoff:
            if self.time_resolution < 1440:
                x = x[(x.index.hour > self.lower_time_cutoff) & (x.index.hour <= self.upper_time_cutoff)]
            x = x[(x.index.weekday >= 0) & (x.index.weekday <= 4)]
        # x[x==2]=0 #drop inactive state
        return x

def ContructDatasetforSVN(parameters_dict,relevant_symbols,cuts, delta_t):
    warnings.filterwarnings('ignore')
    for s, c, d in tqdm_notebook(list(itertools.product(relevant_symbols, cuts, delta_t))):
        parameters_dict['time_resolution'] = d
        parameters_dict['symbol'] = s
        parameters_dict['min_number_trades_cutoff'] = c
        cut = c

        M =PrepareData(parameters_dict)
        A = M.construct_states_vs_t_matrix()
        sym = s.replace("/", "")
        A.to_csv('output\\tosvn\\davidnewtosvn_symbol_' + sym + '_delta_' + str(d) + 'mins' + str(cut) + 'cut.csv')


def LoadDynamicDataforSVN(cut):
    df10 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_10mins'+str(cut)+'cut.csv', parse_dates=['QdfTime'],
                       infer_datetime_format=True)
    df15 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_15mins'+str(cut)+'cut.csv', parse_dates=['QdfTime'],
                       infer_datetime_format=True)
    df30 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_30mins'+str(cut)+'cut.csv', parse_dates=['QdfTime'],
                       infer_datetime_format=True)
    df60 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_60mins'+str(cut)+'cut.csv', parse_dates=['QdfTime'],
                       infer_datetime_format=True)
    df120 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_120mins'+str(cut)+'cut.csv', parse_dates=['QdfTime'],
                        infer_datetime_format=True)
    df180 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_180mins'+str(cut)+'cut.csv', parse_dates=['QdfTime'],
                        infer_datetime_format=True)
    df360 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_360mins'+str(cut)+'cut.csv', parse_dates=['QdfTime'],
                        infer_datetime_format=True)
    df1440 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_1440mins'+str(cut)+'cut.csv',
                         parse_dates=['QdfTime'], infer_datetime_format=True)
    # shape problem
    cutday = df1440.QdfTime[1]
    cutdaystop = df1440.QdfTime[len(df1440) - 2]
    df10 = df10[(df10.QdfTime <= cutdaystop) & (df10.QdfTime >= cutday)].drop('QdfTime', axis=1)
    df15 = df15[(df15.QdfTime <= cutdaystop) & (df15.QdfTime >= cutday) & (df15.QdfTime.dt.weekday.values < 5)].drop(
        'QdfTime', axis=1)
    df30 = df30[(df30.QdfTime <= cutdaystop) & (df30.QdfTime >= cutday) & (df30.QdfTime.dt.weekday.values < 5)].drop(
        'QdfTime', axis=1)
    df60 = df60[(df60.QdfTime <= cutdaystop) & (df60.QdfTime >= cutday) & (df60.QdfTime.dt.weekday.values < 5)].drop(
        'QdfTime', axis=1)
    df120 = df120[(df120.QdfTime <= cutdaystop) & (df120.QdfTime >= cutday)].drop('QdfTime', axis=1)
    df180 = df180[(df180.QdfTime <= cutdaystop) & (df180.QdfTime >= cutday)].drop('QdfTime', axis=1)
    df360 = df360[(df360.QdfTime <= cutdaystop) & (df360.QdfTime >= cutday)].drop('QdfTime', axis=1)
    df1440 = df1440[(df1440.QdfTime < cutdaystop) & (df1440.QdfTime >= cutday)].drop('QdfTime', axis=1)
    print(df10.shape, df15.shape, df30.shape, df60.shape, df120.shape, df180.shape, df360.shape, df1440.shape)
    return df10,df15,df30,df60,df120,df180,df360,df1440


def get_graph_adjascency_files(name, minutes, cut, window="6months_2weeks", mypath="output\\dynamic\\"):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    result = [f for f in onlyfiles if ((str(cut) + "cut" in f) & ("links" in f) & (name in f) & (window in f) & (
                "_" + str(minutes) + "min" in f))]
    result.sort(key=lambda x: int(x.split('_')[8]))
    return result


def clusterise_with_karateclub(method, name, mypath="output\\dynamic\\",
                               output_path="output\\dynamic_clusters\\community_"):
    warnings.filterwarnings('ignore')
    data = pd.read_csv(mypath + name)
    le = LabelEncoder()
    df = data[['i', 'j']]
    if len(df) == 0:
        out = pd.DataFrame()
        out.to_csv(output_path + method + '_' + name + '.csv')
        return
    le.fit(df.stack().unique())
    df['i'] = le.transform(df['i'])
    df['j'] = le.transform(df['j'])
    data[['i', 'j']] = df[['i', 'j']]

    # g= max(nx.connected_component_subgraphs(G), key=len)

    if method == 'EgoNetSplitter':
        model = EgoNetSplitter()
    elif method == 'NNSED':
        model = NNSED()
    elif method == 'DANMF':
        model = DANMF()
    elif method == 'MNMF':
        model = MNMF()
    elif method == 'BigClam':
        model = BigClam()
    elif method == 'SymmNMF':
        model = SymmNMF()
    elif method == 'EdMot':
        model = EdMot()
    elif method == 'LabelPropagation':
        model = LabelPropagation()
    elif method == 'SCD':
        model = SCD()
    elif method == 'GEMSEC':
        model = GEMSEC()
    else:
        raise NotImplementedError("Not implemented ot not existent")
    try:
        g = build_graph_from_df(data)
        model.fit(g)
        cluster_membership = model.get_memberships()
    except:
        try:
            print('cant multigraph')
            g = build_graph_from_df2(data)
            model.fit(g)
            cluster_membership = model.get_memberships()
        except:
            raise NotImplementedError("Cant do")

    trader, cluster = [], []
    for (k, v) in model.get_memberships().items():
        for i in list(np.array([v]).ravel()):
            trader.append(le.inverse_transform([k])[0])
            cluster.append(i)
    out = pd.DataFrame({'cluster': cluster, 'trader': trader})
    out.to_csv(output_path + method + '_' + name + '.csv', index=False)


def clusterise_and_save(name, mins, cut, window, method, mypath="output\\dynamic\\",
                        output_path="output\\dynamic_clusters\\community_"):
    eurolinks = get_graph_adjascency_files(name, mins, cut, window, mypath)
    print(name, mins, cut, window, method, mypath)
    # print(eurolinks)
    for ith_link in tqdm_notebook(eurolinks):
        clusterise_with_karateclub(method, ith_link, mypath, output_path)

def build_graph_from_df(data):
    G = nx.MultiGraph()
    G.add_edges_from(data[data.si==-1][['i','j']].values )
    G.add_edges_from(data[data.si==1][['i','j']].values )
    G.add_edges_from(data[data.si==0][['i','j']].values )
    return G
def build_graph_from_df2(data):   
    G = nx.Graph()
    G.add_edges_from(data[data.si==0][['i','j']].values )
    G.add_edges_from(data[data.si==-1][['i','j']].values )
    G.add_edges_from(data[data.si==1][['i','j']].values )
    return G

def createPclusters(mypath_liste,window_liste,cut_liste,minutes_liste,cluster_methods_liste):
    for mypath in mypath_liste:
        for window in window_liste:
            for cut in cut_liste:
                for mins in minutes_liste:
                    for method in cluster_methods_liste:
                        clusterise_and_save("EURUSD", mins, cut, window, method, mypath)