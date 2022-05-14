import matplotlib.pyplot as plt
import numpy as np
import os
#os.environ["MODIN_ENGINE"] = "dask"
import   pandas as pd
#import modin.pandas as pd
from scipy import stats
from numpy import arange, poly1d, random
import matplotlib.pyplot as plt
from math import sqrt
from abc import ABCMeta
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
from pathlib import Path

import glob
import numba as  nb
import time
from  tqdm import tqdm_notebook
import warnings
import plotly.express as px
from plotly.subplots import make_subplots
class AA(object):
    __metaclass__ = ABCMeta
    def __init__(self, dict):
        #self.dapra_howmanyrows=dict['dapra_howmanyrows']
        #self.dapra_path=dict['dapra_path']
        #self.cluster_dates_path= dict['cluster_dates_path']
        self.prices_path =dict['prices_path']

        self.cluster_cut=dict['cluster_cut']
        #self.cluster_window = dict['cluster_window']
        #self.cluster_is_dynamic =dict['cluster_is_dynamic']
        #self.cluster_static =dict['cluster_static']
        self.cluster_algo =dict['cluster_algo']
        self.cluster_truncate_cluster_cardinality= dict['cluster_truncate_cluster_cardinality']
        self.cluster_resolution =dict['cluster_resolution']
        self.full_seeping=dict['full_seeping']

        self.aa_lr=dict['aa_lr']
        self.aa_losstype= dict['aa_losstype']
        self.aa_rho=dict['aa_rho']
        self.aa_rhobis = dict['aa_rhobis']
        self.aa_u = dict['aa_u']
        self.aa_ubis = dict['aa_ubis']

        self.clusters_datasets_done=0
        #self.inludenongrouped=1
        #self.percentile=50
        #self.add_='_dynamic_'+ str(self.cluster_is_dynamic)+'_res_'+str(self.cluster_resolution)
        self.rerun_=0
        self.cap=1e-03 # cap returns abnormal
        #self.beta_sigmoid=dict['beta_sigmoid']
        #self.power=dict['power']

        self.resolution = self.cluster_resolution
        self.pnl_or_pos = dict['pnl_or_pos']
        self.cutoff = self.cluster_cut
        self.threshold = dict['threshold']

        self.dfnames = {10: 'DAPRA_pnl_netpos_dynamic_clusters_cutoff_10.csv',
                   25: 'DAPRA_pnl_netpos_dynamic_clusters_cutoff_25.csv',
                   40: 'DAPRA_pnl_netpos_dynamic_clusters_cutoff_40.csv',
                   50: 'DAPRA_pnl_netpos_dynamic_clusters_cutoff_50.csv',
                   1000: 'DAPRA_pnl_netpos_dynamic_clusters_most_active_cutoff_1000.csv'
                   }

        self.dfnames2 = {10: 'DAPRA_dynamic_clusters_cutoff_10.csv',
                        25: 'DAPRA_dynamic_clusters_cutoff_25.csv',
                        40: 'DAPRA_dynamic_clusters_cutoff_40.csv',
                        50: 'DAPRA_dynamic_clusters_cutoff_50.csv',
                        1000: 'DAPRA_dynamic_clusters_cutoff_1000.csv'
                        }

        self.cluster_folder_names = {10: 'dynamic_clusters_cutoff_10',
                                25: 'dynamic_clusters_cutoff_25',
                                40: 'dynamic_clusters_cutoff_40',
                                50: 'dynamic_clusters_cutoff_50',
                                1000: 'clusters_cutoff_1000'
                                }
        self.cluster_name = {'netpos': 'cluster_netposition_resolution_' + str(self.resolution) + '_mins_',
                        'pnl': 'cluster_pnl_resolution_' + str(self.resolution) + '_mins_'}









    def dapra_loader(self):



        self.dapra = pd.read_csv('data//'+self.dfnames2[self.cutoff],
                                     parse_dates=['QdfTime'],
                                     infer_datetime_format=True).drop('Unnamed: 0', axis=1)


        self.dapra= self.dapra.replace(np.nan,0)
        self.daprasave=self.dapra.copy()


    def cluster_loader(self,fraction=0.0):


        self.clusters =  get_cluster_files(self.resolution, self.pnl_or_pos, self.cutoff, self.threshold,self.cluster_truncate_cluster_cardinality)


        self.cluster_start_end_dates_df = pd.read_csv('clusters_start_end_dates.csv',
                                                      parse_dates=['start','end','shift_start'],
                                                    infer_datetime_format=True)

        l = int(len(self.clusters) * fraction)

        self.cluster_start_end_dates_df = self.cluster_start_end_dates_df.iloc[l:, :].reset_index()
        self.clusters = self.clusters[-len(self.cluster_start_end_dates_df):]
        self.clusters = self.clusters[:-1]

        ###NOLEAK
        self.cluster_start_end_dates_df['shift_start'] = self.cluster_start_end_dates_df['end'].shift(-1)
        self.cluster_start_end_dates_df = self.cluster_start_end_dates_df.iloc[:-1, :]



    def get_subset_existant_traders(self):

        x = [sum(x, []) for x in self.clusters]
        traders = pd.Series([item for sublist in x for item in sublist])
        uni = list(traders.unique())
        traders = uni + ['QdfTime']
        leave = [x for x in self.dapra.columns if x.split('_')[0] in traders]
        self.dapra = self.dapra[leave]

        a = (self.dapra.drop('QdfTime', axis=1).iloc[:, ::2])
        b = (self.dapra.drop('QdfTime', axis=1).iloc[:, 1::2])
        a.columns = [x.split('_')[0] for x in a.columns]
        b.columns = [x.split('_')[0] for x in b.columns]
        add = a*b
        self.dapra = pd.concat([self.dapra, add], axis=1)
        del a, b


    def delete_inactive_traders(self,test=[]):
        cols1 = [x for x in self.dapra.columns if 'gamma' in x]
        cols2 = [x for x in self.dapra.columns if 'omega' in x]
        cols3 = [x for x in self.dapra.columns if ('_' not in x) and ('Qdf' not in x)]  # loss
        to_reject = abs(self.dapra[cols1]).cumsum().iloc[-1, :]
        to_reject = to_reject[to_reject == 0].index
        keep=[x.split('_')[0] for x in cols1 if x not in to_reject]

        out1=subclusters(self.clusters,keep)

        self.clusters=out1

        keep=['QdfTime']+[x for x in cols1+cols2+cols3 if x.split('_')[0] in keep]
        if len(test)>0:
            out1=subclusters2(self.clusters,test)
            self.clusters = out1
            keep = ['QdfTime'] + [x for x in cols1 + cols2 + cols3 if x.split('_')[0] in test]

        self.dapra=self.dapra[keep]

    def dates_leak(self):

        #NO LEAK
        self.dapra=self.dapra[(self.dapra.QdfTime>=self.cluster_start_end_dates_df['end'].values[0])
                              & ((self.dapra.QdfTime<self.cluster_start_end_dates_df['shift_start'].values[-1]))]




        self.dapra=self.dapra.reset_index().drop('index',axis=1)
    def LS_loss(self,c,eps=0):
        tmp0=1+self.aa_rho*c

        return np.maximum(tmp0,eps)


    def LSD_loss(self,c,eps=0):
        tmp=1+self.aa_rho*np.minimum(c,0)

        return np.maximum(tmp,eps)


    def LSN_loss(self,c,eps=0):
        tmp1=self.aa_rho* self.aa_u/(self.aa_u+self.aa_ubis)*c
        tmp2=self.aa_rhobis*self.aa_ubis/(self.aa_u+self.aa_ubis)*np.minimum(0,c)


        return np.maximum(1+tmp1+tmp2,eps)



    def logloss(self,c):
        tmp0 = 1 + self.aa_rho * c
        tmp0=np.log(1+(np.e-1)*np.maximum(tmp0,0))
        return tmp0
    def loglossdownside(self,c):
        tmp0 = 1 + self.aa_rho * np.minimum(c,0)
        tmp0=np.log(1+(np.e-1)*np.maximum(tmp0,0))
        return tmp0


    def powerloss(self,c):
        tmp0 = 1 + (self.aa_rho * c)**self.power
        tmp0= np.maximum(tmp0,0)
        return tmp0
    def powerlossdownside(self,c):
        tmp0 = 1 +  np.minimum(self.aa_rho *c,0)**self.power
        tmp0=np.maximum(tmp0,0)
        return tmp0

    def loglogloss(self,c):
        tmp0 = 1 + self.aa_rho * c
        tmp0=np.log(np.log(np.e+(np.e**np.e-np.e)*np.maximum(tmp0,0)))
        return tmp0
    def logloglossdownside(self,c):
        tmp0 = 1 + self.aa_rho * np.minimum(c,0)
        tmp0=np.log(np.log(np.e+(np.e**np.e-np.e)*np.maximum(tmp0,0)))
        return tmp0

    def sigmoidloss(self, c):
        tmp0 = 1 + self.aa_rho * c
        tmp0=np.maximum(tmp0, 0)

        return 1 / (1 + ((-self.aa_rho * c) / tmp0) ** self.beta_sigmoid)  


    def sigmoidlossdownside(self, c):
        tmp=np.minimum(c, 0)
        tmp0 = 1 + self.aa_rho * tmp
        tmp1=np.maximum(tmp0, 0)

        return 1 / (1 + ((-self.aa_rho * tmp) / tmp1) ** self.beta_sigmoid)  



    def calculate_loss(self):
            cols=[x for x in  self.dapra.columns if ('_' not in x) and ('Qdf' not in x) and('gamma' not in x) and ('return' not in x)  ]
            losscols = ['AAloss_' + x for x in cols]
            if self.aa_losstype=='LS':
                self.dapra[losscols]=self.LS_loss(self.dapra[cols] ).values
            if self.aa_losstype=='LSD':
                self.dapra[losscols]=self.LSD_loss(self.dapra[cols] ).values
            if self.aa_losstype=='LSN':
                self.dapra[losscols]=self.LSN_loss(self.dapra[cols] ).values

            if self.aa_losstype=='logloss':
                self.dapra[losscols]=self.logloss(self.dapra[cols] ).values
            if self.aa_losstype=='loglossdownside':
                self.dapra[losscols]=self.loglossdownside(self.dapra[cols] ).values
            if self.aa_losstype == 'loglogloss':
                self.dapra[losscols] = self.loglogloss(self.dapra[cols]).values
            if self.aa_losstype == 'logloglossdownside':
                self.dapra[losscols] = self.logloglossdownside(self.dapra[cols]).values
            if self.aa_losstype=='sigmoidloss':
                self.dapra[losscols]=self.sigmoidloss(self.dapra[cols] ).values
            if self.aa_losstype=='sigmoidlossdownside':
                self.dapra[losscols]=self.sigmoidlossdownside(self.dapra[cols] ).values

            if self.aa_losstype=='powerloss':
                self.dapra[losscols]=self.powerloss(self.dapra[cols] ).values
            if self.aa_losstype=='powerlossdownside':
                self.dapra[losscols]=self.powerlossdownside(self.dapra[cols] ).values



    def loss_sleepingAA(self):
        names = ['event', 'Symbol', 'QdfTime', 'bid', 'ask', 'other1', 'other2', 'other3', 'other4']
        prices = pd.read_csv(self.prices_path, header=None, names=names, parse_dates=['QdfTime'],
                             infer_datetime_format=True)

        prices['Time'] = prices.QdfTime
        prices.QdfTime = prices.QdfTime.dt.ceil('1min')
        prices = prices.groupby('QdfTime').agg('last')
        prices['mid'] = prices.ask / 2 + prices.bid / 2
        prices = prices.reset_index()[['QdfTime', 'mid']]

        #############insert interpolated prices TOTEST
        maxi = prices['QdfTime'].max()
        mini = prices['QdfTime'].min()

        prices.set_index('QdfTime', inplace=True)

        def get_dapra_init(minimum, maximum):
            timeindex = pd.date_range(minimum, maximum, periods=(maximum - minimum).total_seconds() / 60 + 1)
            return pd.Series(np.zeros(len(timeindex)), index=timeindex)

        output2 = get_dapra_init(mini, maxi)

        p = pd.DataFrame.from_records(np.zeros((len(output2), 1)), columns=['mid'])
        p.index = output2.index
        p.loc[prices.index, :] = prices.loc[prices.index, :]
        p['mid'] = p['mid'].replace(0, np.NaN)

        p['mid'] = p['mid'].interpolate().fillna(method='bfill')

        prices = p.copy()
        del p, output2
        prices = prices.reset_index().rename(columns={'index': 'QdfTime'})

        ###########

        prices['return_for_loss_sleepingAA'] = (prices.mid / prices.mid.shift(1) - 1).replace(np.nan, 0)
        prices['return_for_AA'] = (prices.mid.shift(-1) / prices.mid - 1).replace(np.nan, 0)

        prices['return_for_AA'] = prices['return_for_AA'].replace(np.nan, 0)
        prices['return_for_loss_sleepingAA'] = prices['return_for_loss_sleepingAA'].replace(np.nan, 0)

        prices = prices.drop('mid', axis=1)

        self.dapra = pd.merge(self.dapra, prices, on=["QdfTime"], how='left')
        self.dapra['return_for_loss_sleepingAA'] = self.dapra['return_for_loss_sleepingAA'].replace(np.nan, 0)
        self.dapra['return_for_AA'] = self.dapra['return_for_AA'].replace(np.nan, 0)

        # cap some values only 100 minutes if 5e-3 only 12 records could be error
        cap = self.cap
        self.dapra['return_for_loss_sleepingAA'][(self.dapra['return_for_loss_sleepingAA']) >= cap] = cap
        self.dapra['return_for_loss_sleepingAA'][(self.dapra['return_for_loss_sleepingAA']) <= -cap] = -cap
        self.dapra['return_for_AA'][self.dapra['return_for_AA'] >= cap] = cap
        self.dapra['return_for_AA'][self.dapra['return_for_AA'] <= -cap] = -cap


    def online_readjust_for_new_traders(self):
        cols = [x for x in self.dapra.columns if 'gamma' in x]
        d = (abs(self.dapra[cols]) > 0) * 1.
        cols = [x for x in self.dapra.columns if 'omega' in x]
        self.dapra[cols] = d.values

    def active_or_not(self, df):

        d = (abs(df) > 0) * 1.


        return d.values


    def AA_return(self):
        #cols = [x for x in self.dapra.columns if ('_' not in x) and ('Qdf' not in x)]  # omega*gamma
        cols=[x for x in self.dapra.columns if ('AAloss'  in x)]#loss
        cols2 = [x for x in self.dapra.columns if 'omega' in x]  # active or not
        cols3 = [x for x in self.dapra.columns if '_gamma' in x]  # gamma

        res, omegaAA, r = calculator_probas_sleeping2(self.dapra[cols3].values, self.dapra[cols].values,
                                                      self.dapra['return_for_loss_sleepingAA'].values,
                                                      self.dapra[cols2].values, self.aa_losstype, self.aa_lr,
                                                      self.aa_rho, self.aa_rhobis, self.aa_u, self.aa_ubis,1,1)

        self.probas = pd.DataFrame(r, columns=cols2)

        mean_portfolio = (self.probas.values > 0) * self.dapra[cols3].values
        mean_portfolio = np.nan_to_num((mean_portfolio.T / (np.sum(mean_portfolio != 0, axis=1))).T.sum(axis=1))

        self.dapra['return_meanportfolio'] = mean_portfolio * self.dapra['return_for_AA'].values

        ew = self.dapra[cols3].values
        ew = np.nan_to_num((ew.T / (np.sum(ew != 0, axis=1))).T.sum(axis=1))

        self.dapra['return_equal_weights'] = ew * self.dapra['return_for_AA'].values

        self.dapra['gammaAA'] = omegaAA

        self.dapra['returnAA'] = self.dapra['gammaAA'] * self.dapra['return_for_AA']

        cols = [x for x in self.dapra.columns if '_gamma' in x]
        cols = [x for x in cols if (x != 'gammaAA')]
        self.save_gamma = self.dapra[cols].copy()

    def gammaCAA(self):
        cols = [x for x in self.dapra.columns if '_gamma' in x]
        #cols = [x for x in cols if (x != 'gammaAA')]
        cols2 = [x for x in self.dapra.columns if 'omega' in x]

        # self.save_gamma=self.dapra[cols].copy()

        for ind, row in (self.cluster_start_end_dates_df.iterrows()):

            s = self.clusters[ind]


            # NOLEAK
            tmp = self.dapra[cols][(self.dapra.QdfTime >= row['end']) & (self.dapra.QdfTime <row['shift_start'])]
            # LEAK
            #tmp = self.dapra[cols][(self.dapra.QdfTime >= row['start']) & (self.dapra.QdfTime < row['shift_start'])]

            for cl in s:
                columns = [x + '_gamma' for x in cl]
                means = tmp[columns].mean(axis=1)
                for c in columns:
                    tmp[c] = means

            self.dapra.loc[tmp.index, cols] = tmp



        self.dapra['gammaCAA'] = (self.probas).mul(self.dapra[cols].values).sum(axis=1)
        self.dapra['returnCAA'] = self.dapra['gammaCAA'] * self.dapra['return_for_AA']
        self.dapra[cols] = self.save_gamma.copy()
        #del  self.save_gamma

    def gammaCAA2(self):
        cols = [x for x in self.dapra.columns if 'gamma' in x]
        cols = [x for x in cols if (x != 'gammaAA')]
        cols2 = [x for x in self.dapra.columns if 'omega' in x]

        # self.save_gamma=self.dapra[cols].copy()

        if self.clusters_datasets_done == 0:

            for ind, row in (self.cluster_start_end_dates_df.iterrows()):

                if ind == 0:
                    # noleak
                    # start=row['end']
                    # leak
                    start = row['start']

                s = self.clusters[ind]
                n = [len(x) for x in s]

                tmp = self.dapra[(self.dapra.QdfTime >= start) & (self.dapra.QdfTime < row['shift_start'])]
                gr = self.clusters[ind]
                percentile = self.percentile
                rs = tmp['return_for_loss_sleepingAA']
                r = tmp['return_for_AA']
                ni = self.inludenongrouped
                tmp = tmp[cols]
                tmp.columns = [x.split('_')[0] for x in tmp.columns]
                out = get_caa_datasets(tmp, gr, percentile, rs, r, ni)
                out.to_csv('groupedAA' + str(ind) + self.add_ + '.csv')

            self.clusters_datasets_done = 1

        returnaa = []
        gammaaa = []
        f = 0
        l = 0
        for i in range(len(self.clusters)):
            df = pd.read_csv('groupedAA' + str(i) + self.add_ + '.csv')
            l = len(df)
            cols = [x for x in df.columns if 'g' in x]
            gamma = df[cols]
            loss = self.calculate_lossCAA2(gamma.apply(lambda x: x * df.rs))
            active_and_not = self.active_or_not(gamma)

            res, gammaAA, r = grouped_calculator_probas_sleeping(gamma.values, loss,
                                                                 df.rs.values, active_and_not, self.aa_losstype,
                                                                 self.aa_lr, self.aa_rho, self.aa_rhobis, self.aa_u,
                                                                 self.aa_ubis)

            df['gammaCAAgrouped'] = gammaAA
            df['returnCAAgrouped'] = df['gammaCAAgrouped'] * df['r']

            gammaaa = gammaaa + list(df['gammaCAAgrouped'].values[f:l])
            returnaa = returnaa + list(df['returnCAAgrouped'].values[f:l])
            f = l

        self.dapra['gammaCAAgrouped'] = gammaaa
        self.dapra['returnCAAgrouped'] = returnaa

        # del  self.save_gamma

    def save(self, s):

        if self.cluster_is_dynamic == 0:

            s2 = 'static_' + str(self.cluster_static) + '_cut_' + str(self.cluster_cut) + '_' + self.cluster_window
            folder = 'static_cut_' + str(self.cluster_cut) + '_window_' + self.cluster_window

        else:

            s2 = 'dynamic_cut_' + str(self.cluster_cut) + '_' + self.cluster_window
            folder = 'dynamic_cut_' + str(self.cluster_cut) + '_window_' + self.cluster_window

        if self.aa_losstype == 'LSN':
            s = s + '_LSN_' + str(self.aa_lr) + '_' + str(self.aa_rho) + '_' + str(self.aa_rhobis) + '_' + str(
                self.aa_u) + '_' + str(self.aa_ubis) + '_'
        else:
            s = s + '_' + self.aa_losstype + '_' + str(self.aa_lr) + '_' + str(self.aa_rho) + '_'

        s3 = '_CL_algo_' + self.cluster_algo + '_CL_res_' + str(self.cluster_resolution) + '_CL_tr_' + str(
            self.cluster_truncate_cluster_cardinality)

        cols = ['gammaAA', 'gammaCAA', 'returnAA', 'returnCAA']
        self.dapra[cols].to_csv(self.cluster_dates_path + folder + '\\' + s + s2 + s3 + '_.csv')

    def reload(self):
        self.dapra = self.daprasave.copy()

    def run_paper(self):
        if self.rerun_ == 1:
            self.calculate_loss()
            self.AA_return()
            self.gammaCAA()


        else:
            self.dapra_loader()
            self.rerun_ = 1
            self.cluster_loader()
            self.get_subset_existant_traders()
            self.delete_inactive_traders()
            self.dates_leak()
            self.loss_sleepingAA()
            self.online_readjust_for_new_traders()
            self.calculate_loss()

            self.AA_return()
            self.gammaCAA()

    def rerun(self):

        self.reload()
        self.cluster_loader()
        self.get_subset_existant_traders()
        self.dates_leak()
        self.calculate_loss()
        self.delete_inactive_traders()
        self.loss_sleepingAA()
        self.online_readjust_for_new_traders()
        self.AA_return()
        self.leader()
        self.followleadingcluster()
        self.gammaCAA()











    def run_paper_scaling_factor(self, lossname, max_scaling, cluster_resolution=0):
        warnings.filterwarnings('ignore')
        self.rerun_ = 0

        if cluster_resolution != 0: self.cluster_resolution = cluster_resolution
        self.aa_losstype = lossname

        l = [0] + list(np.arange(5, max_scaling, 5))
        outLS = []
        outrLS = []
        outsLS = []
        outDD = []

        for i in tqdm_notebook(l):
            if lossname in ['LS', 'LSD','logloss','loglogloss','sigmoidloss','powerloss','loglossdownside','logloglossdownside','sigmoidlossdownside','powerlossdownside']:
                self.aa_rho = i + 1
            else:
                self.aa_rho = i + 1
                self.aa_rhobis = i + 1

            self.run_paper()
            a = self.dapra.returnAA
            b = self.dapra.returnCAA
            c = self.dapra.return_meanportfolio
            f = self.dapra.return_equal_weights

            sa = a[a < 0]
            sb = b[b < 0]
            sc = c[c < 0]
            sf = f[f < 0]

            a = a[a != 0]
            b = b[b != 0]
            c = c[c != 0]
            f = f[f != 0]
            d = np.sqrt(250)

            x = d * np.mean(a) / np.std(a)
            y = d * np.mean(b) / np.std(b)
            z = d * np.mean(c) / np.std(c)
            zz = d * np.mean(f) / np.std(f)

            sx = np.nan_to_num(d * np.mean(a) / np.std(sa))
            sy = np.nan_to_num(d * np.mean(b) / np.std(sb))
            sz = np.nan_to_num(d * np.mean(c) / np.std(sc))
            szz = np.nan_to_num(d * np.mean(f) / np.std(sf))

            r = (self.dapra.returnAA + 1).cumprod().iloc[-1]
            rc = (self.dapra.returnCAA + 1).cumprod().iloc[-1]
            rcc = (self.dapra.return_meanportfolio + 1).cumprod().iloc[-1]
            rf = (self.dapra.return_equal_weights + 1).cumprod().iloc[-1]

            d1 = (self.dapra.returnAA + 1).cumprod()
            d2 = (self.dapra.returnCAA + 1).cumprod()
            d3 = (self.dapra.return_meanportfolio + 1).cumprod()
            d4 = (self.dapra.return_equal_weights + 1).cumprod()

            d1 = dd(d1)
            d2 = dd(d2)
            d3 = dd(d3)
            d4 = dd(d4)

            outrLS.append((r, rc, rcc, rf))
            outLS.append((x, y, z, zz))
            outsLS.append((sx, sy, sz, szz))
            outDD.append((d1,d2,d3,d4))
        self.LSr = outrLS
        self.LSs = outLS
        self.LSss = outsLS
        self.LSdd = outDD

        d = {"LS": "Long Short Loss", "LSD": "Downside Long Short Loss", "LSN": "Mixed Long Short Loss",
             'logloss':"Log Loss",'loglogloss':"Log Log Loss",'sigmoidloss':"Sigmoid Loss",'powerloss':"Power loss",
             'loglossdownside':"Downside Log Loss",'logloglossdownside':"Downside Log Log Loss",'sigmoidlossdownside':"Downside Sigmoid Loss",'powerlossdownside':"Downside Power loss"}
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 16))
        c = ['shr_AA', 'shr_CAA', 'shr_mean_portfolio', 'shr_equal_weights_portfolio']
        sh = pd.DataFrame.from_records(outLS, columns=c, index=l)
        # pd.DataFrame.from_records(outLS, columns=c, index=l).plot(ax=ax1)
        c = ['r_AA', 'r_CAA', 'r_AA_mean_portfolio', 'r_equal_weights_portfolio']
        re = pd.DataFrame.from_records(outrLS, columns=c, index=l)
        # pd.DataFrame.from_records(outrLS, columns=c, index=l).plot(ax=ax2)
        c = ['sr_AA', 'sr_CAA', 'sr_AA_mean_portfolio', 'sr_equal_weights_portfolio']
        sr = pd.DataFrame.from_records(outsLS, columns=c, index=l)

        c = ['dd_AA', 'dd_CAA', 'dd_AA_mean_portfolio', 'dd_equal_weights_portfolio']
        dd_ = pd.DataFrame.from_records(outDD, columns=c, index=l)

        colors = ['blue', 'yellow', "green", 'red']

        fig = make_subplots(rows=1, cols=5, subplot_titles=(
            "Sharpe ratio for " + d[lossname], "Return for " + d[lossname], "Sortino ratio for " + d[lossname],
            "Drawdown for " + d[lossname], "Return/Drawdown for " + d[lossname]))

        for i, n in enumerate(sh.columns):
            fig.add_scatter(x=sh.index, y=sh[sh.columns[i]], name=sh.columns[i], row=1, col=1, mode='lines',
                            line=dict(color=colors[i]))

        fig.update_xaxes(title_text="Scaling factor", row=1, col=1)

        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)

        # fig.show()

        # fig = px.line()
        for i, n in enumerate(re.columns):
            fig.add_scatter(x=re.index, y=re[re.columns[i]], name=re.columns[i], row=1, col=2, mode='lines',
                            line=dict(color=colors[i]))

        fig.update_xaxes(title_text="Scaling factor", row=1, col=2)

        fig.update_yaxes(title_text="1+Return", row=1, col=2)
        # fig.show()

        # fig = px.line()
        for i, n in enumerate(re.columns):
            fig.add_scatter(x=sr.index, y=sr[sr.columns[i]], name=sr.columns[i], row=1, col=3, mode='lines',
                            line=dict(color=colors[i]))

        fig.update_xaxes(title_text="Scaling factor", row=1, col=3)

        fig.update_yaxes(title_text="Sortino Ratio", row=1, col=3)
        # fig.show()
        for i, n in enumerate(dd_.columns):
            fig.add_scatter(x=dd_.index, y=dd_[dd_.columns[i]], name=dd_.columns[i], row=1, col=4, mode='lines',
                            line=dict(color=colors[i]))

        fig.update_xaxes(title_text="Scaling factor", row=1, col=4)

        fig.update_yaxes(title_text="Drawdown", row=1, col=4)

        # fig.show()
        # fig = px.line()

        ddd = re.values / dd_.values
        dddcolumns = ['r/' + x for x in dd_.columns]
        dddindex = dd_.index
        ddd = pd.DataFrame.from_records(ddd, columns=dddcolumns, index=dddindex)
        # fig = px.line()
        for i, n in enumerate(dd_.columns):
            fig.add_scatter(x=ddd.index, y=ddd[ddd.columns[i]], name=ddd.columns[i], row=1, col=5, mode='lines',
                            line=dict(color=colors[i]))

        fig.update_xaxes(title_text="Scaling factor", row=1, col=5)

        fig.update_yaxes(title_text="Return /Drawdown", row=1, col=5)

        fig.update_layout(font=dict(size=10)
                          , template="simple_white", height=600, width=3500
                          )
        fig.show()






        saveto = "noleak_truemean_scalling_fullsleeping_" + str(
            self.full_seeping) + "_loss_" + lossname + "_clusterresolution_" + str(
            cluster_resolution) + "_cutoff_" + str(self.cluster_cut)+'_hierthreshold_'+str(self.threshold)+'_mins_'+str(self.resolution)
        re.to_csv("return" + saveto + '.csv')
        sh.to_csv("sharpe" + saveto + '.csv')
        sr.to_csv("sortino" + saveto + '.csv')
        dd_.to_csv("drawdown" + saveto + '.csv')
        ddd.to_csv("r_drawdown" + saveto + '.csv')

    def rerunsmall(self):
        teste = ['T12936', 'T10804', 'T17761', 'T5121', 'T6963', 'T19791', 'T7095',
                 'T3845', 'T12525', 'T6428', 'T20461', 'T9281', 'T8123', 'T4008', 'T19543', 'T3140', 'T16563',
                 'T5276', 'T8481', 'T1779']  # to change here

        self.reload()
        self.cluster_loader()
        self.get_subset_existant_traders()
        self.dates_leak()
        self.calculate_loss()
        self.delete_inactive_traders(teste)
        self.loss_sleepingAA()
        self.online_readjust_for_new_traders()
        self.AA_return()
        self.leader()
        self.followleadingcluster()
        self.gammaCAA()

    def rerun(self):

        self.reload()
        self.cluster_loader()
        self.get_subset_existant_traders()
        self.dates_leak()
        self.calculate_loss()
        self.delete_inactive_traders()
        self.loss_sleepingAA()
        self.online_readjust_for_new_traders()
        self.AA_return()
        self.leader()
        self.followleadingcluster()
        self.gammaCAA()

    def run(self):

        self.dapra_loader()
        self.cluster_loader()
        self.get_subset_existant_traders()
        self.dates_leak()
        self.calculate_loss()
        self.delete_inactive_traders()
        self.loss_sleepingAA()
        self.online_readjust_for_new_traders()
        self.AA_return()
        self.leader()
        self.followleadingcluster()
        self.gammaCAA()

    def rungrouped(self):

        self.dapra_loader()
        self.cluster_loader()
        self.get_subset_existant_traders()
        self.dates_leak()
        self.calculate_loss()
        self.delete_inactive_traders()
        self.loss_sleepingAA()
        self.online_readjust_for_new_traders()
        self.AA_return()
        self.leader()
        self.followleadingcluster()
        self.gammaCAA()
        self.gammaCAA2()

    def rerungrouped(self):

        self.reload()
        self.cluster_loader()
        self.get_subset_existant_traders()
        self.dates_leak()
        self.calculate_loss()
        self.delete_inactive_traders()
        self.loss_sleepingAA()
        self.online_readjust_for_new_traders()
        self.AA_return()
        self.leader()
        self.followleadingcluster()
        self.gammaCAA()
        self.gammaCAA2()

    def plot(self):

        plt.figure(figsize=(35, 10))
        plt.plot((self.dapra.returnAA + 1).cumprod(), label='AA')
        plt.plot((self.dapra.returnCAA + 1).cumprod(), label='CAA')
        # plt.plot((self.dapra.returnbest_ + 1).cumprod(), label='leader')
        # plt.plot((self.dapra.returnbest_c + 1).cumprod(), label='leadercluster')
        ##plt.plot((self.dapra.returnbest_c_n + 1).cumprod(), label='leaderclusternorm')
        # plt.plot((self.dapra.returnCAAgrouped + 1).cumprod(), label='CAAgrouped')
        # plt.plot((self.dapra.return_meanportfolio + 1).cumprod(), label='return_meanportfolio')
        plt.plot((self.dapra.return_equal_weights + 1).cumprod(), label='return_equal_weights')

        plt.legend()
        plt.show()
        cols = [x for x in self.dapra.columns if ('_' not in x) and ('Qdf' not in x)][:-6]
        cols = self.dapra[cols].cumprod().iloc[-1, :].sort_values().index
        fig, ax = plt.subplots()
        a = (((self.dapra[cols] - 1) / self.aa_rho + 1) ** (1 / self.aa_lr)).cumprod()
        a.plot(figsize=(35, 10), ax=ax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=10)
        plt.show()
        cols = [x for x in self.dapra.columns if ('_' not in x) and ('Qdf' not in x)][:-6]
        c = self.dapra[cols].cumprod().iloc[-1, :].sort_values().index
        cbis = [x + '_gamma' for x in c]
        c = [x + '_omega' for x in c]

        fig, ax = plt.subplots()
        a = self.probas[c]

        a.plot(figsize=(35, 10), ax=ax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=10)
        plt.show()
        del a
        self.dapra.gammaAA.plot(figsize=(35, 6))
        plt.show()


def get_caa_datasets(df, gr, percentile, returnaashift, returnaa, inludenongrouped):
    grnames = ['g' + str(i) for i in range(len(gr))]
    o = pd.DataFrame(np.ones(shape=df.shape), columns=df.columns)
    x = o.iloc[0, :].copy() * np.nan
    out = pd.DataFrame(np.zeros(shape=(df.shape[0], len(gr))), columns=grnames)

    if inludenongrouped:
        s1 = set(df.columns)
        s2 = set(sum(gr, []))
        s = s1 - s2

        if len(s) > 0:
            gr = gr + [list(s)]
            grnames = grnames + ['g' + str(len(gr) - 1)]
            out['g' + str(len(gr) - 1)] = 0.

    for j, g in enumerate(gr):
        x[g] = 1

        if percentile:
            out['g' + str(j)] = np.nanpercentile((o * x) * df, percentile, axis=1)
        else:
            out['g' + str(j)] = np.mean((o * x) * df, axis=1)

        x = x * np.nan

    out.index = df.index
    out['rs'] = returnaashift
    out['r'] = returnaa

    return out


def bestclusterindexes(p, c):
    cols = list(x.split('_')[0] for x in p.columns)
    m = p.idxmax(axis=1).apply(lambda x: x.split('_')[0])

    def f(x, l):
        if len(l) > 0:
            return l
        else:
            return [x]

    g = m.apply(lambda x: (f(x, list(np.array([y for y in c if x in y]).ravel()))))

    def ff(x):
        if x in cols:
            return cols.index(x)
        else:
            return -1

    def h(x):
        return list([ff(y) for y in x])

    g = g.apply(h)
    l = (list(g.values))

    l2 = [(list(np.repeat(i, len(x)))) for i, x in zip(np.arange(len(l)), l)]
    l2 = [x for y in l2 for x in y]
    l3 = [x for y in l for x in y]
    return l2, l3


@nb.njit
def calculator_probas_sleeping(gamma, loss, return_for_loss_sleepingAA,
                               active_and_not, aa_losstype,
                               aa_lr, aa_rho, aa_rhobis, aa_u, aa_ubis):
    omegaAA = np.zeros(loss.shape[0])
    omegabest = np.zeros(loss.shape[0])
    res = np.ones(loss.shape)  # * 1. / loss.shape[1]
    res2 = np.zeros(loss.shape)

    for i in range(0, res.shape[0]):
        if (np.sum(active_and_not[i, :]) == 0):
            if i == 0:
                continue

            else:
                for j in range(res.shape[1]):
                    res[i, j] = res[i - 1, j]
                continue

        else:
            denominator = np.dot(active_and_not[i, :], res[i - 1, :])
            for j in range(res.shape[1]):
                if (active_and_not[i, j] == 1):
                    res[i, j] = res[i - 1, j]
                    # denominator=np.dot(active_and_not[i, :], res[i - 1, :])

                    if denominator > 0:
                        res2[i, j] = res[i, j] * 1.0 / denominator
                        omegaAA[i] += res2[i, j] * gamma[i, j]
                        res[i, j] = res[i, j] * loss[i, j]

                    else:
                        omegaAA[i] += 0

            for j in range(res.shape[1]):
                if (active_and_not[i, j] == 0) & (aa_losstype == 'LS'):
                    x = omegaAA[i] * return_for_loss_sleepingAA[i]
                    tmp0 = np.maximum(1 + aa_rho * x, 0)

                    if aa_lr == 1:
                        res[i, j] = res[i - 1, j] * tmp0

                    if (tmp0 < 0) & (aa_lr != 1):
                        res[i, j] = 0.

                    if (tmp0 >= 0) & (aa_lr != 1):
                        tmp = -np.log(tmp0)
                        res[i, j] = res[i - 1, j] * np.exp(-aa_lr * tmp)

                if (active_and_not[i, j] == 0) & (aa_losstype == 'LSD'):
                    x = omegaAA[i] * return_for_loss_sleepingAA[i]
                    tmp0 = np.maximum(1 + aa_rho * np.minimum(x, 0), 0)

                    if aa_lr == 1:
                        res[i, j] = res[i - 1, j] * tmp0

                    else:
                        tmp = -np.log(tmp0)
                        res[i, j] = res[i - 1, j] * np.exp(-aa_lr * tmp)

                if ((active_and_not[i, j] == 0)) & (aa_losstype == 'LSN'):
                    x = omegaAA[i] * return_for_loss_sleepingAA[i]
                    tmp1 = aa_rho * aa_u / (aa_u + aa_ubis) * x
                    tmp2 = aa_rhobis * aa_ubis / (aa_u + aa_ubis) * np.minimum(0, x)

                    if aa_lr == 1:
                        res[i, j] = res[i - 1, j] * (1 + tmp1 + tmp2)

                    if (1 + tmp1 + tmp2 < 0) & (aa_lr != 1):
                        res[i, j] = 0.

                    else:
                        tmp = -np.log(1 + tmp1 + tmp2)
                        res[i, j] = res[i - 1, j] * np.exp(-aa_lr * tmp)

    return res, omegaAA, res2, omegabest


def subclusters(big, small):
    out1 = []

    if len(small) == 0:
        return big

    for z in big:
        out2 = []

        for x in z:
            out3 = []

            for y in x:
                if y in small:
                    out3.append(y)

            if len(out3):
                out2.append(out3)

        out1.append(out2)
    return out1


def subclusters2(big, small):
    out1 = []
    if len(small) == 0:
        return big

    for z in big:
        out2 = []
        for x in z:
            out3 = []

            for y in x:

                if y in small:
                    out3.append(y)
            if len(out3):
                out2.append(out3)

        if len(out2) == 0:
            t = [[x] for x in small]
            out1.append(t)
        else:
            out1.append(out2)
    return out1


def get_community_files(name, minutes, cut, window, stratname):
    mypath = "output\\dynamic_clusters\\"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    result = [f for f in onlyfiles if
              ((str(cut) + "cut" in f) & ("community" in f) & (stratname in f) & (name in f) & (window in f) & (
                      "_" + str(minutes) + "min" in f))]
    return result


def cluster_dynamics2(name2, minutes, cut, window, stratname, truncate=1):
    eurusdcommunities = get_community_files(name2, minutes, cut, window, stratname)
    eurusdcommunities.sort(key=lambda x: int(x.split('_')[-3]))

    result = []
    M = 0
    plt.ioff()
    mypath = "output\\dynamic_clusters\\"
    output = []
    alltraders = []

    for name in eurusdcommunities:

        d = pd.read_csv(mypath + name)

        le = preprocessing.LabelEncoder()

        if (len(d) == 0): continue

        alltraders.append(d['trader'].values)
    import itertools

    alltraders = list(itertools.chain(*alltraders))

    le2 = preprocessing.LabelEncoder()
    le2.fit(np.unique(np.array(alltraders)))

    for name in eurusdcommunities:

        d = pd.read_csv(mypath + name)

        le = preprocessing.LabelEncoder()
        if (len(d) == 0): continue

        le.fit(d['cluster'].values)
        d['cluster'] = le.transform(d.cluster.values)
        d['trader'] = d.trader.values

        output.append(list(d.groupby('cluster')['trader'].apply(list).values))
    if truncate:
        return [[x for x in output[i] if len(x) >= truncate] for i in range(len(output))]
    return output


@nb.njit
def calculator_probas_sleeping2(gamma, loss, return_for_loss_sleepingAA,
                                active_and_not, aa_losstype,
                                aa_lr, aa_rho, aa_rhobis, aa_u, aa_ubis,beta_sigmoid,power):
    omegaAA = np.zeros(loss.shape[0])
    omegabest = np.zeros(loss.shape[0])
    res = np.ones(loss.shape)  # * 1. / loss.shape[1]
    res2 = np.zeros(loss.shape)
    ones = np.ones(loss.shape[1])

    for i in range(0, res.shape[0]):
        if (np.sum(active_and_not[i, :]) == 0):
            if i == 0:
                continue

            else:
                for j in range(res.shape[1]):
                    res[i, j] = res[i - 1, j]
                omegaAA[i] = 0
                continue

        else:
            denominator = np.dot(active_and_not[i, :], res[i - 1, :])
            if denominator == 0:
                for j in range(res.shape[1]):
                    res[i, j] = res[i - 1, j]
                omegaAA[i] = 0
                continue
            else:
                for j in range(res.shape[1]):
                    if (active_and_not[i, j] == 1):
                        res[i, j] = res[i - 1, j]
                        res2[i, j] = res[i, j] * 1.0 / denominator
                        omegaAA[i] += res2[i, j] * gamma[i, j]
                        res[i, j] = res[i, j] * loss[i, j]

                for j in range(res.shape[1]):
                    if (active_and_not[i, j] == 0) & (aa_losstype == 'LS'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 = np.maximum(1 + aa_rho * x, 0)

                        if aa_lr == 1:
                            res[i, j] = res[i - 1, j] * tmp0

                        if (tmp0 < 0) & (aa_lr != 1):
                            res[i, j] = 0.

                        if (tmp0 >= 0) & (aa_lr != 1):
                            tmp = -np.log(tmp0)
                            res[i, j] = res[i - 1, j] * np.exp(-aa_lr * tmp)

                    if (active_and_not[i, j] == 0) & (aa_losstype == 'LSD'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 = np.maximum(1 + aa_rho * np.minimum(x, 0), 0)

                        if aa_lr == 1:
                            res[i, j] = res[i - 1, j] * tmp0

                        else:
                            tmp = -np.log(tmp0)
                            res[i, j] = res[i - 1, j] * np.exp(-aa_lr * tmp)

                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'LSN'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp1 = aa_rho * aa_u / (aa_u + aa_ubis) * x
                        tmp2 = aa_rhobis * aa_ubis / (aa_u + aa_ubis) * np.minimum(0, x)

                        if aa_lr == 1:
                            res[i, j] = res[i - 1, j] * (1 + tmp1 + tmp2)

                        if (1 + tmp1 + tmp2 < 0) & (aa_lr != 1):
                            res[i, j] = 0.

                        else:
                            tmp = -np.log(1 + tmp1 + tmp2)
                            res[i, j] = res[i - 1, j] * np.exp(-aa_lr * tmp)


                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'logloss'):

                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 = np.maximum(1 + aa_rho * x, 0)


                        res[i, j] = res[i - 1, j] * (np.log(1+(np.e-1)*np.maximum(tmp0,0)) )



                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'loglossdownside'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 =  1 + aa_rho * np.minimum(x, 0)

                        res[i, j] = res[i - 1, j] * (np.log(1 + (np.e - 1) * np.maximum(tmp0, 0)))




                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'loglogloss'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 = np.maximum(1 + aa_rho * x, 0)

                        res[i, j] = res[i - 1, j] * (np.log(np.log(np.e+(np.e**np.e-np.e)*np.maximum(tmp0,0))))

                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'logloglossdownside'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 = 1 + aa_rho * np.minimum(x, 0)
                        res[i, j] = res[i - 1, j] * (np.log(np.log(np.e + (np.e ** np.e - np.e) * np.maximum(tmp0, 0))))



                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'sigmoidloss'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 = np.maximum(1 + aa_rho * x, 0)


                        if tmp0:
                            tmp=1 / (1 + ((- aa_rho * x) / tmp0) ** beta_sigmoid)
                        else:
                            tmp=0

                        res[i, j] = res[i - 1, j] * tmp



                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'sigmoidlossdownside'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp = np.minimum(x, 0)
                        tmp0 = 1 +  aa_rho * tmp
                        tmp1 = np.maximum(tmp0, 0)

                        if tmp1:
                            tmp2 = 1 / (1 + ((- aa_rho * x) / tmp1) ** beta_sigmoid)
                        else:
                            tmp2 = 0

                        res[i, j] = res[i - 1, j] * tmp2


                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'powerloss'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 = 1 + (aa_rho * x)**power
                        tmp0 = np.maximum(tmp0, 0)



                        res[i, j] = res[i - 1, j] * tmp0

                    if ((active_and_not[i, j] == 0)) & (aa_losstype == 'powerlossdownside'):
                        x = omegaAA[i] * return_for_loss_sleepingAA[i]
                        tmp0 = 1 + np.minimum(aa_rho * x,0) ** power
                        tmp0 = np.maximum(tmp0, 0)

                        res[i, j] = res[i - 1, j] * tmp0

    return res, omegaAA, res2


def plot_scaling(lossname, res, alpha):
    mypath = "."

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    result = [f for f in onlyfiles if '_' + lossname + '_' in f and '_' + str(res) + '_' in f]
    s = [x for x in result if ('sharpe' in x) and ("alpha" not in x) and ("restricted" not in x)]
    r = [x for x in result if ('return' in x) and ("alpha" not in x) and ("restricted" not in x)]
    sr = [x for x in result if ('sortino' in x) and ("alpha" not in x) and ("restricted" not in x)]
    sh = pd.read_csv(s[0], index_col=[0])
    re = pd.read_csv(r[0], index_col=[0])
    ss = pd.read_csv(sr[0], index_col=[0])
    d = {"LS": "Long Short Loss", "LSD": "Downside Long Short Loss", "LSN": "Mixed Long Short Loss"}

    fig = px.line()
    for i, n in enumerate(sh.columns):
        fig.add_scatter(x=sh.index, y=sh[sh.columns[i]], name=sh.columns[i])
    fig.update_layout(
        title="Sharpe ratio for " + d[lossname]
        , xaxis_title="Scaling factor"
        , yaxis_title="Sharpe Ratio"
        , font=dict(size=25)
        , template="simple_white"  # "none"
    )

    fig.show()

    fig = px.line()
    for i, n in enumerate(re.columns):
        fig.add_scatter(x=re.index, y=re[re.columns[i]], name=re.columns[i])
    fig.update_layout(
        title="Return for " + d[lossname]
        , xaxis_title="Scaling factor"
        , yaxis_title="1+Return"
        , font=dict(size=25)
        , template="simple_white"  # "none"
    )

    fig.show()

    fig = px.line()
    for i, n in enumerate(re.columns):
        fig.add_scatter(x=ss.index, y=ss[ss.columns[i]], name=ss.columns[i])
    fig.update_layout(
        title="Sortino ratio for " + d[lossname]
        , xaxis_title="Scaling factor"
        , yaxis_title="Sortino ratio"
        , font=dict(size=25)
        , template="simple_white"  # "none"
    )

    fig.show()


def dd(obs):
    i = np.argmax(np.maximum.accumulate(obs) - obs)  # end of the period
    j = np.argmax(obs[:i])  # start of period
    return obs[j] - obs[i]


def get_cluster_files(resolution, pnl_or_pos, cutoff, threshold,truncate=1):
    cluster_folder_names = {10: 'dynamic_clusters_cutoff_10',
                            25: 'dynamic_clusters_cutoff_25',
                            40: 'dynamic_clusters_cutoff_40',
                            50: 'dynamic_clusters_cutoff_50',
                            1000: 'clusters_cutoff_1000'
                            }

    mypath = cluster_folder_names[cutoff]
    cluster_name = {'netpos': 'netposition',
                    'pnl': '_pnl_'}

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    eurusdcommunities = [f for f in onlyfiles if  ((cluster_name[pnl_or_pos] in f) & ('hierthreshold_' + str(threshold) in f) & (
                      "_" + str(resolution) + "_mins" in f))]

    eurusdcommunities.sort(key=lambda x: int(x.split('_')[-1][:-4]))



    result = []
    M = 0

    output = []
    alltraders = []

    for name in eurusdcommunities:

        d = pd.read_csv(mypath+'//' + name)
        d.columns = ['cluster', 'trader']

        le = preprocessing.LabelEncoder()

        if (len(d) == 0): continue

        alltraders.append(d['trader'].values)
    import itertools

    alltraders = list(itertools.chain(*alltraders))

    le2 = preprocessing.LabelEncoder()
    le2.fit(np.unique(np.array(alltraders)))

    for name in eurusdcommunities:

        d = pd.read_csv(mypath + '//' + name)
        d.columns = ['cluster', 'trader']

        le = preprocessing.LabelEncoder()
        if (len(d) == 0): continue

        le.fit(d['cluster'].values)
        d['cluster'] = le.transform(d.cluster.values)
        d['trader'] = d.trader.values

        output.append(list(d.groupby('cluster')['trader'].apply(list).values))
    if truncate:
        return [[x for x in output[i] if len(x) >= truncate] for i in range(len(output))]
    return output




    return result


'''
def cluster_dynamics2(name2, minutes, cut, window, stratname, truncate=1):
    eurusdcommunities = get_cluster_files(resolution, pnl_or_pos, cutoff, threshold)
     

    result = []
    M = 0
    plt.ioff()
    mypath = "output\\dynamic_clusters\\"
    output = []
    alltraders = []

    for name in eurusdcommunities:

        d = pd.read_csv(mypath + name)

        le = preprocessing.LabelEncoder()

        if (len(d) == 0): continue

        alltraders.append(d['trader'].values)
    import itertools

    alltraders = list(itertools.chain(*alltraders))

    le2 = preprocessing.LabelEncoder()
    le2.fit(np.unique(np.array(alltraders)))

    for name in eurusdcommunities:

        d = pd.read_csv(mypath + name)

        le = preprocessing.LabelEncoder()
        if (len(d) == 0): continue

        le.fit(d['cluster'].values)
        d['cluster'] = le.transform(d.cluster.values)
        d['trader'] = d.trader.values

        output.append(list(d.groupby('cluster')['trader'].apply(list).values))
    if truncate:
        return [[x for x in output[i] if len(x) >= truncate] for i in range(len(output))]
    return output'''