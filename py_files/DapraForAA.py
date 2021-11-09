from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
from itertools import groupby





def dapra_row_pnl(subprices, row):
    out = pd.Series(np.zeros(len(subprices)), index=subprices.index)
    if len(subprices) == 0:
        a = row['Side'] * (row['ClosePrice'] - row['OpenPrice']) * row['Amount']
        out = pd.Series(a, index=[row.CloseTime.ceil('1min')])


    elif len(subprices) == 1:
        out[0] = row['Side'] * (row['ClosePrice'] - row['OpenPrice']) * row['Amount']
    elif len(subprices) == 2:
        if row.Side == 1:
            out[0] = (subprices.bid[0] - row['OpenPrice']) * row['Amount']
            out[1] = (row['ClosePrice'] - subprices.bid[0]) * row['Amount']
        else:
            out[0] = -(subprices.ask[0] - row['OpenPrice']) * row['Amount']
            out[1] = -(row['ClosePrice'] - subprices.ask[0]) * row['Amount']


    else:
        if row.Side == 1:
            tmp1 = (subprices.bid - subprices.bid.shift(1)) * row['Amount']
            tmp1[0] = (subprices.bid[0] - row['OpenPrice']) * row['Amount']
            tmp1[-1] = (row['ClosePrice'] - subprices.bid[-2]) * row['Amount']
        else:
            tmp1 = -(subprices.ask - subprices.ask.shift(1)) * row['Amount']
            tmp1[0] = (row['OpenPrice'] - subprices.ask[0]) * row['Amount']
            tmp1[-1] = (-row['ClosePrice'] + subprices.ask[-2]) * row['Amount']
        out = pd.Series(tmp1, index=subprices.index)
    return out

def dapra_row_omega(subprices, row):
    out = pd.Series(np.zeros(len(subprices)), index=subprices.index)
    if len(subprices) == 0:
        a = (row['ClosePrice'] / row['OpenPrice']) - 1
        out = pd.Series(a, index=[row.CloseTime.ceil('1min')])


    elif len(subprices) == 1:
        out[0] = (row['ClosePrice'] / row['OpenPrice']) - 1
    elif len(subprices) == 2:
        if row.Side == 1:
            out[0] = (subprices.bid[0] / row['OpenPrice']) - 1
            out[1] = (row['ClosePrice'] / subprices.bid[0]) - 1
        else:
            out[0] = (subprices.ask[0] / row['OpenPrice']) - 1
            out[1] = (row['ClosePrice'] / subprices.ask[0]) - 1


    else:
        if row.Side == 1:
            tmp1 = (subprices.bid / subprices.bid.shift(1)) - 1
            tmp1[0] = (subprices.bid[0] / row['OpenPrice']) - 1
            tmp1[-1] = (row['ClosePrice'] / subprices.bid[-2]) - 1
        else:
            tmp1 = (subprices.ask / subprices.ask.shift(1)) - 1
            tmp1[0] = (subprices.ask[0] / row['OpenPrice']) - 1
            tmp1[-1] = (row['ClosePrice'] / subprices.ask[-2]) - 1
        out = pd.Series(tmp1, index=subprices.index)
    return out


def dapra_row_omega_mid(subprices, row):
    out = pd.Series(np.zeros(len(subprices)), index=subprices.index)

    if row.TraderId == 'T18771':
        subprices['price'] = subprices.ask / 2 + subprices.bid / 2
        if len(subprices) == 0:
            a = (row['ClosePrice'] / row['OpenPrice']) - 1
            out = pd.Series(a, index=[row.CloseTime.ceil('1min')])


        elif len(subprices) == 1:
            out[0] = (row['ClosePrice'] / row['OpenPrice']) - 1
        elif len(subprices) == 2:

            out[0] = (subprices.price[0] / row['OpenPrice']) - 1
            out[1] = (row['ClosePrice'] / subprices.price[0]) - 1



        else:

            tmp1 = (subprices.price / subprices.price.shift(1)) - 1
            tmp1[0] = (subprices.price[0] / row['OpenPrice']) - 1
            tmp1[-1] = (row['ClosePrice'] / subprices.price[-2]) - 1

            out = pd.Series(tmp1, index=subprices.index)
        return out


    else:

        if len(subprices) == 0:
            a = (row['ClosePrice'] / row['OpenPrice']) - 1
            out = pd.Series(a, index=[row.CloseTime.ceil('1min')])


        elif len(subprices) == 1:
            out[0] = (row['ClosePrice'] / row['OpenPrice']) - 1
        elif len(subprices) == 2:
            if row.Side == 1:
                out[0] = (subprices.bid[0] / row['OpenPrice']) - 1
                out[1] = (row['ClosePrice'] / subprices.bid[0]) - 1
            else:
                out[0] = (subprices.ask[0] / row['OpenPrice']) - 1
                out[1] = (row['ClosePrice'] / subprices.ask[0]) - 1


        else:
            if row.Side == 1:
                tmp1 = (subprices.bid / subprices.bid.shift(1)) - 1
                tmp1[0] = (subprices.bid[0] / row['OpenPrice']) - 1
                tmp1[-1] = (row['ClosePrice'] / subprices.bid[-2]) - 1
            else:
                tmp1 = (subprices.ask / subprices.ask.shift(1)) - 1
                tmp1[0] = (subprices.ask[0] / row['OpenPrice']) - 1
                tmp1[-1] = (row['ClosePrice'] / subprices.ask[-2]) - 1
            out = pd.Series(tmp1, index=subprices.index)
        return out


def dapra_row_count(row, x):
    return pd.Series([1.0] * len(x), index=x.index)


def get_dapra_init(minimum, maximum):
    timeindex = pd.date_range(minimum, maximum, periods=(maximum - minimum).total_seconds() / 60 + 1)
    return pd.Series(np.zeros(len(timeindex)), index=timeindex)


def get_sub_prices(prices_data, from_date, to_date):
    return prices_data[(prices_data.index <= to_date.ceil('1min')) & (prices_data.index > from_date.floor('1min'))]


def sum_two_series(s1, s2):
    if len(s1) < len(s2):
        return s2.add(s1, fill_value=0)
    else:
        return s1.add(s2, fill_value=0)


def dapra_row_order(subprices, row):
    out = pd.Series(np.zeros(len(subprices)), index=subprices.index)
    if len(subprices) == 0:
        a = row['Side'] * row['Amount']
        out = pd.Series(a, index=[row.CloseTime.ceil('1min')])


    else:
        tmp1 = [row['Side'] * row['Amount']] * (len(subprices))
        out = pd.Series(tmp1, index=subprices.index)
    return out





def not_zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([1], np.equal(a, 0).view(np.int8), [1]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 0.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def max_not_zero_runs(a):
    return [max(abs(np.array(list(g)))) for k, g in groupby(a, lambda x: x != 0) if k]


def gamma_trader(a):
    b = a.copy() * 1.
    xx = not_zero_runs(b)
    yy = max_not_zero_runs(b)
    for x, y in zip(xx, yy):
        b[x[0]:x[1]] = b[x[0]:x[1]] / y
    return b

def DAPRA_dynamic_clusters_AA( cutoff):
    assert cutoff in [1000,500]
    df = pd.read_csv('data//Fudged_EURUSDTrades_2014To2016.csv', parse_dates=['OpenTime', 'CloseTime', 'QdfTime'],
                     infer_datetime_format=True)

    names = ['event', 'Symbol', 'QdfTime', 'bid', 'ask', 'other1', 'other2', 'other3', 'other4']
    prices = pd.read_csv('data//FudgeRaw1SecPrices.csv', header=None, names=names, parse_dates=['QdfTime'],
                         infer_datetime_format=True)

    print('--------------------Prepare trades df--------------')

    #min_number_trades_cutoff = cutoff / 2
    df = df[df.ClosePrice > 0]
    df = df[df.CloseTime != df.OpenTime]
    df = df[df.Amount != 0]
    df['Side'] = df['Type'].map({'buy': 1, 'sell': -1})



    #most_activ_clients = df.groupby('TraderId').TraderId.count()
    #most_activ_clients = most_activ_clients[most_activ_clients >= min_number_trades_cutoff]
    df1440 = pd.read_csv('output\\tosvn\\davidnewtosvn_symbol_EURUSD_delta_1440mins' + str(cutoff) + 'cut.csv',
                         parse_dates=['QdfTime'], infer_datetime_format=True).drop("QdfTime",axis=1)

    df = df[df.TraderId.isin(list(df1440.columns))]
    print(df.TraderId.nunique())

    print('--------------------Prepare prices df--------------')
    prices['Time'] = prices.QdfTime
    prices.QdfTime = prices.QdfTime.dt.ceil('1min')
    prices = prices.groupby('QdfTime').agg('last')
    print('--------------------Prepare for DAPRA--------------')
    maximum = df['CloseTime'].max().ceil('1min')
    minimum = df['OpenTime'].min().floor('1min')
    df = df[['OpenTime', 'CloseTime', 'Side', 'OpenPrice', 'ClosePrice', 'Amount', 'TraderId']]
    prices = prices.reset_index()[['QdfTime', 'bid', 'ask']]

    prices.set_index('QdfTime', inplace=True)
    output = get_dapra_init(minimum, maximum)
    print('------------------ LOOP-----------------------')
    output_dict = {'QdfTime': output.index}

    traders = df.TraderId.unique()
    print('ok ')
    for trader in tqdm_notebook(traders):

        new = df[df.TraderId == trader]
        output = get_dapra_init(minimum, maximum)
        out = get_dapra_init(minimum, maximum)
        out2=get_dapra_init(minimum,maximum)
        for index, row in new.iterrows():#  tqdm_notebook(new.iterrows(),total= len(new)):

            subprices = get_sub_prices(prices, row.OpenTime, row.CloseTime)

            # dapra_add=dapra_row(subprices,row)
            dapra_add = dapra_row_omega(subprices, row)
            dapra_add_order = dapra_row_order(subprices, row)
            ones=dapra_row_count(row,dapra_add_order)

            output[dapra_add.index] += dapra_add.values
            out[dapra_add_order.index] += dapra_add_order.values
            out2[dapra_add_order.index]+=ones.values

        output_dict[trader + '_omega'] = output
        # out=(out/abs(out)).replace(np.nan,0)
        out = gamma_trader(out.values)

        output_dict[trader + '_gamma'] = out
        # print(  pd.DataFrame(output_dict).head())
        out2=abs(out2).replace(0,1)
        output_dict[trader + '_omega'] = output_dict[trader + '_omega']   /out2

    save = 'data\\DAPRA_dynamic_clusters_cutoff_' + str(cutoff) + '.csv'
    pd.DataFrame(output_dict).to_csv(save)
