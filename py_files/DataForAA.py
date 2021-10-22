from tqdm import tqdm_notebook
import pandas as pd
import numpy as np

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


def dapra_row(subprices, row):
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


def DAPRA_for_dynamic_clusters_AA(df, prices, cutoff):
    print('--------------------Prepare trades df--------------')
    min_number_trades_cutoff = cutoff / 2
    df = df[df.ClosePrice > 0]
    df = df[df.CloseTime != df.OpenTime]
    df = df[df.Amount != 0]
    df['Side'] = df['Type'].map({'buy': 1, 'sell': -1})
    most_activ_clients = df.groupby('TraderId').TraderId.count()
    most_activ_clients = most_activ_clients[most_activ_clients >= min_number_trades_cutoff]
    df = df[df.TraderId.isin(most_activ_clients.index)]
    print(len(most_activ_clients))

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
    for trader in tqdm_notebook(traders):

        new = df[df.TraderId == trader]
        output = get_dapra_init(minimum, maximum)
        out = get_dapra_init(minimum, maximum)
        for index, row in new.iterrows():  # tqdm(new.iterrows(),total= len(new)):

            subprices = get_sub_prices(prices, row.OpenTime, row.CloseTime)

            dapra_add = dapra_row(subprices, row)
            dapra_add_order = dapra_row_order(subprices, row)
            output[dapra_add.index] += dapra_add.values
            out[dapra_add_order.index] += dapra_add_order.values
        output_dict[trader] = output
        out = (out / abs(out)).replace(np.nan, 0)
        output_dict[trader + '_side'] = output

    save = 'data\\DAPRA_dynamic_clusters_cutoff_' + str(cutoff) + '.csv'
    pd.DataFrame(output_dict).to_csv(save)

