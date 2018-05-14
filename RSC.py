import nse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------- #
def compute_date_intersection(df1, df2, symbol):
    ''' d1 and d2 are Pandas date series '''

    print ('NOT THE SAME: %3.0f, %3.0f, %s' %(len(df1), len(df2), symbol))
    dates = df1[df1.isin(df2)]
    return dates

# ---------------------------------------------------------------------------- #
def compute_RSC50(df, df_index, symbol, window=5):
    global c1,c2,c3
    mask = (df['SYMBOL'] == symbol)
    symbol_date = df.loc[mask,'date']
    symbol_close = df.loc[mask,'CLOSE']

    mask = (df_index['Index Name'] == 'Nifty 500')
    nifty50_date = df_index.loc[mask,'Index Date']
    nifty50_close = df_index.loc[mask,'Closing Index Value']

    if len(symbol_date) ==0:
        c3 += 1
        print('>>> NOT EQ: %s' %(symbol))
        return -1
    if len(nifty50_date) != len(symbol_date):
        c1 +=1
        dates = compute_date_intersection(symbol_date, nifty50_date, symbol)
    else:
        c2 += 1
        dates = symbol_date
        print ('!!!!!!! SAME: %3.0f, %3.0f, %s' %(len(symbol_date), len(nifty50_date), symbol))
        # print (len(nifty50_date.intersection(symbol_date)))

# ---------------------------------------------------------------------------- #
def date_diff(vals):

    N = len(vals)
    max_val = max(vals[:-1])
    index = vals.index(max_val)
    return N - index

# ---------------------------------------------------------------------------- #
def compute_RSC(df, df_index, save_path):

    columns = ['Close', 'Avg Quaterly Traded Volume', 'Avg Yearly Traded Volume' ,'Close High' , 'RSC50_5', 'RSC50_10', 'RSC500_10', 'RSC500_20', 'Close Date Diff', 'RSC50 Date Diff', 'RSC500 Date Diff', 'EMA21', 'EMA50', 'EMA200', 'EMA200 Value','Avg Fortnightly TOTTRDVAL','Avg Quarterly TOTTRDVAL','Avg Yearly TOTTRDVAL', 'ISIN']
    df_RSC = pd.DataFrame(columns=columns)

    isin_numbers = df['ISIN'].unique()
    for isin_number in isin_numbers:
        mask = (df['ISIN'] == isin_number)
        isin_number_dates = df.loc[mask,'date']
        isin_number_close = df.loc[mask,'CLOSE']

        close_high = ''
        RSC50_5 = ''
        RSC50_10 = ''
        RSC500_10 = ''
        RSC500_20 = ''

        if isin_number_close.values[-1] == max(isin_number_close):
            close_high = 'New High'

        mask = (df_index['Index Name'] == 'Nifty 50')
        nifty50_dates = df_index.loc[mask,'Index Date']
        nifty50_close = df_index.loc[mask,'Closing Index Value']
        print ('%3.0f, %3.0f, %s' %(len(isin_number_dates), len(nifty50_dates), isin_number))

        if (len(isin_number_dates) == len(nifty50_dates)) & (len(isin_number_dates) > 200):
            dates = nifty50_dates[nifty50_dates.isin(isin_number_dates)]

            u = isin_number_close[isin_number_dates.isin(dates)].values
            v =  nifty50_close[nifty50_dates.isin(dates)].values
            scale = 100 * v[0] / u[0]
            scale = 1
            rsc = scale * np.divide(u , v)
            ewma5 = pd.ewma(rsc, span=5)
            ewma10 = pd.ewma(rsc, span=10)

            if ewma5[-1] == max(ewma5):
                RSC50_5 = 'New High'

            if ewma10[-1] == max(ewma10):
                RSC50_10 = 'New High'

            mask = (df_index['Index Name'] == 'Nifty 500')
            nifty50_dates = df_index.loc[mask,'Index Date']
            nifty50_close = df_index.loc[mask,'Closing Index Value']

            dates = nifty50_dates[nifty50_dates.isin(isin_number_dates)]

            u = isin_number_close[isin_number_dates.isin(dates)].values
            v =  nifty50_close[nifty50_dates.isin(dates)].values
            scale = 100 * v[0] / u[0]
            scale = 1
            rsc = scale * np.divide(u , v)
            ewma10 = pd.ewma(rsc, span=10)
            ewma20 = pd.ewma(rsc, span=20)

            if ewma10[-1] == max(ewma10):
                RSC500_10 = 'New High'

            if ewma20[-1] == max(ewma20):
                RSC500_20 = 'New High'

            # if RSC50_5 != '' or RSC50_10 != '' or RSC500_10 != '' or RSC500_20 != '':
            if close_high != '' or RSC50_5 != '' or RSC500_10 != '':

                close_val = isin_number_close.values[-1]

                ema21 = pd.ewma(u, span=21)
                ema50 = pd.ewma(u, span=50)
                ema200 = pd.ewma(u, span=200)
                ema200_val = ema200[-1]

                if close_high != '':
                    date_diff_close = date_diff(isin_number_close.values.tolist())
                else:
                    date_diff_close = ''
                if RSC50_5 != '':
                    date_diff_rsc50 = date_diff(ewma5.tolist())
                else:
                    date_diff_rsc50 = ''
                if RSC500_10 != '':
                    date_diff_rsc500 = date_diff(ewma10.tolist())
                else:
                    date_diff_rsc500 = ''

                diff_ema21 = int( 100 * (close_val - ema21[-1]) / ema21[-1] )
                diff_ema50 = int( 100 * (close_val - ema50[-1]) / ema50[-1] )
                diff_ema200 = int( 100 * (close_val - ema200_val) / ema200_val )

                mask_vol = (df['ISIN'] == isin_number)
                isin_number_vol = df.loc[mask_vol,'TOTTRDQTY']
                yearly_vol = isin_number_vol.mean()

                offset_qtr = (pd.Timestamp('today') - pd.DateOffset(days=(90)))
                isin_number_qtrvol = isin_number_vol.loc[(df['date'] > offset_qtr )]
                qtrly_vol = isin_number_qtrvol.mean()

                isin_number_val = df.loc[mask_vol,'TOTTRDVAL']
                yearly_val = isin_number_val.mean()

                isin_number_qtrval = isin_number_val.loc[(df['date'] > offset_qtr )]
                qtrly_val = isin_number_qtrval.mean()

                offset_fortnight = (pd.Timestamp('today') - pd.DateOffset(days=(14)))
                isin_number_fnval = isin_number_val.loc[(df['date'] > offset_fortnight )]
                fortnightly_val = isin_number_fnval.mean()

                isin_number_symbol =  df.loc[mask_vol,'SYMBOL']
                symbol = isin_number_symbol.iloc[-1]

                row = [close_val, qtrly_vol, yearly_vol, close_high, RSC50_5, RSC50_10, RSC500_10, RSC500_20, date_diff_close, date_diff_rsc50, date_diff_rsc500, diff_ema21, diff_ema50, diff_ema200, ema200_val, fortnightly_val, qtrly_val, yearly_val, isin_number]
                df_RSC.loc[symbol] = row


        print(df_RSC.tail())

    df_RSC.to_csv(save_path, encoding='utf-8', index=True)


# ---------------------------------------------------------------------------- #
df = nse.read_NSE('data_NSEdivfixed.csv')
df_index = nse.read_NSEIndex('data_NSEIndex.csv')

compute_RSC(df, df_index, 'data_RSC_20180514.csv')

# plt.figure()
# plt.semilogy(x,rsc)
# plt.semilogy(x,ewma5)
# plt.semilogy(x,ewma10)
# plt.gca().yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.1f"))
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
# plt.grid(True, which="both")
# plt.title(symbol)
# plt.show()
