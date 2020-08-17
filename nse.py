import requests
import pandas as pd
import numpy as np
import os, re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def url_NSEIndex(date=pd.Timestamp('today')):
    """
        Gives URL for NSE Index data csv file for a particular date
    """
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    url_start = 'https://www1.nseindia.com/content/indices/ind_close_all_'
    url = url_start + day + month + year + '.csv'
    return url


def url_NSE(date=pd.Timestamp('today')):
    """
        Gives URL for NSE Index data csv file for a particular date
    """
    year = date.strftime('%Y')
    month = date.strftime('%b').upper()
    day = date.strftime('%d')
    url_start = 'https://www1.nseindia.com/content/historical/EQUITIES/'
    url_end = 'cm' + day + month + year + 'bhav.csv.zip'
    url =  url_start + year + '/' + month + '/' + url_end
    return url


def download_NSEdata(df=None, url=url_NSE(pd.Timestamp('today')), save_path=None, date=None, len_check=False):
    """
        Downloads NSE data csv file for a particular date.
        Default is set to the equity data for today

        Inputs: df - pandas DataFrame to append the downloaded data to
                url - NSE URL to download the respective data from
                save_path - where to save the data if so required
                date - To see when calling the function back in time which date
                        the data has been downloaded for
                len_check - ensures that nearly empty download files are ignored
        Outputs: df - (updated) pandas DataFrame with newly downloaded data
                 downloaded - boolean whether data was successfully downloaded
    """
    if date is None: date = "date not given"
    if url[-3:] == 'zip':
        tempFileName = 'downloadedFile.zip'
    else:
        tempFileName = 'downloadedFile'
    # Attempt to download data without raising errors
    try:
        # download the file contents in binary format
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(tempFileName, "wb") as code:
            code.write(r.content)
        # read and return the zip file as a pandas dataframe
        df_new = pd.read_csv(tempFileName)
        # Remove the downloaded temp file
        os.remove(tempFileName)
        # Check for nearly empty download file
        if len_check:
            if len(df_new) < 10:
                raise ValueError('No data available for this day!')
        # Update DataFrame if given
        if df is not None:
            df = df.append(df_new, ignore_index=True)
            df.drop_duplicates(inplace=True)
        else:
            df = df_new
        # Save the new data frame
        if save_path is not None:
            df.to_csv(save_path, encoding='utf-8', index=False)
        # Print success
        print("Saved data for: %s" %(date))
        downloaded = True
    except:
        # Print failure
        print("!! No data for: %s" %(date))
        downloaded = False
    return df, downloaded


def download_NSEIndex_as_csv(df=None,save_path='NSEIndex_Data.csv',num_days_back=365):
    """
        downloads NSE Index data for a lookback period

        Inputs: df - pandas DataFrame to append
                save_path - where to save downloaded data
                num_days_back - integer of lookback period in days
        Outputs: df - (appended) Pandas DataFrame containing NSE Index data
    """
    for d in range(num_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=d))
        df, _ = download_NSEdata(df, url_NSEIndex(offset_date), save_path, offset_date, True)
    return df


def update_NSEIndex(NSEIndex_file_path, save_path=None, max_days_back=365):
    """
        Loads previously downloaded NSE Index data file and updates it to today

        Inputs: NSEIndex_file_path - path where DataFrame is stored
                save_path - where to save the updated DataFrame
                max_days_back - how far back to update the data file
        Outputs: df - DataFrame with updated NSE Index data
    """
    # If no new save path is given then assume save_path of original data file
    if save_path is None: save_path = NSEIndex_file_path
    downloaded = False
    # Load data
    df = pd.read_csv(NSEIndex_file_path)
    # Get the last date avaiable in the previous data
    last_date_in_data = max(pd.to_datetime(df['Index Date'], format='%d-%m-%Y'))
    # Download data going back from today upto max_days_back
    for d in range(max_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=(d)))
        if offset_date.date() == last_date_in_data.date():
            print('Last date of data reached.')
            break
        df, data_downloaded = download_NSEdata(df, url_NSEIndex(offset_date), date=offset_date, len_check=True)
        # If data_downloaded == True even once then set downloaded = True
        downloaded = downloaded or data_downloaded
    # Save data if something has been downloaded
    if downloaded: df.to_csv(save_path, encoding='utf-8', index=False)
    return df


def download_NSE_as_csv(df=None,save_path='NSE_Data.csv',num_days_back=365):
    """
        downloads NSE Equity data for a lookback period

        Inputs: df - pandas DataFrame to append
                save_path - where to save downloaded data
                num_days_back - integer of lookback period in days
        Outputs: df - (appended) Pandas DataFrame containing NSE Equity data
    """
    for d in range(num_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=d))
        df, _ = download_NSEdata(df, url_NSE(offset_date), save_path, offset_date)
    return df


def update_NSE(NSE_data_path, save_path=None, max_days_back=365):
    """
        Loads previously downloaded NSE Equity data file and updates it to today

        Inputs: NSEIndex_file_path - path where DataFrame is stored
                save_path - where to save the updated DataFrame
                max_days_back - how far back to update the data file
        Outputs: df - DataFrame with updated NSE Equity data
    """
    # If no new save path is given then assume save_path of original data file
    if save_path == None: save_path=NSE_data_path
    downloaded = False
    # Load data
    df = pd.read_csv(NSE_data_path)
    # Get the last date avaiable in the previous data
    last_date_in_data = max(pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y'))
    for d in range(max_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=(d)))
        if offset_date.date() == last_date_in_data.date():
            print('Last date of data reached.')
            break
        df, data_downloaded = download_NSEdata(df, url_NSE(offset_date), date=offset_date)
        # If data_downloaded == True even once then set downloaded = True
        downloaded = downloaded or data_downloaded
    # Save data if something has been downloaded
    if downloaded: df.to_csv(save_path, encoding='utf-8', index=False)
    return df


def get_num(x):
    """
        Takes string and outputs only the digits within it
    """
    return ''.join(ele for ele in x if ele.isdigit())


def add_stockSplitInfo(df_split=None, stockSplit_path=None):
    """
        Input a pandas DataFrame or a file path to load stock split DataFrame
        Outputs updated stock split DataFrame with info on previous face value
    """
    if df_split is None:
        df_split = pd.read_csv(stockSplit_path)
    # Set new column with previous face value to None
    df_split['Previous Face Value(Rs.)'] = None
    # for all stock splits get the previous face value based on file logic
    for x in df_split.index.values:
        val = df_split['Face Value(Rs.)'].loc[x]
        s = df_split['Purpose'].loc[x]
        l = len(s)
        nums = get_num(s)
        # Iterate over the purpose string to find value
        for i in range(l):
            if nums[-i:] == str(val):
                # Assumption that previous value is <= 10
                if nums[-i-1:-i] == str(0):
                    prev_val = int(nums[-i-2:-i])
                else:
                    prev_val = int(nums[-i-1:-i])
        # Update with found value
        df_split['Previous Face Value(Rs.)'].loc[x] = prev_val
    return df_split


def fix_stockSplits(df, df_splits, tol=0.33):
    """
        Fixes raw price data for equities based on stock split information

        Inputs: df - pandas DataFrame with NSE equity data
                df_splits - pandas DataFrame with NSE stock split info
                tol - tolerance of % variance of when to ignore stock split info
        Outputs: df - updated DataFrame with stock splits accounted for in
                      the historical price
    """
    num_splits = len(df_splits)
    for n in range(num_splits):
        # Get info for this symbol in readble format
        symbol = df_splits['Symbol'].loc[n]
        series_type = df_splits['Series'].loc[n]
        faceValue = df_splits['Face Value(Rs.)'].loc[n]
        previous_faceValue = df_splits['Previous Face Value(Rs.)'].loc[n]
        date_executed = df_splits['Ex-Date'].loc[n]
        ratio = faceValue / previous_faceValue
        # Work with ISIN number to overcome issues over series type change
        mask_isin_number = (df['SYMBOL'] == symbol) & (df['SERIES'] == series_type)
        isin_number = df.loc[mask_isin_number, 'ISIN'].unique()
        if len(isin_number) == 0:
            print('No data for %s' %(symbol))
            continue
        isin_number = isin_number[0]
        # Get the close value on the day of execution
        mask_t0 = (df['ISIN'] == isin_number) & (df['date'] == date_executed)
        close_t0 = df.loc[mask_t0, 'CLOSE'].values
        # Get the close value on the previous market open day taking into account closed market days upto 14 days back
        for d in range(14):
            mask_t1 = (df['ISIN'] == isin_number) & (df['date'] == date_executed - pd.DateOffset(days=d+1))
            close_t1 = df.loc[mask_t1, 'CLOSE'].values
            if close_t1.size > 0:
                break
            else:
                close_t1 = close_t0
        # If there is % diff greater than the tolerance then adjust data for stock splits
        if np.abs((close_t1 - close_t0)/close_t1) > tol:
            df.loc[mask_t0, 'PREVCLOSE'] = ratio * df.loc[mask_t0, 'PREVCLOSE']
            mask = (df['ISIN'] == isin_number) & (df['date'] < date_executed)
            df.loc[mask, ['OPEN','CLOSE','LOW','HIGH','LAST','PREVCLOSE']] = ratio * df.loc[mask, ['OPEN','CLOSE','LOW','HIGH','LAST','PREVCLOSE']]
            df.loc[mask, ['TOTTRDQTY']] = (1.0/ratio) * df.loc[mask, ['TOTTRDQTY']]
        else:
            print('Failed for %s on %s' %(symbol, date_executed.date()))
    # Plots a graph when finished to see whether the end result is fine
    # df_symbol = df[df['ISIN'] == isin_number]
    # x = df_symbol['date']
    # y = df_symbol['CLOSE']
    # plt.figure()
    # plt.plot(x,y)
    # plt.title(symbol)
    # plt.show()
    return df


def fix_dividends(df, df_divs, tol=0.33, display=True):
    """
        Fixes stock split fixed price data for equities based on dividend info

        Inputs: df - pandas DataFrame with NSE equity data
                df_divs - pandas DataFrame with NSE dividend info
                tol - tolerance of % variance of when to ignore dividend info
                display - boolean for whether function should output to console
        Outputs: df - updated DataFrame with dividends accounted for in
                      the historical price
    """
    starting_date_df = df['date'].min()
    mask_df_divs_startingDate = pd.to_datetime(df_divs['Ex-Date']) >= starting_date_df
    df_divs = df_divs.loc[mask_df_divs_startingDate]
    # Get num of dividends to compute
    num_divs = len(df_divs)
    # Iterate
    for n in range(num_divs):
        symbol = df_divs['Symbol'].iloc[n]
        series_type = df_divs['Series'].iloc[n]
        date_executed = df_divs['Ex-Date'].iloc[n]
        s = df_divs['Purpose'].iloc[n]
        m = re.findall(r'-?\d+\.?\d*',s)
        # Currently assuming dividend is the largest number and is per share
        if not m:
            print('Failed for %s on %s because %s' %(symbol, date_executed.date(), 'dividend amount not specified!'))
            continue
        dividend = max([float(i) for i in m])
        # Work with ISIN number to overcome issues over series type change
        mask_isin_number = (df['SYMBOL'] == symbol) & (df['SERIES'] == series_type)
        isin_number = df.loc[mask_isin_number, 'ISIN'].unique()
        if isin_number.size > 0:
            isin_number = isin_number[0]
        else:
            e = 'No data for this intrument found in NSE data file'
            if display: print('Failed for %s on %s because %s' %(symbol, date_executed.date(), e))
            continue
        # Get the close value on the day of execution
        mask_t0 = (df['ISIN'] == isin_number) & (df['date'] == date_executed)
        close_t0 = df.loc[mask_t0, 'CLOSE'].values
        if close_t0.size > 1:
            e = 'More than 1 closing value for the date of execution'
            if display: print('Failed for %s on %s because %s, %s' %(symbol, date_executed.date(), e, close_t0))
            continue
        # Iterative over 14 days to find when dividend was given
        for d in range(14):
            mask_t1 = (df['ISIN'] == isin_number) & (df['date'] == date_executed - pd.DateOffset(days=d+1))
            close_t1 = df.loc[mask_t1, 'CLOSE'].values
            if len(close_t1) > 1:
                e = 'multiple Symbols with the same name found'
                break
            if close_t1.size > 0:
                e = 'close_t1 found'
                break
            else:
                e = 'close_t1 not found - i.e. no data available in NSE data file for date of execution of dividend for this intrument'
        # Set close_t0 and close_t1 for error cases
        if e != 'close_t1 found':
            close_t0 = 1
            close_t1 = close_t0 / (1-tol)
        # If there is % diff greater than the tolerance then adjust data for stock splits
        if np.abs((close_t1 - close_t0)/close_t1) < tol:
            ratio = (close_t0 - dividend)/close_t0
            if type(ratio) != float:
                ratio = ratio.item()
            df.loc[mask_t0, 'PREVCLOSE'] = ratio * df.loc[mask_t0, 'PREVCLOSE']
            mask = (df['ISIN'] == isin_number) & (df['date'] < date_executed)
            df.loc[mask, ['OPEN','CLOSE','LOW','HIGH','LAST','PREVCLOSE','TOTTRDVAL']] = ratio * df.loc[mask, ['OPEN','CLOSE','LOW','HIGH','LAST','PREVCLOSE','TOTTRDVAL']]
            if display: print('Succeess for %10s on %s' %(symbol, date_executed.date()))
        else:
            if display: print('Failed for %s on %s because %s' %(symbol, date_executed.date(), e))
    # Plots a graph when finished to see whether the end result is fine
    # if display:
    #     df_symbol = df[df['ISIN'] == isin_number]
    #     x = df_symbol['date']
    #     y = df_symbol['CLOSE']
    #     plt.figure()
    #     plt.plot(x,y)
    #     plt.title(symbol)
    #     plt.show()
    return df


def read_NSE(path):
    """
        Inputs path to read NSE Equity data, fix date column format, sort data
        by symbol and date.
    """
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y')
    df.sort_values(by=['SYMBOL','date'], inplace=True)
    return df


def read_NSEIndex(path):
    """
        Inputs path to read NSE Index data, fix date column format, sort data
        by index name and date.
    """
    df_index = pd.read_csv(path)
    df_index['Index Date'] = pd.to_datetime(df_index['Index Date'], format='%d-%m-%Y')
    df_index.sort_values(by=['Index Name','Index Date'], inplace=True)
    return df_index
# ---------------------------------------------------------------------------- #
def plot_symbol(df, symbol, fig=None, color=None, symbol_name=None, linestyle='solid'):
    """
        Plots price for a given symbol in the DataFrame
    """
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
    if symbol_name == None: symbol_name = symbol
    # Work with ISIN number to overcome issues over series type change
    mask_isin_number = (df['SYMBOL'] == symbol)
    isin_number = df.loc[mask_isin_number, 'ISIN'].unique()
    isin_number = isin_number[0]
    mask = df['ISIN'] == isin_number
    x = df.loc[mask,'date']
    y = df.loc[mask,'CLOSE']
    if color is None:
        plt.semilogy(x,y,label=symbol_name,linestyle=linestyle)
    else:
        plt.semilogy(x,y,color=color,label=symbol_name, linestyle=linestyle)
    # Set labels
    plt.xlabel('date')
    plt.ylabel('Close Price')
    plt.gca().yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.0f"))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    return fig


if __name__ == "__main__":
    data_stock_file_path = './data/data_NSE.csv'
    data_index_file_path = './data/data_NSEIndex.csv'
    ### DOWNLOAD NSE DATA FOR THE FIRST TIME ###
    # download_NSE_as_csv(save_path=data_stock_file_path)
    # download_NSEIndex_as_csv(save_path=data_index_file_path)
    ## UPDATE PREVIOUSLY DOWNLOADED NSE DATA ###
    update_NSE(data_stock_file_path)
    update_NSEIndex(data_index_file_path)
    ### DOWNLOAD SPLIT DATA AND UPDATE THE NSE FILE ###
    url_split = 'https://www1.nseindia.com/corporates/datafiles/CA_LAST_12_MONTHS_SPLIT.csv'
    # df_splits, _ = download_NSEdata(url=url_split, save_path='./data/data_stockSplits.csv')
    # Update previously downloaded split data
    df_splits = pd.read_csv('./data/data_stockSplits.csv')
    df_splits, _ = download_NSEdata(df=df_splits, url=url_split, save_path='./data/data_stockSplits.csv', date=pd.Timestamp('today'))
    # Compute stock split previous face value
    df_splits = add_stockSplitInfo(stockSplit_path='./data/data_stockSplits.csv')
    df_splits['Ex-Date'] = pd.to_datetime(df_splits['Ex-Date'], format='%d-%b-%Y')
    # Load NSE data and fix it for stock splits
    df = read_NSE(data_stock_file_path)
    df = fix_stockSplits(df, df_splits)
    # Drop working date column from NSE data and save it
    df.drop(columns=['date'],inplace=True)
    df.to_csv('./data/data_NSEfixed.csv', encoding='utf-8', index=False)
    print('Stock Split NSE Data saved')
    #
    ### DOWNLOAD DIVIDEND DATA AND UPDATE THE NSE FILE ###
    url_dividend = 'https://www1.nseindia.com/corporates/datafiles/CA_LAST_12_MONTHS_DIVIDEND.csv'
    # df_dividends, _ = download_NSEdata(url=url_dividend, save_path='./data/data_stockDividends.csv')
    # Update previous dividend file
    df_dividends = pd.read_csv('./data/data_stockDividends.csv')
    df_dividends, _ = download_NSEdata(df=df_dividends, url=url_dividend, save_path='./data/data_stockDividends.csv', date=pd.Timestamp('today'))
    # Format date field for working in DataFrame for dividends
    df_dividends['Ex-Date'] = pd.to_datetime(df_dividends['Ex-Date'], format='%d-%b-%Y')
    # Load and fix NSE equity data
    df = read_NSE('./data/data_NSEfixed.csv')
    df = fix_dividends(df, df_dividends)
    # Drop working date column from NSE data and save it
    df.drop(columns=['date'],inplace=True)
    df.to_csv('./data/data_NSEallfixed.csv', encoding='utf-8', index=False)
    ### PLOTTING FIXED VS UNFIXED DATA ###
    symbol = 'PERSISTENT'
    # Plot div fixed data
    df = read_NSE('./data/data_NSEallfixed.csv')
    fig = plot_symbol(df, symbol, fig=None, color='b', symbol_name='Fixed Symbol')
    # Plot only stock split fixed data
    df = read_NSE(data_stock_file_path)
    fig = plot_symbol(df, symbol, fig, color='r', symbol_name='Unfixed Symbol', linestyle='dashed')
    # Add grid and show
    plt.grid(True, which="both")
    plt.legend(loc='upper left')
    plt.title(symbol)
    plt.show()
