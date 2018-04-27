import requests
import pandas as pd
import os

# ---------------------------------------------------------------------------- #
def url_NSEIndex(offset_date=pd.Timestamp('today')):
    url_start = 'https://www.nseindia.com/content/indices/ind_close_all_'
    year = offset_date.strftime('%Y')
    month = offset_date.strftime('%m')
    day = offset_date.strftime('%d')
    url_end = day + month + year + '.csv'
    url = url_start + url_end

    return url

# ---------------------------------------------------------------------------- #
def url_NSE(offset_date=pd.Timestamp('today')):

    year = offset_date.strftime('%Y')
    month = offset_date.strftime('%b').upper()
    day = offset_date.strftime('%d')

    url_join = '/'
    url_start = 'https://www.nseindia.com/content/historical/EQUITIES/'
    url_end = 'cm' + day + month + year + 'bhav.csv.zip'
    url =  url_start + year + url_join + month + url_join + url_end

    return url

# ---------------------------------------------------------------------------- #
def download_NSEdata(df=None,url=url_NSE(pd.Timestamp('today')),save_path=None,date=None):

    if date is None:
        date = '?'
    else:
        date = date.date()

    if url[-3:] == 'zip':
        tempFileName = 'downloadedFile.zip'
    else:
        tempFileName = 'downloadedFile'

    try:
        # download the file contents in binary format
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

        # open method to open a file on your system and write the contents
        with open(tempFileName, "wb") as code:
            code.write(r.content)

        # read and return the zip file as a pandas dataframe
        df_new = pd.read_csv(tempFileName)

        # Remove the downloaded temp file
        os.remove(tempFileName)

        if len(df_new) == 1:
            raise ValueError('No data available for this day!')

        if df is not None:
            df = df.append(df_new, ignore_index=True)
        else:
            df = df_new

        # save the new data frame
        if save_path is not None:
            df.to_csv(save_path, encoding='utf-8', index=False)

        print("Saved data for: %s" %(date))

    except:
        print("No data for :::::::::::: %s" %(date))

    return df

# ---------------------------------------------------------------------------- #
def download_NSEIndex_as_csv(df=None,save_path='NSEIndex_Data.csv',num_days_back=365):

    for d in range(num_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=d))
        df = download_NSEdata(df, url_NSEIndex(offset_date), save_path, offset_date)

    return df

# ---------------------------------------------------------------------------- #
def update_NSEIndex(NSE_data_path, save_path=None, max_days_back=30):

    if save_path == None:
        save_path=NSE_data_path

    downloaded = False

    df = pd.read_csv(NSE_data_path)
    # Get the last date avaiable in the previous data
    last_date = max(pd.to_datetime(df['Index Date'], format='%d-%m-%Y'))

    for d in range(max_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=(d)))

        if offset_date.date() == last_date.date():
            print('Last date of data reached.')
            break

        df = download_NSEdata(df, url_NSEIndex(offset_date), date=offset_date)
        downloaded = True

    # Save data once and if everything is downloaded
    if downloaded: df.to_csv(save_path, encoding='utf-8', index=False)

    return df
# ---------------------------------------------------------------------------- #
def download_NSE_as_csv(df=None,save_path='NSE_Data.csv',num_days_back=365):

    for d in range(num_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=d))
        df = download_NSEdata(df, url_NSE(offset_date), save_path, offset_date)

    return df
# ---------------------------------------------------------------------------- #
def update_NSE(NSE_data_path, save_path=None, max_days_back=30):

    if save_path == None:
        save_path=NSE_data_path

    downloaded = False

    df = pd.read_csv(NSE_data_path)
    # Get the last date avaiable in the previous data
    last_date = max(pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y'))

    for d in range(max_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=(d)))

        if offset_date.date() == last_date.date():
            print('Last date of data reached.')
            break

        df = download_NSEdata(df, url_NSE(offset_date), date=offset_date)
        downloaded = True

    # Save data once and if everything is downloaded
    if downloaded: df.to_csv(save_path, encoding='utf-8', index=False)

    return df

# ---------------------------------------------------------------------------- #

# download_NSE_as_csv(save_path='data_NSE.csv',num_days_back=7)
# update_NSE('data_NSE.csv',max_days_back=10)

save_path = 'data_NSEIndex.csv'
download_NSEIndex_as_csv(save_path=save_path)
update_NSEIndex(save_path)
