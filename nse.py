import requests
import pandas as pd
import os
# ---------------------------------------------------------------------------- #
def download_NSE(df=None,offset_date=pd.Timestamp('today'),save_path=None):

    year = offset_date.strftime('%Y')
    month = offset_date.strftime('%b').upper()
    day = offset_date.strftime('%d')

    url_join = '/'
    url_start = 'https://www.nseindia.com/content/historical/EQUITIES/'
    url_end = 'cm' + day + month + year + 'bhav.csv.zip'
    url =  url_start + year + url_join + month + url_join + url_end

    try:
        # download the file contents in binary format
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

        # open method to open a file on your system and write the contents
        with open("zippedFile.zip", "wb") as code:
            code.write(r.content)

        # read the zip file as a pandas dataframe
        df_new = pd.read_csv('zippedFile.zip')   # pandas version 0.18.1 takes zip files

        # remove downloaded zipfile
        os.remove('zippedFile.zip')

        if len(df_new) == 1:
            raise ValueError('No data avialable for this day!')

        if df is not None:
            df = df.append(df_new, ignore_index=True)
        else:
            df = df_new

        if save_path is not None:
            # save the new data frame
            df.to_csv(save_path, encoding='utf-8', index=False)

        print("Saved NSE data for: %s" %((day+month+year)))

    except:
        print("No NSE data for ::(failed):: %s" %((day+month+year)))

    return df

# ---------------------------------------------------------------------------- #
def download_NSE_as_csv(df=None,save_path='NSE_Data.csv',num_days_back=365):

    for d in range(num_days_back):
        offset_date = (pd.Timestamp('today') - pd.DateOffset(days=d))
        df = download_NSE(df,offset_date,save_path)

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

        df = download_NSE(df,offset_date)
        downloaded = True

    # Save data once and if everything is downloaded
    if downloaded: df.to_csv(save_path, encoding='utf-8', index=False)

    return df

# ---------------------------------------------------------------------------- #
# url = 'https://www.nseindia.com/products/content/sec_bhavdata_full.csv'
# url = 'https://www.nseindia.com/content/historical/EQUITIES/2018/Apr/cm20Apr2018bhav.csv.zip'

# download_NSE_as_csv(save_path='NSE_DataTest.csv',num_days_back=7)
update_NSE('NSE_Data.csv',max_days_back=10)
