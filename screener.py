import requests
import pandas as pd
import numpy as np
import nse
from bs4 import BeautifulSoup
import datetime
# ---------------------------------------------------------------------------- #
def load_symbol_page(symbol):
    url = "https://www.screener.in/company/" + symbol + "/"
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if r.status_code != 404:
        soup = BeautifulSoup(r.content)
    else:
        soup = None

    return soup
# ---------------------------------------------------------------------------- #
def download_screener_data(symbols):

    output_arr = []
    failed_symbols = []

    todays_date = datetime.datetime.today().date()
    today_date_str = todays_date.strftime("%Y-%m-%d")

    for symbol in symbols:
        soup = load_symbol_page(symbol)
        try:
            company_name = soup.h1.text
            four_columns = soup.find_all('li', {'class': 'four columns'})
            market_cap = four_columns[0].b.text
            book_value = four_columns[3].b.text
            pe_ratio = four_columns[4].b.text
            small = soup.find('small')
            sector = small.find('a').text.replace('\n','').strip()
            industry = small.find_all('a')[1].text.replace('\n','').strip()
            if sector == "":
                raise Exception("No Sector information found")
            if industry == "":
                raise Exception("No Industry information found")
            output_arr.append([symbol, company_name, sector, industry, market_cap, book_value, pe_ratio, today_date_str])
            print(symbol + ": " + market_cap + ',' + sector + ',' + industry)
        except:
            print("Unexpected error for: " + symbol )
            failed_symbols.append(symbol)

    print(failed_symbols)
    print("Above is a list of failed symbols")

    columns = ["Symbol", "Company Name", "Sector", "Industry", "Market Cap", "Book Value", "PE Ratio", "Date"]
    df = pd.DataFrame(output_arr, columns=columns)

    df_sectors = df["Sector"].value_counts()
    df_industries = df["Industry"].value_counts()

    df["Sector Size"] = df["Sector"].map(lambda x: df_sectors[x])
    df["Industry Size"] = df["Industry"].map(lambda x: df_industries[x])

    save_file_path = "./data/data_screener_latest.csv"
    df.to_csv(save_file_path, index=False)

    save_file_path = "./data/data_screener_latest.csv"
    df.to_csv(save_file_path, index=False)

    return df
# ---------------------------------------------------------------------------- #
def mapping(df, sector):
    if np.isnan(sector):
        return None

    return df.loc[sector]


if __name__ == "__main__":

    df = nse.read_NSE('./data/data_NSEallfixed.csv')
    symbols = df.SYMBOL.unique()
    df = download_screener_data(symbols)

    df = pd.read_csv("./data/data_screener_latest.csv")

    print(df.head())

    # symbol = "HDFCBANK"
    # load_symbol_page(symbol)
    # print(soup.h1.text)
