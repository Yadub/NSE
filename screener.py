import requests
import pandas as pd
import numpy as np
import nse
from bs4 import BeautifulSoup
import datetime


def load_symbol_page(symbol):
    """
        Inputs: symbol - Name for stock as a string
        Output: soup - BeautifulSoup for screener webpage of the inputted stock
    """
    url = "https://www.screener.in/company/" + symbol + "/"
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if r.status_code != 404:
        soup = BeautifulSoup(r.content, features="html.parser")
    else:
        soup = None
    return soup


def download_screener_data(symbols, file_save_dir=None):
    """
        Downloads data for the given symbols from www.screener.in

        Inputs: symbols - List of strings of the symbols of stocks of interest
                file_save_dir - dir path for downloaded data else saved to root
        Output: Pandas DataFrame containing the following for each symbol
                "Symbol", "Company Name", "Sector", "Industry", "Market Cap",
                "Book Value", "PE Ratio", "Date"
    """
    # Initialize arrays to save downloaded data
    output_arr = []
    failed_symbols = []
    # Get today's date
    todays_date = datetime.datetime.today().date()
    today_date_str = todays_date.strftime("%Y-%m-%d")
    # Load data from the webpage for each symbol
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
    # Display list of failed symbols
    print(failed_symbols)
    print("Above is a list of failed symbols")
    # Set data structure to save to pandas DataFrame
    columns = ["Symbol", "Company Name", "Sector", "Industry", "Market Cap", "Book Value", "PE Ratio", "Date"]
    df = pd.DataFrame(output_arr, columns=columns)
    # Compute number of companies in each sector and each industry for the given list of symbols
    df_sectors = df["Sector"].value_counts()
    df_industries = df["Industry"].value_counts()
    # Save the computed data to new columns in the DataFrame
    df["Sector Size"] = df["Sector"].map(lambda x: df_sectors[x])
    df["Industry Size"] = df["Industry"].map(lambda x: df_industries[x])
    # Save the downloaded data
    if file_save_dir is None: file_save_dir = "./"
    file_save_path = file_save_dir + "data_screener_" + today_date_str + ".csv"
    """
        TODO: Fix date of screener data based on previous market close information
    """
    df.to_csv(file_save_path, index=False)
    return df


if __name__ == "__main__":
    # Set working directory
    data_file_dir = './data/screener/'
    # Load data with list of symbols
    """
        TODO: Automate fetching all symbols available on screener.in
    """
    df = nse.read_NSE('./data/data_NSEallfixed.csv')
    symbols = df.SYMBOL.unique()
    # Get data and print the header for it
    df = download_screener_data(symbols, data_file_dir)
    print(df.head())
