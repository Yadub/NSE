# Download historical data from the NSE website

### Files
* nse.py: Downloads (or updates) available csv NSE exchange data from their website and adjusts the data for stock splits and dividends.
* screener.py: Downloads industry, sector, marketcap and PE ratios of stocks 

### Useful Links
NSE Series Types: https://www.truedata.in/blog/what-do-the-nse-series-eq-be-bl-bt-gc-il-iq-mean/

### Packages Required
###### All files
* requests
* pandas
* numpy
* matplotlib
###### Only screener.py
* bs4

### ToDo
* Add funcationality to perform update on stock splits and dividend fixed files rather than recomputing them on every run
