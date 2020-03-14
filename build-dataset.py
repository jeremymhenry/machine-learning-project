import os
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

START_DATE = "2003-08-01"
END_DATE = "2015-01-01"

# Read in the dataset for all stock prices
def build_stock_dataset(start=START_DATE, end=END_DATE):

    statspath = "intraQuarter/_KeyStats/"
    ticker_list = os.listdir(statspath)

    # Required on macOS
    if ".DS_Store" in ticker_list:
        os.remove(f"{statspath}/.DS_Store")
        ticker_list.remove(".DS_Store")

    # Get Adjusted Close prices
    all_data = pdr.get_data_yahoo(ticker_list, start, end)
    stock_data = all_data["Adj Close"]

    # Remove empty columns
    stock_data.dropna(how="all", axis=1, inplace=True)
    missing_tickers = [
        ticker for ticker in ticker_list if ticker.upper() not in stock_data.columns
    ]

    stock_data.ffill(inplace=True)
    stock_data.to_csv("stock_prices.csv")


# Read in S&P500 Index
def build_sp500_dataset(start=START_DATE, end=END_DATE):

    index_data = pdr.get_data_yahoo("SPY", start=START_DATE, end=END_DATE)
    index_data.to_csv("sp500_index.csv")


if __name__ == "__main__":
    build_stock_dataset()
