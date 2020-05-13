import os
import re
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from tqdm import tqdm

from com.ucdenver.ml.utils.utils import string_to_float

yf.pdr_override()


class HistoricalDataset:
    # Read in the dataset for all stock prices
    START_DATE = "2003-08-01"
    END_DATE = "2015-01-01"

    statspath = "resources/intraQuarter/_KeyStats/"

    # The list of features to parse from the html files
    features = [  # Valuation measures
        "Market Cap",
        "Enterprise Value",
        "Trailing P/E",
        "Forward P/E",
        "PEG Ratio",
        "Price/Sales",
        "Price/Book",
        "Enterprise Value/Revenue",
        "Enterprise Value/EBITDA",
        #  Financial highlights
        "Profit Margin",
        "Operating Margin",
        "Return on Assets",
        "Return on Equity",
        "Revenue",
        "Revenue Per Share",
        "Qtrly Revenue Growth",
        "Gross Profit",
        "EBITDA",
        "Net Income Avl to Common",
        "Diluted EPS",
        "Qtrly Earnings Growth",
        "Total Cash",
        "Total Cash Per Share",
        "Total Debt",
        "Total Debt/Equity",
        "Current Ratio",
        "Book Value Per Share",
        "Operating Cash Flow",
        "Levered Free Cash Flow",
        # Trading information
        "Beta",
        "50-Day Moving Average",
        "200-Day Moving Average",
        "Avg Vol (3 month)",
        "Shares Outstanding",
        "Float",
        "% Held by Insiders",
        "% Held by Institutions",
        "Shares Short (as of",
        "Short Ratio",
        "Short % of Float",
        "Shares Short (prior month",
    ]

    def __init__(self, START_DATE, END_DATE):
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE
        self.build_stock_dataset(start=self.START_DATE, end=self.END_DATE)
        self.build_sp500_dataset(start=self.START_DATE, end=self.END_DATE)

    def __init__(self):
        # self.build_stock_dataset()
        # self.build_sp500_dataset()
        pass

    def build_stock_dataset(self, start=START_DATE, end=END_DATE):
        ticker_list = os.listdir(self.statspath)

        # Required on macOS
        if ".DS_Store" in ticker_list:
            os.remove(f"{self.statspath}/.DS_Store")
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
        stock_data.to_csv("resources/stock_prices.csv")

    # Read in S&P500 Index
    def build_sp500_dataset(self, start=START_DATE, end=END_DATE):
        index_data = pdr.get_data_yahoo("SPY", start=start, end=end)
        index_data.to_csv("resources/sp500_index.csv")

    def preprocess_price_data(self):
        """
        Currently, the sp500 and stock price resources we downloaded do not have any data for
        days when the market was closed (weekends and public holidays). We need to amend this so that
        all rows are included. Doing this now saves a lot of effort when we actually create the
        keystats dataset, which requires that we have stock data every day.
        :return: SP500 and stock dataframes, with no missing rows.
        """
        # Read in SP500 data and stock data, parsing the dates.
        sp500_raw_data = pd.read_csv("resources/sp500_index.csv", index_col="Date", parse_dates=True)
        stock_raw_data = pd.read_csv("resources/stock_prices.csv", index_col="Date", parse_dates=True)

        # We will reindex to include the weekends.
        start_date = str(stock_raw_data.index[0])
        end_date = str(stock_raw_data.index[-1])
        idx = pd.date_range(start_date, end_date)
        sp500_raw_data = sp500_raw_data.reindex(idx)
        stock_raw_data = stock_raw_data.reindex(idx)

        # Now the weekends are NaN, so we fill forward these NaNs
        # (i.e weekends take the value of Friday's adjusted close).
        sp500_raw_data.ffill(inplace=True)
        stock_raw_data.ffill(inplace=True)

        return sp500_raw_data, stock_raw_data

    def parse_keystats(self, sp500_df, stock_df):
        """
        We have downloaded a large number of html files, which are snapshots of a ticker at different times,
        containing the fundamental data (our features). To extract the key statistics, we use regex.
        For supervised machine learning, we also need the data that will form our dependent variable,
        the performance of the stock compared to the SP500.
        :sp500_df: dataframe containing SP500 prices
        :stock_df: dataframe containing stock prices
        :return: a dataframe of training data (i.e features and the components of our dependent variable)
        """
        # The tickers whose data is to be parsed.
        stock_list = [x[0] for x in os.walk(self.statspath)]
        stock_list = stock_list[1:]

        # Creating a new dataframe which we will later fill.
        df_columns = [
                         "Date",
                         "Unix",
                         "Ticker",
                         "Price",
                         "stock_p_change",
                         "SP500",
                         "SP500_p_change",
                     ] + self.features

        df = pd.DataFrame(columns=df_columns)

        # tqdm is a simple progress bar
        for stock_directory in tqdm(stock_list, desc="Parsing progress:", unit="tickers"):
            keystats_html_files = os.listdir(stock_directory)

            # Snippet to get rid of the .DS_Store file in macOS
            if ".DS_Store" in keystats_html_files:
                keystats_html_files.remove(".DS_Store")

            ticker = stock_directory.split(self.statspath)[1]

            for file in keystats_html_files:
                # Convert the datetime format of our file to unix time
                date_stamp = datetime.strptime(file, "%Y%m%d%H%M%S.html")
                unix_time = time.mktime(date_stamp.timetuple())

                # Read in the html file as a string.
                full_file_path = stock_directory + "/" + file

                # This will store the parsed values
                value_list = []

                with open(full_file_path, "r") as source:
                    source = source.read()
                    # Remove commas from the html to make parsing easier.
                    source = source.replace(",", "")

                    # Regex search for the different variables in the html file, then append to value_list
                    for variable in self.features:
                        # Search for the table entry adjacent to the variable name.
                        try:
                            regex = (
                                    r">"
                                    + re.escape(variable)
                                    + r".*?(\-?\d+\.*\d*K?M?B?|N/A[\\n|\s]*|>0|NaN)%?"
                                      r"(</td>|</span>)"
                            )
                            value = re.search(regex, source, flags=re.DOTALL).group(1)

                            # Dealing with number formatting
                            value_list.append(data_string_to_float(value))

                        # The data may not be present. Process accordingly
                        except AttributeError:
                            # In the past, 'Avg Vol' was instead named 'Average Volume'
                            # If 'Avg Vol' fails, search for 'Average Volume'.
                            if variable == "Avg Vol (3 month)":
                                try:
                                    new_variable = ">Average Volume (3 month)"
                                    regex = (
                                            re.escape(new_variable)
                                            + r".*?(\-?\d+\.*\d*K?M?B?|N/A[\\n|\s]*|>0)%?"
                                              r"(</td>|</span>)"
                                    )
                                    value = re.search(regex, source, flags=re.DOTALL).group(
                                        1
                                    )
                                    value_list.append(data_string_to_float(value))
                                except AttributeError:
                                    value_list.append("N/A")
                            else:
                                value_list.append("N/A")

                # We need the stock price and SP500 price now and one year from now.
                # Convert from unix time to YYYY-MM-DD, so we can look for the price in the dataframe
                # then calculate the percentage change.
                current_date = datetime.fromtimestamp(unix_time).strftime("%Y-%m-%d")
                one_year_later = datetime.fromtimestamp(unix_time + 31536000).strftime(
                    "%Y-%m-%d"
                )

                # SP500 prices now and one year later, and the percentage change
                sp500_price = float(sp500_df.loc[current_date, "Adj Close"])
                sp500_1y_price = float(sp500_df.loc[one_year_later, "Adj Close"])
                sp500_p_change = round(
                    ((sp500_1y_price - sp500_price) / sp500_price * 100), 2
                )

                # Stock prices now and one year later. We need a try/except because some data is missing
                stock_price, stock_1y_price = "N/A", "N/A"
                try:
                    stock_price = float(stock_df.loc[current_date, ticker.upper()])
                    stock_1y_price = float(stock_df.loc[one_year_later, ticker.upper()])
                except KeyError:
                    # If stock data is missing, we must skip this datapoint
                    # print(f"PRICE RETRIEVAL ERROR for {ticker}")
                    continue

                stock_p_change = round(
                    ((stock_1y_price - stock_price) / stock_price * 100), 2
                )

                # Append all our data to the dataframe.
                new_df_row = [
                                 date_stamp,
                                 unix_time,
                                 ticker,
                                 stock_price,
                                 stock_p_change,
                                 sp500_price,
                                 sp500_p_change,
                             ] + value_list

                df = df.append(dict(zip(df_columns, new_df_row)), ignore_index=True)

        # Remove rows with missing stock price data
        df.dropna(axis=0, subset=["Price", "stock_p_change"], inplace=True)
        # Output the CSV
        df.to_csv("resources/keystats.csv", index=False)