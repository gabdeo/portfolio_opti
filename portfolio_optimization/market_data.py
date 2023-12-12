from datetime import datetime
from os import makedirs
from os.path import exists
import json
import re
import time
import pandas as pd
import numpy as np
import yfinance as yf


"""
Module containing the MarketData class, which loads and preprocesses
market data from Yahoo Finance, or from cached data. MarketData
objects can be passed to Strategy objects to run strategies on 
processed data.
"""

DIRS = ["data", "data/cache"]


class MarketData:
    """
    Market data loader. Loads data using the yfinance library. Keeps loaded
    data in a cache, used in future loadings to avoid re-downloading data
    or repeating mean and variance estimations when possible. 
    """


    def __init__(self, date_format="%Y-%m-%d") -> None:
        """
        Creates necessary directories.
        date_format: datetime format to use when writing/reading dates
        """

        self.date_format = date_format

        # Creates all necessary directories
        for _dir in DIRS:
            if not exists(_dir):
                makedirs(_dir)
        
        # Defines all attributes
        self.metadata = {}
        self.tickers = []
        self.returns = pd.DataFrame()
        self.avg_ret = pd.DataFrame()
        self.avg_cov = pd.DataFrame()
        self.sample_freq = "1d"
        self.prices_metric = "Adj Close"
        self.log_returns = False
        self.est_window = 1
        self.dates = []
        self.T = 0
        self.n = 0

    def _write_metadata(self, metadata=None):
        """
        Writes metadata to keep track of cached data
        metadata: dict, dictionary to write as a json file. 
        If None, use metadata attribute instead
        """
        if metadata is None:
            metadata = self.metadata

        with open("data/cache/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def _load_metadata(self):
        """
        Loads existing metadata file from cache 
        """
        if not exists("data/cache/metadata.json"):
            # # Creates empty metadata file
            # metadata = {"tickers":[],"start_date":None, "end_date":None, "sample_freq":None}
            # self.metadata = metadata
            # self._write_metadata(metadata)
            raise FileNotFoundError("Metadata doesn't exist")

        else:
            # Loads existing one
            with open("data/cache/metadata.json", "r") as f:
                self.metadata = json.load(f)

            return self.metadata

    def _convert_freq(self, conv_to: str, freq=None):
        """
        Convert string frequency to number of days corresponding or to pandas frequency rule.
        freq: optional, string frequency. Format should be "<number><period>".
        If None passed, uses self.sample_freq instead
        conv_to: string, either \"pandas\" or \"days\".
        Format to convert the yfinance frequency to
        """
        if conv_to not in ["pandas", "days"]:
            raise ValueError(f'"{conv_to}" Coversion format not supported')

        if freq is None:
            freq = self.sample_freq

        num, period = re.split("(?<=[0-9])(?=[a-z])", freq)

        if period == "d":
            if conv_to == "days":
                return int(num)
            else:
                return num + "B"

        elif period == "wk":
            if conv_to == "days":
                return int(num) * 7
            else:
                return num + "W"

        elif period == "mo":
            if conv_to == "days":
                return int(num) * 30
            else:
                return num + "BM"

        elif period == "y":
            if conv_to == "days":
                return int(num) * 365
            else:
                return num + "BY"

        else:
            raise ValueError(f'"{period}" frequency not supported')

    def check_load(self, start_date, end_date):
        """
        Checks if data should be reloaded, using metadata info.
        Specifically, checks if:
        - self.tickers is a subset of data tickers
        - self.sample_freq is the same as the data sampling frequency
        - start_date and end_date are contained in data date range

        start_date: desired data starting date
        end_date: desired data ending date
        """

        # Check tickers
        if any([tic not in self.metadata["tickers"] for tic in self.tickers]):
            # If any ticker is not in stored tickers, can't load from data
            return False

        # Check dates
        if start_date is not None:
            if datetime.strptime(start_date, self.date_format) < datetime.strptime(
                self.metadata["start_date"], self.metadata["date_format"]
            ):
                return False

        if end_date is not None:
            if datetime.strptime(end_date, self.date_format) > datetime.strptime(
                self.metadata["end_date"], self.metadata["date_format"]
            ):
                return False

        # Check price metric
        if self.prices_metric != self.metadata["prices_metric"]:
            return False

        return True

    def _download_data(
        self,
        tickers,
        sample_freq="1d",
        start_date=None,
        end_date=None,
        metric="Adj Close",
        date_format="%Y-%m-%d",
    ):
        """
        Downloads new market data using the yfinance module.

        tickers: list, tickers to download
        sample_freq: sample_frequency to use for metadata
        start_date: start_date to download data from. if None, downloads from the earliest possible
        end_date: end_date to download data until. if None, downloads until the latest possible
        metric: metric to use from the yfinance downloaded data (Open, High, Low, Close, Adj Close)
        date_format: date_format to use for metadata
        """
        main_data = yf.download(tickers, start=start_date, end=end_date, interval="1d")[
            metric
        ]
        # In case only 1 ticker (although tickers should be > 1 for models to make sense)
        main_data = pd.DataFrame(main_data, index=pd.DatetimeIndex(main_data.index))

        # Check that sth was downloaded
        if main_data.empty:
            raise ValueError("No data downloaded")

        main_data.to_csv("data/cache/Prices.csv")

        if start_date is None:
            start_date = main_data.index[0].strftime(date_format)
        if end_date is None:
            end_date = main_data.index[-1].strftime(date_format)

        # Modify metadata to reflect new parameters
        new_metadata = {
            "start_date": start_date,
            "end_date": end_date,
            "sample_freq": sample_freq,
            "date_format": date_format,
            "prices_metric": metric,
            "tickers": tickers,
            "log_returns": True,
            "nan_cutoff": 0,
        }
        self._write_metadata(new_metadata)
        self.metadata = new_metadata

        pd_freq = self._convert_freq("pandas")
        main_data = main_data.resample(pd_freq).last()

        return main_data

    def _subselect_data(
        self, returns, avg_ret, avg_cov, start_date=None, end_date=None, tickers=None
    ):
        """
        Subselects data from processed data

        returns: returns DataFrame to use
        avg_ret: estimated returns DataFrame to use
        avg_cov: estimated covariance DataFrame to use
        start_date: start_date to download data from. if None, downloads from the earliest possible
        end_date: end_date to download data until. if None, downloads until the latest possible
        tickers: tickers to use
        """
        
        if tickers is not None:
            # Subselect tickers that have been kept in preprocessing
            # (some series with a lot of NaN values might have been dropped)
            kept_tickers = [tic for tic in tickers if tic in returns.columns]
            self.tickers = kept_tickers
            returns = returns[kept_tickers]
            avg_ret = avg_ret[kept_tickers]
            avg_cov = avg_cov.loc[(slice(None), kept_tickers), kept_tickers]

        if start_date is not None:
            datetime_start = datetime.strptime(start_date, self.date_format)
            returns = returns.loc[datetime_start:]
            avg_ret = avg_ret.loc[datetime_start:]
            avg_cov = avg_cov.loc[datetime_start:]

        if end_date is not None:
            datetime_end = datetime.strptime(end_date, self.date_format)
            returns = returns.loc[:datetime_end]
            avg_ret = avg_ret.loc[:datetime_end]
            avg_cov = avg_cov.loc[:datetime_end]

        return returns, avg_ret, avg_cov

    def _preprocess_prices(
        self,
        data,
        tickers=None,
        sample_freq="1d",
        start_date=None,
        end_date=None,
        log_returns=False,
        nan_cutoff=1,
        ffill=True,
        ret_conversion={},
    ):
        # Cut tickers
        if tickers is not None:
            data = data[tickers]

        # Resample in case necessary
        resample_freq = self._convert_freq("pandas", sample_freq)
        data = data.resample(resample_freq).last()
        self.metadata["sample_freq"] = sample_freq

        # Cut start and end dates
        if start_date is not None:
            data = data[data.index >= datetime.strptime(start_date, self.date_format)]
        if end_date is not None:
            data = data[data.index <= datetime.strptime(end_date, self.date_format)]

        if ffill:
            data = data.ffill()

        # Clear nans
        nan_rates = data.isna().mean(axis=0)
        keep_tickers = nan_rates[nan_rates <= nan_cutoff].index.values

        data = data[keep_tickers]
        data = data.dropna(axis=0)
        self.tickers = list(data.columns)
        self.metadata["nan_cutoff"] = nan_cutoff

        # Creates returns
        self.returns = pd.DataFrame(np.nan, index=data.index, columns=data.columns)

        # Use normal returns conversion for tickers which don't need special conversion
        normal_conv = [tic for tic in data.columns if tic not in ret_conversion.keys()]
        if log_returns:
            self.returns[normal_conv] = np.log(data[normal_conv]).diff()
        else:
            self.returns[normal_conv] = data[normal_conv].pct_change()

        # Use given conversion on others
        for conv in ret_conversion.keys():
            vec_conv = np.vectorize(ret_conversion[conv])
            self.returns[list(conv)] = vec_conv(data[list(conv)])

        # if log_returns:
        #     self.returns = np.log(data).diff()
        # else:
        #     self.returns = data.pct_change()
        self.metadata["log_returns"] = log_returns

        # Applies forward fill (recommended)

        # Clear nans
        # nan_rates = self.returns.isna().mean(axis = 0)
        # keep_tickers = nan_rates[nan_rates <= nan_cutoff].index.values

        # self.returns = self.returns[keep_tickers]
        self.returns = self.returns.dropna(axis=0)
        # self.tickers = list(self.returns.columns)
        # self.metadata["nan_cutoff"] = nan_cutoff
        # self.metadata["conversion_rules"] = ret_conversion
        # assert self.returns.isna().sum().sum() == 0, "NaN hanging around"

        self._write_metadata()
        self.returns.to_csv("data/cache/Real returns.csv")

        return self.returns

    def _preprocess_returns(self, est_window):
        # Create expected returns
        self.avg_ret = self.returns.rolling(est_window).mean()
        self.avg_ret.dropna(axis=0, inplace=True)
        self.avg_ret.to_csv("data/cache/Expected returns.csv")

        # Create expected covariance
        mu_squares = np.einsum("ij,ik -> ijk", self.avg_ret, self.avg_ret)
        ret_squares = np.einsum("ij,ik -> ijk", self.returns, self.returns)

        start_time = time.time()
        ret_squares_avg = np.ma.average(ret_squares, axis=0)
        avg_cov = ret_squares_avg - mu_squares

        cov_index = pd.MultiIndex.from_product(
            [self.avg_ret.index, self.avg_ret.columns], names=["Dates", "Tickers"]
        )
        cov_values = avg_cov.reshape(
            (self.avg_ret.index * self.avg_ret.columns, self.avg_ret.columns)
        )
        avg_cov_df = pd.DataFrame(
            cov_values, index=cov_index, columns=self.avg_ret.columns
        )
        print(time.time() - start_time)

        start_time = time.time()
        self.avg_cov = self.returns.rolling(est_window).cov()
        print(time.time() - start_time)

        self.avg_cov.dropna(axis=0, inplace=True)

        self.avg_cov.to_csv("data/cache/Expected covariance.csv")

    def preprocess(
        self,
        tickers,
        est_window,
        sample_freq="1d",
        start_date=None,
        end_date=None,
        prices_metric="Adj Close",
        log_returns=True,
        nan_cutoff=0.4,
        ffill=True,
        conversion_rules={},
        reload=False,
    ):
        tickers = sorted(tickers)
        self.tickers = tickers
        self.sample_freq = sample_freq
        self.prices_metric = prices_metric
        self.log_returns = log_returns
        self.est_window = est_window

        # Check if metadata exists - if not, downloads price data (and writes metadata)
        # If reload = True, reloads everything
        if not exists("data/cache/metadata.json") or reload:
            main_data = self._download_data(
                tickers,
                sample_freq,
                start_date,
                end_date,
                prices_metric,
                self.date_format,
            )
            self._preprocess_prices(
                main_data,
                tickers,
                sample_freq,
                start_date,
                end_date,
                log_returns,
                nan_cutoff,
                ffill,
                conversion_rules,
            )
            self._preprocess_returns(est_window)

        else:
            self._load_metadata()

            # Check if main data is loadable
            if self.check_load(start_date, end_date):
                # If it is, check if processed data is loadable
                if (
                    self._convert_freq("days")
                    == self._convert_freq("days", self.metadata["sample_freq"])
                    and (log_returns == self.metadata["log_returns"])
                    and (nan_cutoff == self.metadata["nan_cutoff"])
                    # and (conversion_rules == self.metadata["conversion_rules"])
                ):
                    avg_ret = pd.read_csv(
                        "data/cache/Expected returns.csv", index_col=0, parse_dates=True
                    )
                    avg_cov = pd.read_csv(
                        "data/cache/Expected covariance.csv",
                        index_col=[0, 1],
                        parse_dates=True,
                    )
                    returns = pd.read_csv(
                        "data/cache/Real returns.csv", index_col=0, parse_dates=True
                    )

                    self.returns, self.avg_ret, self.avg_cov = self._subselect_data(
                        returns, avg_ret, avg_cov, start_date, end_date, tickers
                    )

                # Otherwise, opens stored price data and preprocesses it
                else:
                    main_data = pd.read_csv(
                        "data/cache/Prices.csv", index_col=0, parse_dates=True
                    )
                    self._preprocess_prices(
                        main_data,
                        tickers,
                        sample_freq,
                        start_date,
                        end_date,
                        log_returns,
                        nan_cutoff,
                        ffill,
                        conversion_rules,
                    )
                    self._preprocess_returns(est_window)

            else:
                main_data = self._download_data(
                    tickers,
                    sample_freq,
                    start_date,
                    end_date,
                    prices_metric,
                    self.date_format,
                )
                self._preprocess_prices(
                    main_data,
                    tickers,
                    sample_freq,
                    start_date,
                    end_date,
                    log_returns,
                    nan_cutoff,
                    ffill,
                    conversion_rules,
                )
                self._preprocess_returns(est_window)

        # Check that dates match, use them on the main returns
        assert all(
            self.avg_cov.index.get_level_values(0).unique() == self.avg_ret.index
        ), "Dates don't match"
        self.dates = self.avg_ret.index
        self.returns = self.returns.loc[self.dates]
        self.T = len(self.returns.index)
        self.n = len(self.returns.columns)
