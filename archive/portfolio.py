import pandas as pd
import numpy as np
# from scipy.linalg import sqrtm
# from scipy.stats import norm
# from julia import Main as M
import cvxpy as cp
import yfinance as yf
from matplotlib import pyplot as plt
from datetime import datetime
from os.path import exists
import json
import re
from os import makedirs
from typing import List

DIRS = ['data','data/cache']

class MarketData():

    def __init__(self, date_format = '%Y-%m-%d') -> None:
        self.date_format = date_format

        # Creates all necessary directories
        for dir in DIRS:
            if not exists(dir):
                makedirs(dir)

    def write_metadata(self, metadata = None):
        
        if metadata is None:
            metadata = self.metadata

        with open("data/cache/metadata.json", "w") as f:
            json.dump(metadata, f, indent = 4)
    
    def load_metadata(self):

        if not exists('data/cache/metadata.json'):
            # # Creates empty metadata file
            # metadata = {"tickers":[],"start_date":None, "end_date":None, "sample_freq":None}
            # self.metadata = metadata
            # self.write_metadata(metadata)
            raise FileNotFoundError('Metadata doesn\'t exist')

        else:
            # Loads existing one
            with open("data/cache/metadata.json", "r") as f:
                self.metadata = json.load(f)
            
            return self.metadata

    def convert_freq(self, conv_to : str, freq = None):
        """
        Convert string frequency to number of days corresponding or to pandas frequency rule.
        freq: optional, string frequency. Format should be "<number><period>".
        If None passed, uses self.sample_freq instead
        conv_to: string, either \"pandas\" or \"days\". 
        Format to convert the yfinance frequency to
        """
        if conv_to not in ["pandas","days"]:
            raise ValueError(f"\"{conv_to}\" Coversion format not supported")

        if freq is None:
            freq = self.sample_freq

        num, period = re.split("(?<=[0-9])(?=[a-z])", freq)

        if period == 'd':
            if conv_to =="days":
                return int(num)
            else:
                return num + "B"

        elif period == 'wk':
            if conv_to =="days":
                return int(num) * 7
            else:
                return num + "W"

        elif period == 'mo':
            if conv_to =="days":
                return int(num) * 30
            else:
                return num + "BM"
     
        elif period == 'y':
            if conv_to =="days":
                return int(num) * 365
            else:
                return num + "BY"
        
        else:
            raise ValueError(f"\"{period}\" frequency not supported")


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
            if datetime.strptime(start_date, self.date_format) < datetime.strptime(self.metadata["start_date"], self.metadata["date_format"]):
                return False
        
        if end_date is not None:
            if datetime.strptime(end_date, self.date_format) > datetime.strptime(self.metadata["end_date"], self.metadata["date_format"]):
                return False
            
        # Check price metric
        if self.prices_metric != self.metadata["prices_metric"]:
            return False
        
        return True

    def download_data(self, tickers, sample_freq = '1d', start_date = None, end_date = None, metric = 'Adj Close', date_format = "%Y-%m-%d"):
        
        main_data = yf.download(tickers, start = start_date, end = end_date, interval='1d')[metric]
        #In case only 1 ticker (although tickers should be > 1 for models to make sense)
        main_data = pd.DataFrame(main_data, index = pd.DatetimeIndex(main_data.index)) 
    
        #Check that sth was downloaded
        if main_data.empty:
            raise ValueError("No data downloaded")
        
        main_data.to_csv('data/cache/Prices.csv')

        if start_date is None:
            start_date = main_data.index[0].strftime(date_format)
        if end_date is None:
            end_date = main_data.index[-1].strftime(date_format)

        # Modify metadata to reflect new parameters
        new_metadata = {"start_date": start_date, "end_date": end_date, "sample_freq": sample_freq,
                        "date_format": date_format, "prices_metric": metric, "tickers":tickers,
                        "log_returns": True, "nan_cutoff": 0}
        self.write_metadata(new_metadata)
        self.metadata = new_metadata

        pd_freq = self.convert_freq("pandas")
        main_data = main_data.resample(pd_freq).last()

        return main_data

    def subselect_data(self, returns, avg_ret, avg_cov, start_date = None, end_date = None, tickers = None):
        """
        Subselects data from processed data
        """
        if tickers is not None:
            # Subselect tickers that have been kept in preprocessing 
            # (some series with a lot of NaN values might have been dropped)
            kept_tickers = [tic for tic in tickers if tic in returns.columns]
            self.tickers = kept_tickers
            returns = returns[kept_tickers]
            avg_ret = avg_ret[kept_tickers]
            avg_cov = avg_cov.loc[(slice(None),kept_tickers),kept_tickers]
        
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

    def preprocess_prices(self, data, tickers = None, sample_freq = '1d', start_date = None, end_date = None, log_returns = False, nan_cutoff = 1, ffill = True, ret_conversion = {}):
        
        # Cut tickers
        if tickers is not None:
            data = data[tickers]
        
        # Resample in case necessary
        resample_freq = self.convert_freq("pandas", sample_freq)
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
        nan_rates = data.isna().mean(axis = 0)
        keep_tickers = nan_rates[nan_rates <= nan_cutoff].index.values

        data = data[keep_tickers]
        data = data.dropna(axis = 0)
        self.tickers = list(data.columns)
        self.metadata["nan_cutoff"] = nan_cutoff

        # Creates returns
        self.returns = pd.DataFrame(np.nan, index = data.index, columns = data.columns)

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
        self.returns = self.returns.dropna(axis = 0)
        # self.tickers = list(self.returns.columns)
        # self.metadata["nan_cutoff"] = nan_cutoff
        # self.metadata["conversion_rules"] = ret_conversion
        # assert self.returns.isna().sum().sum() == 0, "NaN hanging around"

        self.write_metadata()
        self.returns.to_csv('data/cache/Real returns.csv')

        return self.returns
    
    

    def preprocess_returns(self, est_window):
        
        # Create expected returns
        self.avg_ret = self.returns.rolling(est_window).mean()
        self.avg_ret.dropna(axis = 0, inplace = True)
        self.avg_ret.to_csv('data/cache/Expected returns.csv')

            
        # Create expected covariance
        # squares = self.returns.values 

        self.avg_cov = self.returns.rolling(est_window).cov()
        self.avg_cov.dropna(axis = 0, inplace = True)
        self.avg_cov.to_csv('data/cache/Expected covariance.csv')

    def preprocess(self, tickers, est_window, sample_freq = '1d', start_date = None, end_date = None, prices_metric = 'Adj Close', log_returns = True, nan_cutoff = 0.4, ffill = True, conversion_rules = {}, reload = False):
        tickers = sorted(tickers)
        self.tickers = tickers
        self.sample_freq = sample_freq
        self.prices_metric = prices_metric
        self.log_returns = log_returns
        self.est_window = est_window

        # Check if metadata exists - if not, downloads price data (and writes metadata)
        # If reload = True, reloads everything
        if not exists('data/cache/metadata.json') or reload:
            main_data = self.download_data(tickers, sample_freq, start_date, end_date, prices_metric, self.date_format)
            self.preprocess_prices(main_data, tickers, sample_freq, start_date, end_date, log_returns, nan_cutoff, ffill, conversion_rules)
            self.preprocess_returns(est_window)

            
        else:
            self.load_metadata()

            # Check if main data is loadable
            if self.check_load(start_date, end_date):
                
                # If it is, check if processed data is loadable
                if (self.convert_freq("days") == self.convert_freq("days", self.metadata["sample_freq"])
                    and (log_returns == self.metadata["log_returns"])
                    and (nan_cutoff == self.metadata["nan_cutoff"])
                    # and (conversion_rules == self.metadata["conversion_rules"])
                    ):
                    avg_ret = pd.read_csv('data/cache/Expected returns.csv', index_col=0, parse_dates=True)
                    avg_cov = pd.read_csv('data/cache/Expected covariance.csv', index_col=[0,1], parse_dates=True)
                    returns = pd.read_csv('data/cache/Real returns.csv', index_col=0, parse_dates=True)

                    self.returns, self.avg_ret, self.avg_cov = self.subselect_data(returns, avg_ret, avg_cov, start_date, end_date, tickers)
                    
                # Otherwise, opens stored price data and preprocesses it
                else:
                    main_data = pd.read_csv('data/cache/Prices.csv', index_col=0, parse_dates=True)
                    self.preprocess_prices(main_data, tickers, sample_freq, start_date, 
                                           end_date, log_returns, nan_cutoff, ffill, conversion_rules)
                    self.preprocess_returns(est_window)

            else:
                main_data = self.download_data(tickers, sample_freq, start_date, end_date, prices_metric, self.date_format)
                self.preprocess_prices(main_data, tickers, sample_freq, start_date, end_date, log_returns, nan_cutoff, ffill, conversion_rules)
                self.preprocess_returns(est_window)
            
        # Check that dates match, use them on the main returns
        assert all(self.avg_cov.index.get_level_values(0).unique() == self.avg_ret.index), "Dates don't match"
        self.dates = self.avg_ret.index
        self.returns = self.returns.loc[self.dates]
        self.T = len(self.returns.index)
        self.n = len(self.returns.columns)
    
    ### MODELS ###



class Strategy():

    init_pnl = 100
    name = "Strategy"

    def __init__(self, data: MarketData) -> None:

        try:
            # Keep data
            self.returns = data.returns
            self.avg_ret = data.avg_ret
            self.avg_cov = data.avg_cov

            # Keep some metadata
            self.dates = data.dates
            self.tickers = data.tickers
            self.T = data.T
            self.n = data.n
            self.log_returns = data.log_returns
            self.sample_freq = data.sample_freq

            # Load Gurobi 
            # M.eval("using JuMP, Gurobi")

        except AttributeError as err:
            raise ValueError("The data passed is incomplete. MarketData might not have been preprocessed?") from err



        # keep_attrs = ["returns", "avg_ret", "avg_cov", "dates", "tickers", "T", "n", "log_returns"]
        # for attr in self.keep_attrs:
        #     if attr not in data.__dict__:
        #         raise ValueError("Attributes missing in MarketData. MarketData might not have been preprocessed?")
            
        #     self.__setattr__(attr, data.__getattribute__(attr))

    def init_results(self, **_):
        
        empty_results = {
            "optimums":[0],
            "returns": [0],
            "PnLs": [self.init_pnl]
        }
        
        return empty_results
    
    def prep_model(self, t, **_):

        prep_kwargs = {
            "mu": self.avg_ret.loc[self.dates[t],:].values,
            "sigma": self.avg_cov.loc[self.dates[t], :].values, 
        }
    
        return prep_kwargs 

    def run_model(self, **_):
        raise NotImplementedError("This strategy is incomplete (missing run_model method)")

    def update_results(self, prev_res, new_res, real_ret, **_):
        prev_res["optimums"].append(new_res["optimum"])
        prev_res["weights"].append(new_res["weights"])

        returns = new_res["weights"] @ real_ret
        prev_res["returns"].append(returns)
        if self.log_returns:
            prev_res["PnLs"].append(prev_res["PnLs"][-1] * np.exp(returns))
        else:
            prev_res["PnLs"].append(prev_res["PnLs"][-1] * (1 + returns))

        return prev_res

    def run_strategy(self, init_params = {}, verbose = False, **kwargs):
        
        results = self.init_results(**(kwargs | init_params))


        for t in range(self.T - 1):
            
            if t % 10 == 0 and verbose:
                print('Computing step {} of {}'.format(t, self.T))

            model_kwargs = self.prep_model(t, results=results, **kwargs) 
            new_res = self.run_model(**model_kwargs)
            results = self.update_results(results, new_res, self.returns.iloc[t+1, :])
            
        self.results = results

        return results
    
    def plot(self, res_name, res_tic: int | None = None):
        
        if not hasattr(self, "results"):
            raise ValueError("Strategy needs to be ran first!")
        
        to_plot = self.results[res_name]

        if res_tic is not None:
            res_idx = self.tickers.index(res_tic)
            to_plot = [el[res_idx] for el in to_plot]
            ticker = self.tickers[res_idx]
            plt.title(f"{self.name} strategy - {res_name} results (Ticker: {ticker})")

        else:
            plt.title(f"{self.name} strategy - {res_name} results")

        plt.plot(self.dates, to_plot)
    
    def sharpe(self, rf = None, dates = None):

        if not hasattr(self, "results"):
            raise ValueError("Strategy needs to be ran first!")

        if dates is None:
            dates = self.dates
        
        dates_idx = self.dates.get_indexer(dates)
        
        if isinstance(rf, str):
            rf = self.returns[rf].values[1:]
        elif isinstance(rf, (pd.Series)):
            rf = rf.loc[dates][1:]
        elif isinstance(rf, (float, int, np.ndarray)):
            pass
        elif rf is None:
            rf = 0
        else:
            raise TypeError(f"{type(rf)} is not a supported type for rf")
        
        pnls = [self.results["PnLs"][idx] for idx in dates_idx]
        pct_change = pd.Series(pnls).pct_change().dropna()
        res_sharpe = ((pct_change.values - rf).mean() / pct_change.std())
        
        # Annualize sharpe
        freq_per_year = 365 / MarketData.convert_freq(None, "days", self.sample_freq)
        sharpe_annual = res_sharpe * np.sqrt(freq_per_year)

        return sharpe_annual
    

class SparseMVO(Strategy):

    name = "Sparse MVO"

    def init_results(self, sparsity, init_weights = None, init_ind = None, **_):
        
        empty_results = super().init_results()

        if init_weights is not None:
            weights = init_weights
        else:
            weights = np.zeros(self.n)
            weights[0] = 1
                
        if init_ind is not None:
            inds = init_ind
        else:
            # Find the first n weights equal to 0
            zero_weights = np.where(weights == 0)[0]
            if len(zero_weights) < len(weights) - sparsity:
                raise ValueError("Sparsity is too small, or init_weights is not sparse enough")
            
            inds = np.ones(self.n)
            inds[zero_weights] = 0
        
        empty_results.update({
            "weights":[weights],
            "inds":[inds]
        })

        return empty_results
    
    def prep_model(self, t, results, **kwargs):

        return_kwargs = super().prep_model(t)

        return_kwargs.update({
            "max_sigma": kwargs["max_sigma"],
            "sparsity": kwargs["sparsity"],
            "previous_weights": results["weights"][-1],
            "previous_ind": results["inds"][-1]
        })

        return return_kwargs

    def run_model(self, mu, sigma, max_sigma, sparsity, previous_weights = None, previous_ind = None):
        
        # Pass previous weight to speed up optimization 
        # (because current optimum is probably close to previous one)
        if previous_weights is None:
            previous_weights = np.zeros(self.n)
            previous_weights[0] = 1
            previous_ind = np.zeros(self.n)
            previous_ind[0] = 1

        weights = cp.Variable(self.n)
        indicator = cp.Variable(self.n, boolean = True)
        weights.value = previous_weights
        indicator.value = previous_ind

        # Long only constraint
        long_constraint = [weights[i] >= 0 for i in range(self.n)]

        # l1 regularization
        zero_constraint = [np.ones(self.n) @ weights == 1]
        
        # Sparsity constraint
        sparse_constraint = [weights <= indicator, 
                             np.ones(self.n) @ indicator <= sparsity]
        
        # Variance constraint
        variance_constraint = [cp.quad_form(weights, sigma) <= max_sigma]

        prob = cp.Problem(
            cp.Maximize(mu @ weights),
            long_constraint + sparse_constraint + zero_constraint + variance_constraint
    
        )

        prob.solve(solver = cp.GUROBI)#, warm_start = True

        
        return {"optimum":prob.value, "weights":weights.value, "indicators":indicator.value}
    
    def update_results(self, prev_res, new_res, real_ret):
        
        prev_res = super().update_results(prev_res, new_res, real_ret)
        prev_res["inds"].append(new_res["indicators"])
            
        return prev_res

    def run(self, sparsity, max_sigma, init_params = {}, verbose = False):
        return self.run_strategy(init_params, verbose, sparsity=sparsity, max_sigma = max_sigma)

class MVO(Strategy):

    name = "MVO"

    def init_results(self, init_weights = None, **_):
        empty_results = super().init_results()

        if init_weights is not None:
            weights = init_weights
        else:
            weights = np.ones(self.n) / self.n
                
        empty_results.update({
            "weights":[weights]
        })

        return empty_results

    def prep_model(self, t, results, **kwargs):
        return_kwargs = super().prep_model(t)

        return_kwargs.update({
            "max_sigma": kwargs["max_sigma"],
            "previous_weights": results["weights"][-1],
        })

        return return_kwargs

    def run_model(self, mu, sigma, max_sigma, previous_weights = None):
        # Pass previous weight to speed up optimization 
        # (because current optimum is probably close to previous one)
        if previous_weights is None:
            previous_weights = np.ones(self.n) / self.n

        weights = cp.Variable(self.n)
        weights.value = previous_weights

        # Long only constraint
        long_constraint = [weights[i] >= 0 for i in range(self.n)]

        # l1 regularization
        zero_constraint = [np.ones(self.n) @ weights == 1]
        
        # Variance constraint
        variance_constraint = [cp.quad_form(weights, sigma) <= max_sigma]

        prob = cp.Problem(
            cp.Maximize(mu @ weights),
            long_constraint + zero_constraint + variance_constraint
    
        )

        prob.solve(solver = cp.GUROBI)#, warm_start = True

        
        return {"optimum":prob.value, "weights":weights.value}

    def update_results(self, prev_res, new_res, real_ret):
        # Unnecessary function (defined here for completeness)
        prev_res = super().update_results(prev_res, new_res, real_ret)
            
        return prev_res
    
    def run(self, max_sigma, init_params = {}, verbose = False):
        return self.run_strategy(init_params, verbose, max_sigma = max_sigma)



class EqWeight(Strategy):
    
    name = "Equal-Weighted"

    def __init__(self, data: MarketData) -> None:
        super().__init__(data)
        
        self.const_weights = np.ones(self.n) / self.n
    
    def init_results(self, **_):

        init_kwargs =  super().init_results(**_)
        init_kwargs["weights"] = [self.const_weights]

        return init_kwargs

    def run_model(self, mu, **_):
        
        opt = self.const_weights @ mu
        return {"optimum":opt, "weights":self.const_weights}
    
    def run(self, verbose = False):
        return self.run_strategy(verbose = verbose)

class SingleSec(Strategy):

    # name = "Single Security"

    def __init__(self, data: MarketData, sec_ticker) -> None:
        super().__init__(data)
        
        self.single_sec = sec_ticker
        sec_idx = self.tickers.index(sec_ticker)

        self.sec_weights = np.zeros(self.n)
        self.sec_weights[sec_idx] = 1

        self.name = f"Single Security ({sec_ticker})"
    
    def init_results(self, **_):
        
        init_kwargs =  super().init_results(**_)
        init_kwargs["weights"] = [self.sec_weights]

        return init_kwargs

    def run_model(self, mu, **_):
        
        opt = self.sec_weights @ mu
        return {"optimum":opt, "weights":self.sec_weights}
    
    def run(self, verbose = False):
        return self.run_strategy({}, verbose)
    


def plot_results(strategies: List[Strategy], res_name, res_tic: str | None = None):

    
    if len(strategies) == 0:
        raise ValueError("plot_results needs at least one strategy")
    

    for strat in strategies[1:]:
        if not strat.dates.equals(strategies[0].dates):
            raise ValueError("All strategies must be ran on the same dates")
        elif res_tic is not None:
            if strat.tickers != strategies[0].tickers:
                raise ValueError("All strategies must be ran on the same tickers is res_idx is specifies")        

    if res_tic is not None:

        plt.title(f"{res_name} results (Ticker: {res_tic})")
    else:
        plt.title(f"{res_name} results")

    for strat in strategies:
        if not hasattr(strat, "results"):
            raise ValueError(f"{strat} Strategy needs to be ran first!")

        to_plot = strat.results[res_name]

        if res_tic is not None:
            res_idx = strategies[0].tickers.index(res_tic)
            to_plot = [el[res_idx] for el in to_plot]
            
        plt.plot(strat.dates, to_plot, label = strat.name)
    
    plt.legend()

def plot_sharpe(strategies: List[Strategy], window, rf = None):


    if len(strategies) == 0:
        raise ValueError("plot_results needs at least one strategy")
    

    for strat in strategies[1:]:
        if not strat.dates.equals(strategies[0].dates):
            raise ValueError("All strategies must be ran on the same dates")
        
    
    if isinstance(rf, str):
        rf = strategies[0].returns[rf].values[1:]
    elif isinstance(rf, (pd.Series)):
        rf = rf.loc[1:]
    elif isinstance(rf, (float, int, np.ndarray)):
        pass
    elif rf is None:
        rf = 0
    else:
        raise TypeError(f"{type(rf)} is not a supported type for rf")
        
    all_sharpes = {}

    for strat in strategies:

        if not hasattr(strat, "results"):
            raise ValueError(f"{strat} Strategy needs to be ran first!")
              
        pct_change = pd.Series(strat.results["PnLs"]).pct_change().dropna()
        excess_change = pct_change - rf
        res_sharpe = pct_change.rolling(window).apply(lambda s: (excess_change).mean() / s.std()).dropna()
        
        # Annualize sharpe
        freq_per_year = 365 / MarketData.convert_freq(None, "days", strat.sample_freq)
        sharpe_annual = res_sharpe * np.sqrt(freq_per_year)

        all_sharpes[strat.name] = sharpe_annual
    
        plt.plot(strat.dates[window:], sharpe_annual, label = strat.name)
    
    plt.legend()
    plt.title(f"{window}-rolling Sharpe")

    return all_sharpes
        

        
        
    


if __name__ == "__main__":
    pass
