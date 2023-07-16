import pandas as pd
import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from .market_data import MarketData

"""
Module containing Strategy class and subclasses.

Strategies are used to compare different portfolio optimization methods.
To create a strategy, create a subclass of Strategy and implement the run_model and run methods.
"""


class Strategy():
    """
    Base class for strategies. Create a subclass and implement the run_model and run methods
    to make a custom strategy.
    """

    init_pnl = 100
    name = "Strategy"

    def __init__(self, data: MarketData) -> None:
        """
        Initializes a strategy with the given data.

        Parameters:
        -----------
        data: MarketData, data to use for the strategy
        """

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
        """
        Default function to initialize the results dict. Can be overriden in subclasses.
        """
        
        empty_results = {
            "optimums":[0],
            "returns": [0],
            "PnLs": [self.init_pnl]
        }
        
        return empty_results
    
    def prep_model(self, t, **_):
        """
        Default function to prepare the model parameters at each iteration.
        Can be overriden in subclasses.
        """

        prep_kwargs = {
            "mu": self.avg_ret.loc[self.dates[t],:].values,
            "sigma": self.avg_cov.loc[self.dates[t], :].values, 
        }
    
        return prep_kwargs 

    def run_model(self, **_):
        """
        Model needs to be implemented in subclasses.
        """
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
    """
    Sparse Meav-Variance optimization strategy. This is equivalent to the regular MVO strategy
    with the added constraint that the number of non-zero weights is limited to the sparsity parameter.
    """

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
            "long_only": self.long_only,
            "previous_weights": results["weights"][-1],
            "previous_ind": results["inds"][-1]
        })

        return return_kwargs

    def run_model(self, mu, sigma, max_sigma, sparsity, long_only = True, previous_weights = None, previous_ind = None):
        
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
        indicator.value = previous_ind.round()

        if long_only:
            # Long only constraint
            long_constraint = [weights[i] >= 0 for i in range(self.n)]
        else:
            long_constraint =  [weights[i] <= 1 for i in range(self.n)]
            long_constraint += [weights[i] >= -1 for i in range(self.n)]
            # Sparsity constraint
            sparse_constraint = [weights >= -indicator, 
                             np.ones(self.n) @ indicator <= sparsity]

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

    def run(self, sparsity, max_sigma, long_only = True, init_params = {}, verbose = False):
        self.long_only = long_only
        return self.run_strategy(init_params, verbose, sparsity=sparsity, max_sigma = max_sigma)

class MVO(Strategy):
    """
    Classic Meav-Variance optimization strategy, with hard constraint on the variance.
    """

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
            "long_only": self.long_only
        })

        return return_kwargs

    def run_model(self, mu, sigma, max_sigma, previous_weights = None, long_only = True):
        # Pass previous weight to speed up optimization 
        # (because current optimum is probably close to previous one)
        if previous_weights is None:
            previous_weights = np.ones(self.n) / self.n

        weights = cp.Variable(self.n)
        weights.value = previous_weights

        if long_only:
            # Long only constraint
            long_constraint = [weights[i] >= 0 for i in range(self.n)]
        else:
            # Max +- 100% weight constraint
            long_constraint =  [weights[i] <= 1 for i in range(self.n)]
            long_constraint += [weights[i] >= -1 for i in range(self.n)]

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
    
    def run(self, max_sigma, long_only = True, init_params = {}, verbose = False):

        self.long_only = long_only

        return self.run_strategy(init_params, verbose, max_sigma = max_sigma)

class PenaltyMVO(Strategy):
    """
    Classic Mean-Variance optimization strategy, with penalty on the variance 
    (instead of a hard bound on the variance).
    """

    name = "Penalty MVO"

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
            "penalty": kwargs["penalty"],
            "previous_weights": results["weights"][-1],
        })

        return return_kwargs

    def run_model(self, mu, sigma, penalty, previous_weights = None):
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
        

        prob = cp.Problem(
            cp.Maximize(mu @ weights - penalty * cp.quad_form(weights, sigma)),
            long_constraint + zero_constraint
    
        )

        prob.solve(solver = cp.GUROBI)#, warm_start = True

        
        return {"optimum":prob.value, "weights":weights.value}

    def update_results(self, prev_res, new_res, real_ret):
        # Unnecessary function (defined here for completeness)
        prev_res = super().update_results(prev_res, new_res, real_ret)
            
        return prev_res
    
    def run(self, penalty, init_params = {}, verbose = False):
        return self.run_strategy(init_params, verbose, penalty = penalty)


class PenaltySparseMVO(Strategy):
    """
    Sparse Meav-Variance optimization strategy. This is equivalent to the penalty MVO strategy
    with the added constraint that the number of non-zero weights is limited to the sparsity parameter.
    """


    name = "Sparse Penalty MVO"

    
    def prep_model(self, t, results, **kwargs):

        return_kwargs = super().prep_model(t, results)

        return_kwargs.update({
            "penalty": kwargs["penalty"],
            "sparsity": kwargs["sparsity"],
            "previous_weights": results["weights"][-1],
            "previous_ind": results["inds"][-1]
        })

        return return_kwargs

    def run_model(self, mu, sigma, penalty, sparsity, previous_weights = None, previous_ind = None):
        
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

        prob = cp.Problem(
            cp.Maximize(mu @ weights - penalty * cp.quad_form(weights, sigma)),
            long_constraint + sparse_constraint + zero_constraint
    
        )

        prob.solve(solver = cp.GUROBI)#, warm_start = True

        
        return {"optimum":prob.value, "weights":weights.value, "indicators":indicator.value}
    
    def update_results(self, prev_res, new_res, real_ret):
        
        prev_res = super().update_results(prev_res, new_res, real_ret)
        prev_res["inds"].append(new_res["indicators"])
            
        return prev_res

    def run(self, sparsity, penalty, init_params = {}, verbose = False):
        return self.run_strategy(init_params, verbose, sparsity=sparsity, penalty = penalty) 


class EqWeight(Strategy):
    """
    Equal weighted strategy. All weights are equal to 1/n.
    """

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
    """
    All weights are 0 except for one, which is 1.
    """


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
    