import pandas as pd
import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from typing import List
from .market_data import MarketData
from .strategy import Strategy

"""
Module containing comparator functions for strategies
"""


def plot_results(strategies: List[Strategy], res_name, res_tic: str | None = None):
    """
    Plots an output (PnL, weights, etc.) of several strategies on the same graph

    Parameters:
    -----------
    strategies: List[Strategy], list of strategies to plot
    res_name: str, name of the result to plot
    res_tic: str, ticker of the asset to plot (if None, plots all assets)

    Returns:
    --------
    None, plots the results on the current figure
    """
    
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
    """
    
    Plots the rolling sharpe ratio of several strategies on the same graph
    
    
    Parameters:
    -----------
    strategies: List[Strategy], list of strategies to plot
    window: int, window of the rolling sharpe
    rf: float | str | pd.Series | np.ndarray, risk-free rate to use for the sharpe ratio

    Returns:
    --------
    None, plots the results on the current figure
    """

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