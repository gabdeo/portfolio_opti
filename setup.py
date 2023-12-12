from setuptools import setup, find_packages

setup(
    name="portfolio_optimization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["cvxpy","matplotlib","numpy","pandas","yfinance"],
)
