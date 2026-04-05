import pandas as pd
import yfinance as yf

def load_data(ticker='AAPL', start='2015-01-01', end='2026-01-01', save_path='data/aapl.csv'):
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(save_path)
    return data

def load_csv(path='data/aapl.csv'):
    return pd.read_csv(path, index_col=0, parse_dates=True)