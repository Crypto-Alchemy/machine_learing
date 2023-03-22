import yfinance as yf


def addMarketIndicators(df):
    # Add S&P 500 index as market indicator
    sp500 = yf.Ticker("^GSPC").history(
        period='1d', start=df.index.min().date(), end=df.index.max().date())
    df['sp500_close'] = sp500['Close']
    return df
