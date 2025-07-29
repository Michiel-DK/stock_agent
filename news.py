import yfinance as yf


def get_yahoo_news(ticker):
    dat = yf.Ticker(ticker)
    news = dat.news
    return news