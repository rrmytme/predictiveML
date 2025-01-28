from random import randint
import yfinance as yf
import pandas as pd

# to-do get data using all the important market parameters
# marketAnalyser = ['marketCap', 'revenue','netIncomeToCommon','trailingEps', 'trailingPE','returnOnEquity','dividendYield','totalReturn','freeCashflow','debtToEquity','priceToSalesTrailing12Months']

# this method helps to get top 10 stocks by market cap  
def get_sp500_top_10_marketCap():
    # Download the S&P 500 index data
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(period="1d")
    
    # Get the list of S&P 500 companies
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    
    # Download market cap data for each company
    market_caps = {}
    for ticker in sp500_tickers['Symbol']:
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info['marketCap']
            market_caps[ticker] = market_cap
        except KeyError:
            # Skip if market cap data is not available
            continue
    
    # Sort the companies by market cap in descending order and get the top 10
    top_10_stocks_market_caps = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_10_stocks_market_caps

def createPortfolio():
    top_10_stocks_market_caps = get_sp500_top_10_marketCap()
    stocks = []
    portfolio = {} 
    for stock, market_cap in top_10_stocks_market_caps:
        stocks.append(stock)

    for stock in stocks:
        portfolio[stock] = randint(5, 15)
    return portfolio


def get_sp500_top_10_revenue():
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(period="1d")
    
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    
    revenues = {}
    for ticker in sp500_tickers['Symbol']:
        try:
            stock = yf.Ticker(ticker)
            revenue = stock.info['revenue']
            revenues[ticker] = revenue
        except KeyError:
            continue
    
    top_10_stocks_revenues = sorted(revenues.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_10_stocks_revenues

# Example usage
if __name__ == "__main__":
    top_10_stocks_market_caps = get_sp500_top_10_marketCap()
    print("Top 10 S&P 500 stocks by market capitalization:")
    for stock, market_cap in top_10_stocks_market_caps:
        print(f"{stock}: ${market_cap:,}")