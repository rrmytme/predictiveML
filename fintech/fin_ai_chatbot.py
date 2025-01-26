# Version: 1.0

from fin_ai_assistant import BotAssistant
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import datetime as dt
import yfinance as yf


# #Step1: 
# # Load the portfolio
# portfolio = {'AAPL': 10, 'TSLA': 5, 'AMZN': 2, 'GOOGL': 3, 'MSFT': 4}
# # create a file to store the portfolio
# with open('portfolio.pkl', 'wb') as f:
#     pickle.dump(portfolio, f)

# #Step2:
# # Load the portfolio
with open('portfolio.pkl', 'rb') as f:
    portfolio = pickle.load(f)

# #Step3:
# create mapping for the chatbot
def save_portfolio(portfolio):
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)

def stock_price():
    stock = input('which stock do you want to check: ')
    start_date = dt.datetime.now() - dt.timedelta(days=5)
    end_date = dt.datetime.now()
    stock_data = yf.download(stock, start=start_date, end=end_date)['Close']
    print(f'{stock} price is: {stock_data}')

def add_portfolio():
    stock = input('which stock do you want to add: ')
    quantity = int(input('how many shares do you want to add: '))

    if stock in portfolio.keys():
        portfolio[stock] += quantity
    else:
        portfolio[stock] = quantity
    
    save_portfolio(portfolio)  

def remove_portfolio():
    stock = input('which stock do you want to sell: ')
    quantity = int(input('how many shares do you want to sell: '))

    if stock in portfolio.keys():
        if quantity <= portfolio[stock]:
            portfolio[stock] -= quantity
        else:
            print('you dont have enough stock in portfolio')
    else:
        print('stock not in portfolio')

    save_portfolio(portfolio)

def show_portfolio():
    print("your portfolio:")
    for stock, quantity in portfolio.items():
        print(f'{stock}: {quantity}')

def portfolio_worth():
    for stock, quantity in portfolio.items():
        start_date = dt.datetime.now() - dt.timedelta(days=2)
        end_date = dt.datetime.now()
        stock_price = yf.download(stock, start_date, end_date)['Close']
        print(f'{stock} worth is: {stock_price * quantity} USD')

def portfolio_growth():
    start_date = dt.datetime.now() - dt.timedelta(days=30)
    end_date = dt.datetime.now()
    portfolio_worth = 0
    for stock, quantity in portfolio.items():
        stock_data = yf.download(stock, start_date, end_date)
        stock_data['daily_return'] = stock_data['Close'].pct_change()
        stock_data['investment'] = stock_data['daily_return'] * quantity
        portfolio_worth += stock_data['investment'].sum()
    print(f'your portfolio growth in last 30 days is: {portfolio_worth}\n')  

def plot_chart():
    try:
        stock = input('which stock do you want to plot: ')
        print(f"Fetching data for stock: {stock}")
        start_date = dt.datetime.now() - dt.timedelta(days=30)
        end_date = dt.datetime.now()        
        stock_data = yf.download(stock, start=start_date, end=end_date)
        print(f"Retrieved data: {stock_data.head()}")
        sns.set_style('dark')
        stock_data['Close'].plot(title=f"{stock} Stock Price", figsize=(10, 6), color='blue', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"An error occurred while fetching the stock data: {e}")
        return "Sorry, I couldn't retrieve the stock data at the moment."

def goodbye():
    print('Goodbye! Have a great day!')                

# step4: map the methods to the chatbot
mapping ={
    'stock_price': stock_price,
    'add_portfolio': add_portfolio,
    'remove_portfolio': remove_portfolio,
    'show_portfolio': show_portfolio,
    'portfolio_worth': portfolio_worth,
    'portfolio_growth': portfolio_growth,
    'plot_chart': plot_chart,
    'goodbye': goodbye,
    # to-do: add more methods
    # 'plot_stock': 'plot_stock',
    # 'get_stock': 'get_stock',
    # 'get_news': 'get_news',
    # 'get_price': 'get_price',
    # 'get_data': 'get_data',
    # 'get_chart': 'get_chart',
    # 'get_recommendation': 'get_recommendation',
    # 'get_ticker': 'get_ticker',
    # 'get_dividends': 'get_dividends',
    # 'get_info': 'get_info',
    # 'get_summary': 'get_summary',
    # 'get_company': 'get_company',
    # 'get_balance_sheet': 'get_balance_sheet',
    # 'get_income_statement': 'get_income_statement',
    # 'get_cash_flow': 'get_cash_flow',
    # 'get_earnings': 'get_earnings',
    # 'get_financials': 'get_financials',
    # 'get_history': 'get_history',
    # 'get_quote': 'get_quote',
    # 'get_stats': 'get_stats',
    # 'get_peers': 'get_peers',
    # 'get_profile': 'get_profile',
    # 'get_sector': 'get_sector',
    # 'get_splits': 'get_splits',
    # 'get_volume_by_venue': 'get_volume_by_venue', 
}

# step5: train the chatbot
finbot = BotAssistant('fintech/intents.json', mapping, 'fin_basic_model')
# finbot.train_model()
# finbot.save_model()
finbot.load_model()

# step6: run the chatbot
while True:
    message = input('Hi!: ')
    if message == 'exit':
        break
    finbot.get_response(message)

