# FinAI Assistant[DRAFT VERSION..]

FinAI Assistant is a chatbot designed to assist with financial tasks such as retrieving stock prices, managing portfolios, and plotting stock charts. The assistant uses machine learning to understand user intents and provide appropriate responses.

## Features

- Retrieve stock prices
- Add and remove stocks from a portfolio
- Show portfolio details
- Calculate portfolio worth and growth
- Plot stock price charts

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/fin_ai_assistant.git
    cd fin_ai_assistant
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Install additional dependencies:
    ```sh
    pip install yfinance matplotlib nltk
    ```

## Usage

1. Train the model:
    ```sh
    python fin_ai_chatbot.py
    ```

2. Interact with the chatbot:
    ```sh
    python fin_ai_chatbot.py
    ```

## Project Structure

- [fin_ai_assistant.py](http://_vscodecontentref_/0): Contains the main assistant class and methods for training the model and processing user inputs.
- `fin_ai_chatbot.py`: The main script to run the chatbot.
- [intents.json](http://_vscodecontentref_/1): Contains the intents data used to train the model.
- `requirements.txt`: Lists the required Python packages.

## Example

To plot a stock price chart, you can use the following command in the chatbot: 
I want you to plot a stock price

## License

This project is licensed under the MIT License.