import wrds
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor

# Initialize the app
app = dash.Dash(__name__)

# Establish a connection
# Define a function to read credentials from a file
def read_credentials(filename):
    credentials = {}
    with open(filename, 'r') as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split('=', 1)
                credentials[key] = value
    return credentials

# Use the function to get credentials
credentials = read_credentials('credentials.txt')
username = credentials.get('username')
password = credentials.get('password')

db = wrds.Connection(wrds_username=username, wrds_password=password)  # Using provided credentials

# SQL query to fetch historical daily prices from CRSP
query_historical_prices = """
SELECT date, permno, prc AS price
FROM crsp.dsf
WHERE date >= '2020-01-01' AND date <= '2023-01-01'
ORDER BY permno, date;
"""

# SQL query to fetch financial ratios from IBES
query_financial_ratios = """
SELECT permno, qdate AS date, be AS book_equity, bm AS book_market_ratio, evm AS enterprise_value_multiple, pe_exi AS current_pe_ratio, ps AS price_sales_ratio,
      pcf AS price_cash_flow_ratio, npm AS net_profit_margin, opmbd AS operating_profit_margin, gpm AS gross_profit_margin, roa AS return_on_assets,
      roe AS return_on_equity, debt_ebitda AS debt_ebitda_ratio, cash_debt AS cash_debt_ratio, debt_assets AS debt_assets_ratio, de_ratio AS debt_equity_ratio,
      quick_ratio, curr_ratio AS current_ratio, mktcap AS mktcap_in_millions, ptb AS price_book_ratio, divyield AS dividend_yield_percentage,
      PEG_trailing AS peg_ratio, ticker, ret_crsp AS returns, fcf_ocf AS free_cash_flow_operating_cash_flow_ratio
FROM wrdsapps_finratio_ibes.firm_ratio_ibes
WHERE qdate >= '2022-09-01' AND qdate <= '2022-12-31'
ORDER BY ticker, qdate;
"""

# SQL query to fetch returns from Compustat funda
# query_revenue = """
# SELECT datadate AS date, tic AS ticker, revt AS revenue, exchg AS exchange
# FROM comp.funda
# WHERE datadate >= '2020-01-01' AND datadate <= '2023-01-01'
# ORDER BY exchg, tic, datadate;
# """

# SQL query to fetch S&P 500 monthly returns from CRSP
query_sp500_returns = """
SELECT date, vwretd as market_return
FROM crsp.msi
WHERE date >= '2022-01-01' AND date <= '2022-12-31'
ORDER BY date;
"""

# SQL query to fetch Beta from CRSP
query_beta1 = """
SELECT date, betav AS beta1, permno
FROM crsp_a_indexes.dport6
WHERE date >= '2022-01-01' AND date <= '2022-12-31'
ORDER BY permno, date;
"""

# SQL query to fetch Beta from CRSP
query_beta2 = """
SELECT date, betav AS beta2, permno
FROM crsp_a_indexes.dport8
WHERE date >= '2022-01-01' AND date <= '2022-12-31'
ORDER BY permno, date;
"""

# exchange_codes = {
#     1: 'NYSE',
#     2: 'AMEX',
#     3: 'NASDAQ'
# }
# data_revenue['exchange'] = data_revenue['exchange'].map(exchange_codes)

# Define functions for each query
def fetch_historical_prices():
    return db.raw_sql(query_historical_prices)

def fetch_financial_ratios():
    return db.raw_sql(query_financial_ratios)

def fetch_sp500_returns():
    return db.raw_sql(query_sp500_returns)

def fetch_beta1():
    return db.raw_sql(query_beta1)

def fetch_beta2():
    return db.raw_sql(query_beta2)

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
@lru_cache(maxsize=10)

# Use ThreadPoolExecutor to run tasks concurrently
def fetch_all_data():
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_prices = executor.submit(fetch_historical_prices)
        future_ratios = executor.submit(fetch_financial_ratios)
        future_sp500 = executor.submit(fetch_sp500_returns)
        future_beta1 = executor.submit(fetch_beta1)
        future_beta2 = executor.submit(fetch_beta2)

        try:
            data_historical_prices = future_prices.result()
            data_financial_ratios = future_ratios.result()
            data_sp500_returns = future_sp500.result()
            data_beta1 = future_beta1.result()
            data_beta2 = future_beta2.result()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
        # Create csv file for each database
        data_historical_prices.to_csv('data_historical_price.csv', index=False)
        data_financial_ratios.to_csv('data_financial_ratios.csv', index=False)
        # data_revenue.to_csv('data_revenue.csv', index=False)
        data_sp500_returns.to_csv('data_sp500_returns.csv', index=False)
        data_beta1.to_csv('data_beta1.csv', index=False)
        data_beta2.to_csv('data_beta2.csv', index=False)

        # Merge the beta data with financial ratios on the 'permno' and 'date' columns
        merged_data = pd.merge(data_beta1, data_financial_ratios, on=['permno'], how='right')
        merged_data.to_csv('merged_data.csv', index=False)
        
        # Now merge the above with beta2 data on the 'permno' and 'date' columns
        merged2_data = pd.merge(merged_data, data_beta2, on=['permno'], how='left')
        merged2_data.to_csv('merged2_data.csv', index=False)
        
        # Merge the above data with historical prices with the 'permno' and 'date' columns
        merged3_data = pd.merge(merged2_data, data_historical_prices, on= ['permno', 'date'], how='left')
        merged3_data.to_csv('merged3_data.csv', index=False)
        
        # Merge the above with sp500 returns data on the 'date' column
        final_merged_data = pd.merge(merged3_data, data_sp500_returns, on=['date'], how='left')
        
        # Merge 'date' and 'date_x' columns
        final_merged_data['combined_date'] = final_merged_data['date'].fillna(final_merged_data['date_x'])
        # Drop the original 'date' and 'date_x' columns
        final_merged_data.drop(['date', 'date_x'], axis=1, inplace=True)
        
        # Merge 'beta1' and 'beta2' columns
        final_merged_data['combined_beta'] = final_merged_data['beta1'].fillna(final_merged_data['beta2'])
        # Drop the original 'date' and 'date_x' columns
        final_merged_data.drop(['beta1', 'beta2'], axis=1, inplace=True)
        final_merged_data.to_csv('final_merged_data.csv', index=False)
        
        # Remove duplicates based on specific columns
        final_data = final_merged_data.drop_duplicates(subset=['ticker', 'combined_date'])
        # Reset the index after dropping duplicates
        final_data = final_data.reset_index(drop=True)

        final_data['min_price'] = final_data['price']  # Create a new column 'min_price' with the same data as 'price'
        final_data['max_price'] = final_data['price']  # Similarly, create 'max_price'
        
        # Convert market cap to acutal $$
        final_data['market_capitalization'] = final_data['mktcap_in_millions'] * 1e6
        
        # Sort the final dataframe by ticker and date
        final_data = final_data.sort_values(by=['ticker', 'combined_date'])
        data_historical_prices = data_historical_prices.sort_values(by=['permno', 'date'])

        final_data.to_csv('data_merged_data.csv', index=False)

        # Convert dividend yield to a % value
        final_data['dividend_yield_percentage'] = final_data['dividend_yield_percentage'] * 100

        # Calculate net income
        final_data['net_income'] = final_data['return_on_equity'] * final_data['book_equity']
       
        # Calculate number of shares outstanding
        final_data['number_of_shares'] = final_data['market_capitalization'] / final_data['price']
        # Ensure you handle cases where price is zero to avoid division by zero
        final_data['number_of_shares'] = final_data.apply(lambda row: row['market_capitalization'] / row['price'] if row['price'] != 0 else np.nan, axis=1)

        # Calculate current P/E ratio where P/E ratio data is blank
        final_data['calculated_current_eps'] = final_data['net_income'] / final_data['number_of_shares']
        # Iterate through the DataFrame and add calculated P/E ratio where needed
        for index, row in final_data.iterrows():
            if pd.isna(row['current_pe_ratio']):
                # Calculate the current P/E ratio directly
                if row['calculated_current_eps'] != 0:  # Avoid division by zero
                    calculated_current_pe_ratio = row['price'] / row['calculated_current_eps']
                    # Update the current_pe_ratio in the DataFrame
                    final_data.at[index, 'current_pe_ratio'] = calculated_current_pe_ratio

        # Calculate forward EPS growth rate
        # Check for zero PEG ratio to avoid division by zero
        final_data['forward_eps'] = final_data['current_pe_ratio'] / final_data['peg_ratio'].replace(0, np.nan)

        # Calculate Forward P/E using the projected forward EPS
        final_data['forward_pe'] = final_data['price'] / final_data['forward_eps']

        # Calculate revenue growth
        # Assuming there are approximately 252 trading days in a year
        # trading_days_in_year = 252
        # Calculate revenue growth
        # Compute the prior period's revenue (shifted by approximately one year)
        # final_data['prior_revenue'] = final_data.groupby('ticker')['revenue'].shift(trading_days_in_year)
        # Compute the revenue growth
        # final_data['revenue_growth'] = ((final_data['revenue'] - final_data['prior_revenue']) / final_data['prior_revenue']) * 100

        # Calculate Earnings Yield
        # Handle cases where the P/E ratio might be zero or null to avoid division by zero errors
        final_data['earnings_yield'] = (1 / final_data['current_pe_ratio'].replace(0, np.nan)) * 100

        # Calculate Free Cash Flow
        # Calculate Operating Cash Flow
        final_data['operating_cash_flow'] = final_data['market_capitalization'] / final_data['price_cash_flow_ratio']
        # Calculate Free Cash Flow
        final_data['free_cash_flow'] = final_data['operating_cash_flow'] * final_data['free_cash_flow_operating_cash_flow_ratio']

        # Calculate Sharpe Ratio
        # Constants
        RISK_FREE_RATE = 0.03  # Risk-free rate of 3.0%
        # Calculate the excess returns by subtracting the risk-free rate from the portfolio returns
        final_data['excess_return'] = final_data['returns'] - RISK_FREE_RATE
        # Calculate the standard deviation of the excess returns
        std_dev_excess_return = final_data['excess_return'].std()
        # Calculate the Sharpe Ratio
        final_data['sharpe_ratio'] = final_data['excess_return'].mean() / std_dev_excess_return

        # Calculate Alpha
        # Constants
        RISK_FREE_RATE = 0.03  # Risk-free rate of 3.0%
        EXPECTED_MARKET_RETURN = 0.08  # Expected market return of 8.0%
        # Calculate expected return using CAPM
        final_data['expected_return'] = RISK_FREE_RATE + final_data['combined_beta'] * (EXPECTED_MARKET_RETURN - RISK_FREE_RATE)
        # Calculate alpha
        final_data['alpha'] = final_data['returns'] - final_data['expected_return']

        # Reorder the columns in the final data
        columns_order = ['ticker', 'price'] + [col for col in final_data.columns if col not in ['ticker', 'price']]
        final_data = final_data[columns_order]
        
        # Save to a CSV file
        final_data.to_csv('data_final_data.csv', index=False)

        # Return the final processed data
        return final_data
    
# Define layout
def create_field(label, id, value, type='number'):
    return html.Div([
        html.Label(label, style={'color': 'white'}),
        dcc.Input(id=id, value=value, type=type, style={'color': 'black', 'backgroundColor': 'white'}),
        dcc.Checklist(
            id=f'check-{id}',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            inline=True,
            style={'display': 'inline-block', 'margin-left': '10px'}
        )
    ], style={'width': '48%', 'display': 'inline-block'})

fields = [
    ('Min Price:', 'min_price', 100),
    ('Max Price:', 'max_price', 200),
    ('Max P/E Ratio:', 'forward_pe', 20),
    ('Max PEG Ratio', 'peg_ratio', 1),
    ('Min EPS:', 'forward_eps', 2),
    ('Min Dividend Yield:', 'dividend_yield_percentage', .025), # Use decimal format
    ('Min P/B Ratio:', 'price_book_ratio', 1),
    ('Min Return on Equity:', 'return_on_equity', 0.15), # Use decimal format
    ('Min Current Ratio:', 'current_ratio', 1.2),
#     ('Min Revenue Growth:', 'revenueGrowth', 0.05), # Use decimal format
    ('Min Free Cash Flow Yield:', 'free_cash_flow', 0.025),
    ('Min Operating Margin:', 'operating_profit_margin', 0.15),
    ('Max Price/Sales Ratio:', 'price_sales_ratio', 1),
    ('Min Earnings Yield:', 'earnings_yield', 0.10),
    ('Min Quick Ratio:', 'quick_ratio', 1.5),
    ('Max Debt/Equity Ratio:', 'debt_equity_ratio', 2),
    ('Min Alpha:', 'alpha', 0.10),
    ('Min Beta', 'combined_beta', 1),
    ('Min Sharpe Ratio', 'sharpe_ratio', 1),
    ('Min Market Cap', 'market_capitalization', 1e9),
]

# Define the app layout
app.layout = html.Div([
    html.Div([create_field(*field) for field in fields]),
    html.Button('Screen Stocks', id='screen-button', style={'color': 'black', 'backgroundColor': 'white'}),
    dcc.Loading(
        id="loading",
        type="circle",  
        color="#FFFFFF",  # color of the loading spinner
        style={'margin-top': '-250px'},  # adjust the top margin to move it up
        children=[
            html.Div(id='output-table', style={'color': 'white'}),
            # html.Div(id='hidden-div', style={'display': 'none'})  # Hidden Div to store filtered DataFrame
        ]
    )
], style={'backgroundColor': '#000000', 'textAlign': 'center'})

# Updated callback
@app.callback(
    [Output('output-table', 'children')],
    [Input('screen-button', 'n_clicks')],
    [State(field_id, 'value') for _, field_id, _ in fields] +
    [State(f'check-{field_id}', 'value') for _, field_id, _ in fields]
)
def update_output(n_clicks, *args):
    if n_clicks is None:
        return [dash.no_update]

    # Define the columns to include in the table
    columns_to_include = ['ticker', 'price', 'forward_pe', 'peg_ratio', 'forward_eps', 'dividend_yield_percentage',
                          'price_book_ratio', 'return_on_equity', 'current_ratio', 'free_cash_flow',
                          'operating_profit_margin', 'price_sales_ratio', 'earnings_yield', 'quick_ratio',
                          'debt_equity_ratio', 'alpha', 'combined_beta', 'sharpe_ratio', 'market_capitalization']

    # Fetch the data
    df = fetch_all_data()
    if df is None or df.empty:
        return ['No data to display']

    n = len(fields)
    values = args[:n]
    checkboxes = args[n:]

    # Apply filters based on the input values and checkboxes
    for value, (_, field_id, _), checkbox in zip(values, fields, checkboxes):
        if 'on' not in checkbox:
            continue

        column_name = field_id  # Ensure this matches the DataFrame column name
        if 'Min' in field_id:
            df = df.loc[df[column_name] >= value]  # Use .loc for proper indexing
        elif 'Max' in field_id:
            df = df.loc[df[column_name] <= value]  # Use .loc for proper indexing

    # Select only the columns to include
    df = df[columns_to_include]

    # Round all numeric columns to 4 decimal places
    for col in df.select_dtypes(include=['float64']).columns:
        df.loc[:, col] = df[col].round(4)  # Use .loc for assignment
        
    # Create a Plotly Table to Display Data
    fig = go.Figure(data=[go.Table(
            header=dict(values=columns_to_include,  # Directly use columns_to_include list
                    fill_color='#f5f5f5',
                    align='center',
                    font=dict(size=12),
                    line=dict(color='darkslategray', width=1)),
            cells=dict(values=[df[col] for col in columns_to_include],  # Iterate over columns_to_include list
                    fill_color='white',
                    align='center',
                    font=dict(size=10),
                    line=dict(color='darkslategray', width=1))
    )])
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, b=20, t=20)
    )

    return [dcc.Graph(figure=fig, style={"height": "calc(100vh - 40px)"})]

if __name__ == '__main__':
    app.run_server(debug=True)
    
# Close the connection
db.close()