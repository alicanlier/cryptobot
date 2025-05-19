import hmac
import time
import hashlib
import requests
import json
import decimal
from urllib.parse import urlencode
import config.config as config

""" This is a very simple script working on Binance API

- work with USER_DATA endpoint with no third party dependency
- work with testnet

Provide the API key and secret, and it's ready to go

Because USER_DATA endpoints require signature:
- call `send_signed_request` for USER_DATA endpoints
- call `send_public_request` for public endpoints

```python

python um_futures.py

```

"""


# api_key_dict = config.fetch_credentials()
# API_KEY, API_SECRET = api_key_dict['API_KEY'], api_key_dict['API_SECRET']
API_KEY = "wZlriYGkPIUtAq2hWekXIppzbnKHtN5ZemFcbDV66Jg8WNAglEcnN0PabtPgHmk7"
API_SECRET = "VhKqqUicU3TlQASorCsEZiVp64Lb23Df250NC1NHmCKkscGbt13knZWSiMUC30L6"
BASE_URL = 'https://fapi.binance.com' # production base url
# BASE_URL = "https://testnet.binancefuture.com"  # testnet base url

""" ======  begin of functions, you don't need to touch ====== """

def round_long_decimals(num, prec=4):
    # Convert the float number to a Decimal with the appropriate precision
    decimal.getcontext().prec = prec
    decimal_num = decimal.Decimal(str(num))

    # Round the Decimal number to its last 4 nonzero digits
    rounded_decimal_num = decimal_num.normalize()

    return float(rounded_decimal_num)

def current_price(symbol):
        # Make a request to the Binance API to get the latest price
        response = requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}')

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the response JSON data
            data = response.json()
            # Extract the current price from the response
            current_price = round_long_decimals(float(data['price']))
            # Print the current price
            print(f"{symbol}:{current_price}", end=' |')
            return current_price
        else:
            print(f"Failed to fetch price data from Binance API for {symbol}")

current_price('BNBUSDT')

def place_sell_order(symbol, quantity, target_price):
    """
    Place a sell order for the given symbol and quantity at the target price.

    Args:
        symbol (str): The trading pair symbol (e.g., XRPUSDT).
        quantity (float): The quantity to sell.
        target_price (float): The target price for the sell order.
        api_key (str): Your Binance API key.
        api_secret (str): Your Binance API secret.
    """
    # Binance API endpoint for placing orders
    endpoint = 'https://api.binance.com/api/v3/order'

    # Parameters for the sell order
    params = {
        'symbol': symbol,
        'side': 'SELL',            # Order side (SELL for selling)
        'type': 'LIMIT',           # Order type (LIMIT for limit order)
        'quantity': quantity,      # Quantity to sell
        'price': target_price,     # Target price for the sell order
        'timeInForce': 'GTC',      # Time in force (Good Till Cancelled)
        'recvWindow': 5000         # Optional parameter for time window in milliseconds
    }

    # Add timestamp parameter
    params['timestamp'] = int(time.time() * 1000)
    # params['timestamp'] = get_timestamp()

    # Create query string
    query_string = '&'.join([f"{key}={value}" for key, value in params.items()])

    # Create signature
    signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    # Add signature to params
    params['signature'] = signature

    # Headers for the request (include API key)
    headers = {
        'X-MBX-APIKEY': API_KEY
    }

    # Send the request to place the sell order
    response = requests.post(endpoint, params=params, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("Sell order successfully placed.")
    else:
        print(f"Failed to place sell order. Status code: {response.status_code}")

# Example usage
symbol = 'XRPUSDT'
quantity = 100
target_price = 1.1 * current_price('XRPUSDT')  # Assuming target price is 10% over the current price

# Place a sell order for 100 XRPUSDT at the target price
place_sell_order(symbol, quantity, target_price)


def hashing(query_string):
    return hmac.new(
        API_SECRET.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def get_timestamp():
    return int(time.time() * 1000)


def dispatch_request(http_method):
    session = requests.Session()
    session.headers.update(
        {"Content-Type": "application/json;charset=utf-8", "X-MBX-APIKEY": API_KEY}
    )
    return {
        "GET": session.get,
        "DELETE": session.delete,
        "PUT": session.put,
        "POST": session.post,
    }.get(http_method, "GET")


# used for sending request requires the signature
def send_signed_request(http_method, url_path, payload={}):
    query_string = urlencode(payload)
    # replace single quote to double quote
    query_string = query_string.replace("%27", "%22")
    if query_string:
        query_string = "{}&timestamp={}".format(query_string, get_timestamp())
    else:
        query_string = "timestamp={}".format(get_timestamp())

    url = (
        BASE_URL + url_path + "?" + query_string + "&signature=" + hashing(query_string)
    )
    print("{} {}".format(http_method, url))
    params = {"url": url, "params": {}}
    response = dispatch_request(http_method)(**params)
    return response.json()


# used for sending public data request
def send_public_request(url_path, payload={}):
    query_string = urlencode(payload, True)
    url = BASE_URL + url_path
    if query_string:
        url = url + "?" + query_string
    print("{}".format(url))
    response = dispatch_request("GET")(url=url)
    return response.json()


""" ======  end of functions ====== """

### public data endpoint, call send_public_request #####
# get klines
response = send_public_request(
    "/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "15m"}
)
print(response)


# get account informtion
# if you can see the account details, then the API key/secret is correct
response = send_signed_request("GET", "/fapi/v2/account")
print(555, response)


### USER_DATA endpoints, call send_signed_request #####
# place an order
# if you see order response, then the parameters setting is correct
# if it has response from server saying some parameter error, please adjust the parameters according the market.
params = {
    "symbol": "BNBUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "timeInForce": "GTC",
    "quantity": 0.1,
    "price": "150",
}
response = send_signed_request("POST", "/fapi/v1/order", params)
print(response)

# place batch orders
# if you see order response, then the parameters setting is correct
# if it has response from server saying some parameter error, please adjust the parameters according the market.
params = {
    "batchOrders": [
        {
            "symbol": "BNBUSDT",
            "side": "SELL",
            "type": "STOP",
            "quantity": "0.1",
            "price": "500",
            "timeInForce": "GTC",
            "stopPrice": "510",
        },
        {
            "symbol": "BNBUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "150",
            "timeInForce": "GTC",
        },
    ]
}
response = send_signed_request("POST", "/fapi/v1/batchOrders", params)
print(response)



