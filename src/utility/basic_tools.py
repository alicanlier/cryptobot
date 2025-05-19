import os, re
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.um_futures import UMFutures
from binance.error import ClientError
# from binance_f import RequestClient
# from binance_f.model.constant import PositionSide
import logging
import schedule
from datetime import datetime
import time
import pickle
import config.config as config
import pandas as pd
from collections import OrderedDict
from scipy.signal import argrelextrema


pd.set_option('display.max_columns', None)

# Connect to Binance API
api_key_dict = config.fetch_credentials()
API_KEY, API_SECRET = api_key_dict['API_KEY'], api_key_dict['API_SECRET']
CLIENT = Client(API_KEY, API_SECRET)
FUTURES_CLIENT = UMFutures(API_KEY, API_SECRET)

# Define fixed variable
INTERVAL_DICT = config.fetch_interval_dict(CLIENT)
TIME_DICT = config.time_dict()
FUTURES_SYMBOLS_DICT_DICT_PKL_PATH = r'C:\Users\A\Desktop\my-PT\crypto_bot\data\futures_symbols_dict_dict.pkl'


def get_futures_symbols_dict_dict():
    if not os.path.exists(FUTURES_SYMBOLS_DICT_DICT_PKL_PATH):
        futures_exchange_info = CLIENT.futures_exchange_info()
        futures_symbols_dict_list = futures_exchange_info['symbols']
        futures_symbols_dict_list = [x for x in futures_symbols_dict_list if x['quoteAsset'] == 'USDT' and x['status'] == 'TRADING' and x['contractType'] == 'PERPETUAL']

        futures_symbols_dict1 = OrderedDict({x['symbol']: x for x in futures_symbols_dict_list})
        futures_symbols_dict2 = OrderedDict({x[:-4]: y for x, y in futures_symbols_dict1.items()})
        futures_symbols_dict3 = OrderedDict({x.lower(): y for x, y in futures_symbols_dict2.items()})
        futures_symbols_dict_dict = {**futures_symbols_dict1, **futures_symbols_dict2, **futures_symbols_dict3}

        futures_symbols_dict_dict = OrderedDict(sorted(futures_symbols_dict_dict.items()))

        with open(FUTURES_SYMBOLS_DICT_DICT_PKL_PATH, 'wb') as file:
            pickle.dump(futures_symbols_dict_dict, file)
    else:
        with open(FUTURES_SYMBOLS_DICT_DICT_PKL_PATH, 'rb') as file:
            futures_symbols_dict_dict = pickle.load(file)

    return futures_symbols_dict_dict


def get_futures_symbols_dict_list():
    FUTURES_SYMBOLS_DICT_LIST_PKL_PATH = re.sub(r'dict_dict', 'dict_list', FUTURES_SYMBOLS_DICT_DICT_PKL_PATH)
    if os.path.exists(FUTURES_SYMBOLS_DICT_LIST_PKL_PATH):
        with open(FUTURES_SYMBOLS_DICT_LIST_PKL_PATH, 'wb') as file:
            futures_symbols_dict_list = pickle.load(file)
    else:
        futures_symbols_dict_list = [x for x in list(get_futures_symbols_dict_dict().values()) if 'USDT' in x['symbol']]
        futures_symbols_dict_list = sorted(futures_symbols_dict_list, key=lambda x: x['symbol'])

    # with open(FUTURES_SYMBOLS_DICT_LIST_PKL_PATH, 'wb') as file:
    #     pickle.dump(futures_symbols_dict_list, file)

    return futures_symbols_dict_list


def get_futures_symbols_list():
    futures_symbols_dict_list = get_futures_symbols_dict_list()
    futures_symbols_list = [x['symbol'] for x in futures_symbols_dict_list]
    return sorted(futures_symbols_list)


def get_futures_symbols_dict_dict111():
    futures_symbols_dict_list = get_futures_symbols_dict_list()
    futures_symbols_dict1 = {x['symbol']:x for x in futures_symbols_dict_list}
    futures_symbols_dict2 = {x[:-4]:y for x,y in futures_symbols_dict1.items()}
    futures_symbols_dict3 = {x.lower():y for x,y in futures_symbols_dict2.items()}
    futures_symbols_dict_dict = {**futures_symbols_dict1, **futures_symbols_dict2, **futures_symbols_dict3}

    with open(FUTURES_SYMBOLS_DICT_DICT_PKL_PATH, 'wb') as file:
        pickle.dump(futures_symbols_dict_list, file)

    return futures_symbols_dict_dict


# objective is saving all 15 min interval data since its first appearance on Binance (probably later than 2019, 1, 1).
def save_klines(symbol, time_frame=15):
    limit = 500
    time_frame = TIME_DICT[15] * 60 * 1000  # time_frame in miliseconds
    time_interval = time_frame * limit  # since query limit=1500
    current_date = int(time.time() * 1000)
    first_date = int(datetime(2018, 1, 1).timestamp()) * 1000
    start_date = current_date - time_interval
    end_date = current_date

    df_list = []
    df_last = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    # df_last.set_index('timestamp', inplace=True)
    symbol_pickle_path = os.path.join(r'/data/futures_hist', symbol + '.pkl')
    if os.path.exists(symbol_pickle_path):
        with open(symbol_pickle_path, 'rb') as file:
            df_last = pickle.load(file)
            # df_last.set_index('timestamp', inplace=True)
            head_time = df_last['timestamp'].iloc[0]
            end_date = head_time - time_frame
            start_date = end_date - time_interval
            print(head_time, datetime.fromtimestamp(head_time / 1000))

    while start_date > first_date:
        # klines = FUTURES_CLIENT.continuous_klines(pair='BTCUSDT', interval='1h', contractType='PERPETUAL', startTime=start_date, endTime=end_date, limit=1500)
        klines = FUTURES_CLIENT.klines(symbol=symbol, interval=INTERVAL_DICT[15], startTime=start_date, endTime=end_date, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # df.set_index('timestamp', inplace=True)
        df = df.dropna()
        df_list.append(df)
        # print(df['close'])
        end_date = start_date
        start_date = end_date - time_interval

        # print(datetime.fromtimestamp(start_date / 1000))
        time.sleep(10)

    print(len(df_list))
    df_big = pd.concat(df_list[::-1], ignore_index=True)
    if len(df_last) > 0:
        df_last = pd.concat([df_big, df_last], ignore_index=True)
    else:
        df_last = df_big

    with open(symbol_pickle_path, 'wb') as file:
        df_last.drop_duplicates(inplace=True)
        pickle.dump(df_last, file)

# symbols = get_futures_symbols()
# pkl_files = os.listdir(r'C:\Users\A\Desktop\my-PT\crypto_bot\data\futures_hist')
# pkl_list = [os.path.splitext(pkl_file)[0] for pkl_file in pkl_files]
# symbols = [x['symbol'] for x in symbols if x['symbol'] not in pkl_list]
# for symbol in symbols:
#     print(f'...saving {symbol} klines.')
#     save_klines(symbol)

# print(get_futures_symbols_list()[:40])