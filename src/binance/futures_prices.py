import math
import os
import threading
from pathlib import Path

from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.um_futures import UMFutures
from binance.error import ClientError
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from scipy.signal import argrelextrema
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from collections import OrderedDict
import requests
import re, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import decimal
import config.config as config
import utility.calculation_tools as calculate
import utility.basic_tools as basic
import spot_prices as spot_prices
import pickle

# Connect to Binance API
api_key_dict = config.fetch_credentials()
API_KEY, API_SECRET = api_key_dict['API_KEY'], api_key_dict['API_SECRET']
CLIENT = Client(API_KEY, API_SECRET)
FUTURES_CLIENT = UMFutures(API_KEY, API_SECRET)
INTERVAL_DICT = config.fetch_interval_dict(CLIENT)
TIME_DICT = config.time_dict()


# All 'TRADING', 'PERPETUAL', 'USDT' pairs are fetched.
# futures_symbol_dict_list = basic.get_futures_symbols_dict_list()
# futures_pair_list = [futures_symbol_dict['symbol'] for futures_symbol_dict in futures_symbol_dict_list]
# futures_symbols_dict = basic.get_futures_symbols_dict_dict()  # symbol=symbol_info dict: symbol_info itself is a dict.

# now = int(time.time() * 1000)
# start_date = int(datetime(2023, 1, 1).timestamp()) * 1000
# end_date = int(datetime(2024, 1, 1).timestamp()) * 1000
# # klines = FUTURES_CLIENT.continuous_klines(pair='BTCUSDT', interval='1h', contractType='PERPETUAL', startTime=start_date, endTime=end_date, limit=1500)
# klines = FUTURES_CLIENT.klines(symbol='1000FLOKIUSDT', interval=INTERVAL_DICT[15], startTime=start_date, endTime=end_date, limit=1500)

COINS_DICT = {}
# futures_hist_dirpath = Path('../../data/futures_hist')
futures_hist_dirpath = Path('D:/crypto_bot/data/futures_hist')
futures_hist_dirpath_old = Path('D:/crypto_bot/data/futures_hist/old')
futures_dirpath = Path('../../data/futures')
all_futures_usdt_pairs_pkl_path = futures_dirpath.joinpath("all_futures_usdt_pairs.pkl")
all_usable_futures_pairs_pkl_path = futures_dirpath.joinpath("all_usable_futures_pairs.pkl")
coins_dict_pkl_path = futures_dirpath.joinpath("coins_dict.pkl")
orders_dict_pkl_path = futures_dirpath.joinpath("orders_dict.pkl")
followup_futures_pairs_pkl_path = futures_dirpath.joinpath("followup_futures_pairs.pkl")
futures_pairs_stats_path = futures_dirpath.joinpath("futures_pairs_stats.csv")


def get_all_futures_pairs_df(_futures_dirpath='../../data/futures'):
    # futures_dirpath = _futures_dirpath
    # futures_pairs_stats_path = os.path.join(futures_dirpath, "futures_pairs_stats.csv")
    futures_exchange_info = CLIENT.futures_exchange_info()
    coins_dict = {}
    if os.path.exists(coins_dict_pkl_path):
        coins_dict = save_load_pkl(coins_dict_pkl_path)

    if os.path.exists(futures_pairs_stats_path):
        df = pd.read_csv(futures_pairs_stats_path)
        if 'Unnamed: 0' in df.columns and df.index.name != 'futures_pair':
            df.set_index('Unnamed: 0', inplace=True)
            df.index.name = 'futures_pair'
            df.to_csv(futures_pairs_stats_path, encoding='UTF-8', index_label='futures_pair')
        elif df.index.name != 'futures_pair':
            df.set_index('futures_pair', inplace=True)
            df.to_csv(futures_pairs_stats_path, encoding='UTF-8', index_label='futures_pair')
    else:
        # Find the symbol information for the symbol
        symbol_info_list = [item for item in futures_exchange_info['symbols'] if item['status'] == 'TRADING' and item['quoteAsset'] != 'USDC' and 'DOM' not in item['symbol']]
        symbol_info_list = sorted(symbol_info_list, key=lambda x: x['symbol'])

        # last_price = get_futures_historical_data(CLIENT, symbol, '1m', str(datetime.now(timezone.utc) - timedelta(minutes=1)))['close'].iloc[-1]

        df = pd.DataFrame(symbol_info_list, columns=['symbol', 'pricePrecision', 'quantityPrecision'])
        df.rename(columns={'symbol': 'futures_pair'}, inplace=True)
        df.set_index('futures_pair', inplace=True)

        os.makedirs(futures_dirpath, exist_ok=True)
        df.to_csv(futures_pairs_stats_path, encoding='UTF-8', index_label='futures_pair')

    for index, row in df.iterrows():
        if index not in coins_dict.keys():
            coins_dict[index] = {}
        coins_dict[index]['pricePrecision'] = row['pricePrecision']
        coins_dict[index]['quantityPrecision'] = row['quantityPrecision']

    save_load_pkl(coins_dict_pkl_path, coins_dict)

    return df


def get_new_time_interval(new_kline_interval, file_path='', _df='', is_save=False):
    if file_path:
        df = pd.read_csv(file_path, index_col="datetime", parse_dates=True)
    else:
        df = _df

    new_kline_interval_str = config.time_dict_str()[new_kline_interval]
    new_freq = config.time_dict()[new_kline_interval]

    # Resample by {new_freq} minutes
    df_resampled = df.resample(str(new_freq) + 'min').agg({
        'open': 'first',  # Open value for the first row in each {new_freq}-minute group
        'close': 'last',  # Close value for the last row in each {new_freq}-minute group
        'high': 'max',  # Highest value within the {new_freq}-minute period
        'low': 'min'  # Lowest value within the {new_freq}-minute period
    })

    if is_save:
        dirpath, filename = os.path.split(file_path)
        new_dirpath = os.path.join(os.path.split(dirpath)[0], new_kline_interval_str)
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)
        new_filepath = os.path.join(new_dirpath, filename)
        df_resampled.to_csv(new_filepath, encoding='UTF-8', index_label='datetime')

    return df_resampled


def get_new_time_interval2(new_kline_interval, new_freq, file_path='', _df='', unit='min', is_save=False):
    if file_path:
        df = pd.read_csv(file_path, index_col="datetime", parse_dates=True)
    else:
        df = _df

    direction = 'right'

    if unit == 'min':
        # Ensure datetime index is aligned with multiples of new_freq
        df.index = df.index - pd.to_timedelta(df.index.minute % new_freq, unit=unit)
        if new_freq >= 120:
            direction = 'left'
    elif unit == 'h':
        df.index = df.index - pd.to_timedelta(df.index.hour % new_freq, unit=unit)
        direction = 'left'

    # Resample by {new_freq} minutes
    df_resampled = df.resample(
        str(new_freq) + unit,
        label=direction,  # Aligns labels to the right edge (e.g., 13:05, 13:10, etc.)
        closed=direction  # Treats intervals as closed on the right
    ).agg({
        'open': 'first',  # Open value for the first row in each {new_freq}-minute group
        'close': 'last',  # Close value for the last row in each {new_freq}-minute group
        'high': 'max',  # Highest value within the {new_freq}-minute period
        'low': 'min'  # Lowest value within the {new_freq}-minute period
    })

    # Save the resampled dataframe to a CSV file if required
    if is_save:
        save_path = file_path.replace('.csv', f'_resampled_{new_freq}min.csv')
        df_resampled.to_csv(save_path, encoding='utf-8', index_label='datetime')

    return df_resampled


def get_new_open_time(shift_days=7, _df='', file_path='', is_save=False):
    if file_path:
        df = pd.read_csv(file_path, index_col="datetime", parse_dates=True)
    else:
        df = _df

    last_date = df.index[-1]
    new_start_date = last_date - timedelta(days=shift_days)

    df_filtered = df[df.index > new_start_date]

    if is_save:
        new_file_path = os.path.join(os.path.split(file_path)[0] + f"_{shift_days}days", os.path.split(file_path)[1])
        df_filtered.to_csv(new_file_path, encoding='utf-8', index_label='datetime')

    return df_filtered


# Gets data via Binance API
def get_futures_historical_data(client_me, symbol, _kline_interval='1m', start_str=None, end_str=None):
    kline_interval = INTERVAL_DICT[_kline_interval]

    try:
        klines = client_me.futures_historical_klines(symbol, kline_interval, start_str, end_str)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                           'taker_buy_quote_asset_volume', 'ignore'])

        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['low'] = df['low'].astype(float)
        df['high'] = df['high'].astype(float)

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)

        # if df is not None and not df.empty:
        #     # print(df.iloc[100])
        #     # print(df.loc['2024-11-14 05:53:00'])
        #     # print(df.index[100])
        #     max_high_row_index = df['high'].idxmax()
        #     min_low_row_index = df['low'].idxmin()
        #     print(f"Max high value at time {max_high_row_index}: {df.loc[max_high_row_index, 'high']}")
        #     print(f"Min low value at time {min_low_row_index}: {df.loc[min_low_row_index, 'low']}")

        return df.dropna()

    except BinanceAPIException as e:
        print(f"ERROR for {symbol}: {e}")


# Gets data from saved files on the computer/storage.
def get_futures_record_data(symbol, kline_interval='1m', _start_date=datetime(2009, 1, 1), _end_date=datetime(2039, 1, 1)):
    saving_dirpath = os.path.join(futures_hist_dirpath, kline_interval)
    pattern = rf"(?<!\w){symbol}"
    matching_files = [file for file in os.listdir(saving_dirpath) if os.path.isfile(os.path.join(saving_dirpath, file)) and re.search(pattern, file)]
    matching_files_tuples = [(file, os.path.splitext(file)[0].split("_")[-1]) for file in matching_files]

    print(symbol, 'record_data:', end=' ')

    if matching_files:
        matching_file = max(matching_files_tuples, key=lambda x: datetime.strptime(x[1], "%y%m%d"))[0]
        df = pd.read_csv(os.path.join(saving_dirpath, matching_file), index_col="datetime", parse_dates=True)
        print(f'record data matching_file: {matching_file} | first date: {df.index[0]}', end='|')
        df = df[(df.index > _start_date) & (df.index < _end_date)]
        print(f'first date post-filter: {df.index[0]}')

        return df


# Gets coin prices from either of recorded data or Binance API or combination of those depending on desired dates.
# Purpose is to save speed by depending less on Binance API, though recorded data is updated once this method is called.
def get_futures_recent_data(client_me, symbol, kline_interval='1m', _start_date=datetime(2009, 1, 1), _end_date=datetime(2039, 1, 1)):
    saving_dirpath = os.path.join(futures_hist_dirpath, config.time_dict_str()[kline_interval])
    pattern = rf"(?<!\w){symbol}"
    matching_files = [file for file in os.listdir(saving_dirpath) if os.path.isfile(os.path.join(saving_dirpath, file)) and re.search(pattern, file)]
    matching_files_tuples = [(file, os.path.splitext(file)[0].split("_")[-1]) for file in matching_files]

    print(symbol, "recent_data is being assembled with record data and the newest data.")

    if matching_files:
        matching_file = max(matching_files_tuples, key=lambda x: datetime.strptime(x[1], "%y%m%d"))[0]
        matching_file_date = datetime.strptime(os.path.splitext(matching_file)[0].split("_")[-1], "%y%m%d").date()

        if matching_file_date < _start_date.date():
            # threading.Thread(target=save_futures_historical_data(client_me, kline_interval, symbol), daemon=True).start()
            return get_futures_historical_data(client_me, symbol, kline_interval, str(_start_date), str(_end_date))

        else:
            df_old_temp = get_futures_record_data(symbol, config.time_dict_str()[kline_interval], pd.to_datetime(matching_file_date) - timedelta(days=1))
            df_old = get_futures_record_data(symbol, config.time_dict_str()[kline_interval], _start_date)
            last_timestamp = df_old_temp.iloc[-1].name
            next_timestamp = last_timestamp + timedelta(minutes=1)
            df_new = get_futures_historical_data(client_me, symbol, kline_interval, start_str=str(next_timestamp))

            # threading.Thread(target=save_futures_historical_data(client_me, kline_interval, symbol),  daemon=True).start()
            return pd.concat([df_old, df_new])

    else:
        # threading.Thread(target=save_futures_historical_data(client_me, kline_interval, symbol), daemon=True).start()
        return get_futures_historical_data(client_me, symbol, kline_interval, str(_start_date), str(_end_date))


# Temporary method
def change_index_name(kline_interval='1m'):
    saving_dirpath = os.path.join(futures_hist_dirpath, kline_interval)
    files = [file for file in os.listdir(saving_dirpath) if os.path.isfile(os.path.join(saving_dirpath, file))]
    for file in files:
        file_path = os.path.join(saving_dirpath, file)
        df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
        df.rename_axis("datetime", inplace=True)
        df.to_csv(file_path, encoding='utf-8', index_label='datetime')


def save_futures_historical_data(client_me, _kline_interval, is_symbol='', _start_str=None, _end_str=None):
    kline_interval = config.fetch_interval_dict(client_me)[_kline_interval]
    saving_dirpath = os.path.join(futures_hist_dirpath, kline_interval)
    get_all_futures_pairs_df()  # guarantees formation of COINS_DICT

    # Generate file path to save the data in a csv file
    def get_saving_path(symbol):
        if not os.path.exists(saving_dirpath):
            os.makedirs(saving_dirpath)

        return os.path.join(saving_dirpath, symbol + '_FUTURES_' + current_date.strftime("%y%m%d") + '.csv')

    def save_df_to_csv(symbol):
        saving_path = get_saving_path(symbol)
        pattern = rf"(?<!\w){symbol}"
        matching_files = [file for file in os.listdir(saving_dirpath) if os.path.isfile(os.path.join(saving_dirpath, file)) and re.search(pattern, file)]
        print("matching_files: ", matching_files)

        if not matching_files:
            df = get_futures_historical_data(client_me, symbol, kline_interval, _start_str, _end_str)
            if df is not None and not df.empty:
                df.to_csv(saving_path, encoding='utf-8', index_label='datetime')
                return df
            else:
                print("...No historical klines info for", symbol)
        else:
            df_old = get_futures_record_data(symbol)
            # print(df_old.tail(3))
            # last_timestamp = df_old.index[-1]
            last_timestamp = df_old.iloc[-1].name
            next_timestamp = last_timestamp + timedelta(minutes=1)
            print('...continue from', next_timestamp)
            df_new = get_futures_historical_data(client_me, symbol, kline_interval, start_str=str(next_timestamp))

            df_combined = df_old
            if df_new is not None and not df_new.empty:
                df_combined = pd.concat([df_old, df_new])

            if last_timestamp.date() != current_date.date():
                time.sleep(3)
                for matching_file in matching_files:
                    matching_file_path = os.path.join(saving_dirpath, matching_file)
                    current_date_str = current_date.strftime("%y%m%d")
                    if os.path.exists(matching_file_path) and not re.search(rf'{current_date_str}', matching_file):
                        os.remove(matching_file_path)

            df_combined.to_csv(saving_path, encoding='utf-8', index_label='datetime')

            return df_combined

    if not is_symbol:
        followup_futures_pairs_pkl_path = os.path.join(futures_dirpath, "followup_futures_pairs.pkl")
        all_usable_futures_pairs_pkl_path = os.path.join(futures_dirpath, "all_usable_futures_pairs.pkl")
        all_futures_usdt_pairs_pkl_path = os.path.join(futures_dirpath, "all_futures_usdt_pairs.pkl")

        if os.path.exists(followup_futures_pairs_pkl_path):
            followup_futures_pairs = save_load_pkl(followup_futures_pairs_pkl_path)
            # followup_futures_pairs.extend(['DYMUSDT'])
            followup_futures_pairs.sort()
            print("followup_futures_pairs[:20]:", followup_futures_pairs[:20])
            print("followup_futures_pairs[-20:]:", followup_futures_pairs[-20:])
            futures_pairs = followup_futures_pairs
            print('len(followup_futures_pairs):', len(followup_futures_pairs))
        else:
            futures_exchange_info = client_me.futures_exchange_info()
            futures_symbols = futures_exchange_info['symbols']

            # futures_pairs = [futures_symbol['symbol'] + f'_PERP' for futures_symbol in futures_symbols if futures_symbol['quoteAsset'] == 'USDT']
            all_trading_futures_pairs = [futures_symbol['symbol'] for futures_symbol in futures_symbols if futures_symbol['status'] == 'TRADING']
            usable_futures_pairs = [futures_pair for futures_pair in all_trading_futures_pairs if 'DOMUSDT' not in futures_pair and not re.search(r"_\d{6}$|USDC", futures_pair)]

            futures_usdt_pairs = [futures_pair for futures_pair in usable_futures_pairs if 'USDT' in futures_pair]
            futures_pairs = futures_usdt_pairs + ['ETHBTC']
            futures_pairs.sort()
            followup_futures_pairs = futures_pairs.copy()

            all_trading_futures_pairs.sort()
            usable_futures_pairs.sort()
            futures_usdt_pairs.sort()

            print('all_trading_futures_pairs:', len(all_trading_futures_pairs), all_trading_futures_pairs[:40])
            print('usable_futures_pairs:', len(usable_futures_pairs), usable_futures_pairs[:40])
            print('futures_usdt_pairs:', len(futures_usdt_pairs), futures_usdt_pairs[:40])

            save_load_pkl(all_usable_futures_pairs_pkl_path, usable_futures_pairs)
            save_load_pkl(all_futures_usdt_pairs_pkl_path, futures_usdt_pairs)

        COINS_DICT = {}
        if os.path.exists(coins_dict_pkl_path):
            COINS_DICT = save_load_pkl(coins_dict_pkl_path)

        for i, futures_pair in enumerate(futures_pairs):
            coin_dict = {}
            if COINS_DICT:
                if futures_pair not in COINS_DICT.keys():
                    COINS_DICT[futures_pair] = {}
                coin_dict = COINS_DICT[futures_pair]
            print('\n', i + 1, futures_pair)
            df = save_df_to_csv(futures_pair)

            if df is not None and not df.empty:
                # print(df.iloc[100])
                # print(df.loc['2024-11-14 05:53:00'])
                # print(df.index[100])
                max_high_row_index = df['high'].idxmax()
                min_low_row_index = df['low'].idxmin()
                futures_high = df.loc[max_high_row_index, 'high']
                futures_low = df.loc[min_low_row_index, 'low']

                coin_dict['futures_high'] = futures_high
                coin_dict['futures_low'] = futures_low

                try:
                    spot_pair = futures_pair
                    multiplier = 1
                    match = re.match(r"^(1[0]+)([A-Z]+)$", futures_pair)
                    if match and futures_pair not in ['1000SATSUSDT', '1000CATUSDT']:
                        multiplier = int(match.group(1))  # Convert the number part to an integer
                        spot_pair = match.group(2)  # The text part of the pair

                    print(11,spot_pair)
                    df_spot_1M = spot_prices.get_spot_historical_data(CLIENT, spot_pair, '1mon')
                    spot_max_high_row_index = df_spot_1M['high'].idxmax()
                    spot_min_low_row_index = df_spot_1M['low'].idxmin()
                    print(22,spot_pair)

                    if COINS_DICT:
                        pricePrecision = COINS_DICT[futures_pair]['pricePrecision']
                    else:
                        price_str = str(current_futures_price(futures_pair))
                        if '.' in price_str:
                            decimal_part = price_str.split('.')[1]  # Get the part after the decimal point
                            pricePrecision = len(decimal_part)
                        else:
                            pricePrecision = 0

                    coin_dict['pricePrecision'] = pricePrecision
                    coin_dict['spot_high'] = round_to_significant_figures(df_spot_1M['high'].max()*multiplier, pricePrecision)
                    coin_dict['spot_low'] = round_to_significant_figures(df_spot_1M['low'].min()*multiplier, pricePrecision)
                except:
                    coin_dict['spot_high'] = 0
                    coin_dict['spot_low'] = 10000000

                coin_dict['all_high'] = max(coin_dict['futures_high'], coin_dict['spot_high'])
                coin_dict['all_low'] = min(coin_dict['futures_low'], coin_dict['spot_low'])
                current_price = current_futures_price(futures_pair)

                print(f"Max high value at time {max_high_row_index if coin_dict['futures_high'] > coin_dict['spot_high'] else spot_max_high_row_index}: {coin_dict['all_high']}")
                print(f"Min low value at time {min_low_row_index if coin_dict['futures_low'] < coin_dict['spot_low'] else spot_min_low_row_index}:"
                      f" {coin_dict['all_low']} (max /{round(coin_dict['all_high']/coin_dict['all_low'], 1)})")
                print(f"Current price: {current_price} (min x{round(current_price/coin_dict['all_low'], 1)}) (max /{round(coin_dict['all_high']/current_price, 1)})")

                for window in [200, 100, 50, 25]:
                    coin_dict_extremes = {}; coin_dict_extremes2 = {}; coin_dict_extremes3 = {}
                    try:
                        df_ema = calculate.ema(df, window=window)
                        coin_dict_extremes = calculate.calculate_ema_dev_extremes(df_ema, 'ema' + str(window))
                        coin_dict_extremes2 = calculate.calculate_ema_trend_extremes(df_ema, 'ema' + str(window), trend_extension='_slope', hist_threshold=90)
                        coin_dict_extremes3 = calculate.calculate_ema_trend_extremes(df_ema, 'ema' + str(window), trend_extension='_pct_change', hist_threshold=90)
                    except:
                        pass
                    coin_dict.update({**coin_dict_extremes, **coin_dict_extremes2, **coin_dict_extremes3})

                if not COINS_DICT and os.path.exists(coins_dict_pkl_path):
                    COINS_DICT = save_load_pkl(coins_dict_pkl_path)
                COINS_DICT[futures_pair] = coin_dict
                save_load_pkl(coins_dict_pkl_path, COINS_DICT)

            # Save the list as a pickle file after removing the current pair
            followup_futures_pairs.remove(futures_pair)
            save_load_pkl(followup_futures_pairs_pkl_path, followup_futures_pairs)

        df_coins = get_all_futures_pairs_df()
        df_extremes = pd.DataFrame.from_dict(COINS_DICT, orient='index')

        # Merge or concatenate with the existing DataFrame
        df_coins = pd.concat([df_coins, df_extremes], axis=1)
        futures_pairs_stats_path = os.path.join('../../data/futures', "futures_pairs_stats.csv")
        df_coins.to_csv(futures_pairs_stats_path, encoding='UTF-8', index_label='futures_pair')

        # futures_usdt_pairs = [symbol['symbol'] for symbol in futures_symbols if symbol['quoteAsset'] == 'USDT']
        # diff_fut = list(set(futures_pairs) - set(futures_usdt_pairs))
        # print(6, len(futures_pairs), len(diff_fut), diff_fut)
    else:
        symbol = is_symbol
        df = save_df_to_csv(symbol)
        print(symbol, df[['open', 'high', 'low', 'close', 'volume']].head(1))
        print(symbol, df[['open', 'high', 'low', 'close', 'volume']].tail(1))
        print(df.size)

        # return df


def scan_futures_for_sma(client, time_frame=15, window=200):
    # Binance'taki tüm futures coinlerinin listesini alın
    futures_exchange_info = client.futures_exchange_info()

    symbols = futures_exchange_info['symbols']

    # USDT paritesindeki coinleri seçin
    usdt_pairs = [symbol['symbol'] for symbol in symbols if symbol['quoteAsset'] == 'USDT']

    filter_regex = re.compile(r'(UPUSDT|DOWNUSDT|BULLUSDT|BEARUSDT|DOMUSDT|\d)$|^(TOMO|SRM)')
    usdt_pairs = list(filter(lambda usdt_pair: not re.search(filter_regex, usdt_pair), usdt_pairs))

    # Fetch USDT coin data for the designated time interval, e.g. 15 minute.
    coin_data = {}
    invalid_list = []
    klines = ''
    for symbol in usdt_pairs:
        try:
            klines = client.get_klines(symbol=symbol, interval=INTERVAL_DICT[time_frame], limit=2 * window)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                               'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                               'taker_buy_quote_asset_volume', 'ignore'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            coin_data[symbol] = df.dropna()
        except BinanceAPIException as e:
            invalid_list.append(symbol)
            pass

    # print(f' -- invalid symbol list (size: {len(invalid_list)}): {invalid_list}')

    # Calculate SMA 200 and return df for the last 200 data point (or designated numer: sma_window)
    def calculate_sma(df):
        return df['close'].rolling(window=window).mean()

    # Coinleri SMA 200 ile filtreleme
    over_sma_coins_dict = {}
    sma_coins_dict = {}
    for symbol, df in coin_data.items():
        # for symbol, df in islice(coin_data.items(), 10):
        if len(df) >= window:  # En az 200 veri olmalıdır
            df['close'] = df['close'].astype(float)
            sma_series = calculate_sma(df)
            last_sma = calculate.round_long_decimals(float(sma_series.iloc[-1]))
            last_close_price = calculate.round_long_decimals(float(df['close'].iloc[-1]))
            dev_ratio = calculate.round_long_decimals((last_close_price - last_sma) * 100 / last_sma)
            sma_coins_dict[symbol] = {'last_sma': last_sma, 'last_close_price': last_close_price,
                                      'dev_ratio': dev_ratio}
            if last_close_price > last_sma:  # Son kapanış fiyatı SMA 200'ün üstündeyse
                over_sma_coins_dict[symbol] = {'last_sma': last_sma, 'last_close_price': last_close_price,
                                               'dev_ratio': dev_ratio}

    sma_coins_dict_list = [(key, value) for key, value in sma_coins_dict.items()]
    sma_coins_dict_list = sorted(sma_coins_dict_list, key=lambda x: x[1]['dev_ratio'], reverse=True)
    sma_coins_dict_ordered_dict = OrderedDict({x[0]: x[1] for x in sma_coins_dict_list})

    # Print results
    len1 = len(over_sma_coins_dict)
    len2 = len(sma_coins_dict_list)
    over_all_ratio = len1 * 100 / len2
    print(f"{len1} coins (" + f"{'{:.1f}%'.format(over_all_ratio)} of {len2}" + f") over SMA{window} in {time_frame} "
                                                                                f"%s time frame." % (
              'min' if time_frame in [1, 5, 15, 30] else 'hour'))
    for sma_tuple in sma_coins_dict_list:
        symbol = sma_tuple[0]
        last_sma = calculate.round_long_decimals(sma_tuple[1]['last_sma'], 4)
        last_close_price = calculate.round_long_decimals(sma_tuple[1]['last_close_price'], 4)
        dev_ratio = calculate.round_long_decimals(sma_tuple[1]['dev_ratio'], 4)
        dev_ratio = "{:.1f}%".format(dev_ratio)
        print(
            f"{'{:<12}'.format(symbol)}: last_sma:{'{:<11}'.format(last_sma)}| last close: {'{:<11}'.format(last_close_price)} dev: {dev_ratio}")

    # tuple_: (symbol, last_sma, last_close_price, dev_ratio)
    return sma_coins_dict_ordered_dict


class SingleCoin():
    def __init__(self, client, symbol='', time_frame=1, window=200, add_price_info_bool=True):
        if not re.search('USDT$|BUSD$|USDC$', symbol):
            symbol = symbol + 'USDT'
        self.client = client
        self.symbol = symbol
        self.window = window
        self.time_frame = time_frame
        self.df_symbol = self.get_df_symbol(time_frame, window)
        self.current_price = self.df_symbol['close'].iloc[-1]
        if add_price_info_bool:
            self.df_symbol = self.add_price_info(self.df_symbol, window)

    def get_df_symbol(self, time_frame=None, window=None):
        # # klines = FUTURES_CLIENT.continuous_klines(pair='BTCUSDT', interval='1h', contractType='PERPETUAL', startTime=start_date, endTime=end_date, limit=1500)
        # klines = FUTURES_CLIENT.klines(symbol='1000FLOKIUSDT', interval=INTERVAL_DICT[15], startTime=start_date, endTime=end_date, limit=1500)
        # df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        # df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        # df.set_index('datetime', inplace=True)
        # df.drop('timestamp', axis=1, inplace=True)

        if not time_frame:
            time_frame = self.time_frame
        if not window:
            window = self.window

        now = int(time.time() * 1000)
        start_date = int(datetime(2023, 1, 1).timestamp()) * 1000
        # end_date = int(datetime(2024, 1, 1).timestamp()) * 1000
        end_date = now
        # klines = self.client.klines(symbol=self.symbol, interval=INTERVAL_DICT[time_frame], startTime=start_date, endTime=end_date, limit=1500)
        klines = self.client.klines(symbol=self.symbol, interval=INTERVAL_DICT[time_frame], limit=window)
        df_symbol = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                  'quote_asset_volume',
                                                  'number_of_trades', 'taker_buy_base_asset_volume',
                                                  'taker_buy_quote_asset_volume', 'ignore'])
        df_symbol = df_symbol[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']]

        df_symbol['datetime'] = pd.to_datetime(df_symbol['timestamp'], unit='ms')
        df_symbol.set_index('datetime', inplace=True)
        df_symbol.drop('timestamp', axis=1, inplace=True)
        df_symbol = df_symbol.dropna()

        df_symbol['close'] = df_symbol['close'].astype(float)
        df_symbol['volume'] = df_symbol['volume'].astype(float)
        df_symbol['low'] = df_symbol['low'].astype(float)
        df_symbol['high'] = df_symbol['high'].astype(float)

        return df_symbol

    def add_price_info(self, df_symbol, window=None):
        if not window:
            window = self.window

        time_now = datetime.now()
        time_normalizing_factor = (df_symbol['close_time'].iloc[-1] - df_symbol['close_time'].iloc[-2]) / (int(time_now.timestamp() * 1000) - df_symbol['close_time'].iloc[-2])

        # find percent price change
        df_symbol['price_change'] = (df_symbol['close'].pct_change()).apply(lambda x: calculate.round_long_decimals(x, prec=4) if not pd.isna(x) else float('nan')) * 100
        df_symbol['price_diff'] = df_symbol['close'].diff()

        # find percent volume changes
        n_vol_series = df_symbol['volume'].copy()
        n_vol_series.iloc[-1] = n_vol_series.iloc[-1] * time_normalizing_factor
        df_symbol['vol_sma'] = n_vol_series.rolling(window=5).mean()
        df_symbol['vol_sma_diff'] = df_symbol['vol_sma'].diff()
        # df_symbol['vol_sma_change'] = df_symbol['vol_sma'].pct_change() * 100

        calculate.std(df_symbol, window=200)

        calculate.sma(df_symbol, window=200)
        calculate.sma(df_symbol, window=100)
        calculate.sma(df_symbol, window=50)

        calculate.ema(df_symbol, window=200)
        calculate.ema(df_symbol, window=100)
        calculate.ema(df_symbol, window=50)
        calculate.ema(df_symbol, window=25)

        calculate.rsi(df_symbol, window=7, rsi_column_name='rsi7')
        calculate.rsi(df_symbol, window=14, rsi_column_name='rsi14')
        calculate.rsi(df_symbol, window=21, rsi_column_name='rsi21')

        calculate.macd(df_symbol, short_ema_period=12, long_ema_period=26, window=9)

        # Calculate upper and lower Bollinger Bands
        calculate.bb(df_symbol)

        df_symbol.dropna(inplace=True)

        rounding_columns = ['price_change', 'vol_sma',
                            'std', 'std_diff', 'rsi7', 'rsi14', 'rsi21', 'rsi7_sma', 'rsi14_sma', 'rsi21_sma',
                            'rsi7_diff', 'rsi14_diff', 'rsi21_diff',
                            'macd', 'macd_sma', 'macd_diff',
                            'bb_std', 'bb_sma', 'lower_band', 'upper_band', 'bb_sma_diff', 'bb_std_diff']

        try:
            df_symbol[rounding_columns] = df_symbol[rounding_columns].map(lambda x: calculate.round_long_decimals(x, prec=5))
        except Exception as e:
            print(5555, self.symbol, e)
            print(df_symbol.iloc[-1])
            pass

        return df_symbol


def current_futures_price(symbol):
    # Make a request to the Binance API to get the latest price
    # Binance Futures API URL for Mark Price
    response = requests.get(f'https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}')

    if response.status_code == 200:
        data = response.json()
        # print(data.keys()) # ['symbol', 'markPrice', 'indexPrice', 'estimatedSettlePrice', 'lastFundingRate', 'interestRate', 'nextFundingTime', 'time']
        current_markPrice = calculate.round_long_decimals(float(data['markPrice']))  # Mark price
        current_indexPrice = calculate.round_long_decimals(float(data['indexPrice']))  # Mark price
        lastFundingRate = calculate.round_long_decimals(float(data['lastFundingRate']))  # Mark price
        interestRate = calculate.round_long_decimals(float(data['interestRate']))  # Mark price

        # if 0.085 > current_markPrice > 0.077:
        #     print(f"{symbol} :{round_to_significant_figures(current_markPrice, 3)}", end=' |')
        #     print(f"{symbol} :{round_to_significant_figures(current_indexPrice, 3)}", end=' |')
        #     print(f"{symbol} :{round_to_significant_figures(lastFundingRate*100, 3)}%", end=' |')
        #     print(f"{symbol} :{round_to_significant_figures(interestRate*100, 3)}")
        #
        #     return current_markPrice

        return current_markPrice
    else:
        print(f"Failed to fetch futures price. Status Code: {response.status_code}")


def plot_symbol(symbol, kline_interval, days_range=7, df=None, options=['price']):
    if df is not None or not df.empty:
        start_iloc_index = len(df) - 1 - int(days_range * 24 * 60) - 200
        df = df.iloc[start_iloc_index:]
    else:
        current_date = datetime.now(timezone.utc).replace(tzinfo=None)
        _start_date = current_date - timedelta(days=days_range + 200 * TIME_DICT[kline_interval] / 1440)
        # df = get_futures_record_data(symbol, kline_interval, start_date)
        df = get_futures_recent_data(CLIENT, symbol, kline_interval, _start_date)

    mean_price20 = round_to_significant_figures(df['close'][-20:].mean(), 2)
    print(111, mean_price20)
    plt.figure(figsize=(14, 8))

    if 'price' in options:
        df_price = df.iloc[200:]
        plt.plot(df_price.index, df_price['close'], label='close', color='skyblue')
        # plt.plot(df_price.index, df_price['high'], label='high', color='orange', linewidth=0.7)
        # plt.plot(df_price.index, df_price['low'], label='low', color='orange', linewidth=0.7)
        # plt.plot(df_price.index, df_price['price_diff'] + mean_price20, label='price_diff', color='gray', linestyle='--')
        # plt.plot(df_price.index, (df_price['price_change']+0.1) * mean_price20, label='price_change', color='black')

    if 'bb' in options:
        df_bb = calculate.bb(df).iloc[200:]
        plt.plot(df_bb.index, df_bb['upper_band'], label='ub', color='brown')
        plt.plot(df_bb.index, df_bb['lower_band'], label='lb', color='brown')
        plt.plot(df_bb.index, df_bb['bb_sma'], label='mb', color='lime')

    if 'ema' in options:
        window_list = [200, 100, 50]
        for window in window_list:
            ema_option = ''.join(['ema', str(window)])
            color = config.random_color()
            df_ema = calculate.ema(df, window, ema_option).iloc[200:]
            plt.plot(df_ema.index, df_ema[ema_option], label=ema_option, color=color, linestyle='-')
            # plt.plot(df.index, mean_price20 * (1 + df[option + '_dev_sma']/100), label=option + '_dev_sma', color=color, linestyle='--')
            # plt.plot(df.index, mean_price20 * (1 + df[option + '_dev_sma_diff']/100), label=option + '_dev_sma_diff', color=color, linestyle=':')

    if 'sma' in options:
        window_list = [200, 100, 50]
        for window in window_list:
            sma_option = ''.join(['sma', str(window)])
            color = config.random_color()
            df_sma = calculate.sma(df, window, sma_option).iloc[200:]
            plt.plot(df_sma.index, df_sma[sma_option], label=sma_option, color=color, linestyle='-')
            # plt.plot(df_sma.index, mean_price20 * (1 + df_sma[sma_option + '_dev_sma']/100), label=sma_option + '_dev_sma', color=color, linestyle='--')
            # plt.plot(df_sma.index, mean_price20 * (1 + df_sma[sma_option + '_dev_sma_diff']/100), label=sma_option + '_dev_sma_diff', color=color, linestyle=':')

    if 'rsi' in options:
        df_rsi = calculate.rsi(df).iloc[200:]
        plt.plot(df_rsi.index, (1700 / mean_price20) * (df_rsi['high'] - df_rsi['low'].min()), label='high', color='orange', linewidth=0.7)
        plt.plot(df_rsi.index, (1700 / mean_price20) * (df_rsi['low'] - df_rsi['low'].min()), label='low', color='orange', linewidth=0.7)
        # plt.plot(df.index, (1700/mean_price20/2) * (df['high'] + df['low'] - 2 * df['low'].min()), label='bb_sma', color='orange', linewidth=0.7)
        plt.plot(df_rsi.index, df_rsi['rsi7'], label='rsi7', color='green', linestyle='--')
        plt.plot(df_rsi.index, df_rsi['rsi14'], label='rsi14', color='gray', linestyle='--')
        plt.plot(df_rsi.index, df_rsi['rsi21'], label='rsi21', color='blue', linestyle='--')
        plt.plot(df_rsi.index, df_rsi['rsi7_sma'], label='rsi7_sma', color='green')
        plt.plot(df_rsi.index, df_rsi['rsi14_sma'], label='rsi14_sma', color='gray')
        plt.plot(df_rsi.index, df_rsi['rsi21_sma'], label='rsi21_sma', color='blue')
        # plt.plot(df_rsi.index, df_rsi['rsi7_diff'], label='rsi7_diff', color='green')
        # plt.plot(df_rsi.index, df_rsi['rsi14_diff'], label='rsi14_diff', color='gray')
        # plt.plot(df_rsi.index, df_rsi['rsi21_diff'], label='rsi21_diff', color='blue')

    if 'macd' in options:
        df_macd = calculate.macd(df).iloc[200:]
        plt.plot(df_macd.index, (1700 / mean_price20) * (df_macd['high'] - df_macd['low'].min()), label='high', color='orange', linewidth=0.7)
        plt.plot(df_macd.index, (1700 / mean_price20) * (df_macd['low'] - df_macd['low'].min()), label='low', color='orange', linewidth=0.7)
        # plt.plot(df_macd.index, (40/2) * (df_macd['high'] + df_macd['low'] - 2 * df_macd['low'].min()), label='bb_sma', color='orange', linewidth=0.7)
        plt.plot(df_macd.index, 5000 * df_macd['macd'], label='macd', color='brown', linestyle='--')
        plt.plot(df_macd.index, 5000 * df_macd['macd_sma'], label='macd_sma', color='orange', linestyle='--')
        # plt.plot(df_macd.index, mean_price20 * (1 + df_macd['macd_change'] / 100), label='macd_change', color='gray')
        plt.plot(df_macd.index, 5000 * df_macd['macd_diff'], label='macd_diff', color='red')
        plt.plot(df_macd.index, 5000 * df_macd['macd_sma_diff'], label='macd_sma_diff', color='magenta')

    plt.axhline(y=mean_price20, color='red', linestyle='--', label='mean_price20')
    # plt.axhline(y=order1, color='red', linestyle='--', label=order1)
    # plt.axhline(y=order2, color='green', linestyle='--', label=order2)

    plt.title(f'{symbol}_{config.fetch_real_time_frame(kline_interval)}')
    plt.xlabel('time')
    plt.ylabel('price')
    # plt.xticks(df.index[::5], [str(idx)[-11:-3] for idx in df.index[::5]], fontsize=6)
    # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=20))
    df = df.iloc[200:]
    tick_indices = df.index[::len(df) // 20]  # Roughly 20 ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(tick_indices, rotation=45, fontsize=6)  # Rotate labels for better readability
    plt.text(0.92, 0.6, mean_price20, fontsize=10, color="green", transform=plt.gca().transAxes)

    print(df.index[-1])

    plt.legend()
    plt.grid(False)
    # plt.show(block=False)
    # plt.rcParams['figure.figsize'] = (14.5, 8.5)
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()


def plot_periodical_average(symbol, period_dates='tuesdays', years_back=6, df=None):
    # period = mondays (15m), januaries (4h), 2024 (3d), day (15m), week (1h), month(4h), year (3d)
    current_date = datetime.now(timezone.utc).replace(tzinfo=None)
    _start_date = current_date - relativedelta(years=6, months=1)
    if df is None or df.empty:
        df = get_futures_record_data(symbol)

    df = df[df.index > start_date]

    weekday_dict = {'mondays':0, 'tuesdays':1, 'wednesdays':2, 'thursdays':3, 'fridays':4, 'saturdays':5, 'sundays':6}
    month_dict = {'januaries':1, 'februaries':2, 'marches':3, 'aprils':4, 'mays':5, 'junes':6, 'julies':7, 'augusts':8, 'septembers':9, 'octobers':10, 'novembers':11, 'decembers':12}
    # _start_date = datetime(2024, 11, 20, 18, 30, 0)
    # # start_date = datetime(2024, 11, 20, 18, 30, 0)
    # yester_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    yester_yester_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    yester_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    current_date = datetime.now(timezone.utc).replace(tzinfo=None)
    kline_interval = 1
    start_end_date_tuples = []
    ticks = []

    if period_dates in ['day', 'mondays', 'tuesdays', 'wednesdays', 'thursdays', 'fridays', 'saturdays', 'sundays']:
        kline_interval = 15
        ticks = list(f'{x}:00' for x in range(0,24))

        if period_dates == 'day':
            period_start_date = yester_yester_date.replace(hour=0, minute=0, second=0, microsecond=0)
            time_shift_days = 1
        else:
            days_to_subtract = 7 if current_date.weekday() == weekday_dict[period_dates] else (7 + current_date.weekday() - weekday_dict[period_dates]) % 7
            last_weekday = current_date - timedelta(days=days_to_subtract)
            period_start_date = last_weekday.replace(hour=0, minute=0, second=0, microsecond=0)
            time_shift_days = 7
        period_end_date = period_start_date + timedelta(days=1) - timedelta(minutes=1)

        for _ in range((years_back * 365) // time_shift_days + 1):
            start_end_date_tuples.append((period_start_date, period_end_date))
            period_start_date -= timedelta(days=time_shift_days)
            period_end_date -= timedelta(days=time_shift_days)

    elif period_dates == 'week':
        kline_interval = 60
        days_to_subtract = current_date.weekday()
        last_monday = current_date - timedelta(days=days_to_subtract)
        last_monday_start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        period_start_date = last_monday_start - timedelta(days=7)
        period_end_date = last_monday_start - timedelta(minutes=1)
        time_shift_days = 7
        ticks = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

        for _ in range((years_back * 365) // 7 + 1):  # Include the current month
            start_end_date_tuples.append((period_start_date, period_end_date))
            period_start_date -= relativedelta(days=time_shift_days)
            period_end_date -= relativedelta(days=time_shift_days)

    elif period_dates in ['month', 'januaries', 'februaries', 'marches', 'aprils', 'mays', 'junes', 'julies', 'augusts', 'septembers', 'octobers', 'novembers', 'decembers']:
        kline_interval = 4
        first_day_of_this_month = current_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_year = current_date.year
        ticks = list(x for x in range(1,31))

        if period_dates == 'month':
            period_start_date = first_day_of_this_month - relativedelta(months=1)
            period_end_date = first_day_of_this_month - timedelta(minutes=1)
            time_shift_months = 1
        else:
            asked_month = month_dict[period_dates]
            if current_date.month <= asked_month:
                current_year -= 1
            period_start_date = datetime(current_year, asked_month, 1).replace(tzinfo=None)
            period_end_date = period_start_date + relativedelta(months=1) - timedelta(minutes=1)
            time_shift_months = 12

        for _ in range((years_back * 12) // time_shift_months):  # Include the current month
            start_end_date_tuples.append((period_start_date, period_end_date))
            period_start_date -= relativedelta(months=time_shift_months)
            period_end_date = period_start_date + relativedelta(months=1) - timedelta(minutes=1)

    elif re.search(r'[0-9]{4}', str(period_dates)) or period_dates == 'year':
        kline_interval = 24
        first_day_of_this_year = current_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        ticks = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

        if period_dates == 'year':
            period_start_date = first_day_of_this_year - relativedelta(years=1)
            period_end_date = first_day_of_this_year - timedelta(minutes=1)
            for _ in range(years_back):
                start_end_date_tuples.append((period_start_date, period_end_date))
                period_start_date -= relativedelta(years=1)
                period_end_date -= relativedelta(years=1)
        else:
            period_start_date = datetime(period_dates, 1, 1).replace(tzinfo=None)
            period_end_date = period_start_date + relativedelta(years=1) - timedelta(minutes=1)
            start_end_date_tuples.append((period_start_date, period_end_date))

    df = get_new_time_interval(kline_interval, '', df)
    df_partial_list = []

    for period_start_date, period_end_date in start_end_date_tuples:
        df_partial = df[(df.index >= period_start_date) & (df.index <= period_end_date)].copy()
        if not df_partial.empty:
            mean_close = df_partial['close'].mean()
            if period_start_date > datetime(2024, 12, 1):
                print(period_start_date, mean_close)
            df_partial['deviation'] = ((df_partial['close'] - mean_close) / mean_close) * 100
            # print(11, df_partial.tail(12))
            # print(22, df_partial.index[0], period_start_date, period_end_date)
            if df_partial.index[0] <= period_start_date + timedelta(days=2):
                df_partial_list.append(df_partial)

    num_rows = min(len(df_partial) for df_partial in df_partial_list)  # Assuming all df_partials have the same number of rows
    df_final = pd.DataFrame(index=range(num_rows))  # Use iloc-based indices
    df_final['mean_band'] = 0.0
    df_final['upper_band'] = 0.0
    df_final['lower_band'] = 0.0

    for i in range(num_rows):
        row_values = np.array([df_partial.iloc[i]['close'] for df_partial in df_partial_list])
        row_mean = row_values.mean()
        row_std = row_values.std()

        df_final.loc[i, 'mean_band'] = row_mean
        df_final.loc[i, 'upper_band'] = row_mean + row_std
        df_final.loc[i, 'lower_band'] = row_mean - row_std

    plt.figure(figsize=(14, 8))

    df_price = df.iloc[200:]
    plt.plot(df_final.index, df_final['mean_band'], label='mean_band', color='skyblue')
    plt.plot(df_final.index, df_final['upper_band'], label='upper_band', color='limegreen')
    plt.plot(df_final.index, df_final['lower_band'], label='lower_band', color='limegreen')
    # plt.plot(df_price.index, df_price['high'], label='high', color='orange', linewidth=0.7)
    # plt.plot(df_price.index, df_price['low'], label='low', color='orange', linewidth=0.7)
    # plt.plot(df_price.index, df_price['price_diff'] + mean_price20, label='price_diff', color='gray', linestyle='--')
    # plt.plot(df_price.index, (df_price['price_change']+0.1) * mean_price20, label='price_change', color='black')

    # plt.axhline(y=mean_price20, color='red', linestyle='--', label='mean_price20')
    # plt.axhline(y=order1, color='red', linestyle='--', label=order1)
    # plt.axhline(y=order2, color='green', linestyle='--', label=order2)

    plt.title(f'{symbol}_{period_dates}')
    plt.xlabel('Time')
    plt.ylabel('Price')

    tick_positions = [int(i * len(df_final) / len(ticks)) for i in range(len(ticks))]
    plt.xticks(ticks=[df_final.index[pos] for pos in tick_positions], labels=ticks, rotation=45, fontsize=10)
    # plt.xticks(ticks=range(len(ticks)), labels=ticks, rotation=45, fontsize=10)

    '''plt.xticks(df_final.index[::5], [str(idx)[-11:-3] for idx in df_final.index[::5]], fontsize=6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=20))'''

    '''tick_indices = df.index[::len(df) // 20]  # Roughly 20 ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(tick_indices, rotation=45, fontsize=6)  # Rotate labels for better readability'''

    # plt.text(0.92, 0.6, mean_price20, fontsize=10, color="green", transform=plt.gca().transAxes)

    plt.legend()
    plt.grid(False)
    # plt.show(block=False)
    # plt.rcParams['figure.figsize'] = (14.5, 8.5)
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()


def round_to_significant_figures(num, sig_figs):
    if num == 0:
        return 0
    if abs(num) < 1:
        scale = 10 ** (sig_figs - int(math.floor(math.log10(abs(num)))) - 1)
        return round(num * scale) / scale
    elif abs(num) < 10:
        return round(num, sig_figs)
    else:
        return round(num, sig_figs)


def save_load_pkl(filepath, obj=None):
    if obj is not None:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    else:
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                fetched_obj = pickle.load(f)
                return fetched_obj
        else:
            print(f'There is no file in the path {filepath}')


def add_info_to_futures_pairs(kline_interval='1m', days_range=90):
    df = get_all_futures_pairs_df()
    coin_list = list(df.index)
    # print(coin_list)
    saving_dirpath = os.path.join(futures_hist_dirpath, kline_interval)
    saved_coin_list = [match.group() for file in os.listdir(saving_dirpath) if
                       (match := re.match(r'^[^_]+', file)) and os.path.isfile(os.path.join(saving_dirpath, file))]
    print(saved_coin_list)

    for coin in saved_coin_list:
        # df_coin = get_futures_record_data(coin)
        df_coin = get_futures_record_data(coin, '1m', current_date - timedelta(days=7))
        cp = current_futures_price(coin)

        if not (0.5 <= cp / df_coin.iloc[-1]['close'] <= 2):
            print(coin, cp, df_coin.index[-1], df_coin.iloc[-1]['close'])
        '''
        cp = current_futures_price(coin)
        if cp and 0.083 > cp > 0.079:
            print(coin)
        '''
        '''
        if coin == 'ETHUSDT':
            df_coin = get_futures_record_data(coin, '1m') # , current_date-timedelta(days=90)
            if df_coin is not None and not df_coin.empty:
                # print(df_coin.head(3))
                window = 200
                ema_col_name = 'emab' + str(window)
                ema_dev_col_name = ema_col_name + '_dev'
                df_coin = calculate.emab(df_coin, window, ema_col_name)
                # df_coin = calculate.ema(df_coin, 100, 'ema100')
                print(df_coin[['open', 'high', 'low', 'close', 'volume']].head(3))
                print(df_coin[['open', 'high', 'low', 'close', 'volume']].tail(3))

                max_plus_ema_dev_index = df_coin[ema_dev_col_name].idxmax()
                min_minus_ema_dev_index = df_coin[ema_dev_col_name].idxmin()
                print(f"Ema when max plus_ema_dev at time {max_plus_ema_dev_index}: {round(df_coin.loc[max_plus_ema_dev_index, ema_col_name], 2)}")
                print(f"High at time {max_plus_ema_dev_index}: {df_coin.loc[max_plus_ema_dev_index, 'high']}")
                print('Highest ema_dev(%):', round(df_coin[ema_dev_col_name].max(), 2))
                print('Highest high:', round(df_coin['high'].max(), 2))
                print(f"Ema when min minus_ema_dev at time {min_minus_ema_dev_index}: {round(df_coin.loc[min_minus_ema_dev_index, ema_col_name], 2)}")
                print(f"Low at time {min_minus_ema_dev_index}: {df_coin.loc[min_minus_ema_dev_index, 'low']}")
                print('Lowest ema_dev(%):', round(df_coin[ema_dev_col_name].min(), 2))
                print('Lowest low:', round(df_coin['low'].min(), 2))

                positive_values = df_coin[ema_dev_col_name][df_coin[ema_dev_col_name] > 0]
                negative_values = df_coin[ema_dev_col_name][df_coin[ema_dev_col_name] < 0]

                print('Plus values:', len(positive_values))
                print('Minus values:', len(negative_values))

                get_column_frequent_extremes(df_coin, ema_dev_col_name)
        # '''


if __name__ == '__main__':

    print()

    if os.path.exists(coins_dict_pkl_path):
        COINS_DICT = save_load_pkl(coins_dict_pkl_path)

    start_date = datetime(2009, 1, 1)
    yester_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    current_date = datetime.now(timezone.utc).replace(tzinfo=None)

    save_futures_historical_data(CLIENT, 1,'', str(start_date), str(current_date))

    c0 = save_load_pkl(coins_dict_pkl_path)
    c1 = c0.copy()
    for k, v in c1.items():
        if not re.search(r'[A-Za-z]', str(k)):
            print(k)
            del c0[k]
    save_load_pkl(coins_dict_pkl_path, c0)
    c0 = save_load_pkl(coins_dict_pkl_path)
    print(len(c0))


