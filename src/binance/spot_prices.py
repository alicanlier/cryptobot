import pickle
import re, os, shutil
import pandas as pd
import decimal

import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timezone, timedelta, time
import config.config as config
import utility.calculation_tools as calculate
import mplfinance as mpf


api_key_dict = config.fetch_credentials()
CLIENT = Client(api_key_dict['API_KEY'], api_key_dict['API_SECRET'])
INTERVAL_DICT = config.fetch_interval_dict(CLIENT)
TIME_DICT = config.time_dict()

pd.set_option('display.max_columns', None)


def get_all_spot_pairs_df(_spot_dirpath='../../data/spot', quoteAsset=None):
    spot_dirpath = _spot_dirpath
    spot_pairs_stats_path = os.path.join(spot_dirpath, "spot_pairs_stats.csv")
    spot_exchange_info = CLIENT.get_exchange_info()

    if os.path.exists(spot_pairs_stats_path):
        return pd.read_csv(spot_pairs_stats_path, index_col='spot_pair')
    else:
        # Filter for trading spot pairs
        if quoteAsset:
            symbol_info_list = [item for item in spot_exchange_info['symbols'] if item['status'] == 'TRADING' and item['quoteAsset'] == quoteAsset]
        else:
            symbol_info_list = [item for item in spot_exchange_info['symbols'] if item['status'] == 'TRADING']

        # Filter out unnecessary pairs ending with USDT
        filter_regex = re.compile(r'(UPUSDT|DOWNUSDT|BULLUSDT|BEARUSDT)$')
        symbol_info_list = list(filter(lambda x: not re.search(filter_regex, x['symbol']), symbol_info_list))

        symbol_info_list = sorted(symbol_info_list, key=lambda x: x['symbol'])

        # Create a DataFrame with selected details
        df = pd.DataFrame(symbol_info_list, columns=['symbol', 'baseAsset', 'quoteAsset', 'baseAssetPrecision', 'quotePrecision'])
        df.rename(columns={'symbol': 'spot_pair'}, inplace=True)
        df.set_index('spot_pair', inplace=True)

        # Save to CSV for future use
        os.makedirs(spot_dirpath, exist_ok=True)
        df.to_csv(spot_pairs_stats_path, encoding='UTF-8')

        return df


def get_spot_historical_data(client_me, symbol, _kline_interval='1m', start_str=None, end_str=None):
    kline_interval = INTERVAL_DICT[_kline_interval]
    try:
        klines = client_me.get_historical_klines(symbol, kline_interval, start_str, end_str)
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


def save_spot_historical_data(client_me, _kline_interval, is_symbol='', _start_str=None, _end_str=None):
    kline_interval = config.fetch_interval_dict(client_me)[_kline_interval]
    saving_dirpath = os.path.join(spot_hist_dirpath, kline_interval)

    # Generate file path to save the data in a csv file
    def get_saving_path(symbol):
        if not os.path.exists(saving_dirpath):
            os.makedirs(saving_dirpath)

        return os.path.join(saving_dirpath, symbol + '_SPOT_' + current_date.strftime("%y%m%d") + '.csv')

    def save_df_to_csv(symbol):
        saving_path = get_saving_path(symbol)
        pattern = rf"(?<!\w){symbol}"
        matching_files = [file for file in os.listdir(saving_dirpath) if os.path.isfile(os.path.join(saving_dirpath, file)) and re.search(pattern, file)]
        print("matching_files: ", matching_files)

        if not matching_files:
            df = get_spot_historical_data(client_me, symbol, kline_interval, _start_str, _end_str)
            if df is not None and not df.empty:
                df.to_csv(saving_path, encoding='utf-8')
                return df
            else:
                print("...No historical klines info for", symbol)
        # else:
        #     df_old = get_spot_record_data(symbol)
        #     # print(df_old.tail(3))
        #     # last_timestamp = df_old.index[-1]
        #     last_timestamp = df_old.iloc[-1].name
        #     next_timestamp = last_timestamp + timedelta(minutes=1)
        #     print(next_timestamp)
        #     df_new = get_spot_historical_data(client_me, symbol, kline_interval, start_str=str(next_timestamp))
        #
        #     df_combined = df_old
        #     if df_new is not None and not df_new.empty:
        #         df_combined = pd.concat([df_old, df_new])
        #
        #     if last_timestamp.date() != current_date.date():
        #         time.sleep(3)
        #         for matching_file in matching_files:
        #             matching_file_path = os.path.join(saving_dirpath, matching_file)
        #             current_date_str = current_date.strftime("%y%m%d")
        #             if os.path.exists(matching_file_path) and not re.search(rf'{current_date_str}', matching_file):
        #                 os.remove(matching_file_path)
        #
        #     df_combined.to_csv(saving_path, encoding='utf-8')
        #
        #     return df_combined

    if not is_symbol:
        followup_spot_pairs_pkl_path = os.path.join(spot_dirpath, "followup_spot_pairs.pkl")
        all_usable_spot_pairs_pkl_path = os.path.join(spot_dirpath, "all_usable_spot_pairs.pkl")
        all_spot_usdt_pairs_pkl_path = os.path.join(spot_dirpath, "all_spot_usdt_pairs.pkl")

        if os.path.exists(followup_spot_pairs_pkl_path):
            followup_spot_pairs = save_load_pkl(followup_spot_pairs_pkl_path)
            # followup_spot_pairs.extend(['DYMUSDT'])
            followup_spot_pairs.sort()
            print("followup_spot_pairs[:20]:", followup_spot_pairs[:20])
            print("followup_spot_pairs[-20:]:", followup_spot_pairs[-20:])
            spot_pairs = followup_spot_pairs
            print('len(followup_spot_pairs):', len(followup_spot_pairs))
        else:
            spot_exchange_info = client_me.get_exchange_info()
            spot_symbols = spot_exchange_info['symbols']

            # spot_pairs = [spot_symbol['symbol'] for spot_symbol in spot_symbols if spot_symbol['quoteAsset'] in ['USDT', 'BTC', 'ETH', 'BNB']]
            all_trading_spot_pairs = [spot_symbol['symbol'] for spot_symbol in spot_symbols if spot_symbol['status'] == 'TRADING']
            usable_spot_pairs = [spot_pair for spot_pair in all_trading_spot_pairs if 'DOMUSDT' not in spot_pair and not re.search(r"_\d{6}$|USDC", spot_pair)]

            spot_usdt_pairs = [spot_pair for spot_pair in usable_spot_pairs if 'USDT' in spot_pair]
            spot_pairs = spot_usdt_pairs + ['ETHBTC']
            spot_pairs.sort()
            followup_spot_pairs = spot_pairs.copy()

            all_trading_spot_pairs.sort()
            usable_spot_pairs.sort()
            spot_usdt_pairs.sort()

            print('all_trading_spot_pairs:', len(all_trading_spot_pairs), all_trading_spot_pairs[:40])
            print('usable_spot_pairs:', len(usable_spot_pairs), usable_spot_pairs[:40])
            print('spot_usdt_pairs:', len(spot_usdt_pairs), spot_usdt_pairs[:40])

            save_load_pkl(all_usable_spot_pairs_pkl_path, usable_spot_pairs)
            save_load_pkl(all_spot_usdt_pairs_pkl_path, spot_usdt_pairs)

        COINS_DICT = {}
        for i, spot_pair in enumerate(spot_pairs):
            coin_dict = {}
            print('\n', i + 1, spot_pair)
            df = save_df_to_csv(spot_pair)

            if df is not None and not df.empty:
                # print(df.iloc[100])
                # print(df.loc['2024-11-14 05:53:00'])
                # print(df.index[100])
                max_high_row_index = df['high'].idxmax()
                min_low_row_index = df['low'].idxmin()
                print(f"Max high value at time {max_high_row_index}: {df.loc[max_high_row_index, 'high']}")
                print(f"Min low value at time {min_low_row_index}: {df.loc[min_low_row_index, 'low']}")

                coin_dict['spot_high'] = df.loc[max_high_row_index, 'high']
                coin_dict['spot_low'] = df.loc[min_low_row_index, 'low']

                for window in [200, 100, 50]:
                    coin_dict_extremes = {}
                    try:
                        df_ema = calculate.ema(df, window=window)
                        coin_dict_extremes = calculate.calculate_ema_dev_extremes(df_ema, 'ema' + str(window))
                    except:
                        pass
                    coin_dict.update(coin_dict_extremes)

                coins_dict_pkl_path = '../../data/spot/coins_dict.pkl'
                if not COINS_DICT and os.path.exists(coins_dict_pkl_path):
                    COINS_DICT = save_load_pkl(coins_dict_pkl_path)
                COINS_DICT[spot_pair] = coin_dict
                save_load_pkl(coins_dict_pkl_path, COINS_DICT)

            # Save the list as a pickle file after removing the current pair
            followup_spot_pairs.remove(spot_pair)
            save_load_pkl(followup_spot_pairs_pkl_path, followup_spot_pairs)

        df_coins = get_all_spot_pairs_df()
        df_extremes = pd.DataFrame.from_dict(COINS_DICT, orient='index')

        # Merge or concatenate with the existing DataFrame
        df_coins = pd.concat([df_coins, df_extremes], axis=1)
        spot_pairs_stats_path = os.path.join('../../data/spot', "spot_pairs_stats.csv")
        df_coins.to_csv(spot_pairs_stats_path, encoding='UTF-8')

        # spot_usdt_pairs = [symbol['symbol'] for symbol in spot_symbols if symbol['quoteAsset'] == 'USDT']
        # diff_fut = list(set(spot_pairs) - set(spot_usdt_pairs))
        # print(6, len(spot_pairs), len(diff_fut), diff_fut)
    else:
        symbol = is_symbol
        df = save_df_to_csv(symbol)
        print(symbol, df[['open', 'high', 'low', 'close', 'volume']].head(1))
        print(symbol, df[['open', 'high', 'low', 'close', 'volume']].tail(1))
        print(df.size)

        # return df


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


def fetch_all_spot_pairs(currency='USDT', refresh=False):
    df_pairs = pd.DataFrame()
    spot_pairs = []

    all_pairs_filepath = '../../data/all_pairs.csv'
    if os.path.exists(all_pairs_filepath):
        df_pairs = pd.read_csv(all_pairs_filepath)
        spot_pairs = df_pairs['USDT_spot_pairs'].tolist()

    if not os.path.exists(all_pairs_filepath) or refresh:
        # Binance'taki tüm coinlerin listesini alın
        spot_exchange_info = CLIENT.get_exchange_info()
        spot_symbols = spot_exchange_info['symbols']

        # Filter out inactive coins
        spot_symbols = [spot_symbol for spot_symbol in spot_symbols if spot_symbol['status'] == 'TRADING']

        # Filter out unnecessary pairs ending with USDT
        filter_regex = re.compile(r'(UPUSDT|DOWNUSDT|BULLUSDT|BEARUSDT)$')
        spot_symbols = list(filter(lambda spot_symbol: not re.search(filter_regex, spot_symbol['symbol']), spot_symbols))

        # Filter out unrelated currency pairs
        if currency:
            spot_symbols = [spot_symbol for spot_symbol in spot_symbols if spot_symbol['quoteAsset'] == currency]
        else:
            spot_symbols = [spot_symbol for spot_symbol in spot_symbols if spot_symbol['quoteAsset'] in ['USDT', 'BTC', 'ETH', 'BNB']]

        # Extract pair name list
        spot_pairs = [spot_symbol['symbol'] for spot_symbol in spot_symbols]

        df_pairs['USDT_spot_pairs'] = spot_pairs

    df_pairs.to_csv(all_pairs_filepath, encoding='utf-8', index_label=False, index=False)

    return spot_pairs


def fetch_all_spot_data(time_frame=15, currency='USDT', limit=200, start_date_str=''):
    current_date = datetime.now()
    start_date = datetime.strptime(start_date_str, "%y%m%d") if start_date_str else ''

    spot_pairs = fetch_all_spot_pairs(currency)

    # USDT paritesindeki coinlerin 15 dakikalık grafiğini alın
    coin_data = {}
    for symbol in spot_pairs:
        # print(symbol)
        if start_date:
            klines = CLIENT.get_historical_klines(symbol=symbol, interval=INTERVAL_DICT[time_frame], limit=limit)
            # klines = CLIENT.get_historical_klines(symbol=symbol, interval=INTERVAL_DICT[time_frame], start_str=start_date, end_str=current_date)
        else:
            klines = CLIENT.get_historical_klines(symbol=symbol, interval=INTERVAL_DICT[time_frame], limit=limit)
            # klines = CLIENT.get_historical_klines(symbol=symbol, interval=INTERVAL_DICT[time_frame], start_str="Jan 1, 2010")
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Choose numeric columns to avoid timestamp column
        for column in ['open', 'high', 'low', 'close', 'volume']:
            df[column] = df[column].astype(float)

        coin_data[symbol] = df

    return coin_data


def plot_graph(df):
    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)

    # Plot the candlestick chart
    mpf.plot(df, type='candle', style='charles', volume=True)


def save_all_spot_data(time_frame=1, currency='USDT', start_date='', refresh_data=False, *args):
    current_date = datetime.now()
    upper_dirpath = '../../data/spot_hist_data/'
    dirpath = upper_dirpath + current_date.strftime("%y%m%d")
    coin_data = fetch_all_spot_data(time_frame=time_frame, currency=currency)

    # Fetch functions by args elements, e.g. 'macd', 'rsi', 'bb', 'ema', 'sma'
    def apply_extra_operations(df):
        function_dict = config.operation_dict(calculate)
        if args:
            series_list = list(map(lambda func: func(df), function_dict.values()))
            for series in series_list:
                df = pd.concat([df, series], axis=1)
            return df

    # Generate file path to save the data in a csv file per coin
    for symbol, df in coin_data.items():
        if args:
            df = apply_extra_operations(df)

        saving_path = dirpath + '/' + symbol + '.csv'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        df.to_csv(saving_path, encoding='utf-8')

    if refresh_data:
        for root, dirs, files in os.walk(upper_dirpath):
            for dir in dirs:
                if re.search(r'^\d{6}$', dir):
                    dir_date = datetime.strptime(dir, "%y%m%d")
                    if dir_date < current_date:
                        older_dirpath = os.path.join(root, dir)
                        shutil.rmtree(older_dirpath)

        print(f'{len(coin_data)} coin data has been replaced by recent data.')
        return

    print(f'{len(coin_data)} coin data has been saved.')


def fetch_single_spot_data(symbol, time_frame=15, limit=400):
    klines = CLIENT.get_klines(symbol=symbol, interval=INTERVAL_DICT[time_frame], limit=limit)
    df_symbol = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df_symbol['timestamp'] = pd.to_datetime(df_symbol['timestamp'], unit='ms')
    df_symbol.set_index('timestamp', inplace=True)
    df_symbol = df_symbol.dropna()

    # Choose numeric columns to avoid timestamp column
    for column in ['open', 'high', 'low', 'close', 'volume']:
        df_symbol[column] = df_symbol[column].astype(float)

    sma = calculate.sma(df_symbol['close'])
    last_sma = sma.iloc[-1]

    ema = calculate.ema(df_symbol['close'])
    last_ema = ema.iloc[-1]

    # Calculate upper and lower Bollinger Bands
    df_bb = calculate.bb(df_symbol['close'], 20)
    last_lower_band = df_bb['lower_band'].iloc[-1]
    last_upper_band = df_bb['upper_band'].iloc[-1]
    # print(df_bb)

    df_rsi = calculate.rsi(df_symbol['close'])
    last_rsi = df_rsi['rsi'].iloc[-1]

    last_close_price = df_symbol['close'].iloc[-1]
    minus_2_close_price = df_symbol['close'].iloc[-2]
    last_high_price = df_symbol['high'].iloc[-1]
    last_low_price = df_symbol['low'].iloc[-1]

    sma_deviation = (last_close_price - last_sma) * 100 / last_sma
    ema_deviation = (last_close_price - last_ema) * 100 / last_ema
    last_price_change = (last_close_price - minus_2_close_price) * 100 / minus_2_close_price
    last_volatility = 9

    price_dict = {'last_sma':last_sma, 'last_close_price':last_close_price, 'sma_deviation':sma_deviation}
    price_dict2 = {'last_ema':last_ema, 'last_lower_band':last_lower_band, 'last_upper_band':last_upper_band}
    price_dict.update(price_dict2)

    return price_dict

def round_to_last4(num, prec=4):
    # Convert the float number to a Decimal with the appropriate precision
    decimal.getcontext().prec = prec
    decimal_num = decimal.Decimal(str(num))

    # Round the Decimal number to its last 4 nonzero digits
    rounded_decimal_num = decimal_num.normalize()

    return float(rounded_decimal_num)


def current_spot_price(symbol):
    # Make a request to the Binance API to get the latest price
    response = requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}')

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        current_price = calculate.round_long_decimals(float(data['price']))
        print(f"{symbol}:{current_price}", end=' |')
        return current_price
    else:
        print(f"Failed to fetch price data from Binance API for {symbol}")


if __name__ == '__main__':
    COINS_DICT = {}
    # spot_hist_dirpath = '../../data/spot_hist'
    spot_hist_dirpath = 'D:/crypto_bot/data/spot_hist'
    spot_hist_dirpath_old = 'D:/crypto_bot/data/spot_hist/old'
    spot_dirpath = '../../data/spot'
    spot_pairs_stats_path = os.path.join(spot_dirpath, "spot_pairs_stats.csv")
    followup_spot_pairs_pkl_path = os.path.join(spot_dirpath, "followup_spot_pairs.pkl")
    all_usable_spot_pairs_pkl_path = os.path.join(spot_dirpath, "all_usable_spot_pairs.pkl")
    all_spot_usdt_pairs_pkl_path = os.path.join(spot_dirpath, "all_spot_usdt_pairs.pkl")
    print()

    # Get the current date/time and previous day's date
    start_date_0 = datetime(2009, 1, 1)
    # start_date = datetime(2024, 11, 20, 18, 30, 0)
    start_date = start_date_0
    yester_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    current_date = datetime.now(timezone.utc).replace(tzinfo=None)

    # save_spot_historical_data(CLIENT, 1, '', str(start_date), str(current_date))

    # coin_data = fetch_all_spot_data(time_frame)
    # print(fetch_all_spot_pairs())
    print(get_all_spot_pairs_df().head())

    # calculate.vol_change_ranking(coin_data, 200, 0.1, '-')
    # calculate.merge_tools(coin_data, real_time_frame, 200, 0.1, '-', ['rsi', 'macd', 'ema'])
    # calculate.sma_ema_common_ranking(coin_data)

    # ratio = CLIENT.futures_position_information(symbol='XRPUSDT')
    # print(ratio)

    # print(fetch_single_coin_data('BNBUSDT'))
    # save_all_spot_data(1, 'USDT', '111111')
    # fetch_all_spot_data(15, '')
    # start_date_str = '240205'
    # current_date = datetime.now()
    # start_date = datetime.strptime(start_date_str, "%y%m%d")
    # print(current_date, start_date)
    #
    # timestamp1 = current_date.timestamp()*1000
    # timestamp2 = start_date.timestamp()*1000
    #
    # milliseconds_since_epoch = int(time.time() * 1000)
    #
    # # Print the converted timestamps
    # print("Timestamp 1:", timestamp1)
    # print("Timestamp 2:", timestamp2)

    # Get the current date and time
    # current_date = datetime.now()

    # Format the current date and time as a string
    # current_date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")

    # print("Current date and time as string:", current_date_str)
