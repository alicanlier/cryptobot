import os
from pathlib import Path

from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.um_futures import UMFutures
from binance.error import ClientError
# from binance_f import RequestClient
# from binance_f.model.constant import PositionSide
import logging
import schedule
from datetime import datetime, timezone, timedelta
import time
import random
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import config.config as config
import utility.calculation_tools as calculate
import futures_prices as futures
import utility.strategy_tools as strategy
from futures_prices import save_load_pkl
from src.binance.futures_prices import SingleCoin
from scipy.signal import argrelextrema

from utility.calculation_tools import get_saved_ema_extremes

# Connect to Binance API
api_key_dict = config.fetch_credentials()
API_KEY, API_SECRET = api_key_dict['API_KEY'], api_key_dict['API_SECRET']
CLIENT = Client(API_KEY, API_SECRET)
INTERVAL_DICT = config.fetch_interval_dict(CLIENT)
TIME_DICT = config.time_dict()

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

def single_future_order(symbol, trade_side, time_frame=15, window=200, order_amount=12):
    futures_exchange_info = CLIENT.futures_exchange_info()

    # Find the symbol information for the symbol
    symbol_info = next(item for item in futures_exchange_info['symbols'] if item['symbol'] == symbol)

    # Extract precision requirements for quantity and current_price
    quantity_precision = int(symbol_info['quantityPrecision'])
    price_precision = int(symbol_info['pricePrecision'])

    print(symbol, quantity_precision, price_precision)
    print(type(symbol_info['quantityPrecision']))

    coin = futures.SingleCoin(CLIENT, symbol, time_frame, window)

    current_price = round(coin.CURRENT_PRICE, price_precision)
    last_close_price = round(coin.LAST_CLOSE_PRICE, price_precision)
    last_sma = round(coin.LAST_SMA, price_precision)
    last_ema = round(coin.LAST_EMA, price_precision)
    last_std = coin.LAST_STD
    upper_band = round(coin.UPPER_BAND, price_precision)
    lower_band = round(coin.LOWER_BAND, price_precision)
    price_dev_sma = current_price - last_sma
    price_dev_ema = current_price - last_ema

    def place_order(side, quantity, price):
        limit_order = CLIENT.futures_create_order(
            symbol=symbol,
            side=side,
            type='LIMIT',  # 'MARKET', 'LIMIT', etc. Check Binance API docs for all options
            quantity=quantity,
            price=price,
            timeInForce='GTC')
        return limit_order

    if current_price and type(current_price) == float:
        quantity = round((order_amount/current_price), quantity_precision)  # Quantity to buy or sell 12 USDT
        if price_dev_sma > last_std * 2 * 0.95 and trade_side == 'SELL':
            TRADE_SIDE = 'SELL'
        elif price_dev_sma < 0 and abs(price_dev_sma) > last_std * 2 * 0.95 and trade_side == 'BUY':
            TRADE_SIDE = 'BUY'

        if TRADE_SIDE:
            print(f'-- Preparing {TRADE_SIDE} limit order for {quantity, symbol} at {current_price} USDT.')
            try:
                limit_order = place_order(TRADE_SIDE, quantity, current_price)
                print(f"\n  --- {TRADE_SIDE} Order Placed: {limit_order}")
            except BinanceAPIException as e:
                print('ERROR:', e)
                if 'APIError(code=-1111)' in str(e):
                    single_future_order(symbol, 0)

    else:
        pass

def list_future_order():
    # Define the method you want to call every 5 minutes
    print(time.strftime("%m-%d %H:%M:%S"))

    sma1min_coins_ordered_dict = futures.scan_futures_for_sma(CLIENT, 1, 200)
    sma5min_coins_ordered_dict = futures.scan_futures_for_sma(CLIENT, 5, 200)
    sma15min_coins_ordered_dict = futures.scan_futures_for_sma(CLIENT, 15, 200)
    sma4h_coins_ordered_dict = futures.scan_futures_for_sma(CLIENT, 4, 200)

    narrow_sell_symbol_list = []
    narrow_buy_symbol_list = []
    wide_sell_symbol_list = []
    wide_buy_symbol_list = []

    for symbol, price_dict in sma5min_coins_ordered_dict.items():
        try:
            if price_dict['dev_ratio'] > 2:
                if sma1min_coins_ordered_dict[symbol]['dev_ratio'] > 0.01 and sma15min_coins_ordered_dict[symbol]['dev_ratio'] > 4:
                    narrow_sell_symbol_list.append(symbol)
            elif 0.4 > price_dict['dev_ratio'] > 0.1:
                if sma1min_coins_ordered_dict[symbol]['dev_ratio'] < -0.01 and 1 > sma15min_coins_ordered_dict[symbol]['dev_ratio'] > 0.2:
                    narrow_buy_symbol_list.append(symbol)
            elif -0.1 > price_dict['dev_ratio'] > -0.4:
                if sma1min_coins_ordered_dict[symbol]['dev_ratio'] > 0.01 and -0.2 > sma15min_coins_ordered_dict[symbol]['dev_ratio'] > -1 :
                    narrow_sell_symbol_list.append(symbol)
            elif price_dict['dev_ratio'] < -2:
                if sma1min_coins_ordered_dict[symbol]['dev_ratio'] < -0.01 and sma15min_coins_ordered_dict[symbol]['dev_ratio'] < -4:
                    narrow_buy_symbol_list.append(symbol)
        except BinanceAPIException as e:
            print(f'ERROR for {symbol}: {e}')

    for symbol, price_dict in sma15min_coins_ordered_dict.items():
        if symbol in sma4h_coins_ordered_dict.keys():
            try:
                if price_dict['dev_ratio'] > 10:
                    if sma5min_coins_ordered_dict[symbol]['dev_ratio'] > 0.05 and sma4h_coins_ordered_dict[symbol]['dev_ratio'] > 20:
                        wide_sell_symbol_list.append(symbol)
                elif 2 > price_dict['dev_ratio'] > 0.5:
                    if sma5min_coins_ordered_dict[symbol]['dev_ratio'] < -0.05 and 5 > sma4h_coins_ordered_dict[symbol]['dev_ratio'] > 1:
                        wide_buy_symbol_list.append(symbol)
                elif -0.5 > price_dict['dev_ratio'] > -2:
                    if sma5min_coins_ordered_dict[symbol]['dev_ratio'] > 0.05 and -1 > sma4h_coins_ordered_dict[symbol]['dev_ratio'] > -5 :
                        wide_sell_symbol_list.append(symbol)
                elif price_dict['dev_ratio'] < -10:
                    if sma5min_coins_ordered_dict[symbol]['dev_ratio'] < -0.05 and sma4h_coins_ordered_dict[symbol]['dev_ratio'] < -20:
                        wide_buy_symbol_list.append(symbol)
            except ValueError as ve:
                print(f'ERROR for {symbol}: {ve}')
            except BinanceAPIException as be:
                print(f'ERROR for {symbol}: {be}')


    # symbol_list = ['ATOMUSDT', 'FILUSDT', 'SEIUSDT', 'SUIUSDT', 'XAIUSDT', 'QTUMUSDT', 'IDUSDT', 'AIUSDT', 'WAVESUSDT']
    for symbol in narrow_sell_symbol_list[:]:
        sd = sma5min_coins_ordered_dict[symbol]
        print(f"{'{:<12}'.format('NARROW SELL:')} {'{:<11}'.format(symbol)}: last_sma:{'{:<9}'.format(sd['last_sma'])} last_close:{'{:<9}'.format(sd['last_close_price'])} dev:{sd['dev_ratio']}%")
        # single_future_order(symbol, 'SELL', 5, WINDOW, ORDER_AMOUNT)
    for symbol in narrow_buy_symbol_list[:]:
        sd = sma5min_coins_ordered_dict[symbol]
        print(f"{'{:<12}'.format('NARROW BUY:')} {'{:<11}'.format(symbol)}: last_sma:{'{:<9}'.format(sd['last_sma'])} last_close:{'{:<9}'.format(sd['last_close_price'])} dev:{sd['dev_ratio']}%")
        # single_future_order(symbol, 'BUY', 5, WINDOW, ORDER_AMOUNT)
    for symbol in wide_sell_symbol_list[:]:
        sd = sma15min_coins_ordered_dict[symbol]
        print(f"{'{:<12}'.format('WIDE SELL:')} {'{:<11}'.format(symbol)}: last_sma:{'{:<9}'.format(sd['last_sma'])} last_close:{'{:<9}'.format(sd['last_close_price'])} dev:{sd['dev_ratio']}%")
        # single_future_order(symbol, 'SELL', 15, WINDOW, ORDER_AMOUNT)
    for symbol in wide_buy_symbol_list[:]:
        sd = sma15min_coins_ordered_dict[symbol]
        print(f"{'{:<12}'.format('WIDE BUY:')} {'{:<11}'.format(symbol)}: last_sma:{'{:<9}'.format(sd['last_sma'])} last_close:{'{:<9}'.format(sd['last_close_price'])} dev:{sd['dev_ratio']}%")
        # single_future_order(symbol, 'BUY', 15, WINDOW, ORDER_AMOUNT)
    print('\n')


def select_future_order():
    select_list1 = ['1INCHUSDT', 'AAVEUSDT', 'ADAUSDT', 'AGIXUSDT', 'AIUSDT', 'ALGOUSDT', 'ALICEUSDT', 'ALTUSDT', 'APEUSDT', 'APTUSDT', 'AXSUSDT']
    select_list2 = ['ARKMUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BAKEUSDT', 'BALUSDT', 'BANDUSDT', 'BCHUSDT', 'BELUSDT', 'BLZUSDT', 'C98USDT', 'CAKEUSDT', 'CELOUSDT', 'CHRUSDT', 'COTIUSDT', 'CRVUSDT', 'CYBERUSDT']
    select_list3 = ['DENTUSDT', 'DYDXUSDT', 'EDUUSDT', 'EOSUSDT', 'FETUSDT', 'FILUSDT', 'FLMUSDT', 'GASUSDT', 'GLMUSDT', 'GRTUSDT', 'HIFIUSDT', 'HIGHUSDT']
    select_list4 = ['JASMYUSDT', 'JUPUSDT', 'KAVAUSDT', 'KNCUSDT', 'LDOUSDT', 'LPTUSDT', 'MAGICUSDT', 'MASKUSDT', 'MATICUSDT', 'MBOXUSDT', 'MDTUSDT', 'MEMEUSDT']
    select_list5 = ['NEARUSDT', 'NFPUSDT', 'NMRUSDT', 'OPUSDT', 'PHBUSDT', 'PIXELUSDT', 'RNDRUSDT', 'SANDUSDT', 'SEIUSDT', 'SPELLUSDT', 'STRAXUSDT', 'SUIUSDT', 'SUSHIUSDT']
    select_list6 = ['TIAUSDT', 'UNFIUSDT', 'UNIUSDT', 'WAVESUSDT', 'WLDUSDT', 'XAIUSDT', 'XVGUSDT', 'ZILUSDT']
    select_list7_1000 = ['1000BONKUSDT', '1000FLOKIUSDT', '1000PEPEUSDT', '1000SHIBUSDT', '1000BIGTIMEUSDT']
    # select_list = select_list1 + select_list2 + select_list3 + select_list4 + select_list5 + select_list6 + select_list7_1000
    select_list = ['UXLINKUSDT', 'XRPUSDT', 'REIUSDT', 'WIFUSDT', 'FIDAUSDT', 'WLDUSDT']

    COINS_DICT = {}
    if os.path.exists(futures.coins_dict_pkl_path):
        COINS_DICT = save_load_pkl(futures.coins_dict_pkl_path)

    for symbol in select_list:
        coin_dict = COINS_DICT.get(symbol, {})
        symbol_position = next((pos for pos in all_positions if pos['symbol'] == symbol), None)
        print(symbol, symbol_position)
        if symbol_position and (not symbol_position['positionAmt'] or  float(symbol_position['positionAmt']) == 0):
            symbol_position = None

        critical_buy_price = coin_dict.get('critical_buy_price', None)
        critical_sell_price = coin_dict.get('critical_sell_price', None)

        try:
            continuous_bollinger_order3(symbol, critical_buy_price=critical_buy_price, critical_sell_price=critical_sell_price, symbol_position=symbol_position)
        except ValueError as ve:
            print(f'  --------  ERROR: {ve}')


def continuous_bollinger_order3(symbol, critical_buy_price=None, critical_sell_price=None, symbol_position=None, time_frame=1, general_sentiment='neutral'):
    print()

    futures_client = UMFutures(key=API_KEY, secret=API_SECRET)
    # trades = client.trades(symbol)
    # print('trades:',trades)

    # check 15min rsi, macd, ema_dev and ema_slope values of BTC and ETH for short-term general_sentiment.
    current_date = datetime.now(timezone.utc).replace(tzinfo=None)
    if current_date.minute % 10 == 0:
        if 'general_sentiment' in COINS_DICT.keys():
            general_sentiment = COINS_DICT['general_sentiment']
        BTC_obj = SingleCoin(futures_client, 'BTCUSDT', time_frame=5, window=210)
        df_BTC = BTC_obj.df_symbol
        ETH_obj = SingleCoin(futures_client, 'ETHUSDT', time_frame=5, window=210)
        df_ETH = ETH_obj.df_df_symbol

    coin_order_dict = {}
    
    if not symbol_position and symbol in REALIZED_ORDERS_DICT.keys():
        coin_order_dict = REALIZED_ORDERS_DICT[symbol]
    elif symbol_position:
        coin_order_dict['real_order_value'] = float(symbol_position['notional']) - float(symbol_position['unRealizedProfit'])
        coin_order_dict['real_order_quantity'] = float(symbol_position['positionAmt'])
        coin_order_dict['entryPrice'] = float(symbol_position['entryPrice']) if symbol_position['entryPrice'] is not None else None
        if symbol in REALIZED_ORDERS_DICT.keys():
            coin_order_dict = REALIZED_ORDERS_DICT[symbol]
            if coin_order_dict['real_order_quantity'] > 0 and not coin_order_dict['last_buy_price']:
                coin_order_dict['last_buy_price'] = coin_order_dict['entryPrice'] if coin_order_dict['entryPrice'] is not None else None
                coin_order_dict['last_buy_time'] = current_date - timedelta(minutes=1) if coin_order_dict['last_buy_time'] is None else None
            if coin_order_dict['real_order_quantity'] < 0 and not coin_order_dict['last_sell_price']:
                coin_order_dict['last_sell_price'] = coin_order_dict['entryPrice'] if coin_order_dict['entryPrice'] is not None else None
                coin_order_dict['last_sell_time'] = current_date - timedelta(minutes=1) if coin_order_dict['last_buy_time'] is None else None
        else:
            coin_order_dict['last_buy_time'] = current_date - timedelta(minutes=1) if coin_order_dict['real_order_value'] < 0 else None
            coin_order_dict['last_sell_time'] = current_date - timedelta(minutes=1) if coin_order_dict['real_order_value'] > 0 else None
            coin_order_dict['last_buy_price'] = coin_order_dict['entryPrice'] if coin_order_dict['real_order_quantity'] > 0 else None
            coin_order_dict['last_sell_price'] = coin_order_dict['entryPrice'] if coin_order_dict['real_order_quantity'] < 0 else None
        REALIZED_ORDERS_DICT[symbol] = coin_order_dict
    else:
        coin_order_dict['real_order_value'] = 0
        coin_order_dict['real_order_quantity'] = 0
        coin_order_dict['entryPrice'] = None
        coin_order_dict['last_buy_time'] = None
        coin_order_dict['last_sell_time'] = None
        coin_order_dict['last_buy_price'] = None
        coin_order_dict['last_sell_price'] = None
        REALIZED_ORDERS_DICT[symbol] = coin_order_dict

    real_order_quantity = coin_order_dict['real_order_quantity']
    real_order_value = coin_order_dict['real_order_value']
    entryPrice = coin_order_dict['entryPrice']
    last_sell_time = coin_order_dict['last_sell_time']
    last_sell_price = coin_order_dict['last_sell_price']
    last_buy_time = coin_order_dict['last_buy_time']
    last_buy_price = coin_order_dict['last_buy_price']
    real_order_value_total = REALIZED_ORDERS_DICT['real_order_value_total']
    orders_total_limit = REALIZED_ORDERS_DICT['orders_total_limit']

    if COINS_DICT and symbol in COINS_DICT.keys():
        symbol_dict = COINS_DICT[symbol]
        pricePrecision = int(symbol_dict['pricePrecision'])
        quantityPrecision = int(symbol_dict['quantityPrecision'])
        top_ema200_devs = get_saved_ema_extremes(symbol, 200)['top_ema200_devs']
        dip_ema200_devs = get_saved_ema_extremes(symbol, 200)['dip_ema200_devs']
        pos_ema200_slopes = get_saved_ema_extremes(symbol, 200)['pos_ema200_slopes']
        neg_ema200_slopes = get_saved_ema_extremes(symbol, 200)['neg_ema200_slopes']
        top_ema100_devs = get_saved_ema_extremes(symbol, 100)['top_ema100_devs']
        dip_ema100_devs = get_saved_ema_extremes(symbol, 100)['dip_ema100_devs']
        pos_ema100_slopes = get_saved_ema_extremes(symbol, 100)['pos_ema100_slopes']
        neg_ema100_slopes = get_saved_ema_extremes(symbol, 100)['neg_ema100_slopes']
    else:
        pricePrecision = 4
        quantityPrecision = 0
        top_ema200_devs = [0.75, 1, 1.25]
        dip_ema200_devs = [-0.75, -1, -1.25]
        pos_ema200_slopes = [0.001, 0.006, 0.011, 0.024, 0.35] # referenced by UXLINK values
        neg_ema200_slopes = [-0.001, -0.006, -0.011, -0.024, -0.35]
        top_ema100_devs = [0.5, 0.75, 1]
        dip_ema100_devs = [-0.5, -0.75, -1]
        pos_ema100_slopes = [0.002, 0.009, 0.016, 0.035, 0.52]
        neg_ema100_slopes = [-0.002, -0.009, -0.016, -0.035, -0.52]

    def place_order(symbol, side, quantity, price):
        try:
            response = futures_client.new_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                quantity=quantity,
                timeInForce="GTC",
                price=price,
            )
            logging.info(response)
            pass
        except ClientError as error:
            logging.error("Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )

    symbol_obj = SingleCoin(futures_client, symbol, time_frame=time_frame, window=210)
    df_symbol = symbol_obj.df_symbol

    if config.time_dict()[time_frame] < 5:
        symbol_obj_longer = SingleCoin(futures_client, symbol, time_frame=5, window=210)
        df_symbol_longer = symbol_obj_longer.df_symbol
    else:
        time_frame_longer = 15 if config.time_dict()[time_frame] == 5 else 60
        symbol_obj_longer = SingleCoin(futures_client, symbol, time_frame=time_frame_longer, window=210)
        df_symbol_longer = symbol_obj_longer.df_symbol

    current_price = symbol_obj.current_price

    last_ema200 = df_symbol['ema200'].iloc[-1]
    last_ema200_dev = df_symbol['ema200_dev'].iloc[-1]
    last_ema200_slope = df_symbol['ema200_slope'].iloc[-1]
    last_ema100 = df_symbol['ema100'].iloc[-1]
    last_ema100_dev = df_symbol['ema100_dev'].iloc[-1]
    last_ema100_slope = df_symbol['ema100_slope'].iloc[-1]
    last_ema50 = df_symbol['ema50'].iloc[-1]
    last_ema50_dev = df_symbol['ema50_dev'].iloc[-1]
    last_ema50_slope = df_symbol['ema50_slope'].iloc[-1]
    last_ema25 = df_symbol['ema25'].iloc[-1]
    last_ema25_dev = df_symbol['ema25_dev'].iloc[-1]
    last_ema25_slope = df_symbol['ema25_slope'].iloc[-1]

    '''last_local_ema_extreme_dict = {}
    for term in ['25', '50', '100', '200']:
        ema_column_name = 'ema' + term
        # local_max_indices, _ = find_peaks(df_symbol[ema_column_name])
        # local_min_indices, _ = find_peaks(-df_symbol[ema_column_name])
        # last_local_max = df_symbol.iloc[local_max_indices[-1]] if len(local_max_indices) > 0 else None
        # last_local_min = df_symbol.iloc[local_min_indices[-1]] if len(local_min_indices) > 0 else None
        #
        # last_local_max_date = df_symbol.index[local_max_indices[-1]] if len(local_max_indices) > 0 else None
        # last_local_min_date = df_symbol.index[local_min_indices[-1]] if len(local_min_indices) > 0 else None

        slope = np.gradient(df_symbol[ema_column_name])
        abs_slope = np.abs(slope)
        local_min_indices = (np.diff(np.sign(np.diff(abs_slope))) > 0).nonzero()[0] + 1  # Local minima in abs_slope
        last_local_min = df_symbol.iloc[local_min_indices[-1]] if len(local_min_indices) > 0 else None
        local_max_indices = (np.diff(np.sign(np.diff(-abs_slope))) > 0).nonzero()[0] + 1  # Local maxima in -abs_slope
        last_local_max = df_symbol.iloc[local_max_indices[-1]] if len(local_max_indices) > 0 else None

        last_local_max_date = df_symbol.index[local_max_indices[-1]] if len(local_max_indices) > 0 else None
        last_local_min_date = df_symbol.index[local_min_indices[-1]] if len(local_min_indices) > 0 else None

        last_local_ema_extreme_dict['last_local_max' + ema_column_name] = last_local_max, last_local_max_date
        last_local_ema_extreme_dict['last_local_min' + ema_column_name] = last_local_min, last_local_min_date

    last_local_max_ema200 = last_local_ema_extreme_dict['last_local_max_ema200'][0]
    last_local_min_ema200 = last_local_ema_extreme_dict['last_local_min_ema200'][0]
    last_local_max_ema100 = last_local_ema_extreme_dict['last_local_max_ema100'][0]
    last_local_min_ema100 = last_local_ema_extreme_dict['last_local_min_ema100'][0]
    last_local_max_ema50 = last_local_ema_extreme_dict['last_local_max_ema50'][0]
    last_local_min_ema50 = last_local_ema_extreme_dict['last_local_min_ema50'][0]
    last_local_max_ema25 = last_local_ema_extreme_dict['last_local_max_ema25'][0]
    last_local_min_ema25 = last_local_ema_extreme_dict['last_local_min_ema25'][0]'''

    sell_price_factor = 1
    buy_price_factor = 1
    sell_quantity_factor = 1
    buy_quantity_factor = 1

    if general_sentiment.lower() == 'bull':
        sell_price_factor = 1.01
        sell_quantity_factor = 0.9
        buy_quantity_factor = 1.5
    elif general_sentiment.lower() == 'bear':
        buy_price_factor = 1.01
        buy_quantity_factor = 0.9
        sell_quantity_factor = 1.5

    coin_sentiment = 'neutral'
    ok_to_sell = False
    ok_to_buy = False

    if last_ema200_slope < 0 and last_ema50 > last_ema100 > last_ema200 and current_price < last_ema50:
        ok_to_buy = True
    if last_ema100_slope < last_ema200_slope < 0 and current_price > last_ema50:
        ok_to_sell = True
    if last_ema25 > last_ema50 > last_ema100 > last_ema200:
        pass

    min_price4 = df_symbol['low'].tail(4).min()
    min_price9 = df_symbol['low'].tail(9).min()
    min_price29 = df_symbol['low'].tail(29).min()
    min_price119 = df_symbol['low'].tail(119).min()
    min_price_longer = df_symbol_longer['low'].min()
    min_price_index = df_symbol.index[df_symbol['low'] == min_price9].tolist()[-1]
    min_price_bool = min_price9 < df_symbol['lower_band'].loc[min_price_index]

    max_price4 = df_symbol['high'].tail(4).max()
    max_price9 = df_symbol['high'].tail(9).max()
    max_price29 = df_symbol['high'].tail(29).max()
    max_price119 = df_symbol['high'].tail(119).max()
    max_price_longer = df_symbol_longer['high'].max()
    max_price_index = df_symbol.index[df_symbol['high'] == max_price9].tolist()[-1]
    max_price_bool = max_price9 > df_symbol['upper_band'].loc[max_price_index]

    upper_band = df_symbol['upper_band'].iloc[-1]
    lower_band = df_symbol['lower_band'].iloc[-1]
    bb_sma = calculate.round_long_decimals(df_symbol['bb_sma'].iloc[-1], prec=4)
    bb_std = df_symbol['bb_std'].iloc[-1]
    bb_sma_diff = df_symbol['bb_sma_diff'].iloc[-1]
    mean_bb_sma_diff = calculate.round_long_decimals(df_symbol['bb_sma_diff'][-3:].mean(), prec=4)

    upper_band_longer = df_symbol_longer['upper_band'].iloc[-1]
    lower_band_longer = df_symbol_longer['lower_band'].iloc[-1]
    bb_sma_longer = calculate.round_long_decimals(df_symbol_longer['bb_sma'].iloc[-1], prec=4)
    bb_std_longer = df_symbol_longer['bb_std'].iloc[-1]
    bb_sma_diff_longer = df_symbol_longer['bb_sma_diff'].iloc[-1]
    mean_bb_sma_diff_longer = calculate.round_long_decimals(df_symbol_longer['bb_sma_diff'][-3:].mean(), prec=4)

    if not critical_buy_price: # or (entryPrice > 0 and critical_buy_price > 0.9995 * entryPrice):
        if real_order_quantity < 0:
            critical_buy_price = 0.998 * entryPrice
        elif real_order_quantity == 0:
            critical_buy_price = calculate.round_to_significant_figures(min_price119 * 0.999, pricePrecision)
        else:
            min_buy_price = min_price119 * 0.995
            if last_buy_price:
                min_buy_price = min(min_buy_price, last_buy_price * 0.999)
            critical_buy_price = calculate.round_to_significant_figures(min_buy_price, pricePrecision)

    if not critical_sell_price: # or (entryPrice > 0 and critical_sell_price < 1.0005 * entryPrice):
        if real_order_quantity > 0:
            critical_sell_price = 1.002 * entryPrice
        elif real_order_quantity == 0:
            critical_sell_price = calculate.round_to_significant_figures(max_price119 * 1.001, pricePrecision)
        else:
            max_sell_price = max_price119 * 1.005
            if last_sell_price:
                max_sell_price = max(max_sell_price, last_sell_price * 1.001)
            critical_sell_price = calculate.round_to_significant_figures(max_sell_price, pricePrecision)

    print(f'*** {symbol} current_price:{current_price} | ', end='')
    print(f"max_price9: {max_price9} | min_price9: {min_price9} | upper_band: {upper_band} | lower_band: {lower_band} | mean_bb_sma_diff/bb_sma_diff: {mean_bb_sma_diff}/{bb_sma_diff}")
    print(f'last_buy_price, last_buy_time: {last_buy_price, last_buy_time} | last_sell_price, last_sell_time: {last_sell_price, last_sell_time}')

    '''
    SELL scenarios:
    horizontal bb_sma turns down
    diverging from top, converging to ema
    '''

    def buy(buy_price_factor=1, buy_quantity_factor=1):
        order_quantity = round((100 * buy_quantity_factor / current_price), quantityPrecision)
        order_quantity = 1 if order_quantity == 0 and current_price < 400 else order_quantity
        order_quantity = 0.002 if order_quantity < 0.002 and symbol in ['BTCUSDT', 'ETHBTC'] else order_quantity
        order_quantity = 0.01 if order_quantity < 0.01 and symbol == 'ETHUSDT' else order_quantity
        buy_price = calculate.round_long_decimals(0.999 * current_price * buy_price_factor, prec=pricePrecision)
        order_value = order_quantity * buy_price

        if (abs(real_order_value + order_value) + real_order_value_total) / orders_total_limit <= 1.1:
            # place_order(symbol, 'BUY', order_quantity, buy_price)
            pass

        coin_order_dict['real_order_quantity'] += order_quantity
        coin_order_dict['real_order_value'] += order_value
        coin_order_dict['last_buy_time'] = current_date
        coin_order_dict['last_buy_price'] = buy_price

        print(f'  --------  BOUGHT {order_quantity} USDT of {symbol} at {buy_price}.')

    def sell(sell_price_factor=1, sell_quantity_factor=1):
        order_quantity = round((100 * sell_quantity_factor / current_price), quantityPrecision)
        order_quantity = 1 if order_quantity == 0 and current_price < 400 else order_quantity
        order_quantity = 0.002 if order_quantity < 0.002 and symbol in ['BTCUSDT', 'ETHBTC'] else order_quantity
        order_quantity = 0.01 if order_quantity < 0.01 and symbol == 'ETHUSDT' else order_quantity
        sell_price = calculate.round_long_decimals(0.999 * current_price * sell_price_factor, prec=pricePrecision)
        order_value = order_quantity * sell_price

        if (abs(real_order_value + order_value) + real_order_value_total) / orders_total_limit <= 1.1:
            # place_order(symbol, 'SELL', order_quantity, sell_price)
            pass

        coin_order_dict['real_order_quantity'] -= order_quantity
        coin_order_dict['real_order_value'] -= order_value
        coin_order_dict['last_sell_time'] = current_date
        coin_order_dict['last_sell_price'] = sell_price

        print(f'  --------  SOLD {order_value} USDT of {symbol} at {sell_price}.')

    try:
        # Rebuying sold coin at short/long positions
        if real_order_value < 0 or (real_order_value > 0 and last_sell_time is not None):
            local_buy_quantity_factor = 1 if real_order_value < 0 else 0.3
            local_buy_price_factor = 1 if real_order_value < 0 else 0.996
            print(999, 'rebuy the sold')

            if current_price < last_sell_price * 0.998 * local_buy_price_factor and (last_buy_time is None or last_buy_time < last_sell_time):
                if (current_date - last_buy_time).total_seconds() > 170:
                    buy(buy_quantity_factor=local_buy_quantity_factor)
                elif current_price / last_sell_price < 0.985 * local_buy_price_factor:
                    buy(buy_quantity_factor=local_buy_quantity_factor)
                elif current_price / last_sell_price < 0.995 * local_buy_price_factor and last_ema200_dev < dip_ema200_devs[1] / local_buy_price_factor:
                    buy(buy_quantity_factor=local_buy_quantity_factor)
                elif current_price / last_sell_price < 0.995 * local_buy_price_factor and last_ema200_dev < dip_ema200_devs[2] / local_buy_price_factor:
                    buy(buy_quantity_factor=2 * local_buy_quantity_factor)

        # Buy from zero or buy extra at long position
        elif real_order_value == 0 or (real_order_value > 0 and current_price < last_buy_price * 0.98):
            local_buy_quantity_factor = 1 if real_order_value == 0 else 0.3 # since first buy is difficult, succeeding buy is easier
            local_buy_price_factor = 0.99 if real_order_value == 0 else 1

            if mean_bb_sma_diff < 0 and current_price < bb_sma_longer - 0.9 * bb_std_longer: # buy dip of 5 min bb
                if current_price < bb_sma - 0.6 * bb_std:
                    print(999, 'buy new/extra 1')
                    buy(buy_price_factor=local_buy_price_factor, buy_quantity_factor=local_buy_quantity_factor)

            elif current_price <= critical_buy_price and current_price < bb_sma - 0.97 * bb_std and current_price < min_price4:
                if last_buy_time is None or (current_date - last_buy_time).total_seconds() > 170 or current_price/last_buy_price < 0.975:
                    print(999, 'buy new/extra 2')
                    buy(buy_quantity_factor=local_buy_quantity_factor)
                    buy(buy_price_factor=local_buy_price_factor, buy_quantity_factor=local_buy_quantity_factor)

        # Reselling bought coin at short/long positions
        if real_order_value > 0 or (real_order_value < 0 and last_buy_time is not None):
            local_sell_quantity_factor = 1 if real_order_value > 0 else 0.3
            local_sell_price_factor = 1 if real_order_value > 0 else 1.004
            print(999, 'resell the bought')

            if current_price > last_buy_price * 1.002 * local_sell_price_factor and (last_sell_time is None or last_buy_time > last_sell_time):
                if (current_date - last_buy_time).total_seconds() > 170:
                    sell(sell_quantity_factor=local_sell_quantity_factor)
                elif current_price / last_buy_price > 1.015 * local_sell_price_factor:
                    sell(sell_quantity_factor=local_sell_quantity_factor)
                elif current_price / last_buy_price > 1.005 * local_sell_price_factor and last_ema200_dev > top_ema200_devs[1] * local_sell_price_factor:
                    sell(sell_quantity_factor=local_sell_quantity_factor)
                elif current_price / last_buy_price > 1.005 * local_sell_price_factor and last_ema200_dev > top_ema200_devs[2] * local_sell_price_factor:
                    sell(sell_quantity_factor=2 * local_sell_quantity_factor)

        # Sell from zero or sell extra at short position
        elif real_order_value == 0 or (real_order_value < 0 and current_price > last_sell_price * 1.02):
            local_sell_quantity_factor = 1 if real_order_value == 0 else 0.3  # since first buy is difficult, succeeding buy is easier
            local_sell_price_factor = 1.01 if real_order_value == 0 else 1

            if mean_bb_sma_diff > 0 and current_price > bb_sma_longer + 0.9 * bb_std_longer:  # buy dip of 5 min bb
                if current_price > bb_sma + 0.6 * bb_std:
                    print(999, 'sell new/extra 1')
                    sell(sell_price_factor=local_sell_price_factor, sell_quantity_factor=local_sell_quantity_factor)

            elif current_price >= critical_sell_price and current_price > bb_sma + 0.97 * bb_std and current_price > max_price4:
                if last_buy_time is None or (current_date - last_buy_time).total_seconds() > 170 or current_price / last_sell_price > 1.025:
                    print(999, 'sell new/extra 2')
                    buy(sell_quantity_factor=local_sell_quantity_factor)
                    buy(sell_price_factor=local_sell_price_factor, sell_quantity_factor=local_sell_quantity_factor)

        REALIZED_ORDERS_DICT['real_order_value_total'] -= abs(REALIZED_ORDERS_DICT[symbol]['real_order_value'])
        REALIZED_ORDERS_DICT['real_order_value_total'] += abs(coin_order_dict['real_order_value'])
        REALIZED_ORDERS_DICT[symbol] = coin_order_dict
        save_load_pkl(orders_dict_pkl_path, REALIZED_ORDERS_DICT)

    except ValueError as ve:
        print(f'  --------  ERROR: {ve}')
        # place_order(symbol, 'SELL', 1, bb_dict['SELL'])
        # place_order(symbol, 'BUY', 1, bb_dict['BUY'])

    # max_price9, min_price9, bw, ub, lb, ml, trend = 'max_price9', 'min_price9', 'bw', 'ub', 'lb', 'ml', 'trend'
    #
    # bb1 = strategy.get_bb_dict(CLIENT, symbol, 1)
    # bb5 = strategy.get_bb_dict(CLIENT, symbol, 5)
    # if bb1 and bb5:
    #     bb1 = {key: calculate.round_long_decimals(value) for key, value in bb1.items()}
    #     bb5 = {key: calculate.round_long_decimals(value) for key, value in bb5.items()}
    #     current_price = bb5['current_price']
    #     if symbol in select_list7_1000:
    #         current_price = 1000 * current_price
    #         bb5['SELL'] = 1000 * bb5['SELL']
    #         bb5['BUY'] = 1000 * bb5['BUY']
    #
    #     print(f'trying order for {symbol}... {bb5}', end='')
    #
    #     if symbol in select_list7_1000:
    #         symbol = str(1000) + symbol
    #
    #     try:
    #         min_price9 = bb5[min_price9]
    #         lb = bb5[lb]
    #         if current_price > min_price9 and current_price:
    #             order_value = 8
    #             order_quantity = int(round((order_value / current_price), 0))  # Quantity to buy or sell 12 USDT
    #             place_order(symbol, 'SELL', order_quantity, bb5['SELL'])
    #             place_order(symbol, 'BUY', order_quantity, bb5['BUY'])
    #             print(f'  --------  orders have been implemented.\n')
    #     except ValueError as ve:
    #         print(f'  --------  ERROR: {ve}')
    #         place_order(symbol, 'SELL', 1, bb5['SELL'])
    #         place_order(symbol, 'BUY', 1, bb5['BUY'])


def continuous_bollinger_order2(time_frame=5):
    print()
    select_list1 = ['1INCHUSDT', 'AAVEUSDT', 'ADAUSDT', 'AGIXUSDT', 'AIUSDT', 'ALGOUSDT', 'ALICEUSDT', 'ALTUSDT', 'APEUSDT', 'APTUSDT', 'ARBUSDT', 'AXSUSDT']
    select_list2 = ['ARKMUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BAKEUSDT', 'BALUSDT', 'BANDUSDT', 'BCHUSDT', 'BELUSDT', 'BIGTIMEUSDT', 'BLZUSDT', 'BOMEUSDT', 'C98USDT', 'CAKEUSDT', 'CELOUSDT', 'CHRUSDT', 'COTIUSDT', 'CRVUSDT', 'CYBERUSDT']
    select_list3 = ['DENTUSDT', 'DYDXUSDT', 'EDUUSDT', 'EOSUSDT', 'FETUSDT', 'FILUSDT', 'FLMUSDT', 'GASUSDT', 'GLMUSDT', 'GRTUSDT', 'HIFIUSDT', 'HIGHUSDT']
    select_list4 = ['JASMYUSDT', 'JUPUSDT', 'KAVAUSDT', 'KNCUSDT', 'LDOUSDT', 'LPTUSDT', 'MAGICUSDT', 'MASKUSDT', 'MATICUSDT', 'MDTUSDT', 'MEMEUSDT']
    select_list5 = ['NEARUSDT', 'NFPUSDT', 'NMRUSDT', 'OPUSDT', 'PHBUSDT', 'PIXELUSDT', 'RNDRUSDT', 'SANDUSDT', 'SEIUSDT', 'SPELLUSDT', 'SUIUSDT', 'SUSHIUSDT']
    select_list6 = ['TIAUSDT', 'UNFIUSDT', 'UNIUSDT', 'WAVESUSDT', 'WIFUSDT', 'WLDUSDT', 'XAIUSDT', 'XLMUSDT', 'XRPUSDT', 'XVGUSDT', 'ZILUSDT']
    select_list7_1000 = ['1000BONKUSDT', '1000FLOKIUSDT', '1000PEPEUSDT', '1000SHIBUSDT']
    select_list = select_list1 + select_list2 + select_list3 + select_list4 + select_list5 + select_list6 + select_list7_1000
    # select_list = ['XRPUSDT', 'XLMUSDT', 'ARBUSDT']
    select_list = ['APTUSDT', 'JASMYUSDT']

    # config_logging(logging, logging.DEBUG)

    client = UMFutures(key=API_KEY, secret=API_SECRET)
    # trades = client.trades('XRPUSDT')
    # print('trades:',trades)

    def place_order(symbol, side, quantity, price):
        try:
            response = client.new_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                quantity=quantity,
                timeInForce="GTC",
                price=price,
            )
            logging.info(response)
            pass
        except ClientError as error:
            logging.error("Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )

    def plot_symbol(df, order1='', order2=''):
        df = df[-300:]
        mean_price20 = df['close'][-20:].mean()
        plt.figure(figsize=(14, 8))

        def plot_price():
            plt.plot(df.index, df['close'], label='close', color='skyblue')
            plt.plot(df.index, df['high'], label='high', color='orange', linewidth=0.7)
            plt.plot(df.index, df['low'], label='low', color='orange', linewidth=0.7)
            plt.plot(df.index, df['price_diff'] + mean_price20, label='price_diff', color='gray', linestyle='--')
            # plt.plot(df.index, (df['price_change']+0.1) * mean_price20, label='price_change', color='black')

        def plot_bb():
            plt.plot(df.index, df['upper_band'], label='ub', color='brown')
            plt.plot(df.index, df['lower_band'], label='lb', color='brown')
            plt.plot(df.index, df['bb_sma'], label='mb', color='lime')

        def plot_sma_ema(type_list=['sma','ema'], window_list=[200,100,50]):
            type_list = ['ema']
            options = [''.join([x, str(y)]) for x in type_list for y in window_list]
            for option in options:
                color = config.random_color()
                plt.plot(df.index, df[option], label=option, color=color, linestyle='-')
                # plt.plot(df.index, mean_price20 * (1 + df[option + '_dev_sma']/100), label=option + '_dev_sma', color=color, linestyle='--')
                # plt.plot(df.index, mean_price20 * (1 + df[option + '_dev_sma_diff']/100), label=option + '_dev_sma_diff', color=color, linestyle=':')

        def plot_rsi():
            plt.plot(df.index, (1700/mean_price20) * (df['high'] - df['low'].min()), label='high', color='orange', linewidth=0.7)
            plt.plot(df.index, (1700/mean_price20) * (df['low'] - df['low'].min()), label='low', color='orange', linewidth=0.7)
            # plt.plot(df.index, (1700/mean_price20/2) * (df['high'] + df['low'] - 2 * df['low'].min()), label='bb_sma', color='orange', linewidth=0.7)
            plt.plot(df.index, df['rsi7'], label='rsi7', color='green', linestyle='--')
            plt.plot(df.index, df['rsi14'], label='rsi14', color='gray', linestyle='--')
            plt.plot(df.index, df['rsi21'], label='rsi21', color='blue', linestyle='--')
            plt.plot(df.index, df['rsi7_sma'], label='rsi7_sma', color='green')
            plt.plot(df.index, df['rsi14_sma'], label='rsi14_sma', color='gray')
            plt.plot(df.index, df['rsi21_sma'], label='rsi21_sma', color='blue')
            # plt.plot(df.index, df['rsi7_diff'], label='rsi7_diff', color='green')
            # plt.plot(df.index, df['rsi14_diff'], label='rsi14_diff', color='gray')
            # plt.plot(df.index, df['rsi21_diff'], label='rsi21_diff', color='blue')

        def plot_macd():
            plt.plot(df.index, (1700 / mean_price20) * (df['high'] - df['low'].min()), label='high', color='orange', linewidth=0.7)
            plt.plot(df.index, (1700 / mean_price20) * (df['low'] - df['low'].min()), label='low', color='orange', linewidth=0.7)
            # plt.plot(df.index, (40/2) * (df['high'] + df['low'] - 2 * df['low'].min()), label='bb_sma', color='orange', linewidth=0.7)
            plt.plot(df.index, 5000 * df['macd'], label='macd', color='brown', linestyle='--')
            plt.plot(df.index, 5000 * df['macd_sma'], label='macd_sma', color='orange', linestyle='--')
            # plt.plot(df.index, mean_price20 * (1 + df['macd_change'] / 100), label='macd_change', color='gray')
            plt.plot(df.index, 5000 * df['macd_diff'], label='macd_diff', color='red')
            plt.plot(df.index, 5000 * df['macd_sma_diff'], label='macd_sma_diff', color='magenta')

        plt.axhline(y=mean_price20, color='red', linestyle='--', label='mean_price20')
        # plt.axhline(y=order1, color='red', linestyle='--', label=order1)
        # plt.axhline(y=order2, color='green', linestyle='--', label=order2)

        plot_price()
        plot_bb()
        plot_sma_ema()
        # plot_rsi()
        # plot_macd()

        plt.title(f'{symbol[:-4]} {config.fetch_real_time_frame(time_frame)}')
        plt.xticks(df.index[::5], [str(idx)[-11:-3] for idx in df.index[::5]], fontsize=6)
        plt.xlabel('time')
        plt.ylabel('price')
        plt.legend()
        plt.grid(True)
        # plt.show(block=False)
        plt.rcParams['figure.figsize'] = (14.5, 8.5)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()

    # print(f"{'{:<10}'.format('symbol')}: {'{:<18}'.format('current_price')} | {'{:<18}'.format('5min bb_sma')} | {'{:<18}'.format('5min upper_band')} | 5min lower_band\n{'-' * 88}")
    for symbol in select_list:
        # time_frame = 5
        symbol_obj = SingleCoin(client, symbol, time_frame=time_frame, window=200)
        df_symbol = symbol_obj.df_symbol
        # symbol_obj5 = SingleCoin(client, symbol, time_frame=5, window=200)
        # df_symbol5 = symbol_obj5.df_symbol

        # bb_dict = strategy.get_bb_dict(CLIENT, symbol, 1)
        # if bb_dict:
        #     bb_dict = {key: calculate.round_long_decimals(value) for key, value in bb_dict.items()}
        #     current_price = bb_dict['current_price']
        #     if symbol in select_list7_1000:
        #         current_price = 1000 * current_price
        #         bb_dict['SELL'] = 1000 * bb_dict['SELL']
        #         bb_dict['BUY'] = 1000 * bb_dict['BUY']

        current_price = df_symbol['close'].iloc[-1]

        min_price4 = df_symbol['low'].tail(4).min()
        min_price9 = df_symbol['low'].tail(9).min()
        min_price19 = df_symbol['low'].tail(19).min()
        min_price_index = df_symbol.index[df_symbol['low'] == min_price9].tolist()[-1]
        min_price_bool = min_price9 < df_symbol['lower_band'].loc[min_price_index]

        max_price4 = df_symbol['high'].tail(4).max()
        max_price9 = df_symbol['high'].tail(9).max()
        max_price19 = df_symbol['high'].tail(19).max()
        max_price_index = df_symbol.index[df_symbol['high'] == max_price9].tolist()[-1]
        max_price_bool = max_price9 > df_symbol['upper_band'].loc[max_price_index]

        upper_band = df_symbol['upper_band'].iloc[-1]
        lower_band = df_symbol['lower_band'].iloc[-1]
        bb_sma = calculate.round_long_decimals(df_symbol['bb_sma'].iloc[-1], prec=4)
        bb_std = df_symbol['bb_std'].iloc[-1]
        bb_sma_diff = df_symbol['bb_sma_diff'].iloc[-1]
        mean_bb_sma_diff = calculate.round_long_decimals(df_symbol['bb_sma_diff'][-3:].mean(), prec=4)

        print(f'Trying {symbol} >> price:{current_price} | ', end='')
        print(f"bb_sma_diff: {mean_bb_sma_diff}/{bb_sma_diff} | max: {max_price9} | min: {min_price9} | upper_band: {upper_band} | lower_band: {lower_band}")


        '''
        SELL scenarios:
        horizontal bb_sma turns down
        diverging from top, converging to ema
        '''


        try:
            order_amount = 10
            # plot_symbol(df_symbol, 0.5, 0.49)
            quantity = int(round((order_amount / current_price), 0))  # Quantity to buy or sell 12 USDT
            if current_price > bb_sma + 0.9 * bb_std and current_price < max_price9 and max_price_bool and mean_bb_sma_diff < 0.07: #SELL
                sell_price = calculate.round_long_decimals(0.999 * current_price, prec=4)
                place_order(symbol,'SELL', quantity, sell_price)
                print(f'  --------  SOLD {symbol[:-4]} at {sell_price}.')
            elif bb_sma - 0.2 * bb_std < current_price < bb_sma and current_price < max_price4 < max_price19 * 0.995 and mean_bb_sma_diff < -0.09:
                sell_price = calculate.round_long_decimals(0.999 * current_price, prec=4)
                place_order(symbol,'SELL', quantity, sell_price)
                print(f'  --------  SOLD {symbol[:-4]} at {sell_price}.')
                place_order(symbol, 'BUY', quantity, lower_band)
                print(f'  --------  BUY ORDER {symbol[:-4]} at {lower_band}.')
                # plot_symbol(df_symbol, sell_price, lower_band)
            elif current_price < bb_sma - 0.9 * bb_std and current_price > min_price9 and min_price_bool and mean_bb_sma_diff > -0.07:
                buy_price = calculate.round_long_decimals(1.001 * current_price, prec=4)
                place_order(symbol, 'BUY', quantity, buy_price)
                print(f'  --------  BOUGHT {symbol[:-4]} at {buy_price}.')
                place_order(symbol, 'SELL', quantity, bb_sma)
                print(f'  --------  SELL ORDER {symbol[:-4]} at {bb_sma}.')
                # plot_symbol(df_symbol, bb_sma, buy_price)
            elif bb_sma + 0.2 * bb_std > current_price > bb_sma and current_price > min_price4 > min_price19 * 1.005 and mean_bb_sma_diff > 0.09:
                buy_price = calculate.round_long_decimals(1.001 * current_price, prec=4)
                place_order(symbol, 'BUY', quantity, buy_price)
                print(f'  --------  BOUGHT {symbol[:-4]} at {buy_price}.')
                place_order(symbol, 'SELL', quantity, upper_band)
                print(f'  --------  SELL ORDER {symbol[:-4]} at {upper_band }.')
                # plot_symbol(df_symbol, upper_band, buy_price)
            else:
                # print(f'  --------  NO {symbol[:-4]} ORDERS YET')
                pass

        except ValueError as ve:
            print(f'  --------  ERROR: {ve}')
            # place_order(symbol, 'SELL', 1, bb_dict['SELL'])
            # place_order(symbol, 'BUY', 1, bb_dict['BUY'])

    # max_price9, min_price9, bw, ub, lb, ml, trend = 'max_price9', 'min_price9', 'bw', 'ub', 'lb', 'ml', 'trend'
    #
    # bb1 = strategy.get_bb_dict(CLIENT, symbol, 1)
    # bb5 = strategy.get_bb_dict(CLIENT, symbol, 5)
    # if bb1 and bb5:
    #     bb1 = {key: calculate.round_long_decimals(value) for key, value in bb1.items()}
    #     bb5 = {key: calculate.round_long_decimals(value) for key, value in bb5.items()}
    #     current_price = bb5['current_price']
    #     if symbol in select_list7_1000:
    #         current_price = 1000 * current_price
    #         bb5['SELL'] = 1000 * bb5['SELL']
    #         bb5['BUY'] = 1000 * bb5['BUY']
    #
    #     print(f'trying order for {symbol}... {bb5}', end='')
    #
    #     if symbol in select_list7_1000:
    #         symbol = str(1000) + symbol
    #
    #     try:
    #         min_price9 = bb5[min_price9]
    #         lb = bb5[lb]
    #         if current_price > min_price9 and current_price:
    #             order_amount = 8
    #             quantity = int(round((order_amount / current_price), 0))  # Quantity to buy or sell 12 USDT
    #             place_order(symbol, 'SELL', quantity, bb5['SELL'])
    #             place_order(symbol, 'BUY', quantity, bb5['BUY'])
    #             print(f'  --------  orders have been implemented.\n')
    #     except ValueError as ve:
    #         print(f'  --------  ERROR: {ve}')
    #         place_order(symbol, 'SELL', 1, bb5['SELL'])
    #         place_order(symbol, 'BUY', 1, bb5['BUY'])


def continuous_bollinger_order(symbol, time_frame=5):
    # REALIZED_ORDERS_DICT.keys = [balance, orders_total_limit, real_order_value_total, real_buys_total, real_sells_total, symbol]
    symbol_orders_dict = {
        'buys': {'1m': {'quantity':0, 'amount':0, 'buy_price':0, 'last_order_time':None},
                '5m': {'quantity':0, 'amount':0, 'buy_price':0, 'last_order_time':None},
                '15m': {'quantity':0, 'amount':0, 'buy_price':0, 'last_order_time':None}},
        'sells': {'1m': {'quantity':0, 'amount':0, 'sell_price':0, 'last_order_time':None},
                '5m': {'quantity':0, 'amount':0, 'sell_price':0, 'last_order_time':None},
                '15m': {'quantity':0, 'amount':0, 'sell_price':0, 'last_order_time':None}},
        'total_order_quantity': 0,
        'total_order_amount': 0,
        'latest_order_time': None,
        'latest_order_type': None
    }

    realized_orders_dict_path = '../../data/futures'
    REALIZED_ORDERS_DICT = futures.save_load_pkl(realized_orders_dict_path)
    if symbol in REALIZED_ORDERS_DICT.keys():
        symbol_orders_dict = REALIZED_ORDERS_DICT[symbol]

    order_amount = 0
    if time_frame in [1, '1min']:
        order_amount = 11;
        if symbol == 'ETHUSDT':
            order_amount = 33
        if symbol == 'BTCUSDT':
            order_amount = 220
    elif time_frame in [5, '5min']:
        order_amount = 40
        if symbol == 'ETHUSDT':
            order_amount = 120
        if symbol == 'BTCUSDT':
            order_amount = 330
    elif time_frame in [15, '15min']:
        order_amount = 100
        if symbol == 'ETHUSDT':
            order_amount = 300
        if symbol == 'BTCUSDT':
            order_amount = 750
    else:
        order_amount = 500

    time_frame_str = config.time_dict_str()[time_frame]
    real_order_value_total = REALIZED_ORDERS_DICT['real_order_value_total']
    orders_total_limit = REALIZED_ORDERS_DICT['orders_total_limit']
    latest_order_type = symbol_orders_dict['latest_order_type']
    latest_order_time = symbol_orders_dict['latest_order_time']
    order_limiting_mins = config.order_limiting_mins_dict()[time_frame]
    pricePrecision = COINS_DICT[symbol]['pricePrecision']
    quantityPrecision = COINS_DICT[symbol]['quantityPrecision']

    print()
    # config_logging(logging, logging.DEBUG)

    client = UMFutures(key=API_KEY, secret=API_SECRET)
    # trades = client.trades('XRPUSDT')
    # print('trades:',trades)

    def place_order(symbol, side, quantity, price, order_type="LIMIT"):
        try:
            response = client.new_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                timeInForce="GTC",
                price=price,
            )
            logging.info(response)
            pass
        except ClientError as error:
            logging.error("Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )

    # print(f"{'{:<10}'.format('symbol')}: {'{:<18}'.format('current_price')} | {'{:<18}'.format('5min bb_sma')} | {'{:<18}'.format('5min upper_band')} | 5min lower_band\n{'-' * 88}")

    # time_frame = 5
    symbol_obj = SingleCoin(client, symbol, time_frame=time_frame, window=200)
    df_symbol = symbol_obj.df_symbol
    df_symbol_1min = symbol_obj.df_symbol_dict['1m']
    df_symbol_5min = symbol_obj.df_symbol_dict['5m']
    df_symbol_15min = symbol_obj.df_symbol_dict['15m']
    # symbol_obj5 = SingleCoin(client, symbol, time_frame=5, window=200)
    # df_symbol5 = symbol_obj5.df_symbol

    # bb_dict = strategy.get_bb_dict(CLIENT, symbol, 1)
    # if bb_dict:
    #     bb_dict = {key: calculate.round_long_decimals(value) for key, value in bb_dict.items()}
    #     current_price = bb_dict['current_price']
    #     if symbol in select_list7_1000:
    #         current_price = 1000 * current_price
    #         bb_dict['SELL'] = 1000 * bb_dict['SELL']
    #         bb_dict['BUY'] = 1000 * bb_dict['BUY']

    current_price = df_symbol['close'].iloc[-1]
    min_price4 = df_symbol['low'].tail(4).min()
    min_price9 = df_symbol['low'].tail(9).min()
    min_price19 = df_symbol['low'].tail(19).min()
    min_price_index = df_symbol.index[df_symbol['low'] == min_price9].tolist()[-1]
    min_price_bool = min_price9 < df_symbol['lower_band'].loc[min_price_index]

    max_price4 = df_symbol['high'].tail(4).max()
    max_price9 = df_symbol['high'].tail(9).max()
    max_price19 = df_symbol['high'].tail(19).max()
    max_price_index = df_symbol.index[df_symbol['high'] == max_price9].tolist()[-1]
    max_price_bool = max_price9 > df_symbol['upper_band'].loc[max_price_index]

    upper_band = df_symbol['upper_band'].iloc[-1]
    lower_band = df_symbol['lower_band'].iloc[-1]
    bb_sma = calculate.round_long_decimals(df_symbol['bb_sma'].iloc[-1], prec=4)
    bb_std = df_symbol['bb_std'].iloc[-1]
    bb_sma_diff = df_symbol['bb_sma_diff'].iloc[-1]
    mean_bb_sma_diff = calculate.round_long_decimals(df_symbol['bb_sma_diff'][-3:].mean(), prec=4)

    print(f'Trying {symbol} >> price:{current_price} | ', end='')
    print(f"bb_sma_diff: {mean_bb_sma_diff}/{bb_sma_diff} | max: {max_price9} | min: {min_price9} | upper_band: {upper_band} | lower_band: {lower_band}")

    try:
        current_date = calculate.current_date()
        time_difference = current_date - latest_order_time
        quantity = calculate.round_long_decimals(order_amount / current_price, prec=quantityPrecision)
        ema200_dev = df_symbol_1min['ema200_dev'].iloc[-1]
        ema100_dev = df_symbol_1min['ema100_dev'].iloc[-1]
        top_ema200_dev_mean = COINS_DICT[symbol]['top_ema200_dev_mean']
        top_ema100_dev_mean = COINS_DICT[symbol]['top_ema100_dev_mean']
        sell_bool = False
        buy_bool = False

        if real_order_value_total < orders_total_limit or latest_order_type != 'SELL':
            if time_difference > timedelta(minutes=order_limiting_mins) or latest_order_type != 'SELL':
                sell_bool = True

        # if ema_sell_bool(symbol, df_symbol_1min): # check 1) direction of emas, divergence-convergence, their intersection, deviation from ema(ema_dev)
        if sell_bool and (ema200_dev >= top_ema200_dev_mean or ema100_dev >= top_ema100_dev_mean):
            if current_price > bb_sma + 0.9 * bb_std and current_price < max_price9 and max_price_bool and mean_bb_sma_diff < 0.07:
                sell_price = calculate.round_long_decimals(0.999 * current_price, prec=pricePrecision)
                place_order(symbol,'SELL', quantity, sell_price)

                partial_order_dict = symbol_orders_dict['sells'][time_frame_str]
                partial_order_dict['amount'] -= order_amount
                partial_order_dict['quantity'] -= quantity
                partial_order_dict['sell_price'] = sell_price
                partial_order_dict['last_time'] = current_date
                symbol_orders_dict['total_order_quantity'] -= quantity
                symbol_orders_dict['total_order_amount'] -= order_amount
                symbol_orders_dict['last_order_type'] = 'SELL'
                symbol_orders_dict['last_order_time'] = current_date
                real_order_value_total += abs(symbol_orders_dict['total_order_amount'])
                REALIZED_ORDERS_DICT['real_order_value_total'] = real_order_value_total
                REALIZED_ORDERS_DICT[symbol] = symbol_orders_dict
                futures.save_load_pkl(realized_orders_dict_path, REALIZED_ORDERS_DICT)

                print(f'  --------  SOLD {symbol[:-4]} at {sell_price}.')

        elif bb_sma - 0.2 * bb_std < current_price < bb_sma and current_price < max_price4 < max_price19 * 0.995 and mean_bb_sma_diff < -0.09:
            sell_price = calculate.round_long_decimals(0.999 * current_price, prec=4)
            place_order(symbol,'SELL', quantity, sell_price)
            print(f'  --------  SOLD {symbol[:-4]} at {sell_price}.')
            place_order(symbol, 'BUY', quantity, lower_band)
            print(f'  --------  BUY ORDER {symbol[:-4]} at {lower_band}.')
            # plot_symbol(df_symbol, sell_price, lower_band)
        elif current_price < bb_sma - 0.9 * bb_std and current_price > min_price9 and min_price_bool and mean_bb_sma_diff > -0.07:
            buy_price = calculate.round_long_decimals(1.001 * current_price, prec=4)
            place_order(symbol, 'BUY', quantity, buy_price)
            print(f'  --------  BOUGHT {symbol[:-4]} at {buy_price}.')
            place_order(symbol, 'SELL', quantity, bb_sma)
            print(f'  --------  SELL ORDER {symbol[:-4]} at {bb_sma}.')
            # plot_symbol(df_symbol, bb_sma, buy_price)
        elif bb_sma + 0.2 * bb_std > current_price > bb_sma and current_price > min_price4 > min_price19 * 1.005 and mean_bb_sma_diff > 0.09:
            buy_price = calculate.round_long_decimals(1.001 * current_price, prec=4)
            place_order(symbol, 'BUY', quantity, buy_price)
            print(f'  --------  BOUGHT {symbol[:-4]} at {buy_price}.')
            place_order(symbol, 'SELL', quantity, upper_band)
            print(f'  --------  SELL ORDER {symbol[:-4]} at {upper_band }.')
            # plot_symbol(df_symbol, upper_band, buy_price)
        else:
            # print(f'  --------  NO {symbol[:-4]} ORDERS YET')
            pass

    except ValueError as ve:
        print(f'  --------  ERROR: {ve}')
        # place_order(symbol, 'SELL', 1, bb_dict['SELL'])
        # place_order(symbol, 'BUY', 1, bb_dict['BUY'])

    # max_price9, min_price9, bw, ub, lb, ml, trend = 'max_price9', 'min_price9', 'bw', 'ub', 'lb', 'ml', 'trend'
    #
    # bb1 = strategy.get_bb_dict(CLIENT, symbol, 1)
    # bb5 = strategy.get_bb_dict(CLIENT, symbol, 5)
    # if bb1 and bb5:
    #     bb1 = {key: calculate.round_long_decimals(value) for key, value in bb1.items()}
    #     bb5 = {key: calculate.round_long_decimals(value) for key, value in bb5.items()}
    #     current_price = bb5['current_price']
    #     if symbol in select_list7_1000:
    #         current_price = 1000 * current_price
    #         bb5['SELL'] = 1000 * bb5['SELL']
    #         bb5['BUY'] = 1000 * bb5['BUY']
    #
    #     print(f'trying order for {symbol}... {bb5}', end='')
    #
    #     if symbol in select_list7_1000:
    #         symbol = str(1000) + symbol
    #
    #     try:
    #         min_price9 = bb5[min_price9]
    #         lb = bb5[lb]
    #         if current_price > min_price9 and current_price:
    #             order_amount = 8
    #             quantity = int(round((order_amount / current_price), 0))  # Quantity to buy or sell 12 USDT
    #             place_order(symbol, 'SELL', quantity, bb5['SELL'])
    #             place_order(symbol, 'BUY', quantity, bb5['BUY'])
    #             print(f'  --------  orders have been implemented.\n')
    #     except ValueError as ve:
    #         print(f'  --------  ERROR: {ve}')
    #         place_order(symbol, 'SELL', 1, bb5['SELL'])
    #         place_order(symbol, 'BUY', 1, bb5['BUY'])


# def get_info(symbol=''):
#     symbol = 'XRPUSDT'
# 
#     # Get the position information for the specified symbol
#     position_info = request_client.get_position(symbol=symbol)
#     for position in position_info:
#         if position.positionSide == PositionSide.LONG:
#             print(f"Long position for {symbol}: {position.positionAmt}")
#         elif position.positionSide == PositionSide.SHORT:
#             print(f"Short position for {symbol}: {position.positionAmt}")
# 
#     # Get open orders for the specified symbol
#     open_orders = request_client.get_open_orders(symbol=symbol)
#     for order in open_orders:
#         print(f"Order ID: {order.orderId}, Symbol: {order.symbol}, Side: {order.side}, Type: {order.type}, Price: {order.price}, Quantity: {order.origQty}")


def ema_sell_bool(symbol, df):
    ema_dict = COINS_DICT[symbol]
    ema200_pct_change = df['ema200_pct_change'].iloc[-1]
    ema100_pct_change = df['ema100_pct_change'].iloc[-1]
    ema50_pct_change = df['ema50_pct_change'].iloc[-1]
    ema25_pct_change = df['ema25_pct_change'].iloc[-1]

    ema200_dev = df['ema200_dev'].iloc[-1]
    ema100_dev = df['ema100_dev'].iloc[-1]
    ema50_dev = df['ema50_dev'].iloc[-1]
    ema25_dev = df['ema25_dev'].iloc[-1]

    ema200 = df['ema200'].iloc[-1]
    ema100 = df['ema100'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]
    ema25 = df['ema25'].iloc[-1]

    if ema200_dev >= ema_dict['top_ema200_dev_mean']:
        if ema200_pct_change > 0 and ema100_pct_change > 0:
            return True

    if ema200_pct_change > 0 and ema200_pct_change >= df['ema200_pct_change'].iloc[-7] * 0.9997:
        if ema100 > ema200 and df['ema100'].iloc[-7] > df['ema200'].iloc[-7]:
            return True
        if ema100_pct_change > 0 and ema100_pct_change >= df['ema100_pct_change'].iloc[-5] * 0.9997:
            if ema50_pct_change > 0 and ema50_pct_change >= df['ema50_pct_change'].iloc[-5] * 0.9997:
                return
        # if ema25 > ema50 > ema100 > ema200 and :
    if ema200_dev >= ema_dict['top_ema200_dev_binmode']:
        return True


def ema_buy_bool(symbol, df):
    ema_dict = COINS_DICT[symbol]
    if df['ema200_dev'].iloc[-1] <= ema_dict['dip_ema200_dev_binmode']:
        return True


def get_futures_balance_info(symbol_asset=''):
    if symbol_asset and symbol_asset[-4:] in ['USDT', 'USDC']:
        if symbol_asset not in ['USDT', 'USDC']:
            symbol_asset = symbol_asset[:-4]
    try:
        futures_balance = CLIENT.futures_account_balance()
        balance_dict = {}
        total_futures_usdt_balance = 0.0

        for asset in futures_balance:
            balance = float(asset['balance'])
            if asset['asset'] == 'USDT':
                total_futures_usdt_balance += balance
                balance_dict['USDT'] = balance
                if symbol_asset == 'USDT':
                    return balance
            elif asset['asset'] == 'USDC':
                total_futures_usdt_balance += balance
                balance_dict['USDC'] = balance
                if symbol_asset == 'USDC':
                    return balance
            else:
                symbol = f"{asset['asset']}USDT"
                try:
                    price_data = CLIENT.futures_symbol_ticker(symbol=symbol)
                    current_price = float(price_data['price'])
                    # Convert the asset balance to USDT
                    total_futures_usdt_balance += balance * current_price
                    balance_dict[asset['asset']] = balance * current_price
                    if symbol_asset == asset['asset']:
                        return balance * current_price
                except Exception as e:
                    print(f"Could not fetch price for {symbol}: {e}")

        print(f"Total futures USDT equivalent balance: {total_futures_usdt_balance}")
        return total_futures_usdt_balance, balance_dict
    except Exception as e:
        print(f"Error: {e}")
        return

def get_spot_balance_info(symbol_asset=''):
    if symbol_asset and symbol_asset[-4:] in ['USDT', 'USDC']:
        if symbol_asset not in ['USDT', 'USDC']:
            symbol_asset = symbol_asset[:-4]
    try:
        spot_account_info = CLIENT.get_account()
        balance_dict = {}
        total_spot_usdt_balance = 0.0

        for asset in spot_account_info['balances']:
            balance = float(asset['free'])
            if asset['asset'] == 'USDT':
                total_spot_usdt_balance += balance
                if symbol_asset == 'USDT':
                    return balance
                balance_dict['USDT'] = balance
            elif asset['asset'] == 'USDC':
                total_spot_usdt_balance += balance
                if symbol_asset == 'USDC':
                    return balance
                balance_dict['USDC'] = balance
            else:
                symbol = f"{asset['asset']}USDT"
                try:
                    # Fetch the price of the asset in USDT
                    price_data = CLIENT.get_symbol_ticker(symbol=symbol)
                    current_price = float(price_data['price'])
                    # Convert the asset balance to USDT equivalent
                    total_spot_usdt_balance += balance * current_price
                    if symbol_asset == asset['asset']:
                        return balance * current_price
                    if balance > 0:
                        balance_dict[asset['asset']] = balance * current_price
                except Exception as e:
                    print(f"Could not fetch price for {symbol}: {e}")

        print(f"Total spot USDT equivalent balance: {total_spot_usdt_balance}")
        return total_spot_usdt_balance, balance_dict
    except Exception as e:
        # print(f"Error: {e}")
        return


if __name__ == '__main__':
    all_positions = CLIENT.futures_position_information()
    REALIZED_ORDERS_DICT = {}

    if os.path.exists(orders_dict_pkl_path):
        REALIZED_ORDERS_DICT = save_load_pkl(orders_dict_pkl_path)
        if 'real_order_value_total' not in REALIZED_ORDERS_DICT.keys():
            REALIZED_ORDERS_DICT['real_order_value_total'] = 0
    else:
        REALIZED_ORDERS_DICT['real_order_value_total'] = sum(abs(float(pos['notional']) - float(pos['unRealizedProfit'])) for pos in all_positions)

    REALIZED_ORDERS_DICT['balance'] = get_futures_balance_info()[0]
    REALIZED_ORDERS_DICT['orders_total_limit'] = 3 * REALIZED_ORDERS_DICT['balance']
    save_load_pkl(orders_dict_pkl_path, REALIZED_ORDERS_DICT)

    time_now = datetime.now(timezone.utc)
    print(f"\n.... Running schedule at {time_now.strftime('%Y.%m.%d %H:%M:%S')} UTC\n")

    select_future_order()
    schedule.every(1).minutes.do(select_future_order)

    # Run the scheduler loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)  # Sleep for a short duration to avoid high CPU usage
        except:
            pass

    '''
    futures_client = UMFutures(key=API_KEY, secret=API_SECRET)
    symbol = ('mavia' + 'USDT').upper()
    symbol_obj = SingleCoin(futures_client, symbol, time_frame=1, window=210)
    df_symbol = symbol_obj.df_symbol
    print(df_symbol.columns)
    print(df_symbol['ema200'].iloc[-1])
    '''



