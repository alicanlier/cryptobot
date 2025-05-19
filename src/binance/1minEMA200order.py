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
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config.config as config
import utility.calculation_tools as calculate
import futures_prices as futures
import utility.strategy_tools as strategy
from src.binance.futures_prices import SingleCoin
from scipy.signal import argrelextrema

# Connect to Binance API
api_key_dict = config.fetch_credentials()
API_KEY, API_SECRET = api_key_dict['API_KEY'], api_key_dict['API_SECRET']
CLIENT = Client(API_KEY, API_SECRET)
INTERVAL_DICT = config.fetch_interval_dict(CLIENT)
TIME_DICT = config.time_dict()

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

    sma1min_coins_ordered_dict = futures.scan_futures_for_sma(CLIENT, 1, WINDOW)
    sma5min_coins_ordered_dict = futures.scan_futures_for_sma(CLIENT, 5, WINDOW)
    sma15min_coins_ordered_dict = futures.scan_futures_for_sma(CLIENT, TIME_FRAME, WINDOW)
    sma4h_coins_ordered_dict = futures.scan_futures_for_sma(CLIENT, 4, WINDOW)

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


def get_balance_info():
    # account_info = CLIENT.get_account()
    futures_balance = CLIENT.futures_account_balance()

    for asset in futures_balance:
        if float(asset['balance']) > 0:
            print(asset, type(asset))
            print(f"{asset['asset']}: {asset['balance']}")

    # return


def select_future_order():
    select_list1 = ['1INCHUSDT', 'AAVEUSDT', 'ADAUSDT', 'AGIXUSDT', 'AIUSDT', 'ALGOUSDT', 'ALICEUSDT', 'ALTUSDT', 'APEUSDT', 'APTUSDT', 'AXSUSDT']
    select_list2 = ['ARKMUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BAKEUSDT', 'BALUSDT', 'BANDUSDT', 'BCHUSDT', 'BELUSDT', 'BLZUSDT', 'C98USDT', 'CAKEUSDT', 'CELOUSDT', 'CHRUSDT', 'COTIUSDT', 'CRVUSDT', 'CYBERUSDT']
    select_list3 = ['DENTUSDT', 'DYDXUSDT', 'EDUUSDT', 'EOSUSDT', 'FETUSDT', 'FILUSDT', 'FLMUSDT', 'GASUSDT', 'GLMUSDT', 'GRTUSDT', 'HIFIUSDT', 'HIGHUSDT']
    select_list4 = ['JASMYUSDT', 'JUPUSDT', 'KAVAUSDT', 'KNCUSDT', 'LDOUSDT', 'LPTUSDT', 'MAGICUSDT', 'MASKUSDT', 'MATICUSDT', 'MBOXUSDT', 'MDTUSDT', 'MEMEUSDT']
    select_list5 = ['NEARUSDT', 'NFPUSDT', 'NMRUSDT', 'OPUSDT', 'PHBUSDT', 'PIXELUSDT', 'RNDRUSDT', 'SANDUSDT', 'SEIUSDT', 'SPELLUSDT', 'STRAXUSDT', 'SUIUSDT', 'SUSHIUSDT']
    select_list6 = ['TIAUSDT', 'UNFIUSDT', 'UNIUSDT', 'WAVESUSDT', 'WLDUSDT', 'XAIUSDT', 'XVGUSDT', 'ZILUSDT']
    select_list7_1000 = ['BONKUSDT', 'FLOKIUSDT', 'PEPEUSDT', 'SHIBUSDT']
    select_list8_nospot = ['BIGTIMEUSDT']
    select_list = select_list1 + select_list2 + select_list3 + select_list4 + select_list5 + select_list6 + select_list7_1000
    select_list = ['XRPUSDT']

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
        except ClientError as error:
            logging.error("Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )

        # limit_order = CLIENT.futures_create_order(
        #     symbol=symbol,
        #     side=side,
        #     type='LIMIT',  # 'MARKET', 'LIMIT', etc. Check Binance API docs for all options
        #     quantity=quantity,
        #     price=price,
        #     timeInForce='GTC')
        # return limit_order

    # print(f"{'{:<10}'.format('symbol')}: {'{:<18}'.format('current_price')} | {'{:<18}'.format('5min bb_sma')} | {'{:<18}'.format('5min upper_band')} | 5min lower_band\n{'-' * 88}")
    for symbol in select_list:
        symbol_obj = SingleCoin(client, symbol, time_frame=1, window=200)
        df_symbol = symbol_obj.df_symbol
        symbol_obj5 = SingleCoin(client, symbol, time_frame=5, window=200)
        df_symbol5 = symbol_obj5.df_symbol
        
        order_dict = strategy.bb_check(CLIENT, symbol)
        if order_dict:
            order_dict = {key: calculate.round_long_decimals(value) for key, value in order_dict.items()}
            current_price = order_dict['current_price']
            if symbol in select_list7_1000:
                current_price = 1000 * current_price
                order_dict['SELL'] = 1000 * order_dict['SELL']
                order_dict['BUY'] = 1000 * order_dict['BUY']

            print(f'trying order for {symbol}... {order_dict}', end='')

            if symbol in select_list7_1000:
                symbol = str(1000) + symbol

            try:
                order_amount = 8
                quantity = int(round((order_amount / current_price), 0))  # Quantity to buy or sell 12 USDT
                place_order(symbol,'SELL', quantity, order_dict['SELL'])
                place_order(symbol, 'BUY', quantity, order_dict['BUY'])
                print(f'  --------  orders have been implemented.\n')
            except ValueError as ve:
                print(f'  --------  ERROR: {ve}')
                place_order(symbol, 'SELL', 1, order_dict['SELL'])
                place_order(symbol, 'BUY', 1, order_dict['BUY'])

def continuous_bollinger_order(time_frame=5):
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
        max_price19 = df_symbol['high'].tail(9).max()
        max_price_index = df_symbol.index[df_symbol['high'] == max_price9].tolist()[-1]
        max_price_bool = max_price9 > df_symbol['upper_band'].loc[max_price_index]
        upper_band = df_symbol['upper_band'].iloc[-1]
        lower_band = df_symbol['lower_band'].iloc[-1]
        bb_sma = futures.round_long_decimals(df_symbol['bb_sma'].iloc[-1], prec=4)
        bb_std = df_symbol['bb_std'].iloc[-1]
        bb_sma_diff = df_symbol['bb_sma_diff'].iloc[-1]
        mean_bb_sma_diff = futures.round_long_decimals(df_symbol['bb_sma_diff'][-3:].mean(), prec=4)

        print(f'Trying {symbol} >> price:{current_price} | ', end='')
        print(f"bb_sma_diff: {mean_bb_sma_diff}/{bb_sma_diff} | max: {max_price9} | min: {min_price9} | upper_band: {upper_band} | lower_band: {lower_band}")

        try:
            order_amount = 10
            # plot_symbol(df_symbol, 0.5, 0.49)
            quantity = int(round((order_amount / current_price), 0))  # Quantity to buy or sell 12 USDT
            if current_price > bb_sma + 0.9 * bb_std and current_price < max_price9 and max_price_bool and mean_bb_sma_diff < 0.07:
                sell_price = futures.round_long_decimals(0.999 * current_price, prec=4)
                place_order(symbol,'SELL', quantity, sell_price)
                print(f'  --------  SOLD {symbol[:-4]} at {sell_price}.')
                place_order(symbol, 'BUY', quantity, bb_sma)
                print(f'  --------  BUY ORDER {symbol[:-4]} at {bb_sma}.')
                # plot_symbol(df_symbol, sell_price, bb_sma)
            elif bb_sma - 0.2 * bb_std < current_price < bb_sma and current_price < max_price4 < max_price19 * 0.995 and mean_bb_sma_diff < -0.09:
                sell_price = futures.round_long_decimals(0.999 * current_price, prec=4)
                place_order(symbol,'SELL', quantity, sell_price)
                print(f'  --------  SOLD {symbol[:-4]} at {sell_price}.')
                place_order(symbol, 'BUY', quantity, lower_band)
                print(f'  --------  BUY ORDER {symbol[:-4]} at {lower_band}.')
                # plot_symbol(df_symbol, sell_price, lower_band)
            elif current_price < bb_sma - 0.9 * bb_std and current_price > min_price9 and min_price_bool and mean_bb_sma_diff > -0.07:
                buy_price = futures.round_long_decimals(1.001 * current_price, prec=4)
                place_order(symbol, 'BUY', quantity, buy_price)
                print(f'  --------  BOUGHT {symbol[:-4]} at {buy_price}.')
                place_order(symbol, 'SELL', quantity, bb_sma)
                print(f'  --------  SELL ORDER {symbol[:-4]} at {bb_sma}.')
                # plot_symbol(df_symbol, bb_sma, buy_price)
            elif bb_sma + 0.2 * bb_std > current_price > bb_sma and current_price > min_price4 > min_price19 * 1.005 and mean_bb_sma_diff > 0.09:
                buy_price = futures.round_long_decimals(1.001 * current_price, prec=4)
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


if __name__ == '__main__':
    TIME_FRAME = 15  # minute; the string format '15min' is also good
    WINDOW = 200  # number of data points
    ORDER_AMOUNT = 12  # USDT
    time_now = datetime.now()

    print(f"\n.... Running schedule at {time_now.strftime('%Y.%m.%d %H:%M:%S')}\n")

    # select_future_order()
    # get_balance_info()
    continuous_bollinger_order(1)

    # '''
    
    # Schedule the method to be called every 10 minutes
    schedule.every(2).minutes.do(continuous_bollinger_order)

    # Run the scheduler loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)  # Sleep for a short duration to avoid high CPU usage
        except:
            pass
    # '''

