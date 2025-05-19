import math
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xmlrpc.client import boolean

from dateutil import tz
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import requests
import re, os, time
import pandas as pd
import numpy as np
import decimal
import config.config as config

pd.set_option('display.max_columns', None)

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


def trendline(df, short_ema_period='', long_ema_period='', window=200):
    # macd_name = 'macd' + ''.join([str(x) for x in [short_ema_period, long_ema_period, window]])  # originally 'macd_histogram'
    if not short_ema_period or not long_ema_period or not window:
        short_ema_period = 12 if not short_ema_period else short_ema_period
        long_ema_period = 26 if not long_ema_period else long_ema_period
        window = 9 if not window else window

    def inner_calculate(df):
        df_macd = pd.DataFrame()

        # Calculate short-term and long-term exponential moving averages (EMA)
        df_macd['short_ema'] = df['close'].ewm(span=short_ema_period).mean()
        df_macd['long_ema'] = df['close'].ewm(span=long_ema_period).mean()

        # Calculate MACD line
        df_macd['macd_line'] = df_macd['short_ema'] - df_macd['long_ema']

        # Calculate signal line (EMA of MACD line)
        df_macd['signal_line'] = df_macd['macd_line'].ewm(span=window).mean()

        # Calculate MACD histogram
        df['macd'] = df_macd['macd_line'] - df_macd['signal_line']

        macd

        # Return the DataFrame with MACD values
        return df

    if isinstance(df, pd.Series):
        df = inner_calculate(df.to_frame())
        return df['macd']
    else:
        return inner_calculate(df)


def rsi(df, window=14, rsi_column_name='rsi', smooth_range=5):
    def inner_calculate(df):
        df_rsi = pd.DataFrame()
        # Calculate daily price changes
        df_rsi['price_change'] = df['close'].diff()

        # Calculate gains (positive price changes) and losses (negative price changes)
        df_rsi['gain'] = np.where(df_rsi['price_change'] > 0, df_rsi['price_change'], 0)
        df_rsi['loss'] = np.where(df_rsi['price_change'] < 0, -df_rsi['price_change'], 0)

        # Calculate average gain and average loss over a specified period (e.g., 14 days)
        df_rsi['avg_gain'] = df_rsi['gain'].rolling(window=window).mean()
        df_rsi['avg_loss'] = df_rsi['loss'].rolling(window=window).mean()

        # Calculate Relative Strength (RS)
        df_rsi['rs'] = df_rsi['avg_gain'] / df_rsi['avg_loss']

        # Calculate RSI
        df[rsi_column_name] = 100 - (100 / (1 + df_rsi['rs']))

        df[rsi_column_name + '_sma'] = df[rsi_column_name].rolling(window=smooth_range).mean()
        df[rsi_column_name + '_diff'] = df[rsi_column_name].diff()
        
        # Return the DataFrame RSI values
        return df

    if isinstance(df, pd.Series):
        df = inner_calculate(df.to_frame())
        return df[rsi_column_name]
    else:
        return inner_calculate(df)


def macd(df, short_ema_period='', long_ema_period='', window='', smooth_range=5):
    # macd_name = 'macd' + ''.join([str(x) for x in [short_ema_period, long_ema_period, window]])  # originally 'macd_histogram'
    if not short_ema_period or not long_ema_period or not window:
        short_ema_period = 12 if not short_ema_period else short_ema_period
        long_ema_period = 26 if not long_ema_period else long_ema_period
        window = 9 if not window else window

    def inner_calculate(df):
        df_macd = pd.DataFrame()

        # Calculate short-term and long-term exponential moving averages (EMA)
        df_macd['short_ema'] = df['close'].ewm(span=short_ema_period).mean()
        df_macd['long_ema'] = df['close'].ewm(span=long_ema_period).mean()

        # Calculate MACD line
        df_macd['macd_line'] = df_macd['short_ema'] - df_macd['long_ema']

        # Calculate signal line (EMA of MACD line)
        df_macd['signal_line'] = df_macd['macd_line'].ewm(span=window).mean()

        # Calculate MACD histogram
        df['macd'] = df_macd['macd_line'] - df_macd['signal_line']
        df['macd_sma'] = df['macd'].rolling(window=smooth_range).mean()
        df['macd_diff'] = df['macd'].diff()
        df['macd_sma_diff'] = df['macd_sma'].diff()

        # Return the DataFrame with MACD values
        return df

    if isinstance(df, pd.Series):
        df = inner_calculate(df.to_frame())
        return df['macd']
    else:
        return inner_calculate(df)


def ema(df, window=200, ema_column_name=''):
    if not ema_column_name:
        ema_column_name = 'ema' + str(window)

    def inner_calculate(df):
        # Calculate EMA values
        df[ema_column_name] = df['close'].ewm(span=window).mean()
        df[ema_column_name + '_diff'] = df[ema_column_name].diff()
        df[ema_column_name + '_diff'].rolling(window=3).mean()
        df[ema_column_name + '_slope'] = 100 * df[ema_column_name + '_diff'].rolling(window=3).mean()/df['close'].rolling(window=2*window, min_periods=1).mean()
        df[ema_column_name + '_pct_change'] = df[ema_column_name].pct_change() * 100

        df[ema_column_name + '_dev'] = ((df['close'] - df[ema_column_name])/df[ema_column_name]) * 100
        df[ema_column_name + '_dev_sma'] = df[ema_column_name + '_dev'].rolling(window=5).mean()
        df[ema_column_name + '_dev_sma_diff'] = df[ema_column_name + '_dev_sma'].diff()
        # df[ema_column_name + '_dev_change'] = df[ema_column_name + '_dev'].pct_change() * 100

        # Return the DataFrame with EMA values
        return df

        # closing_prices = np.array([float(kline[4]) for kline in historical_klines])
        # ema = np.convolve(closing_prices, np.ones(window), mode='valid') / window
        # return df['close'].ewm(span=window, min_periods=window, adjust=False).mean()

    if isinstance(df, pd.Series):
        df = inner_calculate(df.to_frame())
        return df[ema_column_name]
    else:
        return inner_calculate(df)


def emab(df, window=200, ema_column_name=''):
    if not ema_column_name:
        ema_column_name = 'emab' + str(window)

    def inner_calculate(df):
        # Calculate EMA values
        df[ema_column_name] = df['close'].ewm(span=window).mean()
        df[ema_column_name + '_diff'] = df[ema_column_name].diff()
        # df[ema_column_name + '_change'] = df[ema_column_name].pct_change() * 100

        # 1. Calculate smoothed values for 'high' and 'low'
        df['smooth_high'] = df['high'].ewm(span=3).mean()
        df['smooth_low'] = df['low'].ewm(span=3).mean()

        # 2. Calculate averages of smoothed values
        df['avg_smooth'] = (df['smooth_high'] + df['smooth_low']) / 2

        # 3. Conditional calculation for ema_dev
        def calculate_ema_dev(row):
            if row['smooth_high'] > row[ema_column_name] and row['avg_smooth'] > row[ema_column_name]:
                return ((row['high'] - row[ema_column_name]) / row[ema_column_name]) * 100
            elif row['smooth_low'] < row[ema_column_name] and row['avg_smooth'] < row[ema_column_name]:
                return ((row['low'] - row[ema_column_name]) / row[ema_column_name]) * 100
            else:
                return 0  # Or set another default value

        df[ema_column_name + '_dev'] = df.apply(calculate_ema_dev, axis=1)
        # df[ema_column_name + '_dev'] = ((df['close'] - df[ema_column_name])/df[ema_column_name]) * 100
        df[ema_column_name + '_dev_sma'] = df[ema_column_name + '_dev'].rolling(window=5).mean()
        df[ema_column_name + '_dev_sma_diff'] = df[ema_column_name + '_dev_sma'].diff()
        # df[ema_column_name + '_dev_change'] = df[ema_column_name + '_dev'].pct_change() * 100

        # Return the DataFrame with EMA values
        return df

        # closing_prices = np.array([float(kline[4]) for kline in historical_klines])
        # ema = np.convolve(closing_prices, np.ones(window), mode='valid') / window
        # return df['close'].ewm(span=window, min_periods=window, adjust=False).mean()

    if isinstance(df, pd.Series):
        df = inner_calculate(df.to_frame())
        return df[ema_column_name]
    else:
        return inner_calculate(df)


def sma(df, window=200, sma_column_name=''):
    if not sma_column_name:
        sma_column_name = 'sma' + str(window)

    def inner_calculate(df):
        df[sma_column_name] = df['close'].rolling(window=window).mean()
        df[sma_column_name + '_diff'] = df[sma_column_name].diff()
        # df[sma_column_name + '_change'] = df[sma_column_name].pct_change() * 100

        df[sma_column_name + '_dev'] = ((df['close'] - df[sma_column_name])/df[sma_column_name]) * 100
        df[sma_column_name + '_dev_sma'] = df[sma_column_name + '_dev'].rolling(window=5).mean()
        df[sma_column_name + '_dev_sma_diff'] = df[sma_column_name + '_dev_sma'].diff()
        # df[sma_column_name + '_dev_change'] = df[sma_column_name + '_dev'].pct_change() * 100

        return df

    if isinstance(df, pd.Series):
        df = inner_calculate(df.to_frame())
        return df[sma_column_name]
    else:
        return inner_calculate(df)


def sma_ranking_list(coin_data, window=200, lowest_change=0.1, ranking_type='+', print_option=True):
    value_keywords = ['last_sma', 'last_close_price', 'dev_ratio_sma']
    sma_df = pd.DataFrame(columns=['index_symbol', 'symbol'] + value_keywords)
    sma_df.set_index('index_symbol', inplace=True)

    def add_new_row(sma_df, symbol, sma_dict):
        sma_dict['index_symbol'] = symbol
        sma_dict['symbol'] = symbol
        sma_df.loc[symbol] = sma_dict

        return sma_df

    for symbol, df in coin_data.items():
        if len(df) >= window:  # En az veri sınırlaması
            sma_series = sma(df['close'])
            last_sma = sma_series.iloc[-1]
            if np.isnan(last_sma):  # değeri olmayanları atlayın
                continue
            last_close_price = df['close'].iloc[-1]
            dev_ratio_sma = (last_close_price - last_sma) * 100 / last_sma
            sma_dict = dict(zip(value_keywords, [last_sma, last_close_price, dev_ratio_sma]))

            if ranking_type.lower() in ['+', 'positive', 'over', 'above']:
                if last_close_price > last_sma and dev_ratio_sma >=  lowest_change:
                    sma_df = add_new_row(sma_df, symbol, sma_dict)
            elif ranking_type.lower() in ['-', 'negative', 'under', 'below']:
                if last_close_price < last_sma and dev_ratio_sma <= -abs(lowest_change):
                    sma_df = add_new_row(sma_df, symbol, sma_dict)
            else:
                sma_df = add_new_row(sma_df, symbol, sma_dict)

    sma_df.dropna(axis=1, how='all')
    ascend = True if ranking_type.lower() in ['-', 'negative', 'under', 'below'] else False
    sorted_sma_df = sma_df.sort_values(by='dev_ratio_sma', ascending=ascend)

    if print_option:
        # Sonuçları yazdırma
        print(f"\n\tCoin price compared to SMA{window} ({len(sma_df)}/{len(coin_data)}):")
        print(f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('last_sma')} {'{:<11}'.format('last_close')} dev\n{'-'*48}")
        for index, row in sorted_sma_df.iterrows():
            symbol = index
            last_sma = round_long_decimals(row['last_sma'], 4)
            last_close_price = round_long_decimals(row['last_close_price'], 4)
            dev_ratio_sma = round_long_decimals(row['dev_ratio_sma'], 4)
            dev_ratio_sma = "{:.1f}%".format(dev_ratio_sma)
            print(f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(last_sma)} {'{:<11}'.format(last_close_price)} {dev_ratio_sma}")

    return sorted_sma_df


def ema_ranking_list(coin_data, window=200, lowest_change=0.1, ranking_type='+', print_option=True):
    value_keywords = ['last_ema', 'last_close_price', 'dev_ratio_ema']
    ema_df = pd.DataFrame(columns=['index_symbol', 'symbol'] + value_keywords)
    ema_df.set_index('index_symbol', inplace=True)

    def add_new_row(ema_df, symbol, ema_dict):
        ema_dict['index_symbol'] = symbol
        ema_dict['symbol'] = symbol
        ema_df.loc[symbol] = ema_dict

        return ema_df

    for symbol, df in coin_data.items():
        if len(df) >= window:  # En az veri sınırlaması
            ema_series = ema(df['close'])
            last_ema = ema_series.iloc[-1]
            if np.isnan(last_ema):  # değeri olmayanları atlayın
                continue
            last_close_price = df['close'].iloc[-1]
            dev_ratio_ema = (last_close_price - last_ema) * 100 / last_ema
            ema_dict = dict(zip(value_keywords, [last_ema, last_close_price, dev_ratio_ema]))

            ema_df.dropna(axis=1, how='all')

            if ranking_type.lower() in ['+', 'positive', 'over', 'above']:
                if last_close_price > last_ema and dev_ratio_ema >=  lowest_change:
                    ema_df = add_new_row(ema_df, symbol, ema_dict)
            elif ranking_type.lower() in ['-', 'negative', 'under', 'below']:
                if last_close_price < last_ema and dev_ratio_ema <= -abs(lowest_change):
                    ema_df = add_new_row(ema_df, symbol, ema_dict)
            else:
                ema_df = add_new_row(ema_df, symbol, ema_dict)

    ascend = True if ranking_type.lower() in ['-', 'negative', 'under', 'below'] else False
    sorted_ema_df = ema_df.sort_values(by='dev_ratio_ema', ascending=ascend)

    if print_option:
        # Sonuçları yazdırma
        print(f"\n\tCoin price compared to EMA{window} ({len(ema_df)}/{len(coin_data)}):")
        print(f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('last_ema')} {'{:<11}'.format('last_close')} dev\n{'-'*48}")
        for index, row in sorted_ema_df.iterrows():
            symbol = index
            last_ema = round_long_decimals(row['last_ema'], 4)
            last_close_price = round_long_decimals(row['last_close_price'], 4)
            dev_ratio_ema = round_long_decimals(row['dev_ratio_ema'], 4)
            dev_ratio_ema = "{:.1f}%".format(dev_ratio_ema)
            print(f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(last_ema)} {'{:<11}'.format(last_close_price)} {dev_ratio_ema}")

    return sorted_ema_df


def sma_ema_common_ranking(coin_data, window=200, lowest_change=0.1, ranking_type='+'):
    sma_rank_df = sma_ranking_list(coin_data, window, lowest_change, ranking_type)
    ema_rank_df = ema_ranking_list(coin_data, window, lowest_change, ranking_type)
    print(ema_rank_df.columns)
    print(sma_rank_df.columns)
    print(ema_rank_df.columns)

    common_df = sma_rank_df.merge(ema_rank_df[['symbol', 'last_ema', 'dev_ratio_ema']], on='index_symbol', how='inner', suffixes=('_sma','_ema'))

    # Sonuçları yazdırma
    print(f"\n\tSMA{window}&EMA{window} intersection list ({len(common_df)}/{len(coin_data)}):")
    print(f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('last_close')} {'{:<11}'.format('last_ema')} {'{:<11}'.format('ema_dev')} {'{:<11}'.format('last_sma')} sma_dev\n{'-'*48}")
    for index, row in common_df.iterrows():
        symbol = index
        last_close_price = round_long_decimals(row['last_close_price'], 4)
        last_sma = round_long_decimals(row['last_sma'], 4)
        sma_dev = round_long_decimals(row['dev_ratio_sma'], 4)
        sma_dev = "{:.1f}%".format(sma_dev)
        last_ema = round_long_decimals(row['last_ema'], 4)
        ema_dev = round_long_decimals(row['dev_ratio_ema'], 4)
        ema_dev = "{:.1f}%".format(ema_dev)
        print(f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(last_close_price)} {'{:<11}'.format(last_ema)} {'{:<11}'.format(ema_dev)} {'{:<11}'.format(last_sma)} {sma_dev}")

    return common_df


def macd_ranking_list(coin_data, window=200, lowest_change=0.1, ranking_type='+', print_option=True):
    value_keywords = ['last_close_price', 'minus2_macd', 'last_macd', 'dev_ratio_macd']
    macd_df = pd.DataFrame(columns=['symbol'] + value_keywords)
    macd_df.set_index('symbol', inplace=True)

    def add_new_row(macd_df, symbol, macd_dict):
        macd_dict['symbol'] = symbol
        macd_df.loc[symbol] = macd_dict

        return macd_df

    for symbol, df in coin_data.items():
        if len(df) >= window:  # En az veri sınırlaması
            macd_series = macd(df['close'])
            last_macd = macd_series.iloc[-1]
            if np.isnan(last_macd):  # değeri olmayanları atlayın
                continue
            minus2_macd = macd_series.iloc[-2]
            last_close_price = df['close'].iloc[-1]
            dev_ratio_macd = (last_macd - minus2_macd) * 100 / minus2_macd
            macd_dict = dict(zip(value_keywords, [last_close_price, minus2_macd, last_macd, dev_ratio_macd]))

            macd_df.dropna(axis=1, how='all')

            if ranking_type.lower() in ['+', 'positive', 'over', 'above']:
                if dev_ratio_macd >= lowest_change:
                    macd_df = add_new_row(macd_df, symbol, macd_dict)
            elif ranking_type.lower() in ['-', 'negative', 'under', 'below']:
                if dev_ratio_macd <= -abs(lowest_change):
                    macd_df = add_new_row(macd_df, symbol, macd_dict)
            else:
                macd_df = add_new_row(macd_df, symbol, macd_dict)

    ascend = True if ranking_type.lower() in ['-', 'negative', 'under', 'below', 'descend'] else False
    sorted_macd_df = macd_df.sort_values(by='dev_ratio_macd', ascending=ascend)

    if print_option:
        # Sonuçları yazdırma
        print(f"\n\tCoin price compared to MACD{window} ({len(macd_df)}/{len(coin_data)}):")
        print(f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('last_close')} {'{:<11}'.format('minus2_macd')} {'{:<11}'.format('last_macd')} dev\n{'-'*48}")
        for index, row in sorted_macd_df.iterrows():
            symbol = index
            last_macd = round_long_decimals(row['last_macd'], 4)
            minus2_macd = round_long_decimals(row['minus2_macd'], 4)
            last_close_price = round_long_decimals(row['last_close_price'], 4)
            dev_ratio_macd = round_long_decimals(row['dev_ratio_macd'], 4)
            dev_ratio_macd = "{:.1f}%".format(dev_ratio_macd)
            print(f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(last_close_price)} {'{:<11}'.format(minus2_macd)} {'{:<11}'.format(last_macd)} {dev_ratio_macd}")

    return sorted_macd_df


def rsi_ranking_list(coin_data, window=200, lowest_change=0.1, ranking_type='+', print_option=True):
    value_keywords = ['last_close_price', 'minus2_rsi', 'last_rsi', 'dev_ratio_rsi']
    rsi_df = pd.DataFrame(columns=['symbol'] + value_keywords)
    rsi_df.set_index('symbol', inplace=True)

    def add_new_row(rsi_df, symbol, rsi_dict):
        rsi_dict['symbol'] = symbol
        rsi_df.loc[symbol] = rsi_dict

        return rsi_df

    for symbol, df in coin_data.items():
        if len(df) >= window:  # En az veri sınırlaması
            rsi_series = rsi(df['close'])
            last_rsi = rsi_series.iloc[-1]
            if np.isnan(last_rsi):  # değeri olmayanları atlayın
                continue
            minus2_rsi = rsi_series.iloc[-2]
            last_close_price = df['close'].iloc[-1]
            dev_ratio_rsi = (last_rsi - minus2_rsi) * 100 / minus2_rsi
            rsi_dict = dict(zip(value_keywords, [last_close_price, minus2_rsi, last_rsi, dev_ratio_rsi]))

            rsi_df.dropna(axis=1, how='all')

            if ranking_type.lower() in ['+', 'positive', 'over', 'above']:
                if dev_ratio_rsi >= lowest_change:
                    rsi_df = add_new_row(rsi_df, symbol, rsi_dict)
            elif ranking_type.lower() in ['-', 'negative', 'under', 'below']:
                if dev_ratio_rsi <= -abs(lowest_change):
                    rsi_df = add_new_row(rsi_df, symbol, rsi_dict)
            else:
                rsi_df = add_new_row(rsi_df, symbol, rsi_dict)

    ascend = True if ranking_type.lower() in ['-', 'negative', 'under', 'below', 'descend'] else False
    sorted_rsi_df = rsi_df.sort_values(by='dev_ratio_rsi', ascending=ascend)

    if print_option:
        # Sonuçları yazdırma
        print(f"\n\tCoin price compared to RSI{window} ({len(rsi_df)}/{len(coin_data)}):")
        print(f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('last_close')} {'{:<11}'.format('minus2_rsi')} {'{:<11}'.format('last_rsi')} rsi_dev\n{'-'*48}")
        for index, row in sorted_rsi_df.iterrows():
            symbol = index
            minus2_rsi = round_long_decimals(row['minus2_rsi'], 4)
            last_rsi = round_long_decimals(row['last_rsi'], 4)
            last_close_price = round_long_decimals(row['last_close_price'], 4)
            dev_ratio_rsi = round_long_decimals(row['dev_ratio_rsi'], 4)
            dev_ratio_rsi = "{:.1f}%".format(dev_ratio_rsi)
            print(f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(last_close_price)} {'{:<11}'.format(minus2_rsi)} {'{:<11}'.format(last_rsi)} {dev_ratio_rsi}")

    return sorted_rsi_df


def price_change_ranking(coin_data, window=200, lowest_change=0.1, ranking_type='+', print_option=True):
    value_keywords = ['minus2_close_price', 'last_close_price', 'last_price_change']
    price_change_df = pd.DataFrame(columns=['symbol'] + value_keywords)
    price_change_df.set_index('symbol', inplace=True)

    def add_new_row(price_change_df, symbol, price_change_dict):
        price_change_dict['symbol'] = symbol
        price_change_df.loc[symbol] = price_change_dict

        return price_change_df

    for symbol, df in coin_data.items():
        if len(df) >= window:  # En az veri sınırlaması
            price_change_series = df['close'].pct_change() * 100
            price_change_series.dropna()
            last_price_change = price_change_series.iloc[-1]
            if np.isnan(last_price_change):  # değeri olmayanları atlayın
                continue
            last_close_price = df['close'].iloc[-1]
            minus2_close_price = df['close'].iloc[-2]

            price_change_dict = dict(zip(value_keywords, [minus2_close_price, last_close_price, last_price_change]))

            price_change_df.dropna(axis=1, how='all')

            if ranking_type.lower() in ['+', 'positive', 'over', 'above']:
                if last_price_change >= lowest_change:
                    price_change_df = add_new_row(price_change_df, symbol, price_change_dict)
            elif ranking_type.lower() in ['-', 'negative', 'under', 'below']:
                if last_close_price <= -abs(lowest_change):
                    price_change_df = add_new_row(price_change_df, symbol, price_change_dict)
            else:
                price_change_df = add_new_row(price_change_df, symbol, price_change_dict)

    ascend = True if ranking_type.lower() in ['-', 'negative', 'under', 'below'] else False
    sorted_price_change_df = price_change_df.sort_values(by='last_price_change', ascending=ascend)

    if print_option:
        # Sonuçları yazdırma
        print(f"\n\tCoin price change ratio ({len(price_change_df)}/{len(coin_data)}):")
        print(f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('minus2_close')} {'{:<11}'.format('last_close')} dev\n{'-'*48}")
        for index, row in sorted_price_change_df.iterrows():
            symbol = index
            last_close_price = round_long_decimals(row['last_close_price'], 4)
            minus2_close_price = round_long_decimals(row['minus2_close_price'], 4)
            last_price_change = round_long_decimals(row['last_price_change'], 4)
            last_price_change = "{:.1f}%".format(last_price_change)
            print(f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(minus2_close_price)} {'{:<11}'.format(last_close_price)} {last_price_change}")

    return sorted_price_change_df


def vol_change_ranking(coin_data, window=200, lowest_change=0.1, ranking_type='+', print_option=True):
    value_keywords = ['last_close_price', 'minus2_volume', 'last_normalized_volume', 'last_vol_change']
    vol_change_df = pd.DataFrame(columns=['symbol'] + value_keywords)
    vol_change_df.set_index('symbol', inplace=True)

    def add_new_row(vol_change_df, symbol, vol_change_dict):
        vol_change_dict['symbol'] = symbol
        vol_change_df.loc[symbol] = vol_change_dict

        return vol_change_df

    for symbol, df in coin_data.items():
        if len(df) >= window:  # En az veri sınırlaması
            time_interval_series = df['close_time'].diff()
            time_interval_series.dropna()
            time_interval_series.iloc[-1] = int(time.time() * 1000) - df['close_time'].iloc[-2]

            per_time_vol_series = df['volume']/time_interval_series
            normalized_vol_change_series = per_time_vol_series.pct_change() * 100

            last_vol_change = normalized_vol_change_series.iloc[-1]
            last_normalized_volume = df['volume'].iloc[-1] * (time_interval_series.iloc[-2]/time_interval_series.iloc[-1])
            minus2_volume = df['volume'].iloc[-2]

            if np.isnan(last_vol_change):  # değeri olmayanları atlayın
                continue
            last_close_price = df['close'].iloc[-1]

            vol_change_dict = dict(zip(value_keywords, [last_close_price, minus2_volume, last_normalized_volume, last_vol_change]))

            vol_change_df.dropna(axis=1, how='all')

            if ranking_type.lower() in ['+', 'positive', 'over', 'above']:
                if last_vol_change >= lowest_change:
                    vol_change_df = add_new_row(vol_change_df, symbol, vol_change_dict)
            elif ranking_type.lower() in ['-', 'negative', 'under', 'below']:
                if last_vol_change <= -abs(lowest_change):
                    vol_change_df = add_new_row(vol_change_df, symbol, vol_change_dict)
            else:
                vol_change_df = add_new_row(vol_change_df, symbol, vol_change_dict)

    ascend = True if ranking_type.lower() in ['-', 'negative', 'under', 'below'] else False
    sorted_vol_change_df = vol_change_df.sort_values(by='last_vol_change', ascending=ascend)

    if print_option:
        # Sonuçları yazdırma
        print(f"\n\tCoin volume change ratio ({len(vol_change_df)}/{len(coin_data)}):")
        print(f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('last_close')} {'{:<14}'.format('minus2_volume')} {'{:<14}'.format('last_norm_vol')} dev\n{'-' * 68}")
        for index, row in sorted_vol_change_df.iterrows():
            symbol = index
            last_close_price = round_long_decimals(row['last_close_price'], 4)
            minus2_volume = round_long_decimals(row['minus2_volume'], 4)
            last_normalized_volume = round_long_decimals(row['last_normalized_volume'], 4)
            last_vol_change = round_long_decimals(row['last_vol_change'], 4)
            last_vol_change = "{:.1f}%".format(last_vol_change)
            print(f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(last_close_price)} {'{:<14}'.format(minus2_volume)} {'{:<14}'.format(last_normalized_volume)} {last_vol_change}")

    return sorted_vol_change_df


def std(df, window=20, std_column_name='std'):

    is_series = isinstance(df, pd.Series)

    def inner_calculate(df):
        df[std_column_name] = df['close'].rolling(window=window).std()
        if not is_series:
            df[std_column_name + '_diff'] = df[std_column_name].diff()
            df[std_column_name + '_change'] = df[std_column_name].pct_change() * 100

        return df

    if is_series:
        df = df.to_frame(name='close')
    elif not is_series and 'close' not in df.columns:
        raise ValueError("DataFrame/Series must contain a 'close' column")

    # Apply the inner function
    df = inner_calculate(df)

    return df[std_column_name] if is_series else df


def bb(df, window=20, range=2):
    def inner_calculate(df):
        df['bb_std'] = std(df, window, std_column_name='bb_std')['bb_std'] * range
        df['bb_sma'] = sma(df, window, sma_column_name='bb_sma')['bb_sma']
        df['lower_band'] = (df['bb_sma'] - df['bb_std']).apply(lambda x: round_long_decimals(x, prec=6) if not pd.isna(x) else float('nan'))
        df['upper_band'] = (df['bb_sma'] + df['bb_std']).apply(lambda x: round_long_decimals(x, prec=6) if not pd.isna(x) else float('nan'))
        df['bb_sma_diff'] = df['bb_sma'].diff()
        df['bb_std_diff'] = df['bb_std'].diff()
        # df['bb_sma_change'] = (df['bb_sma'].pct_change()).apply(lambda x: round_long_decimals(x, prec=6) if not pd.isna(x) else float('nan')) * 100
        # df['bb_std_change'] = (df['bb_std'].pct_change()).apply(lambda x: round_long_decimals(x, prec=6) if not pd.isna(x) else float('nan')) * 100

        return df

    if isinstance(df, pd.Series):
        return inner_calculate(df.to_frame())
    else:
        return inner_calculate(df)


def fib(swing_high, swing_low, current_price=0):
    fib_levels = [0, 14.6, 23.6, 38.2, 50, 61.8, 76.4, 78.6, 85.4, 100, 127.2, 161.8, 261.8, 423.6]
    # fib_levels = [0, 23.6, 38.2, 50, 61.8, 78.6, 85.4, 100]
    # resistance_levels = [swing_high - (swing_high - swing_low) * level/100 for level in fib_levels if level <= 100]
    # support_levels = [swing_low + (swing_high - swing_low) * level/100 for level in fib_levels if level <= 100]
    # extension_levels = [swing_high + (swing_high - swing_low) * (level/100 - 1.0) for level in fib_levels if level > 100]

    resistance_levels = {f"{level}%": round_to_significant_figures((swing_high - (swing_high - swing_low) * level/100), 1) for level in fib_levels if level <= 100}
    support_levels = {f"{level}%": round_to_significant_figures((swing_low + (swing_high - swing_low) * level/100), 1) for level in fib_levels if level <= 100}
    extension_levels = {f"{level}%": round_to_significant_figures((swing_high + (swing_high - swing_low) * (level/100 - 1.0)), 1) for level in fib_levels if level > 100}
    closest_fib_level = 0

    if current_price > 0:
        # combined_levels = resistance_levels.copy()  # Start with resistance levels
        # for key, value in support_levels.items():
        #     # If the key already exists, choose the smaller value (support vs. resistance)
        #     if key in combined_levels:
        #         combined_levels[key] = min(combined_levels[key], value)
        #     else:
        #         combined_levels[key] = value
        # print(10, combined_levels)

        resistance_levels2 = {(item[1], 'resistance'):item[1] for item in resistance_levels.items()}
        support_levels2 = {(item[1], 'support'):item[1] for item in support_levels.items()}
        combined_levels = {**resistance_levels2, **support_levels2}

        # Sort the combined levels by their values in ascending order
        sup_res_levels = dict(sorted(combined_levels.items(), key=lambda item: item[1]))
        closest_fib_level = min(sup_res_levels.items(), key=lambda x: abs(x[1] - current_price))
        closest_fib_level_res = min(resistance_levels2.items(), key=lambda x: abs(x[1] - current_price))
        closest_fib_level_sup = min(support_levels2.items(), key=lambda x: abs(x[1] - current_price))
        if closest_fib_level_res[1] == closest_fib_level_sup[1]:
            if current_price > closest_fib_level_sup[1]:
                closest_fib_level = closest_fib_level_sup
            else:
                closest_fib_level = closest_fib_level_res

    return {"resistance_fibs": resistance_levels, "support_fibs": support_levels, "extension_fibs": extension_levels, "closest_fib_level": closest_fib_level}


def round_long_decimals(num, prec=6):

    def round_to_nonzero_digits(inner_num, inner_prec=prec):
        # Convert the float number to a Decimal with the appropriate precision
        decimal.getcontext().prec = inner_prec
        decimal_num = decimal.Decimal(str(inner_num))

        # Round the Decimal number to its last 'inner_prec' nonzero digits
        rounded_decimal_num = decimal_num.normalize()
        return float(rounded_decimal_num)

    try:
        if num < 10000:
            if num >= 1:
                num = round_to_nonzero_digits(num, inner_prec=prec)
            else:
                num = round_to_nonzero_digits(num * 1e8, inner_prec=prec) / 1e8
        else:
            num = float(int(num))

        return num

    except Exception as e:
        print(6666, e, num)


def merge_tools(coin_data, real_time_frame='15min', window=200, lowest_change=0.1, ranking_type='+', options=[]):
    options1 = ['last_price_change', 'last_vol_change', 'dev_ratio_rsi', 'dev_ratio_macd', 'dev_ratio_sma', 'dev_ratio_ema']
    options2 = ['price', 'volume', 'rsi', 'macd', 'sma', 'ema']
    functions = [price_change_ranking, vol_change_ranking, rsi_ranking_list, macd_ranking_list, sma_ranking_list, ema_ranking_list]

    dict1 = dict(zip(options1, functions))
    dict2 = dict(zip(options2, functions))
    function_dict = {**dict1, **dict2}
    options_dict = dict(zip(options1, options2))
    options_dict_reverse = {value:key for key, value in options_dict.items()}

    if not options:
        options = options1
    else:
        for i, option in enumerate(options):
            if option in options2:
                print(option)
                options[i] = options_dict_reverse[option]

    options = [option for option in options if option not in ['last_price_change', 'price']]

    df_versatile = pd.DataFrame(columns=['symbol', 'last_close_price', 'last_price_change'])
    df_versatile.set_index('symbol', inplace=True)
    df_price = price_change_ranking(coin_data, window, lowest_change, ranking_type, False)
    df_versatile[['last_close_price', 'last_price_change']] = df_price[['last_close_price', 'last_price_change']]

    for option in options:
        df_option = function_dict[option](coin_data, window, lowest_change, ranking_type, False)
        df_versatile = df_versatile.merge(df_option[[option]], on='symbol', how='inner')

    df_versatile.sort_values(by='last_price_change', ascending=False)

    # Sonuçları yazdırma
    len1 = len(df_versatile)
    len2 = len(coin_data)
    over_all_ratio = len1 * 100 / len2

    print("\n" + "\t"*3 + f"INDICATOR VALUE CHANGES FOR ({len(df_versatile)} COINS ({'{:.1f}%'.format(over_all_ratio)} of {len(coin_data)}) - {real_time_frame} time_frame):" + f"\n{'-' * 92}")
    head_line = f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('last_close')} {'{:<11}'.format('price_dev')}"
    for option in options:
        head_line += f" {'{:<11}'.format(options_dict[option] + '_dev')}"

    print(head_line + f"\n{'-' * 92}")
    # print(f"{'{:<12}'.format('symbol')}: {'{:<11}'.format('last_close')} {'{:<11}'.format('price_dev')} {'{:<11}'.format('vol_dev')} {'{:<11}'.format('vol_dev')} {'{:<11}'.format('vol_dev')} {'{:<11}'.format('vol_dev')} {'{:<11}'.format('vol_dev')} dev\n{'-' * 68}")
    for index, row in df_versatile.iterrows():
        symbol = index
        last_close_price = round_long_decimals(row['last_close_price'], 4)
        last_price_change = round_long_decimals(row['last_price_change'], 4)

        in_line = f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(last_close_price)} {'{:<11}'.format(last_price_change)}"

        for option in options:
            option_value = "{:.1f}%".format(round_long_decimals(row[option], 4))
            # last_vol_change = "{:.1f}%".format(last_vol_change)
            in_line += f" {'{:<11}'.format(option_value)}"

        print(in_line)

        # print(f"{'{:<12}'.format(symbol)}: {'{:<11}'.format(last_close_price)} {'{:<11}'.format(last_price_change)} {'{:<11}'.format(last_normalized_volume)} {last_vol_change}")


def current_futures_price(symbol):
    # Make a request to the Binance API to get the latest price
    # Binance Futures API URL for Mark Price
    response = requests.get(f'https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}')

    if response.status_code == 200:
        data = response.json()
        # print(data.keys()) # ['symbol', 'markPrice', 'indexPrice', 'estimatedSettlePrice', 'lastFundingRate', 'interestRate', 'nextFundingTime', 'time']
        current_markPrice = round_long_decimals(float(data['markPrice']))  # Mark price
        current_indexPrice = round_long_decimals(float(data['indexPrice']))  # Mark price
        lastFundingRate = round_long_decimals(float(data['lastFundingRate']))  # Mark price
        interestRate = round_long_decimals(float(data['interestRate']))  # Mark price

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


def calculate_ema_dev_extremes(df, ema_col_name, streak_length=10):
    ema_dev_col_name = ema_col_name + '_dev'
    positive_max_values = []
    negative_min_values = []

    current_streak = []
    current_sign = None
    coin_dict = {}

    # Iterate through the ema_dev column
    for index, value in df[ema_dev_col_name].items():
        if value > 0:  # Positive value
            if current_sign == "positive" or current_sign is None:
                current_streak.append(value)
            else:
                # If the streak changes, check if it qualifies and reset
                # if len(current_streak) >= streak_length:  # At least 10 consecutive
                #     if current_sign == "positive":
                #         positive_max_values.extend(get_list_frequent_extremes(current_streak, 'top'))
                #     elif current_sign == "negative":
                #         negative_min_values.extend(get_list_frequent_extremes(current_streak, 'dip'))
                # current_streak = [value]
                # current_sign = "positive"
                negative_min_values.extend(get_list_frequent_extremes(current_streak, 'dip'))
                current_streak = [value]
            current_sign = "positive"
        elif value < 0:  # Negative value
            if current_sign == "negative" or current_sign is None:
                current_streak.append(value)
            else:
                # If the streak changes, check if it qualifies and reset
                # if len(current_streak) >= streak_length:  # At least 6 consecutive
                #     if current_sign == "positive":
                #         positive_max_values.extend(get_list_frequent_extremes(current_streak, 'top'))
                #     elif current_sign == "negative":
                #         negative_min_values.extend(get_list_frequent_extremes(current_streak, 'dip'))
                # current_streak = [value]
                # current_sign = "negative"
                positive_max_values.extend(get_list_frequent_extremes(current_streak, 'top'))
                current_streak = [value]
            current_sign = "negative"
        else:  # Value is 0, reset streak
            # if len(current_streak) >= streak_length:
            #     if current_sign == "positive":
            #         positive_max_values.extend(get_list_frequent_extremes(current_streak, 'top'))
            #     elif current_sign == "negative":
            #         negative_min_values.extend(get_list_frequent_extremes(current_streak, 'dip'))
            if current_sign == "positive":
                positive_max_values.extend(get_list_frequent_extremes(current_streak, 'top'))
            elif current_sign == "negative":
                negative_min_values.extend(get_list_frequent_extremes(current_streak, 'dip'))
            current_streak = []
            current_sign = None

    # Handle remaining streak at the end
    # if len(current_streak) >= streak_length:
    #     if current_sign == "positive":
    #         positive_max_values.extend(get_list_frequent_extremes(current_streak, 'top'))
    #     elif current_sign == "negative":
    #         negative_min_values.extend(get_list_frequent_extremes(current_streak, 'dip'))
    if current_sign == "positive":
        positive_max_values.extend(get_list_frequent_extremes(current_streak, 'top'))
    elif current_sign == "negative":
        negative_min_values.extend(get_list_frequent_extremes(current_streak, 'dip'))

    positive_max_values = [round_to_significant_figures(num, 3) for num in positive_max_values]
    negative_min_values = [round_to_significant_figures(num, 3) for num in negative_min_values]
    positive_max_values.sort(reverse=True)
    negative_min_values.sort()
    bin_interval = 0.2

    top_max = max(positive_max_values)
    top_mean = round(sum(positive_max_values) / len(positive_max_values), 3)
    top_binmode = get_histogram_bin_mode(positive_max_values, bin_interval=bin_interval)
    top_median = positive_max_values[len(positive_max_values) // 2]
    dip_min = min(negative_min_values)
    dip_mean = round(sum(negative_min_values) / len(negative_min_values), 3)
    dip_binmode = get_histogram_bin_mode(negative_min_values, bin_interval=bin_interval)
    dip_median = negative_min_values[len(negative_min_values) // 2]

    coin_dict[f'top_{ema_dev_col_name}_max'] = top_max
    coin_dict[f'top_{ema_dev_col_name}_mean'] = top_mean
    coin_dict[f'top_{ema_dev_col_name}_binmode'] = top_binmode
    coin_dict[f'top_{ema_dev_col_name}_median'] = top_median
    coin_dict[f'dip_{ema_dev_col_name}_min'] = dip_min
    coin_dict[f'dip_{ema_dev_col_name}_mean'] = dip_mean
    coin_dict[f'dip_{ema_dev_col_name}_binmode'] = dip_binmode
    coin_dict[f'dip_{ema_dev_col_name}_median'] = dip_median

    # print("Positive top values from streaks:", positive_max_values[:20])
    # print("Negative dip values from streaks:", negative_min_values[:20])
    # print("Max top value from streaks:", top_max)
    # print("Min dip value from streaks:", dip_min)
    # print("Bin mode of top values from streaks:", top_binmode)
    # print("Bin mode of dip values from streaks:", dip_binmode)
    # print("Median of top values from streaks:", top_median)
    # print("Median of dip values from streaks:", dip_median)
    # print("Mean of top values from streaks:", top_mean)
    # print("Mean of dip values from streaks:", dip_mean)

    return coin_dict


def calculate_ema_trend_extremes(df, ema_col_name, trend_extension='_slope', hist_threshold=90):
    ema_trend_col_name = ema_col_name + trend_extension
    positive_trend_values = []
    negative_trend_values = []

    current_streak = []
    current_sign = None
    coin_dict = {}

    # Iterate through the ema_dev column
    for index, value in df[ema_trend_col_name].items():
        if value > 0.0001:  # Positive value
            if current_sign == "positive":
                current_streak.append(value)
            else:
                if current_sign == "negative":
                    negative_trend_values.extend(current_streak)
                current_streak = [value]
            current_sign = "positive"
        elif value < -0.0001:  # Negative value
            if current_sign == "negative":
                current_streak.append(value)
            else:
                if current_sign == "positive":
                    positive_trend_values.extend(current_streak)
                current_streak = [value]
            current_sign = "negative"
        else:  # abs(value) <= 0.001, reset streak
            if current_sign == "positive":
                positive_trend_values.extend(current_streak)
            elif current_sign == "negative":
                negative_trend_values.extend(current_streak)
            current_streak = []
            current_sign = None

    # Handle remaining streak at the end
    if current_sign == "positive":
        positive_trend_values.extend(current_streak)
    elif current_sign == "negative":
        negative_trend_values.extend(current_streak)

    positive_trend_values = [round_to_significant_figures(num, 4) for num in positive_trend_values]
    negative_trend_values = [round_to_significant_figures(num, 4) for num in negative_trend_values]
    positive_trend_values.sort(reverse=True)
    negative_trend_values.sort()
    bin_interval = 0.002

    pos_mean = round(sum(positive_trend_values) / len(positive_trend_values), 4)
    pos_binmode = get_histogram_bin_mode(positive_trend_values, bin_interval=bin_interval)
    pos_binmode_majority = round(get_histogram_bin_boundary(positive_trend_values, bin_interval=bin_interval, pct_threshold=hist_threshold), 4)
    pos_median = positive_trend_values[len(positive_trend_values) // 2]
    pos_max = max(positive_trend_values)
    neg_mean = round(sum(negative_trend_values) / len(negative_trend_values), 4)
    neg_binmode = get_histogram_bin_mode(negative_trend_values, bin_interval=bin_interval)
    neg_binmode_majority = round(get_histogram_bin_boundary(negative_trend_values, bin_interval=bin_interval, pct_threshold=hist_threshold), 4)
    neg_median = negative_trend_values[len(negative_trend_values) // 2]
    neg_min = min(negative_trend_values)

    coin_dict[f'pos_{ema_trend_col_name}_max'] = pos_max
    coin_dict[f'pos_{ema_trend_col_name}_mean'] = pos_mean
    coin_dict[f'pos_{ema_trend_col_name}_median'] = pos_median
    coin_dict[f'pos_{ema_trend_col_name}_binmode'] = pos_binmode
    coin_dict[f'pos_{ema_trend_col_name}_binmode{hist_threshold}'] = pos_binmode_majority

    coin_dict[f'neg_{ema_trend_col_name}_min'] = neg_min
    coin_dict[f'neg_{ema_trend_col_name}_mean'] = neg_mean
    coin_dict[f'neg_{ema_trend_col_name}_mean'] = neg_mean
    coin_dict[f'neg_{ema_trend_col_name}_median'] = neg_median
    coin_dict[f'neg_{ema_trend_col_name}_binmode'] = neg_binmode
    coin_dict[f'neg_{ema_trend_col_name}_binmode{hist_threshold}'] = neg_binmode_majority

    '''print(f"Positive{trend_extension} values:", positive_trend_values[:20])
    print(f"Negative{trend_extension} values:", negative_trend_values[:20])
    print(f'Positive_max{trend_extension}:', pos_max)
    print(f'Negative_min{trend_extension}:', neg_min)
    print(f"{hist_threshold}% majority boundary of positive{trend_extension}:", pos_binmode_majority)
    print(f"{hist_threshold}% majority boundary of negative{trend_extension}:", neg_binmode_majority)
    print(f"Mean of positive{trend_extension}:", pos_mean)
    print(f"Mean of negative{trend_extension}:", neg_mean)
    print(f"Median of positive{trend_extension}:", pos_median)
    print(f"Median of negative{trend_extension}:", neg_median)
    print(f"Bin mode of positive{trend_extension}:", pos_binmode)
    print(f"Bin mode of negative{trend_extension}:", neg_binmode)

    plot_list_histogram(positive_trend_values, bin_interval=bin_interval)
    plot_list_histogram(negative_trend_values, bin_interval=bin_interval)'''

    return coin_dict


def get_saved_ema_extremes(symbol, window=200):
    dic = save_load_pkl(coins_dict_pkl_path)

    # dic2 = dic.copy()
    # for k, v in dic2.items():
    #     if not re.search(r'USDT|ETH', str(k)):
    #         del dic[k]

    symbol_dic = dic[symbol]

    top_keys = [f'top_ema{window}_dev_mean', f'top_ema{window}_dev_binmode', f'top_ema{window}_dev_median']
    tops = sorted([symbol_dic[key] for key in top_keys])
    # print(f'top_ema{window}_devs:', tops)
    dip_keys = [f'dip_ema{window}_dev_mean', f'dip_ema{window}_dev_binmode', f'dip_ema{window}_dev_median']
    dips = sorted([symbol_dic[key] for key in dip_keys], reverse=True)
    # print(f'dip_ema{window}_dev:', dips)
    pos_slope_keys = [f'pos_ema{window}_slope_max', f'pos_ema{window}_slope_mean', f'pos_ema{window}_slope_median', f'pos_ema{window}_slope_binmode', f'pos_ema{window}_slope_binmode90']
    pos_slopes = sorted([symbol_dic[key] for key in pos_slope_keys])
    # print(f'pos_ema{window}_slope:', pos_slopes)
    neg_slope_keys = [f'neg_ema{window}_slope_min', f'neg_ema{window}_slope_mean', f'neg_ema{window}_slope_median', f'neg_ema{window}_slope_binmode', f'neg_ema{window}_slope_binmode90']
    neg_slopes = sorted([symbol_dic[key] for key in neg_slope_keys], reverse=True)
    # print(f'neg_ema{window}_slope:', neg_slopes)
    pos_pct_change_keys = [f'pos_ema{window}_pct_change_max', f'pos_ema{window}_pct_change_mean', f'pos_ema{window}_pct_change_median', f'pos_ema{window}_pct_change_binmode', f'pos_ema{window}_pct_change_binmode90']
    pos_pct_changes = sorted([symbol_dic[key] for key in pos_pct_change_keys])
    # print(f'pos_ema{window}_pct_change:', pos_pct_changes)
    # neg_pct_change_keys = [f'neg_ema{window}_pct_change_min', f'neg_ema{window}_pct_change_mean', f'neg_ema{window}_pct_change_median', f'neg_ema{window}_pct_change_binmode', f'neg_ema{window}_pct_change_binmode90']
    # neg_pct_changes = sorted([symbol_dic[key] for key in neg_pct_change_keys], reverse=True)
    # print(f'neg_ema{window}_pct_change:', neg_pct_changes)

    return {f'top_ema{window}_devs':tops, f'dip_ema{window}_devs':dips, f'pos_ema{window}_slopes':pos_slopes, f'neg_ema{window}_slopes':neg_slopes} # , f'pos_ema{window}_pct_changes':pos_pct_changes, f'neg_ema{window}_pct_changes':neg_pct_changes


def get_ema_slope_trend_type(df, symbol):
    trend_dict = {}
    ema200_pos_slopes = get_saved_ema_extremes(symbol, 200)['pos_ema200_slopes']
    ema200_neg_slopes = get_saved_ema_extremes(symbol, 200)['neg_ema200_slopes']
    ema100_pos_slopes = get_saved_ema_extremes(symbol, 100)['pos_ema100_slopes']
    ema100_neg_slopes = get_saved_ema_extremes(symbol, 100)['neg_ema100_slopes']
    ema50_pos_slopes = get_saved_ema_extremes(symbol, 50)['pos_ema50_slopes']
    ema50_neg_slopes = get_saved_ema_extremes(symbol, 50)['neg_ema50_slopes']

    for window in [200, 100, 50]:
        ema_column_name = f'ema{window}'
        trend_dict[f'{ema_column_name}_last'] = df[ema_column_name].iloc[-1]
        trend_dict[f'{ema_column_name}_10'] = df[ema_column_name].iloc[-11]
        trend_dict[f'{ema_column_name}_20'] = df[ema_column_name].iloc[-21]
        trend_dict[f'{ema_column_name}_dev_last'] = df[f'{ema_column_name}_dev'].iloc[-1]
        trend_dict[f'{ema_column_name}_dev_10'] = df[f'{ema_column_name}_dev'].iloc[-11]
        trend_dict[f'{ema_column_name}_dev_20'] = df[f'{ema_column_name}_dev'].iloc[-21]
        trend_dict[f'{ema_column_name}_slope_last'] = df[f'{ema_column_name}_slope'].iloc[-1]
        trend_dict[f'{ema_column_name}_slope_10'] = df[f'{ema_column_name}_slope'].iloc[-11]
        trend_dict[f'{ema_column_name}_slope_20'] = df[f'{ema_column_name}_slope'].iloc[-21]

    if trend_dict['ema200_slope_last'] > 0 and trend_dict['ema200_slope_20'] > 0: # upward ema200
        if trend_dict['ema200_slope_last'] > np.mean(ema200_pos_slopes[2:4]):
            if trend_dict['ema50_last'] > trend_dict['ema100_last'] > trend_dict['ema200_last']:
                if trend_dict['ema50_slope_last'] > trend_dict['ema100_slope_last'] > trend_dict['ema200_slope_last']:
                    return 'fast_trend_up'
        elif ema200_pos_slopes[0] < trend_dict['ema200_slope_last'] < ema200_pos_slopes[2]:
            return 'trend_up'
    elif trend_dict['ema200_slope_last'] < 0 and trend_dict['ema200_slope_20'] < 0: # downward ema200
        pass


def relate_ema_forward_price(df):
    df['100f_pct_change'] = (df['close'].shift(-100) - df['close']) / df['close'] * 100
    df['200f_pct_change'] = (df['close'].shift(-200) - df['close']) / df['close'] * 100

    def plot(df):
        # Assuming df has 'ema200_slope' and '200f_pct_change' columns

        # 1. Define bins for ema200_slope (optional)
        bins = np.linspace(df['ema200_slope'].min(), df['ema200_slope'].max(), 50)  # 50 bins
        df['ema200_slope_bin'] = pd.cut(df['ema200_slope'], bins)

        # 2. Group by bins and calculate the bands
        grouped = df.groupby('ema200_slope_bin')['200f_pct_change']
        middle_band = grouped.mean()
        upper_band = middle_band + grouped.std()
        lower_band = middle_band - grouped.std()

        # 3. Plotting
        plt.figure(figsize=(12, 6))
        bin_centers = [interval.mid for interval in middle_band.index]  # Extract bin centers

        plt.plot(bin_centers, middle_band, label='Middle Band (Mean)', color='blue', linestyle='--')
        plt.plot(bin_centers, upper_band, label='Upper Band (Mean + Std)', color='green', linestyle='-')
        plt.plot(bin_centers, lower_band, label='Lower Band (Mean - Std)', color='red', linestyle='-')

        plt.title('Bollinger Bands of 200f_pct_change by ema200_slope')
        plt.xlabel('ema200_slope')
        plt.ylabel('200f_pct_change')
        plt.legend()
        plt.grid()
        plt.show()

    plot(df)

    return df



def get_list_frequent_extremes(_value_list, extreme_type):
    if extreme_type == 'top':
        # Filter only sufficiently large top values (greater than half of max top)
        max_value = max(_value_list)
        value_list = [value for value in _value_list if value > max_value / 2]
        # Find local tops
        value_list = [value_list[i] for i in range(1, len(value_list) - 1) if
                      value_list[i] > value_list[i - 1] and value_list[i] > value_list[i + 1]]
        local_tops = [value_list[i] for i in range(1, len(value_list) - 1) if
                      value_list[i] > value_list[i - 1] and value_list[i] > value_list[i + 1]]
        # print("Local Tops:", local_tops)

        return local_tops

    elif extreme_type == 'dip':
        # Filter only sufficiently low dip values (less than half of min dip)
        min_value = min(_value_list)
        value_list = [value for value in _value_list if value < min_value / 2]
        # Find local dips
        value_list = [value_list[i] for i in range(1, len(value_list) - 1) if
                      value_list[i] < value_list[i - 1] and value_list[i] < value_list[i + 1]]
        local_dips = [value_list[i] for i in range(1, len(value_list) - 1) if
                      value_list[i] < value_list[i - 1] and value_list[i] < value_list[i + 1]]
        # print("Local Dips:", local_dips)

        return local_dips


def get_histogram_bin_mode(value_list, bin_interval=0.2):
    bins = [i * bin_interval for i in range(int(min(value_list) / bin_interval), int(max(value_list) / bin_interval) + 1)]

    counts, bin_edges = np.histogram(value_list, bins=bins)

    # Find the bin with the highest frequency
    max_bin_index = np.argmax(counts)

    # Get the edges of the most frequent bin
    bin_start = bin_edges[max_bin_index]
    bin_end = bin_edges[max_bin_index + 1]

    # Filter data within the most frequent bin
    most_frequent_bin_values = [x for x in value_list if bin_start <= x < bin_end]

    # Calculate the average of values in the most frequent bin
    average_of_most_frequent_bin = round(np.mean(most_frequent_bin_values), 4)

    return average_of_most_frequent_bin


def get_histogram_bin_boundary(value_list, bin_interval=0.02, pct_threshold=90):
    threshold = pct_threshold/100

    # Define the bins
    bins = [i * bin_interval for i in range(int(min(value_list) / bin_interval), int(max(value_list) / bin_interval) + 1)]

    # Calculate the histogram
    counts, bin_edges = np.histogram(value_list, bins=bins)

    if value_list[len(value_list)//2] > 0:
        cumulative_sum = np.cumsum(counts)
        cumulative_proportion = cumulative_sum / cumulative_sum[-1]
        threshold_index = np.where(cumulative_proportion >= threshold)[0][0]
        boundary_value = bin_edges[threshold_index]

    elif value_list[len(value_list)//2] < 0:
        # Reverse the cumulative sum of the frequencies for negative values
        cumulative_sum = np.cumsum(counts[::-1])
        # Normalize cumulative sum to get proportions (from majority end)
        cumulative_proportion = cumulative_sum / cumulative_sum[-1]
        # Find the first bin where the cumulative proportion exceeds the threshold
        threshold_index = len(cumulative_proportion) - np.where(cumulative_proportion >= threshold)[0][0] - 1
        # Get the bin start for the threshold index
        boundary_value = bin_edges[threshold_index + 1]

    return boundary_value


def plot_list_histogram(value_list, bin_interval=0.2, value_list2=[]):
    # value_list += value_list2
    bins = [i * bin_interval for i in range(int(min(value_list) / bin_interval), int(max(value_list) / bin_interval) + 1)]

    counts, bin_edges = np.histogram(value_list, bins=bins)

    # Find the bin with the highest frequency
    max_bin_index = np.argmax(counts)

    # Get the edges of the most frequent bin
    bin_start = bin_edges[max_bin_index]
    bin_end = bin_edges[max_bin_index + 1]

    # Filter data within the most frequent bin
    most_frequent_bin_values = [x for x in value_list if bin_start <= x < bin_end]

    # Calculate the average of values in the most frequent bin
    average_of_most_frequent_bin = round(np.mean(most_frequent_bin_values), 4)
    plt.axvline(average_of_most_frequent_bin, color="orange", linestyle="--", label="ave_of_most_freq_bin")
    plt.axvline(get_histogram_bin_boundary(value_list, bin_interval=bin_interval), color="lightsalmon", linestyle="--", label="ave_of_most_freq_bin")

    # Plot the histogram
    plt.hist(value_list, bins=bins, edgecolor='blue', alpha=0.7)
    if value_list2:
        bins2 = [i * bin_interval for i in range(int(min(value_list2) / bin_interval), int(max(value_list2) / bin_interval) + 1)]
        counts2, bin_edges2 = np.histogram(value_list2, bins=bins2)

        # Find the bin with the highest frequency
        max_bin_index2 = np.argmax(counts2)

        # Get the edges of the most frequent bin
        bin_start2 = bin_edges2[max_bin_index2]
        bin_end2 = bin_edges2[max_bin_index2 + 1]

        # Filter data within the most frequent bin
        most_frequent_bin_values2 = [x for x in value_list2 if bin_start2 <= x < bin_end2]

        # Calculate the average of values in the most frequent bin
        average_of_most_frequent_bin2 = round(np.mean(most_frequent_bin_values2), 4)

        plt.axvline(average_of_most_frequent_bin2, color="green", linestyle="--", label="ave_of_most_freq_bin2")
        plt.axvline(get_histogram_bin_boundary(value_list2, bin_interval=bin_interval), color="lightgreen", linestyle="--", label="ave_of_most_freq_bin2")
        plt.hist(value_list2, bins=bins, edgecolor='red', alpha=0.7)
    plt.xlabel('Intervals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Float Numbers')

    # Show the plot
    plt.show()


def plot_df(symbol, df, kline_interval='1m', days_range=7, options=['price']):
    current_date = datetime.now(timezone.utc).replace(tzinfo=None)
    _start_date = current_date - timedelta(days=days_range)
    mean_price20 = df['close'][-20:].mean()
    plt.figure(figsize=(14, 8))

    if 'price' in options:
        df_price = df.iloc[200:]
        plt.plot(df_price.index, df_price['close'], label='close', color='skyblue')
        # plt.plot(df_price.index, df_price['high'], label='high', color='orange', linewidth=0.7)
        # plt.plot(df_price.index, df_price['low'], label='low', color='orange', linewidth=0.7)
        # plt.plot(df_price.index, df_price['price_diff'] + mean_price20, label='price_diff', color='gray', linestyle='--')
        # plt.plot(df_price.index, (df_price['price_change']+0.1) * mean_price20, label='price_change', color='black')

    if 'bb' in options:
        df_bb = bb(df).iloc[200:]
        plt.plot(df_bb.index, df_bb['upper_band'], label='ub', color='brown')
        plt.plot(df_bb.index, df_bb['lower_band'], label='lb', color='brown')
        plt.plot(df_bb.index, df_bb['bb_sma'], label='mb', color='lime')

    if 'ema' in options:
        window_list = [200, 100, 50]
        for window in window_list:
            ema_option = ''.join(['ema', str(window)])
            color = config.random_color()
            df_ema = ema(df, window, ema_option).iloc[200:]
            plt.plot(df_ema.index, df_ema[ema_option], label=ema_option, color=color, linestyle='-')
            # plt.plot(df.index, mean_price20 * (1 + df[option + '_dev_sma']/100), label=option + '_dev_sma', color=color, linestyle='--')
            # plt.plot(df.index, mean_price20 * (1 + df[option + '_dev_sma_diff']/100), label=option + '_dev_sma_diff', color=color, linestyle=':')

    if 'sma' in options:
        window_list = [200, 100, 50]
        for window in window_list:
            sma_option = ''.join(['sma', str(window)])
            color = config.random_color()
            df_sma = sma(df, window, sma_option).iloc[200:]
            plt.plot(df_sma.index, df_sma[sma_option], label=sma_option, color=color, linestyle='-')
            # plt.plot(df_sma.index, mean_price20 * (1 + df_sma[sma_option + '_dev_sma']/100), label=sma_option + '_dev_sma', color=color, linestyle='--')
            # plt.plot(df_sma.index, mean_price20 * (1 + df_sma[sma_option + '_dev_sma_diff']/100), label=sma_option + '_dev_sma_diff', color=color, linestyle=':')

    if 'rsi' in options:
        df_rsi = rsi(df).iloc[200:]
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
        df_macd = macd(df).iloc[200:]
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

    plt.title(f'{symbol[:-4]} {config.fetch_real_time_frame(kline_interval)}')
    plt.xticks(df.index[::5], [str(idx)[-11:-3] for idx in df.index[::5]], fontsize=6)
    plt.xlabel('time')
    plt.ylabel('price')
    plt.legend()
    plt.grid(True)
    # plt.show(block=False)
    plt.rcParams['figure.figsize'] = (14.5, 8.5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
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


def get_new_time_interval(new_kline_interval, new_freq, file_path='', _df='', is_save=False):
    if file_path:
        df = pd.read_csv(file_path, index_col="datetime", parse_dates=True)
    else:
        df = _df

    # Resample by {new_freq} minutes
    df_resampled = df.resample(str(new_freq) + 'min').agg({
        'open': 'first',  # Open value for the first row in each {new_freq}-minute group
        'close': 'last',  # Close value for the last row in each {new_freq}-minute group
        'high': 'max',  # Highest value within the {new_freq}-minute period
        'low': 'min'  # Lowest value within the {new_freq}-minute period
    })

    if is_save:
        dirpath, filename = os.path.split(file_path)
        new_dirpath = os.path.join(os.path.split(dirpath)[0], new_kline_interval)
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)
        new_filepath = os.path.join(new_dirpath, filename)
        df_resampled.to_csv(new_filepath, encoding='UTF-8')

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
        df_filtered.to_csv(new_file_path, encoding='utf-8')

    return df_filtered

def current_date():
    return datetime.now(timezone.utc).replace(tzinfo=None)


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


if __name__ == '__main__':
    prec = 5
    if os.path.exists(orders_dict_pkl_path):
        REALIZED_ORDERS_DICT = save_load_pkl(orders_dict_pkl_path)
        print(REALIZED_ORDERS_DICT)

