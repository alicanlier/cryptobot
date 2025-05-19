import utility.calculation_tools as calculate
from src.binance.futures_prices import SingleCoin
import decimal
from binance.exceptions import BinanceAPIException
from scipy.signal import argrelextrema


def rsi_check(last_rsi, last_rsi_change):
    trade_direction = ''
    if 55 < last_rsi < 70 and 2 < last_rsi_change < 10:  # percent(%) change:
        trade_direction = 'BUY'
    elif last_rsi > 80 and last_rsi_change < -2:
        trade_direction = 'SELL'
    elif 30 < last_rsi < 45 and -2 > last_rsi_change > -10:
        trade_direction = 'SELL'
    elif last_rsi < 20 and last_rsi_change > 2:
        trade_direction = 'BUY'

    return trade_direction


def get_bb_dict(client, symbol, time_frame, _window=20):
    current_price = 0
    window = _window
    n = 0

    while True:
        try:
            symbol_obj = SingleCoin(client, symbol, time_frame, window)
            break
        except:
            # correction of data length window for recently listed coins
            window = window // 2
            n += 1
            if n > 3:
                return

    current_price = float(symbol_obj.current_price)

    df_symbol = symbol_obj.df_symbol
    bb_df = calculate.bb(df_symbol)

    bb_df.loc[:, 'upper_band'] = bb_df['upper_band'].astype(float)
    bb_df.loc[:, 'lower_band'] = bb_df['lower_band'].astype(float)
    bb_df.loc[:, 'bb_sma'] = bb_df['bb_sma'].astype(float)
    bb_df.loc[:, 'std'] = bb_df['std'].astype(float)

    band_width = 2 * bb_df['std'].iloc[-1]

    max_price, min_price, bw, ub, lb, ml, trend = 'max_price', 'min_price', 'bw', 'ub', 'lb', 'ml', 'trend'

    # form a dictionary
    bb_dict = {bw: band_width, ub: bb_df['upper_band'].iloc[-1], lb: bb_df['lower_band'].iloc[-1], ml: bb_df['bb_sma'].iloc[-1]}
    ml_last6_avg = bb_df['bb_sma'].tail(6).mean()
    bb_dict[trend] = (bb_dict[ml] - ml_last6_avg) * 100 / ml_last6_avg

    # get min, max prices among last 6 values
    min_price_last6 = df_symbol['close'].tail(6).min()
    max_price_last6 = df_symbol['close'].tail(6).max()

    bb_dict['min_price'] = min_price_last6
    bb_dict['max_price'] = max_price_last6
    bb_dict['current_price'] = current_price

    bb_dict = {key: round_excess_decimals(value) for key, value in bb_dict.items()}

    return bb_dict


def bb_check(client, symbol):
    bb5 = get_bb_dict(client, symbol, 5)
    bb30 = get_bb_dict(client, symbol, 30)
    bb1h = get_bb_dict(client, symbol, 60)
    bb4h = {}
    try:
        bb4h = get_bb_dict(client, symbol, 240)
    except Exception as e:
        print(f'{symbol} error: {e}')
        bb4h = bb1h
    except BinanceAPIException as be:
        print(be)

    # print(current_price); print('bb5:',bb5); print('bb30:',bb30); print('bb1h:',bb1h); print('bb4h:',bb4h);

    # print(f"{'{:<10}'.format(symbol)}: {'{:<18}'.format(current_price)} | {'{:<18}'.format(bb5[ml])} | {'{:<18}'.format(bb5[ub])} | {bb5[lb]}")
    current_price, max_price, min_price, bw, ub, lb, ml, trend = 'current_price', 'max_price', 'min_price', 'bw', 'ub', 'lb', 'ml', 'trend'
    current_price = bb5[current_price]
    order_dict = {}

    if current_price != 0:
        '''current_price around upper band of 4h Bollinger is likely a SELL if there isn't a huge HYPE/FOMO or vice versa (lb:BUY).
        1) Upper band, lower than max, down trend; sell soon? buy middle line?
        '''
        if bb30.get(max_price, 0) * 0.995 > current_price > (bb30[ml] + bb30[bw] * 0.9) and bb30[trend] < -2 and bb1h[trend] < -1 and bb4h[trend] < 0:
            order_dict = {'SELL':(bb5[ub] + bb5[bw] * 2.4), 'BUY':(bb30[ml] - bb5[bw] * 2.2), 'current_price':current_price, 'condition':1}  #'1 ub'
        elif current_price > (bb30[ml] + bb30[bw] * 0.95) and bb30[trend] > 0 and bb1h[trend] > 0 and bb4h[trend] > 0:
            if current_price < bb5.get(max_price, 0) * 0.995:
                order_dict = {'SELL': (bb4h[ml] + bb4h[bw] * 2.0), 'BUY': (bb1h[ml] - bb5[bw] * 2.2), 'current_price': current_price, 'condition':21}  # '2 ub'
            else:
                order_dict = {'SELL': (bb4h[ml] + bb4h[bw] * 4.0), 'BUY': (bb1h[ml] - bb5[bw] * 2.6), 'current_price': current_price, 'condition':22}  # '2 ub'
        elif bb30.get(min_price, 0) < current_price < (bb30[ml] - bb30[bw] * 0.9) and bb30[trend] > 2 and bb1h[trend] > 1 and bb4h[trend] > 0:
            order_dict = {'SELL':(bb30[ml] + bb5[bw] * 2.2), 'BUY':(bb5[lb] - bb5[bw] * 2.4), 'current_price':current_price, 'condition':3}  #'3 lb'
        elif current_price < (bb30[ml] - bb30[bw] * 0.95) and bb30[trend] < 0 and bb1h[trend] < 0 and bb4h[trend] < 0:
            if current_price > bb5.get(min_price, 0) * 1.005:
                order_dict = {'SELL': (bb1h[ml] + bb5[bw] * 2.2), 'BUY': (bb4h[ml] - bb4h[bw] * 2.0), 'current_price': current_price, 'condition':41}  # '4 lb'
            else:
                order_dict = {'SELL': (bb1h[ml] + bb5[bw] * 2.6), 'BUY': (bb4h[ml] - bb4h[bw] * 4.0), 'current_price': current_price, 'condition':42}  # '4 lb'

        if order_dict:
            # print(order_dict)
            return order_dict
        else:
            '''current price around middle line'''
            if (bb30[ml] + bb30[bw] * 0.25) > current_price > (bb30[ml] + bb30[bw] * 0.05):
                if current_price > bb5[min_price] * 1.005 and bb30[trend] > 1.0 and bb1h[trend] > 0.5 and bb4h[trend] > 0.05:
                    order_dict = {'SELL':(bb30[ub] + bb5[bw] * 2), 'BUY':(bb5[ml] - bb5[bw] * 1.2), 'current_price':current_price, 'condition':5}  #'3 ml u+'
            elif (bb30[ml] + bb30[bw] * 0.05) > current_price > (bb30[ml] + bb30[bw] * 0.01):
                if current_price < bb5[max_price] * 0.995 and bb30[trend] < -2 and bb1h[trend] < -1 and bb4h[trend] < 0:
                    order_dict = {'SELL':(bb5[ml] + bb5[bw] * 1.8), 'BUY':(bb30[lb] - bb5[bw] * 1.8), 'current_price':current_price, 'condition':6}  #'4 ml u-'
            elif (bb30[ml] - bb30[bw] * 0.25) < current_price < (bb30[ml] - bb30[bw] * 0.05):
                if current_price < bb5[max_price] * 0.995 and bb30[trend] < -1.0 and bb1h[trend] < -0.5 and bb4h[trend] < -0.05:
                    order_dict = {'SELL':(bb5[ml] + bb5[bw] * 1.2), 'BUY':(bb30[lb] - bb5[bw] * 2), 'current_price':current_price, 'condition':7}  #'5 ml d-'
            elif (bb30[ml] - bb30[bw] * 0.05) < current_price < (bb30[ml] - bb30[bw] * 0.01):
                if current_price > bb5[min_price] * 1.005 and bb30[trend] > 2 and bb1h[trend] > 1 and bb4h[trend] > 0:
                    order_dict = {'SELL':(bb30[ub] + bb5[bw] * 1.8), 'BUY':(bb5[ml] - bb5[bw] * 1.8), 'current_price':current_price, 'condition':8}  #'6 ml d+'

            if order_dict:
                # print(order_dict)
                return order_dict
    else:
        return None

    '''obj4h = SingleCoin(client, symbol, 4)
    obj1h = SingleCoin(client, symbol, 60)
    obj30 = SingleCoin(client, symbol, 30)
    obj5 = SingleCoin(client, symbol, 5)

    current_price = float(obj30.current_price)
    df_symbol_4h = obj4h.df_symbol
    df_symbol_1h = obj1h.df_symbol
    df_symbol_30 = obj30.df_symbol
    df_symbol_5 = obj5.df_symbol

    bb4h_df = calculate.bb(df_symbol_4h)
    bb1h_df = calculate.bb(df_symbol_1h)
    bb30_df = calculate.bb(df_symbol_30)
    bb5_df = calculate.bb(df_symbol_5)

    bb30_df.loc[:, 'upper_band'] = bb30_df['upper_band'].astype(float)
    bb30_df.loc[:, 'lower_band'] = bb30_df['lower_band'].astype(float)
    bb30_df.loc[:, 'bb_sma'] = bb30_df['bb_sma'].astype(float)
    bb5_df.loc[:, 'upper_band'] = bb5_df['upper_band'].astype(float)
    bb5_df.loc[:, 'lower_band'] = bb5_df['lower_band'].astype(float)
    bb5_df.loc[:, 'bb_sma'] = bb5_df['bb_sma'].astype(float)

    band_width30 = 2 * bb30_df['std'].astype(float).iloc[-1]
    band_width5 = 2 * bb5_df['std'].astype(float).iloc[-1]

    bw, ub, lb, ml, trend = 'bw', 'ub', 'lb', 'ml', 'trend'

    bb30 = {bw: band_width30, ub: bb30_df['upper_band'].iloc[-1], lb: bb30_df['lower_band'].iloc[-1], ml: bb30_df['bb_sma'].iloc[-1]}
    bb30 = {key:round_excess_decimals(value) for key,value in bb30.items()}
    ml30_last7_avg = bb30_df['bb_sma'].tail(7).mean()
    bb30[trend] = (bb30[ml] - ml30_last7_avg) * 100 / ml30_last7_avg

    bb5 = {bw: band_width5, ub: bb5_df['upper_band'].iloc[-1], lb: bb5_df['lower_band'].iloc[-1], ml: bb5_df['bb_sma'].iloc[-1]}
    bb5 = {key:round_excess_decimals(value) for key,value in bb5.items()}
    ml5_last7_avg = bb5_df['bb_sma'].tail(7).mean()
    bb5[trend] = (bb5[ml] - ml5_last7_avg) * 100 / ml5_last7_avg'''


def bb_check2(client, symbol):
    obj4 = SingleCoin(client, symbol, 4)
    obj30 = SingleCoin(client, symbol, 30)
    obj5 = SingleCoin(client, symbol, 5)

    current_price = obj4.CURRENT_PRICE
    df_symbol_4h = obj4.df_symbol
    df_symbol_30 = obj30.df_symbol
    df_symbol_5 = obj5.df_symbol

    bb4h_df = calculate.bb(df_symbol_4h)
    bb30_df = calculate.bb(df_symbol_30)
    bb5_df = calculate.bb(df_symbol_5)

    band_width4h = 2 * bb4h_df['std']
    band_width30 = 2 * bb30_df['std']
    band_width5 = 2 * bb5_df['std']

    bw, ub, lb, ml, trend = 'bw', 'ub', 'lb', 'ml', 'trend'
    bb4h = {bw:band_width4h, ub:bb4h_df['upper_band'].iloc[-1], lb:bb4h_df['lower_band'].iloc[-1], ml:bb4h_df['bb_sma'].iloc[-1]}
    ml4h_last7_avg = bb4h_df['bb_sma'].tail(7).mean()
    # 4h bb_sma (like sma) trend (slope) is inferred: expecting >0.3 bor buy, <-0.3 for sell
    bb4h[trend] = (bb4h_df[ml] - ml4h_last7_avg) * 100 / ml4h_last7_avg

    bb30 = {bw:band_width30, ub:bb30_df['upper_band'].iloc[-1], lb:bb30_df['lower_band'].iloc[-1], ml:bb30_df['bb_sma'].iloc[-1]}
    bb5 = {bw:band_width5, ub:bb5_df['upper_band'].iloc[-1], lb:bb5_df['lower_band'].iloc[-1], ml:bb5_df['bb_sma'].iloc[-1]}

    buy_zone_list = []
    sell_zone_list = []

    '''current_price around upper band of 4h Bollinger is likely a SELL if there is no huge HYPE/FOMO.'''
    if current_price > bb4h[ml] + bb4h[bw] * 1.0 and bb4h[trend] > 3.0: # STRONG_SELL
        pass

    elif current_price > bb4h[ml] + bb4h[bw] * 0.9 and -0.3 < bb4h[trend] < 0: # SELL > may evolve to BUY, better not involve
        pass

    elif current_price > bb4h[ml] + bb4h[bw] * 0.9 and bb4h[trend] < -1:  # SELL > may evolve to BUY
        pass

    '''current_price just above bb_sma of 4h Bollinger is likely a BUY if trend and other factors (rsi, macd etc.) push upwards.'''
    if bb4h[ml] + bb4h[bw] * 0.25 > current_price > bb4h[ml] + bb4h[bw] * 0.1 and bb4h[trend] > 0.3: # BUY, confirm BTC trend
        pass

    elif bb4h[ml] + bb4h[bw] * 0.25 > current_price > bb4h[ml] + bb4h[bw] * 0.1 and bb4h[trend] < -0.3:  # STRONG_BUY
        pass

    elif bb4h[ml] + bb4h[bw] * 0.25 > current_price > bb4h[ml] + bb4h[bw] * 0.1 and bb4h[trend] > 0.3:
        pass

    '''current_price just below bb_sma of 4h Bollinger is likely a SELL if trend and other factors (rsi, macd etc.) push downwards.'''
    if bb4h[ml] + bb4h[bw] * 0.25 > current_price > bb4h[ml] + bb4h[bw] * 0.1 and bb4h[trend] > 0.3:  # BUY, confirm BTC trend
        pass

    elif bb4h[ml] + bb4h[bw] * 0.25 > current_price > bb4h[ml] + bb4h[bw] * 0.1 and bb4h[trend] < -0.3:  # STRONG_BUY
        pass

    elif bb4h[ml] + bb4h[bw] * 0.25 > current_price > bb4h[ml] + bb4h[bw] * 0.1 and bb4h[trend] > 0.3:
        pass

    elif bb4h[ml] + bb4h[bw] * 0.25 > current_price > bb4h[ml] + bb4h[bw] * 0.1 and bb4h[trend] > 0.3:
        pass

    '''current_price around lower band of 4h Bollinger is likely a BUY if there is no huge FUD.'''
    if current_price < bb4h[ml] - bb4h[bw] * 0.95 and bb4h[trend] < -1.0: # STRONG_BUY
        pass

    elif current_price < bb4h[ml] - bb4h[bw] * 0.9 and 0.3 > bb4h[trend] > 0: # WEAK_BUY > may evolve to SELL, better not involve
        pass

    elif current_price < bb4h[ml] - bb4h[bw] * 0.9 and bb4h[trend] > 1:  # BUY > may evolve to SELL
        pass

def macd_check(last_macd, last_macd_change):
    trade_direction = ''
    if last_macd < 0 and last_macd_change > 2:
        trade_direction = 'BUY'
    elif last_macd > 0 and last_macd_change < -2:
        trade_direction = 'SELL'

    return trade_direction


def sma_check(sma_dev):
    trade_direction = ''
    if 2 < sma_dev < 5:
        trade_direction = 'BUY'
    elif sma_dev > 50:
        trade_direction = 'SELL'
    elif -5 < sma_dev < -2:
        trade_direction = 'SELL'
    elif sma_dev < -50:
        trade_direction = 'SELL'

    return trade_direction


def ema_check(ema_dev):
    trade_direction = ''
    if 2 < ema_dev < 5:
        trade_direction = 'BUY'
    elif ema_dev > 50:
        trade_direction = 'SELL'
    elif -5 < ema_dev < -2:
        trade_direction = 'SELL'
    elif ema_dev < -50:
        trade_direction = 'SELL'

    return trade_direction


def round_excess_decimals(num, prec=3):
    if num > 999.999:
        prec = len(str(int(num)))
    elif 50 > num > 9.999:
        prec = 4
    decimal.getcontext().prec = prec
    decimal_num = decimal.Decimal(str(num))
    rounded_decimal_num = decimal_num.normalize()

    return float(rounded_decimal_num)


