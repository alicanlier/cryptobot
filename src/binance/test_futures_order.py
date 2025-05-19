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
import config.config as config
import utility.calculation_tools as calculate
import futures_prices as futures
import utility.strategy_tools as strategy

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
    price_deviation_sma = current_price - last_sma
    price_deviation_ema = current_price - last_ema

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
        quantity = round((order_amount / current_price), quantity_precision)  # Quantity to buy or sell 12 USDT
        if price_deviation_sma > last_std * 2 * 0.95 and trade_side == 'SELL':
            TRADE_SIDE = 'SELL'
        elif price_deviation_sma < 0 and abs(price_deviation_sma) > last_std * 2 * 0.95 and trade_side == 'BUY':
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
            if price_dict['deviation_ratio'] > 2:
                if sma1min_coins_ordered_dict[symbol]['deviation_ratio'] > 0.01 and sma15min_coins_ordered_dict[symbol][
                    'deviation_ratio'] > 4:
                    narrow_sell_symbol_list.append(symbol)
            elif 0.4 > price_dict['deviation_ratio'] > 0.1:
                if sma1min_coins_ordered_dict[symbol]['deviation_ratio'] < -0.01 and 1 > \
                        sma15min_coins_ordered_dict[symbol]['deviation_ratio'] > 0.2:
                    narrow_buy_symbol_list.append(symbol)
            elif -0.1 > price_dict['deviation_ratio'] > -0.4:
                if sma1min_coins_ordered_dict[symbol]['deviation_ratio'] > 0.01 and -0.2 > \
                        sma15min_coins_ordered_dict[symbol]['deviation_ratio'] > -1:
                    narrow_sell_symbol_list.append(symbol)
            elif price_dict['deviation_ratio'] < -2:
                if sma1min_coins_ordered_dict[symbol]['deviation_ratio'] < -0.01 and \
                        sma15min_coins_ordered_dict[symbol]['deviation_ratio'] < -4:
                    narrow_buy_symbol_list.append(symbol)
        except BinanceAPIException as e:
            print(f'ERROR for {symbol}: {e}')

    for symbol, price_dict in sma15min_coins_ordered_dict.items():
        if symbol in sma4h_coins_ordered_dict.keys():
            try:
                if price_dict['deviation_ratio'] > 10:
                    if sma5min_coins_ordered_dict[symbol]['deviation_ratio'] > 0.05 and \
                            sma4h_coins_ordered_dict[symbol]['deviation_ratio'] > 20:
                        wide_sell_symbol_list.append(symbol)
                elif 2 > price_dict['deviation_ratio'] > 0.5:
                    if sma5min_coins_ordered_dict[symbol]['deviation_ratio'] < -0.05 and 5 > \
                            sma4h_coins_ordered_dict[symbol]['deviation_ratio'] > 1:
                        wide_buy_symbol_list.append(symbol)
                elif -0.5 > price_dict['deviation_ratio'] > -2:
                    if sma5min_coins_ordered_dict[symbol]['deviation_ratio'] > 0.05 and -1 > \
                            sma4h_coins_ordered_dict[symbol]['deviation_ratio'] > -5:
                        wide_sell_symbol_list.append(symbol)
                elif price_dict['deviation_ratio'] < -10:
                    if sma5min_coins_ordered_dict[symbol]['deviation_ratio'] < -0.05 and \
                            sma4h_coins_ordered_dict[symbol]['deviation_ratio'] < -20:
                        wide_buy_symbol_list.append(symbol)
            except ValueError as ve:
                print(f'ERROR for {symbol}: {ve}')
            except BinanceAPIException as be:
                print(f'ERROR for {symbol}: {be}')

    # symbol_list = ['ATOMUSDT', 'FILUSDT', 'SEIUSDT', 'SUIUSDT', 'XAIUSDT', 'QTUMUSDT', 'IDUSDT', 'AIUSDT', 'WAVESUSDT']
    for symbol in narrow_sell_symbol_list[:]:
        sd = sma5min_coins_ordered_dict[symbol]
        print(
            f"{'{:<12}'.format('NARROW SELL:')} {'{:<11}'.format(symbol)}: last_sma:{'{:<9}'.format(sd['last_sma'])} last_close:{'{:<9}'.format(sd['last_close_price'])} deviation:{sd['deviation_ratio']}%")
        # single_future_order(symbol, 'SELL', 5, WINDOW, ORDER_AMOUNT)
    for symbol in narrow_buy_symbol_list[:]:
        sd = sma5min_coins_ordered_dict[symbol]
        print(
            f"{'{:<12}'.format('NARROW BUY:')} {'{:<11}'.format(symbol)}: last_sma:{'{:<9}'.format(sd['last_sma'])} last_close:{'{:<9}'.format(sd['last_close_price'])} deviation:{sd['deviation_ratio']}%")
        # single_future_order(symbol, 'BUY', 5, WINDOW, ORDER_AMOUNT)
    for symbol in wide_sell_symbol_list[:]:
        sd = sma15min_coins_ordered_dict[symbol]
        print(
            f"{'{:<12}'.format('WIDE SELL:')} {'{:<11}'.format(symbol)}: last_sma:{'{:<9}'.format(sd['last_sma'])} last_close:{'{:<9}'.format(sd['last_close_price'])} deviation:{sd['deviation_ratio']}%")
        # single_future_order(symbol, 'SELL', 15, WINDOW, ORDER_AMOUNT)
    for symbol in wide_buy_symbol_list[:]:
        sd = sma15min_coins_ordered_dict[symbol]
        print(
            f"{'{:<12}'.format('WIDE BUY:')} {'{:<11}'.format(symbol)}: last_sma:{'{:<9}'.format(sd['last_sma'])} last_close:{'{:<9}'.format(sd['last_close_price'])} deviation:{sd['deviation_ratio']}%")
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
    select_list1 = ['1INCHUSDT', 'AAVEUSDT', 'ADAUSDT', 'AGIXUSDT', 'AIUSDT', 'ALGOUSDT', 'ALICEUSDT', 'ALTUSDT',
                    'APEUSDT', 'APTUSDT', 'AXSUSDT']
    select_list2 = ['ARKMUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BAKEUSDT', 'BALUSDT', 'BANDUSDT', 'BCHUSDT', 'BELUSDT',
                    'BLZUSDT', 'C98USDT', 'CAKEUSDT', 'CELOUSDT', 'CHRUSDT', 'COTIUSDT', 'CRVUSDT', 'CYBERUSDT']
    select_list3 = ['DENTUSDT', 'DYDXUSDT', 'EDUUSDT', 'EOSUSDT', 'FETUSDT', 'FILUSDT', 'FLMUSDT', 'GASUSDT', 'GLMUSDT',
                    'GRTUSDT', 'HIFIUSDT', 'HIGHUSDT']
    select_list4 = ['JASMYUSDT', 'JUPUSDT', 'KAVAUSDT', 'KNCUSDT', 'LDOUSDT', 'LPTUSDT', 'MAGICUSDT', 'MASKUSDT',
                    'MATICUSDT', 'MBOXUSDT', 'MDTUSDT', 'MEMEUSDT']
    select_list5 = ['NEARUSDT', 'NFPUSDT', 'NMRUSDT', 'OPUSDT', 'PHBUSDT', 'PIXELUSDT', 'RNDRUSDT', 'SANDUSDT',
                    'SEIUSDT', 'SPELLUSDT', 'STRAXUSDT', 'SUIUSDT', 'SUSHIUSDT']
    select_list6 = ['TIAUSDT', 'UNFIUSDT', 'UNIUSDT', 'WAVESUSDT', 'WLDUSDT', 'XAIUSDT', 'XVGUSDT', 'ZILUSDT']
    select_list7_1000 = ['BONKUSDT', 'FLOKIUSDT', 'PEPEUSDT', 'SHIBUSDT']
    select_list8_nospot = ['BIGTIMEUSDT']
    select_list = select_list1 + select_list2 + select_list3 + select_list4 + select_list5 + select_list6 + select_list7_1000
    # select_list = ['XRPUSDT']

    # config_logging(logging, logging.DEBUG)

    um_futures_client = UMFutures(key=API_KEY, secret=API_SECRET)

    # trades = um_futures_client.trades('XRPUSDT')
    # print('trades:',trades)

    def place_order(symbol, side, quantity, price):
        try:
            response = um_futures_client.new_order(
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

    # print(f"{'{:<10}'.format('symbol')}: {'{:<18}'.format('current_price')} | {'{:<18}'.format('5min middle_line')} | {'{:<18}'.format('5min upper_band')} | 5min lower_band\n{'-' * 88}")
    for symbol in select_list:
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
                place_order(symbol, 'SELL', quantity, order_dict['SELL'])
                place_order(symbol, 'BUY', quantity, order_dict['BUY'])
                print(f'  --------  orders have been implemented.\n')
            except ValueError as ve:
                print(f'  --------  ERROR: {ve}')
                place_order(symbol, 'SELL', 1, order_dict['SELL'])
                place_order(symbol, 'BUY', 1, order_dict['BUY'])


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

    print(f"\n.... Running schedule at {time_now}\n")

    select_future_order()
    # get_balance_info()

    # '''

    # Schedule the method to be called every 10 minutes
    schedule.every(5.0).minutes.do(select_future_order)

    # Run the scheduler loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)  # Sleep for a short duration to avoid high CPU usage
        except:
            pass
    # '''

