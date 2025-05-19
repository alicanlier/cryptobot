import configparser
import random
import ast


def fetch_credentials():
    config = configparser.ConfigParser()
    config.read('../../config/config.properties')

    # API_KEY = ast.literal_eval(config['credentials']['API_KEY'].strip())
    # API_SECRET = ast.literal_eval(config['credentials']['API_SECRET'].strip())
    API_KEY = ''
    API_SECRET = ''
    # API_KEY_futures = ast.literal_eval(config['credentials']['API_KEY_futures'].strip())
    # API_SECRET_futures = ast.literal_eval(config['credentials']['API_SECRET_futures'].strip())
    API_KEY_futures = ''
    API_SECRET_futures = ''

    return {'API_KEY':API_KEY, 'API_SECRET':API_SECRET, 'API_KEY_futures':API_KEY_futures, 'API_SECRET_futures':API_SECRET_futures}


def fetch_interval_dict(client):
    interval_dict = {'1m':client.KLINE_INTERVAL_1MINUTE, '1min':client.KLINE_INTERVAL_1MINUTE, 1:client.KLINE_INTERVAL_1MINUTE,
                     '3m': client.KLINE_INTERVAL_3MINUTE, '3min': client.KLINE_INTERVAL_3MINUTE, 3: client.KLINE_INTERVAL_3MINUTE,
                     '5m':client.KLINE_INTERVAL_5MINUTE, '5min':client.KLINE_INTERVAL_5MINUTE, 5:client.KLINE_INTERVAL_5MINUTE,
                     '15m':client.KLINE_INTERVAL_15MINUTE, '15min':client.KLINE_INTERVAL_15MINUTE, 15:client.KLINE_INTERVAL_15MINUTE,
                     '30m':client.KLINE_INTERVAL_30MINUTE, '30min':client.KLINE_INTERVAL_30MINUTE, 30:client.KLINE_INTERVAL_30MINUTE,
                     '1h':client.KLINE_INTERVAL_1HOUR, '1hour':client.KLINE_INTERVAL_1HOUR, 60:client.KLINE_INTERVAL_1HOUR,
                     '2h': client.KLINE_INTERVAL_2HOUR, '2hour': client.KLINE_INTERVAL_2HOUR, 120: client.KLINE_INTERVAL_2HOUR, 2: client.KLINE_INTERVAL_2HOUR,
                     '4h':client.KLINE_INTERVAL_4HOUR, '4hour':client.KLINE_INTERVAL_4HOUR, 240:client.KLINE_INTERVAL_4HOUR, 4:client.KLINE_INTERVAL_4HOUR,
                     '1d':client.KLINE_INTERVAL_1DAY, '1day':client.KLINE_INTERVAL_1DAY, 24:client.KLINE_INTERVAL_1DAY,
                     '3d':client.KLINE_INTERVAL_3DAY, '3day':client.KLINE_INTERVAL_3DAY, 72:client.KLINE_INTERVAL_3DAY,
                     '1w':client.KLINE_INTERVAL_1WEEK, '1week':client.KLINE_INTERVAL_1WEEK, 7:client.KLINE_INTERVAL_1WEEK,
                     '1mon':client.KLINE_INTERVAL_1MONTH, '1month':client.KLINE_INTERVAL_1MONTH, '1M':client.KLINE_INTERVAL_1MONTH, 31:client.KLINE_INTERVAL_1MONTH}

    return interval_dict


def time_dict():
    time_dict = {'1m':1, '1min':1, 1:1,
                 '3m':3, '3min':3, 3:3,
                 '5m':5, '5min':5, 5:5,
                 '15m':15, '15min':15, 15:15,
                 '30m':30, '30min':30, 30:30,
                 '1h':60, '1hour':60, 60:60,
                 '2h': 120, '2hour': 120, 120: 120, 2: 120,
                 '4h':240, '4hour':240, 240:240, 4:240,
                 '1d':1440, '1day':1440, 24:1440,
                 '3d':4320, '3day':4320, 72:4320,
                 '1w':10080, '1week':10080, 7:10080,
                 '1mon':43800, '1month':43800, '1M':43800}

    return time_dict


def time_dict_str():
    time_dict_str = {'1m':'1m', '1min':'1m', 1:'1m',
                    '3m':'3m', '3min':'3m', 3:'3m',
                    '5m':'5m', '5min':'5m', 5:'5m',
                    '15m':'15m', '15min':'15m', 15:'15m',
                    '30m':'30m', '30min':'30m', 30:'30m',
                    '1h': '1h', '1hour': '1h', 60: '1h',
                    '2h': '2h', '2hour': '2h', 120: '2h', 2: '2h',
                    '4h': '4h', '4hour': '4h', 240: '4h', 4: '4h',
                    '1d': '1d', '1day': '1d', 24: '1d',
                    '3d': '3d', '3day': '3d', 72: '3d',
                    '1w': '1w', '1week': '1w', 7: '1w',
                    '1mon': '1M', '1month': '1M', '1M': '1M'}

    return time_dict_str


def order_limiting_mins_dict():
    order_limiting_mins_dict = {'1m':5, '1min':5, 1:5,
                 '3m':10, '3min':10, 3:10,
                 '5m':15, '5min':15, 5:15,
                 '15m':30, '15min':30, 15:30,
                 '30m':60, '30min':60, 30:60,
                 '1h':120, '1hour':120, 60:120,
                 '2h': 240, '2hour': 240, 120: 240, 2: 240,
                 '4h':480, '4hour':480, 240:480, 4:480,
                }

    return order_limiting_mins_dict


def fetch_real_time_frame(time_frame):
    real_time_frame = time_dict()[time_frame]
    if real_time_frame < 60:
        return str(real_time_frame) + 'min'
    elif real_time_frame < 1440:
        return str(int(real_time_frame/60)) + 'hour'
    else:
        return str(int(real_time_frame/60/24)) + 'day'


def operation_dict(module):
    oper_dict = {'ema':module.ema, 'sma':module.sma, 'rsi':module.rsi, 'macd':module.macd, 'bb':module.bb}
    return oper_dict


def random_color():
    color_key = random.choice(list(color_dict.keys()))
    return color_key


dict_keys   = ['symbol', 'pair', 'contractType', 'deliveryDate', 'onboardDate', 'status', 'maintMarginPercent', 'requiredMarginPercent', 'baseAsset', 'quoteAsset', 'marginAsset', 'pricePrecision', 'quantityPrecision', 'baseAssetPrecision', 'quotePrecision', 'underlyingType', 'underlyingSubType', 'settlePlan', 'triggerProtect', 'liquidationFee', 'marketTakeBound', 'maxMoveOrderLimit']
dict_values = ['BCHUSDT', 'BCHUSDT', 'PERPETUAL', 4133404800000, 1569398400000, 'TRADING', '2.5000',            '5.0000',                'BCH', 'USDT', 'USDT',                     2,                3,                   8,                    8,                'COIN',           ['PoW'],             0,            '0.0500',         '0.015000',       '0.05',            10000,            ]
important_keys = ['pricePrecision', 'quantityPrecision', {'filters':[{'tickSize':0.01},{'stepSize': '0.001', 'minQty': '0.001'}]}, 'orderTypes', 'timeInForce']

filters = [{'maxPrice': '100000', 'tickSize': '0.01', 'minPrice': '13.93', 'filterType': 'PRICE_FILTER'}, {'minQty': '0.001', 'filterType': 'LOT_SIZE', 'maxQty': '10000', 'stepSize': '0.001'}, {'stepSize': '0.001', 'filterType': 'MARKET_LOT_SIZE', 'minQty': '0.001', 'maxQty': '850'}, {'limit': 200, 'filterType': 'MAX_NUM_ORDERS'}, {'limit': 10, 'filterType': 'MAX_NUM_ALGO_ORDERS'}, {'notional': '20', 'filterType': 'MIN_NOTIONAL'}, {'multiplierDecimal': '4', 'filterType': 'PERCENT_PRICE', 'multiplierUp': '1.0500', 'multiplierDown': '0.9500'}]
orderTypes = ['LIMIT', 'MARKET', 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET']
timeInForce = ['GTC', 'IOC', 'FOK', 'GTX', 'GTD']

futures_exchange_info_keys = ['timezone', 'serverTime', 'futuresType', 'rateLimits', 'exchangeFilters', 'assets', 'symbols']
symbols_symbol_keys = ['symbol', 'pair', 'contractType', 'deliveryDate', 'onboardDate', 'status', 'maintMarginPercent', 'requiredMarginPercent', 'baseAsset', 'quoteAsset', 'marginAsset', 'pricePrecision', 'quantityPrecision', 'baseAssetPrecision', 'quotePrecision', 'underlyingType',
                       'underlyingSubType', 'settlePlan', 'triggerProtect', 'liquidationFee', 'marketTakeBound', 'maxMoveOrderLimit', 'filters', 'orderTypes', 'timeInForce']
pricePrecision, quantityPrecision, baseAssetPrecision, quotePrecision = {1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3}, {8}, {8}
status, contractType, quoteAsset = 'TRADING', 'PERPETUAL', 'USDT'
futures_symbols_pkl_path = r'C:\Users\A\Desktop\my-PT\crypto_bot\data\futures_symbols.pkl'

color_dict = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0), 'cyan': (0, 255, 255),
'magenta': (255, 0, 255), 'orange': (255, 165, 0), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'teal': (0, 128, 128),
'brown': (165, 42, 42), 'navy': (0, 0, 128), 'olive': (128, 128, 0), 'maroon': (128, 0, 0), 'lavender': (230, 230, 250),
'turquoise': (64, 224, 208), 'indigo': (75, 0, 130), 'gold': (255, 215, 0), 'silver': (192, 192, 192), 'black': (0, 0, 0)}
