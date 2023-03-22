import pandas as pd
import ta


def preprocess(ohlcv):
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Remove duplicate rows
    df = df[~df.index.duplicated(keep='first')]

    # Remove missing data
    df.dropna(inplace=True)

    # Create additional features
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Add 50 EMA and 200 EMA
    df['ema_50'] = ta.trend.EMAIndicator(
        df['close'], window=50).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(
        df['close'], window=200).ema_indicator()

    # Add volume indicators
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        df['close'], df['volume']).on_balance_volume()
    df['chaikin'] = ta.volume.ChaikinMoneyFlowIndicator(
        df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()

    return df
