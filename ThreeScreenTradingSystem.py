## FirstScreen

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.figure as fig
from Investar import Analyzer

mk = Analyzer.MarketDB()
df = mk.get_daily_price('엔씨소프트', '2017-01-01', '2019-12.01')

ema60 = df.close.ewm(span=60).mean()
ema130 = df.close.ewm(span=130).mean()
macd = ema60 - ema130
signal = macd.ewm(span=45).mean()
macdhist = macd - signal

df = df.assign(ema130=ema130, ema60=ema60, macd=macd, signal=signal, macdhist=macdhist).dropna()
df['number'] = df.index.map(mdates.date2num)
ohlc = df[['number', 'open', 'high', 'low', 'close']]
ndays_high = df.high.rolling(window=14, min_periods=1).max()
ndays_low = df.low.rolling(window=14, min_periods=1).min()

fast_k = (df.close - ndays_low) / (ndays_high - ndays_low) * 100
slow_d = fast_k.rolling(window=3).mean()
df = df.assign(fast_k=fast_k, slow_d=slow_d).dropna()

# Slow stochastic
slow_fast_k = (slow_d)
slow_slow_d = slow_d.rolling(window=3).mean()
df = df.assign(slow_fast_k=slow_fast_k, slow_slow_d=slow_slow_d).dropna()

plt.figure(figsize=(9,9))
p1 = plt.subplot(311)
plt.title('Triple Screen Trading (NCSOFT)')
plt.grid(True)
candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.plot(df.number, df['ema130'], color='c', label='EMA130')
plt.ylim([220000, 550000])
for i in range(1, len(df.close)):
    if df.ema130.values[i-1] < df.ema130.values[i] and \
        df.slow_d.values[i] < 20:
        plt.plot(df.number.values[i], 250000, 'r^')
    elif df.ema130.values[i-1] > df.ema130.values[i] and \
        df.slow_d.values[i] > 80:
        plt.plot(df.number.values[i], 250000, 'bv')
plt.legend(loc='best')

p2 = plt.subplot(312)
plt.grid(True)
p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.bar(df.number, df['macdhist'], color='m', label='MACD-Hist')
plt.plot(df.number, df['macd'], color='b', label='MACD')
plt.plot(df.number, df['signal'], 'g--', label='MACD-Signal')
plt.legend(loc='best')

p3 = plt.subplot(313)
plt.grid(True)
p3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.yticks([0, 20, 80, 100])
plt.plot(df.number, df.fast_k, color='c', label='Fast_K')
plt.plot(df.number, df.slow_d, color='k', label='Slow_D')
plt.legend(loc='best')

# Apply Slow Stochastic
plt.figure(figsize=(9,9))
p1 = plt.subplot(311)
plt.title('Triple Screen Trading (NCSOFT)')
plt.grid(True)
candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.plot(df.number, df['ema130'], color='c', label='EMA130')
plt.ylim([220000, 550000])
for i in range(1, len(df.close)):
    if df.ema130.values[i-1] < df.ema130.values[i] and \
        df.slow_slow_d.values[i] < 20:
        plt.plot(df.number.values[i], 250000, 'r^')
    elif df.ema130.values[i-1] > df.ema130.values[i] and \
        df.slow_slow_d.values[i] > 80:
        plt.plot(df.number.values[i], 250000, 'bv')
plt.legend(loc='best')

p4 = plt.subplot(313)
plt.grid(True)
p4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.yticks([0, 20, 80, 100])
plt.plot(df.number, df.slow_fast_k, color='c', label='Slow_Stocastic:Fast_K')
plt.plot(df.number, df.slow_slow_d, color='k', label='Slow_Stocastic:Slow_D')
plt.show()
