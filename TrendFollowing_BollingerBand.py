### Trading strategy
# Bollinger Band
# 2020-12-21

import matplotlib.pyplot as plt
from Investar import Analyzer

mk = Analyzer.MarketDB()
df = mk.get_daily_price('NAVER', '2020-04-02')

df['MA20'] = df['close'].rolling(window=20).mean()
df['stddev'] = df['close'].rolling(window=20).std()
df['upper'] = df['MA20'] + (2 * df['stddev'])
df['lower'] = df['MA20'] - (2 * df['stddev'])

# With %B
df['PB'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])
# print(df.head(40)) row 1~19 NaN
df = df[19:]

# MFI
df['TP'] = (df['high'] + df['low'] + df['close']) / 3 # Typical price
df['PMF'] = 0 # Positive money flow
df['NMF'] = 0
for i in range(len(df.close)-1):
    if df.TP.values[i] < df.TP.values[i+1]:
        df.PMF.values[i+1] = df.TP.values[i+1] * df.volume.values[i+1]
        df.NMF.values[i+1] = 0
    else:
        df.NMF.values[i+1] = df.TP.values[i+1] * df.volume.values[i+1]
        df.PMF.values[i+1] = 0
df['MFR'] = df.PMF.rolling(window=10).sum() / df.NMF.rolling(window=10).sum()
df['MFI10'] = 100 - 100 / (1 + df['MFR'])

plt.figure(figsize=(9, 5))
plt.subplot(211)
plt.plot(df.index, df['close'], color='#0000FF', label='Close')
plt.plot(df.index, df['upper'], 'r--', label='Upper band')
plt.plot(df.index, df['MA20'], 'k--', label='Moving Average 20')
plt.plot(df.index, df['lower'], 'c--', label='Lower band')
plt.fill_between(df.index, df['upper'], df['lower'], color='0.8')
plt.legend(loc='best')
plt.title('NAVER Bollinger Band (20 day, 2 std) - Trend Following')
for i in range(len(df['close'])):
    if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
        plt.plot(df.index.values[i], df.close.values[i], 'r^')
    elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
        plt.plot(df.index.values[i], df.close.values[i], 'bv')


plt.subplot(212) 
plt.plot(df.index, df['PB'] * 100, color='b', label='%B x 100')
plt.plot(df.index, df['MFI10'], 'g--', label='MFI 10day')
plt.yticks([-20, 0, 20, 40, 60, 80, 100, 120])
plt.grid(True)
plt.legend(loc='best')
for i in range(len(df['close'])):
    if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
        plt.plot(df.index.values[i], 0, 'r^')
    elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
        plt.plot(df.index.values[i], 0, 'bv')
plt.show()