# SK하이닉스 Bollinger band, Intraday intensity

import matplotlib.pyplot as plt
from Investar import Analyzer

mk = Analyzer.MarketDB()
df = mk.get_daily_price('SK하이닉스', '2019-01-01', '2019.11.30')

df['MA20'] = df['close'].rolling(window=20).mean()
df['stddev'] = df['close'].rolling(window=20).std()
df['upper'] = df['MA20'] + (df['stddev'] * 2)
df['lower'] = df['MA20'] - (df['stddev'] * 2)
df['PB'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])

df['II'] = (2*df['close']-df['high']-df['low']) \
    / (df['high'] - df['low']) * df['volume']
df['IIP21'] = (df['II'].rolling(window=21).sum()) \
    / (df['volume'].rolling(window=20).sum()) * 100
df = df.dropna()

plt.figure(figsize=(9,9))
plt.subplot(311)
plt.title('SK Hynix Bollinger Band(20 day, 2 std) - Reversals')
plt.plot(df.index, df['close'], 'b', label='Close')
plt.plot(df.index, df['upper'], 'r--', label='Upper band')
plt.plot(df.index, df['lower'], 'c--', label='Lower band')
plt.plot(df.index, df['MA20'], 'k--', label='Moving Average 20')
plt.fill_between(df.index, df['upper'], df['lower'], color = '0.8')
for i in range(0, len(df['close'])):
    if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
        plt.plot(df.index.values[i], df.close.values[i], 'r^')
    elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:
        plt.plot(df.index.values[i], df.close.values[i], 'bv')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(312)
plt.plot(df.index, df['PB'], 'b', label='%b')
plt.grid(True)
plt.legend(loc='best')

plt.subplot(313)
plt.bar(df.index, df['IIP21'], color='g', label='Intraday intencity')
for i in range(0, len(df['close'])):
    if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
        plt.plot(df.index.values[i], 0, 'r^')
    elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:
        plt.plot(df.index.values[i], 0, 'bv')
plt.grid(True)
plt.legend(loc='best')
plt.show()