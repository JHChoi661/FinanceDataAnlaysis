"""print(r'c:\window\system32\notepad.exe') #raw string 지정
# 따옴표는 \', \" 로 표시 ''' ''', """ """로도 가능
'''로 주석도 표시가능'''"""

#.extend()로 다중리스트 원소만 리스트 뒤에 추가 가능
'''--------------------20201207----------------------'''
"""def getCAGR(first, last, years):
    return (last/first)**(1/years)-1

print('sec CAGR : {:.2%}'.format(getCAGR(65300, 2669000, 20)))"""

#help('modules')
#__path__ 속성이 있으면 패키지다.
#print(keyword.__file__)파일 경로 추적

"""클래스
class A:
    def methodA(self):
        print('aa')
    def method(self):
        print('a')


class B:
    def methodB(self):
        print('b')


class C(A, B):
    def methodC(self):
        print('cc')
    def method(self):
        print('c')
        super().method()
zzz
c = C()
c.methodA()
c.methodC()
c.method()"""
""" 이미지처리
import requests as rq

url = 'https://postfiles.pstatic.net/20110808_216/hunji1_131278362721489XC2_JPEG/35.jpg?type=w2'
r = rq.get(url, stream=True).raw

from PIL import Image

img = Image.open(r)
#img.show()
img.save('src.png')

#print(img.get_format_mimetype)

BUF_SIZE = 1024 #일정한 길이로 나누어서 복사하는 것이 좋다.
with open('src.png', 'rb') as sf, open('dst.png', 'wb') as df:  #따로 파일객체를 close()할 필요가 없다.
    while True:
        data = sf.read(BUF_SIZE)
        if not data:
            break #읽을 데이터가 없다.
        df.write(data)
#img2 = Image.open('dst.png')
#img2.show()

import hashlib

sha_src = hashlib.sha256()
sha_dst = hashlib.sha256()

with open('src.png', 'rb') as sf, open('dst.png', 'rb') as df:
    sha_src.update(sf.read())
    sha_dst.update(df.read())

#print("'src.png's hash : {}".format(sha_src.hexdigest()))
#print("'dst.png's hash : {}".format(sha_dst.hexdigest()))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dst_img = mpimg.imread('dst.png')
#print(dst_img.shape)
pseude_img = dst_img[:, :, 0]
#print(pseude_img.shape)

plt.suptitle('Image Processing', fontsize=18)
plt.subplot(1,2,1)#121도 가능!
plt.title('Original Image')
plt.imshow(mpimg.imread('src.png'))

plt.subplot(1,2,2)
plt.title('Pseudocolor Image')
plt.imshow(mpimg.imread('dst.png'))
plt.show()"""

#numpy
"""
import numpy as np

A = np.array([[1,2], [3,4]])
print(A) # .ndim : dimension, .shape : size, .dtype : elements type
#.max(), .mean(), .min(), .sum() 가능
# 인덱싱 A[1][1] = A[1, 1]
# A[A>1] 등 조건에 맞는 원소만 인덱싱 가능
# .transpose(), .flatten() - 1차원으로 바꿈, 평탄화, .dot()는 내적 곱 - 수학의 행렬의 곱과 같음
"""

# pandas
"""
import pandas as pd

s= pd.Series([0.0, 3.6, 2.0, 5.8, 4.2, 8.0])
#print(s)
s.index = pd.Index([0.0, 1.2, 1.8, 3.0, 3.6, 4.8])
s.index.name = 'MY_IDX'
s.name = 'MY_SERIES'
#print(s)
s[5.9] = 5.5 #series에 데이터 추가 : dictionary와 유사하다.
#print(s)
ser = pd.Series([6.7, 4.2], index=[6.8, 8.0]) # series 생성 시 index 지정 가능(index와 data의 개수가 다르면 에러)
s = s.append(ser)
#print(s)
#s.values[IDX], s.index[IDX], s.loc[index], s.iloc[index]
#s.iloc와 s.values는 둘 다 값을 반환하지만, 복수의 값은 iloc는 series로 반환, values는 배열로 반환한다.
#s.drop(index)로 데이터 삭제 가능
#use describe() to check the information of the series object - num of elements, mean, standard deviation, min, max, quartiles(사분위수)
print(s.describe())

import matplotlib.pyplot as plt
plt.title("WAVE")
plt.plot(s, 'bs--')
plt.xticks(s.index)
plt.yticks(s.values)
plt.grid(True)
plt.show()"""
'''--------------------20201208----------------------'''
"""
#pandas dataframe
import pandas as pd
'''
df = pd.DataFrame({'KOSPI': [1915, 1961, 2026, 2467, 2041], 
    'KOSDAQ': [542, 682, 631, 798, 675]}, 
    index = range(2014, 2019))
print(df)
print(df.describe())'''

#can make a dataframe with multiple series
'''
kospi = pd.Series([1915, 1961, 2026, 2467, 2041], index=range(2014, 2019), name='KOSPI')
kospi.index.name = 'Year'
print(kospi)
kosdaq = pd.Series([542, 682, 631, 798, 675], index=range(2014, 2019), name='KOSDAQ')
kosdaq.index.name = 'Year'
print(kosdaq)
df = pd.DataFrame({kospi.name: kospi, kosdaq.name: kosdaq})
print(df)
'''
#can make a dataframe with adding each rows
'''
col = ['KOSPI', 'KOSDAQ']
index = range(2014, 2019)
rows = []
rows.append([1915, 542])
rows.append([1961, 682])
rows.append([2026, 631])
rows.append([2467, 798])
rows.append([2041, 675])
df = pd.DataFrame(rows, columns=col, index=index)  # this method doesn't need {}
print(df)
'''
#traversal
'''
for i in df.index:
    print(i, df['KOSPI'][i], df['KOSDAQ'][i])
for row in df.itertuples(name='KRX'):
    print(row[0], row[1], row[2])
'''
#get stock prices of Samsung elec. and Microsoft
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
yf.pdr_override() #type always

sec = pdr.get_data_yahoo('005930.KS', start='2018-05-04') #pandas.dataframe type
msft = pdr.get_data_yahoo('MSFT', start='2018-05-04')
tmp_msft = msft.drop(columns='Volume')
'''
plt.plot(sec.index, sec.Close, 'b', label='Samsung Electronics')
plt.plot(tmp_msft.index, tmp_msft.Close, 'r', label='Microsoft')
plt.grid('ON')
plt.title('what the fxxk?')
plt.legend(loc='best')
plt.show()'''   #msft:$, selc:won ==> should compare with daily percent change
#P_today = ((R_t - R_(t-1))/R_(t-1)) * 100
#print(sec.Close) # sec.Close==sec['Close']
sec_dpc = (sec.Close / sec.Close.shift(1) - 1) * 100
sec_dpc.iloc[0] = 0
#print(sec_dpc.head())

msft_dpc = (msft.Close / msft.Close.shift(1) - 1) * 100
msft_dpc.iloc[0] = 0
#print(msft_dpc.head())
'''
plt.hist(sec_dpc, bins=18, label='Samsung daily percent change')
plt.grid(True)
plt.legend(loc='best')
plt.show()  #leptokurtic distribution(급첨분포:가운데가 더 뾰족하다), fat tail
            #the price of stock moves in very narrow range, very big price changing happens compare with the normal distribution
'''
#get a Cumulative Sum
sec_dpc_cs = sec_dpc.cumsum()
msft_dpc_cs = msft_dpc.cumsum()
plt.plot(sec_dpc_cs.index, sec_dpc_cs, 'b', label='SEC cumulative sum')
plt.plot(msft_dpc_cs.index, msft_dpc_cs, 'r-', label='MSFT cumulative sum')
plt.grid(True)
plt.legend(loc='best')
plt.show()
"""
'''--------------------20201209----------------------'''
"""
#MDD: Maximum Drawdown : The biggest lost in particular period
#==> 1-(highest point/lowest point)
from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt
'''
kospi = pdr.get_data_yahoo('^KS11', '2004-01-04')

window = 252
peak = kospi['Adj Close'].rolling(window, min_periods=1).max()
drawdown = kospi['Adj Close']/peak - 1.0 # percent change
max_dd = drawdown.rolling(window, min_periods=1).min()

plt.figure(figsize=(9, 7))
plt.subplot(211)
kospi['Close'].plot(label='KOSPI', title='KOSPI MDD', grid=True, legend=True)
plt.subplot(212)
drawdown.plot(c='blue', label='KOSPI DD', grid=True, legend=True)
max_dd.plot(c='red', label='KOSPI MDD', grid=True, legend=True)
plt.show()'''

dow = pdr.get_data_yahoo('^DJI', '2000-01-04')
kospi = pdr.get_data_yahoo('^KS11', '2000-01-04')

d = 100 * dow.Close/dow.Close.iloc[0]# should 'indexize' both indexes to compare
k = 100 * kospi.Close/kospi.Close.iloc[0]
'''
plt.figure(figsize=(9,5))
plt.plot(d.index, d, 'r', label='Dow Jones Industrial')
plt.plot(k.index, k, 'b', label='KOSPI')
plt.grid(True)
plt.legend(loc='Best')
plt.show() '''

# Scatter plot analysing x:DJI, y:KOSPI
#plt.scatter(dow, kospi, marker='.')==> size error occur
df = pd.DataFrame({'DOW':dow['Close'], 'KOSPI':kospi['Close']}) # sizing with NaN
df = df.fillna(method='bfill')
df = df.fillna(method='ffill')
'''
plt.figure(figsize=(7,7))
plt.scatter(df['DOW'], df['KOSPI'], marker='.')
plt.show() # if the graph is similar with y=x, two indexes have a correlation
'''

from scipy import stats

regr = stats.linregress(df['DOW'], df['KOSPI'])
# also, can get a correlation coefficient with df.corr():dataframe, 
#                                              df['DOW'].corr(df['KOSPI']):series
"""
'''--------------------20201211----------------------'''
"""
from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
import matplotlib.pylab as plt
yf.pdr_override()

from scipy import stats
# linear regression analysis with DJI, KOSPI
'''
dow = pdr.get_data_yahoo('^DJI', '2000-01-04')
kospi = pdr.get_data_yahoo('^KS11', '2000-01-04')

df = pd.DataFrame({'X': dow['Close'], 'Y': kospi['Close']})
df = df.fillna(method='bfill')
df = df.fillna(method='ffill')

regr = stats.linregress(df.X, df.Y)
regr_line = f'Y = {regr.slope:.2f} * X + {regr.intercept:.2f}'

plt.figure(figsize=(7, 7))
plt.plot(df.X, df.Y, '.')
plt.plot(df.X, regr.slope * df.X + regr.intercept, 'r') # y = slope*x + intercept
plt.legend(['DOW x KOSPI', regr_line])
plt.title(f'DOW x KOSPI (R = {regr.rvalue:.2f})')
plt.xlabel('DJI Average')
plt.ylabel('KOSPI')
plt.show()
'''
# linear regression analysis with TLT, KOSPI
tlt = pdr.get_data_yahoo('TLT', '2002-07-30')
kospi = pdr.get_data_yahoo('^KS11', '2002-07-30')

df = pd.DataFrame({'X':tlt['Close'], 'Y':kospi['Close']})
df = df.fillna(method='bfill')
df = df.fillna(method='ffill')

regr = stats.linregress(df.X, df.Y)
regr_line = f'Y = {regr.slope:.2f} * X + {regr.intercept:.2f}'

plt.figure(figsize=(7,7))
plt.plot(df.X, df.Y, '.')
plt.plot(df.X, regr.slope*df.X + regr.intercept, 'r')
plt.legend(['TLT x KOSPI', regr_line])
plt.legend(loc='best')
plt.title(f'TLT x KOSPI (R = {regr.rvalue:.2f})')
plt.xlabel('TLT')
plt.ylabel('KOSPI')
plt.show()
# making a portfolio with objects that have low regression value will
# reduce my risks
"""
'''--------------------20201212----------------------'''
'''
# Web scraping
import pandas as pd
krx_list = pd.read_html(r'C:\coding\vscode files\python_dateAnalysis\상장법인목록.xls')
df = pd.read_html('https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13')[0]
#print(type(df))
krx_list[0].종목코드 = krx_list[0].종목코드.map('{:06d}'.format)

krx_list[0] = krx_list[0].sort_values(by='종목코드')
#print(krx_list[0])

from bs4 import BeautifulSoup
from urllib.request import urlopen
url = 'https://finance.naver.com/item/sise_day.nhn?code=068270&page=1'
with urlopen(url) as doc:
    html = BeautifulSoup(doc, 'lxml')
    pgrr = html.find('td', class_='pgRR')
    s = str(pgrr.a['href']).split('=')
    lastPage = s[-1]

df = pd.DataFrame()
sise_url = 'https://finance.naver.com/item/sise_day.nhn?code=068270'

for entire pages
for page in range(1, int(lastPage)+1):
    pageUrl = '{}&page={}'.format(sise_url, page)
    df = df.append(pd.read_html(pageUrl, header=0)[0])

for page in range(1, 31):
    pageUrl = '{}&page={}'.format(sise_url, page)
    df = df.append(pd.read_html(pageUrl, header=0)[0])
df = df.dropna()
df = df.iloc[0:30]
df = df.sort_values(by='날짜')

import matplotlib.pyplot as plt

plt.title('Celltrion (close)')
plt.xticks(rotation=45)
plt.plot(df.날짜, df.종가, 'co-')
plt.grid(color='gray', linestyle='--')
plt.show()
df = df.rename(columns={'날짜':'Date', '시가':'Open', '고가':'High', '저가':'Low', 
'종가':'Close', '거래량':'Volume'})
df.index = pd.to_datetime(df.Date)
df = df[['Close', 'Open', 'High', 'Low', 'Volume']]
print(df)
import mplfinance as mpf
# mpf.plot(df, title='Celltrion candle', type='candle')
'''
'''--------------------20201214----------------------'''

""" # connect with database(mariaDB, mySQL)
import pymysql

connection = pymysql.connect(host='localhost', port=3306, db='INVESTAR',
    user='root', passwd='tlqkfanjdi1', autocommit=True)

cursor = connection.cursor()
cursor.execute("SELECT VERSION();")
result = cursor.fetchone()

print(result)

connection.close() """
'''--------------------20201215----------------------'''
"""
# KRX 상장 목록에서 코넥스 기업들을 DB에서 제거
import pymysql, calendar, time, json
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.request import urlopen
from threading import Timer
conn = pymysql.connect(host='localhost', user='root',
        password='tlqkfanjdi1', db='INVESTAR', charset='utf8', autocommit=True)

url = "C:/coding/vscode_files/python_dataAnalysis/상장법인목록(코넥스).xls"

krx_KONEX = pd.read_html(url, header=0)[0]
krx_KONEX = krx_KONEX[['종목코드']]
krx_KONEX = krx_KONEX.rename(columns={'종목코드':'code'})
krx_KONEX.code = krx_KONEX.code.map('{:06d}'.format)
print(krx_KONEX)

with conn.cursor() as curs:
    for  _, code in enumerate(krx_KONEX.code):
        sql = f"DELETE FROM company_info WHERE CODE='{code}'"
        curs.execute(sql)
"""
'''--------------------20201229----------------------'''


a = {'a':30, 'b':[20]}
a = list(a)
print(a[0][1])
print('123123')