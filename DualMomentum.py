# Dual Momentum Class
import pandas as pd
import pymysql
from datetime import datetime
from datetime import timedelta
from Investar import Analyzer
class DualMomentum:
    def __init__(self):
        """생성자 : KRX 종목코드(codes)를 구하기 위한 MarketDB 객체 생성"""
        self.mk = Analyzer.MarketDB()
    
    def get_rltv_momentum(self, start_date, end_date, stock_count):
        """특정 기간 동안 수익률이 제일 높았던 stock_count 개의 종목들 (상대 모멘텀)
            - start_date  : 상대 모멘텀을 구할 시작일자 ('2020-01-01')
            - end_date    : 상대 모멘텀을 구할 종료일자 ('2020-12-24')
            - stock_count : 상대 모멘텀을 구할 종목 수
            ***********유/무상 증자, 액면 분할에 주가 변동 처리 등 필요...***************
            코넥스 기업 DB 제거 완료
        """
        connection = pymysql.connect(host='localhost', port=3306, \
            db='INVESTAR', user='root', passwd='tlqkfanjdi1', autocommit=True)
        cursor = connection.cursor()

        sql =f"SELECT max(date) from daily_price where date <= '{start_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()
        if result[0] is None:
            print("start_date : {} -> returned None".format(sql))
            return
        start_date = result[0].strftime('%Y-%m-%d')

        sql = f"SELECT max(date) from daily_price where date <= '{end_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()
        if result[0] is None:
            print("end_date : {} -> returned None".format(sql))
            return
        end_date = result[0].strftime('%Y-%m-%d')

        rows = []
        columns = ['code', 'company', 'old_price', 'new_price', 'returns']
        for _, code in enumerate(self.mk.codes):
            sql = f"SELECT code FROM daily_price "\
                f"WHERE code='{code}' and date >= '{start_date}' and date <= '{end_date}' and open=0"
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is not None:
                continue
            sql = f"SELECT close FROM daily_price "\
                f"WHERE code='{code}' AND date='{start_date}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                continue
            old_price = int(result[0])
            sql = f"SELECT close FROM daily_price "\
                f"WHERE code='{code}' AND date='{end_date}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                continue
            new_price = int(result[0])
            returns = (new_price / old_price - 1) * 100
        
            rows.append([code, self.mk.codes[code], old_price, new_price, returns])
        df = pd.DataFrame(rows, columns=columns)
        df = df[['code', 'company', 'old_price', 'new_price', 'returns']] #????
        df = df.sort_values(by='returns', ascending=False)
        df = df.head(stock_count)
        df.index = pd.Index(range(stock_count))
        connection.close()
        print(df)
        print(f"\nRelative momentum ({start_date} ~ {end_date}) : "\
            f"{df['returns'].mean():.2f}% \n")
        return df



    def get_abs_momentum(self, rltv_momentum, start_date, end_date):
        """특정 기간 동안 상대 모멘텀에 투자했을 때의 평균 수익률 (절대 모멘텀)
            - rltv_momentum : get_rltv_momentum() 함수의 리턴값 (상대 모멘텀)
            - start_date  : 절대 모멘텀을 구할 매수일 ('2020-01-01')
            - end_date    : 절대 모멘텀을 구할 매도일 ('2020-12-24')
        """
        stocklist = list(rltv_momentum['code'])
        connection = pymysql.connect(host='localhost', port=3306, \
            db='INVESTAR', user='root', passwd='tlqkfanjdi1', autocommit=True)
        cursor = connection.cursor()

        sql = f"SELECT max(date) from daily_price where date <= '{start_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()
        if result[0] is None:
            print("start_date : {} -> returned None".format(sql))
            return
        start_date = result[0].strftime('%Y-%m-%d')

        sql = f"SELECT max(date) from daily_price where date <= '{end_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()
        if result[0] is None:
            print("end_date : {} -> returned None".format(sql))
            return
        end_date = result[0].strftime('%Y-%m-%d')

        rows = []
        columns = ['code', 'company', 'old_price', 'new_price', 'returns']
        for _, code in enumerate(stocklist):
            # 유/무상 증자 배제
            sql = f"SELECT code FROM daily_price "\
                f"WHERE code='{code}' and date >= '{start_date}' and date <= '{end_date}' and open=0" 
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is not None:
                continue

            sql = f"SELECT close FROM daily_price "\
                f"WHERE code='{code}' AND date='{start_date}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                continue
            old_price = int(result[0])

            sql = f"SELECT close FROM daily_price "\
                f"WHERE code='{code}' AND date='{end_date}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                continue
            new_price = int(result[0])

            returns = (new_price / old_price - 1) * 100
            rows.append([code, self.mk.codes[code], old_price, new_price, returns])

        df = pd.DataFrame(rows, columns=columns)
        df = df[['code', 'company', 'old_price', 'new_price', 'returns']] #????
        df = df.sort_values(by='returns', ascending=False)
        df = df.head(len(stocklist))
        df.index = pd.Index(range(len(stocklist)))
        connection.close()
        print(df)
        print(f"\nAbsolute momentum ({start_date} ~ {end_date}) : "\
            f"{df['returns'].mean():.2f}% \n")
        return


if __name__ == "__main__":
    DM = DualMomentum()
    rm = DM.get_rltv_momentum('2020-06-15', '2020-09-15', 10)
    DM.get_abs_momentum(rm, '2020-09-15', '2020-12-15')