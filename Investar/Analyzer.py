import pymysql
import pandas as pd
from datetime import datetime
from datetime import timedelta
import re
import calendar

class MarketDB:
    def __init__(self):
        """생성자: MariaDB 연결 및 종목코드 딕셔너리 생성"""
        self.conn = pymysql.connect(host='localhost', user='root',
            password='tlqkfanjdi1', db='INVESTAR', charset='utf8')
        self.codes = {}     # == self.codes = dict()
        self.get_comp_info()

    def __del__(self):
        """소멸자: MariaDB 연결 해제"""
        self.conn.close()

    def get_comp_info(self):
        """company_info 테이블에서 읽어와서 codes에 저장"""
        sql = "SELECT * FROM company_info"
        krx = pd.read_sql(sql, self.conn)
        for idx in range(len(krx)):
            self.codes[krx['code'].values[idx]] = krx.company.values[idx]

    def get_daily_price(self, code, start_date=None, end_date=None):
        """KRX 종목별 시세를 데이터프레임 형태로 반환
            - code       : KRX 종목코드('005930'), ('삼성전자')
            - start_date : 조회 시작일('2020-01-01') 미입력시 1년 전 오늘
            - end_date   : 조회 종료일('2020-09-30') 미입력시 오늘
        """
        if start_date is None:
            one_year_ago = datetime.today() - timedelta(days=365)
            start_date = one_year_ago.strftime('%Y-%m-%d')
            print("start_date is initialized to '{}'".format(start_date))
        else:
            start_lst = re.split('[^0-9]+', start_date) # split by non-numeric chars
            if start_lst[0] == '':
                start_lst = start_lst[1:]
            start_year = int(start_lst[0])
            start_month = int(start_lst[1])
            start_day = int(start_lst[2])
            if start_year < 1900 or start_year > 2200:
                print(f"ValueError: start_year({start_year:d} is out of range")
                return
            if start_month < 1 or start_month > 12:
                print(f"ValueError: start_month({start_month:d} is out of range")
                return
            if start_day < 1 or start_day > calendar.monthrange(start_year, start_month)[1]:
                print(f"ValueError: start_day({start_day:d} is out of range")
                return
            start_date = f"{start_year:04d}-{start_month:02d}-{start_day:02d}"

        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            print("end_date is initialized to '{}'".format(end_date))
        else:
            end_lst = re.split('[^0-9]+', end_date)
            if end_lst[0] == '':
                end_lst = end_lst[1:]
            end_year = int(end_lst[0])
            end_month = int(end_lst[1])
            end_day = int(end_lst[2])
            if end_year < 1900 or end_year > 2200:
                print(f"ValueError: end_year({end_year:d} is out of range")
                return
            if end_month < 1 or end_month > 12:
                print(f"ValueError: end_month({end_month:d} is out of range")
                return
            if end_day < 1 or end_day > calendar.monthrange(end_year, end_month)[1]:
                print(f"ValueError: end_day({end_day:d} is out of range")
                return
            end_date = f"{end_year:04d}-{end_month:02d}-{end_day:02d}"

        codes_keys = list(self.codes.keys())
        codes_values = list(self.codes.values())
        if code in codes_keys:
            pass
        elif code in codes_values:
            idx = codes_values.index(code)
            code = codes_keys[idx]
        else:
            print("ValueError: Code({}) doesn't exist.".format(code))

        sql = f"SELECT * FROM daily_price WHERE code = '{code}'"\
            f" and date >= '{start_date}' and date <= '{end_date}'"
        df = pd.read_sql(sql, self.conn)
        df.index = df['date']
        return df


#For debugging
"""
if __name__ == '__main__':
    mk = MarketDB()
    print(mk.get_daily_price('code', 'start_date', 'end_date')) 
"""