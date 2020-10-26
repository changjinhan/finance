import matplotlib.pyplot as plt
import FinanceDataReader as fdr  # finance-datareader library

'''
* Overview
The FinanceDataReader is financial data reader(crawler) for finance.
The main functions are as follows.

- KRX Stock Symbol listings: 'KRX', 'KOSPI', 'KODAQ', 'KONEX'
- Global Stock Symbol listings: 'NASDAQ', 'NYSE', 'AMEX' and 'S&P500', 'SSE'(상해), 'SZSE'(심천), 'HKEX'(홍콩), 'TSE'(도쿄)
- KRX delistings: 'KRX-DELISTING'(상장폐지종목), 'KRX-ADMINISTRATIVE' (관리종목)
- ETF Symbol listings(for multiple countries): 'KR', 'US', 'JP'
- Stock price(KRX): '005930'(Samsung), '091990'(Celltrion Healthcare) ...
- Stock price(Word wide): 'AAPL', 'AMZN', 'GOOG' ... (you can specify exchange(market) and symbol)
- Indexes: 'KS11'(코스피지수), 'KQ11'(코스닥지수), 'DJI'(다우존스지수), 'IXIC'(나스닥지수), 'US500'(S&P 500지수) ...
- Exchanges: 'USD/KRX', 'USD/EUR', 'CNY/KRW' ... (조합가능한 화폐별 환율 데이터 일자별)
- Cryptocurrency price data: 'BTC/USD' (Bitfinex), 'BTC/KRW' (Bithumb)

* Install
$ pip install finance-datareader

'''


# 차트 설정
plt.rcParams["font.family"] = 'nanummyeongjo'
plt.rcParams["figure.figsize"] = (14,4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams["axes.grid"] = True

# 한국거래소 상장종목 전체
df_krx = fdr.StockListing('KRX')
print(df_krx.head())
print('한국거래소 상장종목 수:', len(df_krx))

# NASDAQ 종목 전체
df_ndq = fdr.StockListing('NASDAQ')
print(df_ndq.head())

# ETF 종목 리스트
df_etf = fdr.EtfListing('KR')
print(df_etf.head(10))

# 애플(AAPL), 2018-01-01 ~ 2020-01-30
df = fdr.DataReader('AAPL', '2018-01-01', '2020-01-30')
print(df.tail())

# KS11 (KOSPI 지수), 2015년~현재
df = fdr.DataReader('KS11', '2015-01-01')
df['Close'].plot().figure.savefig('KOSPI.png')