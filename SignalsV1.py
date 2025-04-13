import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import seaborn as sns
import pprint
from prettytable import PrettyTable
from tabulate import tabulate
from dateutil.relativedelta import relativedelta
class MarketRegimes:
    def __init__(self, path1, path2, months_back):
        self.price_data = pd.read_excel(path1)
        self.valuation_data = pd.read_excel(path2)
        # print(self.valuation_data.columns)
        self.cols = list(set(self.price_data.columns) - set(['dates']))
        self.price_data.dates = pd.to_datetime(self.price_data.dates, infer_datetime_format=True)
        rel = relativedelta(months=months_back)
        index =  self.price_data[self.price_data.dates >= (self.price_data.dates.iat[-1] -rel)].index[0]
        self.month_start_index = index - self.price_data.dates.index[-1]
        if months_back == 0:
            self.month_start_index = -1   
        print(f"Price Data Last Date is {self.price_data.dates.iat[-1]}")
        print(f"{months_back} Months Back Index is {index} & date on that Index is {self.price_data.dates.iat[self.month_start_index]}")

    def calculate_rolling(self, month=3, year = 0):
        rel = relativedelta(years=year, months=month)
        numerator_data_loc = self.price_data[self.price_data.dates >= (self.price_data.dates.iloc[0]+rel)].index[0]
        print(numerator_data_loc)

        if month !=0:
            returns = self.price_data[self.cols]/self.price_data[self.cols].shift(1*numerator_data_loc) -1

        else:
            returns = (self.price_data[self.cols]/self.price_data[self.cols].shift(1*numerator_data_loc))**(1/year) -1

        returns['dates'] = self.price_data.dates
        # print(returns)

        return returns.dropna().reset_index(drop = True)
    
    def calculate_volatility(self, x = 12, y = 36):
        returns = self.price_data[self.cols].pct_change()
        returns['dates'] = self.price_data.dates

        vol_ratio = returns[self.cols].rolling(22*x).std()
        # /returns[self.cols].rolling(22*y).std()
        vol_ratio['dates'] = self.price_data.dates

        return vol_ratio.dropna().reset_index(drop = True)
    
    def calculate_momentum(self, days1=100, days2=200):
        rolling100 = self.price_data[self.cols].rolling(window=days1).mean()
        rolling200 = self.price_data[self.cols].rolling(window=days2).mean()

        rolling_100_dist = self.price_data[self.cols].subtract(rolling100)/self.price_data[self.cols]
        rolling_200_dist = self.price_data[self.cols].subtract(rolling200)/self.price_data[self.cols]
        rolling_100_dist['dates'] = self.price_data.dates
        rolling_200_dist['dates'] = self.price_data.dates


        return rolling_100_dist.dropna().reset_index(drop = True), rolling_200_dist.dropna().reset_index(drop = True)
    
    def calculate_dma(self, days1=100, days2=200):
        rolling100 = self.price_data[self.cols].rolling(window=days1).mean()
        rolling200 = self.price_data[self.cols].rolling(window=days2).mean()

        rolling100['dates'] = self.price_data.dates
        rolling200['dates'] = self.price_data.dates


        return rolling100.dropna().reset_index(drop = True), rolling200.dropna().reset_index(drop = True)
    
    def calculate_rsi(self, periods = 14):
        difference = self.price_data[self.cols].diff()
        # print(difference)
        rolling = pd.DataFrame()
        for index in self.cols:
            rolling['gains_'+index] = np.where(difference[index]>0, difference[index], 0)
            rolling['loss_'+index] = np.where(difference[index]<0, -difference[index], 0)
            rolling[index] = 100 - (100/(1+rolling['gains_'+index].rolling(window=periods).mean()/rolling['loss_'+index].rolling(window=periods).mean()))

        # print(rolling['Nifty Smallcap 250 - TRI'])
        rolling['dates'] = self.price_data.dates
        
        return rolling

       

    
    def calculate_valuation(self):
        data = self.valuation_data.copy()
        # print(data.columns)
        data['month']= data.dates.dt.month
        data['year']= data.dates.dt.year
        cols = list(set(data.columns) - set(['NIFTY SMALLCAP 250 PB','dates', 'month', 'year']))

        monthly = data.groupby(['year', 'month'])[cols].mean().reset_index()
        # print(monthly.columns)
        rolling12m = monthly[cols].rolling(12).mean()
        rolling12m[['month', 'year']] = monthly[['month', 'year']]


        rolling36m = monthly[cols].rolling(36).mean()
        rolling36m[['month', 'year']] = monthly[['month', 'year']]

        return monthly, rolling12m, rolling36m
    
    def percentile_calculator(self, pe_year = 2023, pe_month = 11):
        timeframe = self.month_start_index
        print(f"{timeframe}")
        print(f"...............Taking {timeframe//22} Month data to check where do we stand.....\n\n")
        print("...............Initialising Rolling Return Analysis For Multiple Period Rolling Return.....\n")
        years = [(0,3),(1,0), (5,0)]
        # ,(3,0),(5,0),(7,0),(10,0)
        for year,month in years:
            rol = self.calculate_rolling(month=month, year=year)
            print(f".................For {year} Year {month} Month.......................")
            res = {'index':[], 'percentile':[]}
            # print(rol)
            for index in self.cols:
                last_val = rol[index].iat[timeframe]
                pct = len(rol[rol[index]>last_val])/len(rol)
                res['index'].append(index)
                res['percentile'].append(1-pct)
            res = pd.DataFrame(res)
            res.sort_values(by = 'percentile', ascending=False,inplace=True)
            print(tabulate(res, headers='keys', tablefmt='psql', showindex=False))

        print("..........Finishing Rolling Return Analysis....\n ")

        print("...............Initialising Momentum Analysis.....\n")
        momentum100, momentum200 = self.calculate_momentum()
        print("................(Price - DMA)/Price...............")
        new_df = momentum100.iloc[[timeframe]].T.reset_index()
        new_df.columns = ['index', 'price-dma/price']
        print(tabulate(new_df, headers='keys', tablefmt='psql', showindex=False))
        print("..............Where are we Today(Percentile) as Compared to 100DMA.........")
        
        res_momentum = {'index':[], 'percentile':[]}
        for index in self.cols:
            last_val = momentum100[index].iat[timeframe]
            pct = len(momentum100[momentum100[index]>last_val])/len(momentum100)
            res_momentum['index'].append(index)
            res_momentum['percentile'].append(1-pct)
        res_momentum = pd.DataFrame(res_momentum)
        res_momentum.sort_values(by = 'percentile', ascending=False,inplace=True)
        print(tabulate(res_momentum, headers='keys', tablefmt='psql', showindex=False))

        print("\n ...................RSI......................\n")
        rsi = self.calculate_rsi(periods=100)
        res_rsi = {'index':[], 'RSI':[]}
        for col in self.cols:
            res_rsi['index'].append(col)
            res_rsi['RSI'].append(rsi[col].iat[timeframe])

        res_rsi = pd.DataFrame(res_rsi)
        res_rsi.sort_values(by='RSI', ascending=False, inplace=True)
        print(tabulate(res_rsi, headers='keys', tablefmt='psql', showindex=False))



        print("..........Finishing Momentum Analysis....\n ")

        print("...............Initialising PE/PB Analysis.....\n")
        monthly, rolling12m, rolling36m = self.calculate_valuation()
        x = PrettyTable()
        x.field_names = ['Index', 'Current_PE', '12M_Rolling_PE', '36M_Rolling_PE']
        x.add_rows(
            [
                ['NIFTY SMALLCAP 250 PE', monthly.loc[(monthly.year==pe_year) & (monthly.month==pe_month), 'NIFTY SMALLCAP 250 PE'].iat[0], rolling12m.loc[(rolling12m.year==pe_year) & (rolling12m.month==pe_month), 'NIFTY SMALLCAP 250 PE'].iat[0], rolling36m.loc[(rolling36m.year==pe_year) & (rolling36m.month==pe_month), 'NIFTY SMALLCAP 250 PE'].iat[0]], 
                
                ['NIFTY MIDCAP 150 PE', monthly.loc[(monthly.year==pe_year) & (monthly.month==pe_month), 'NIFTY MIDCAP 150 PE'].iat[0], rolling12m.loc[(rolling12m.year==pe_year) & (rolling12m.month==pe_month), 'NIFTY MIDCAP 150 PE'].iat[0], rolling36m.loc[(rolling36m.year==pe_year) & (rolling36m.month==pe_month), 'NIFTY MIDCAP 150 PE'].iat[0]],

                ['NIFTY 100 PE', monthly.loc[(monthly.year==pe_year) & (monthly.month==pe_month), 'NIFTY 100 PE'].iat[0], rolling12m.loc[(rolling12m.year==pe_year) & (rolling12m.month==pe_month), 'NIFTY 100 PE'].iat[0], rolling36m.loc[(rolling36m.year==pe_year) & (rolling36m.month==pe_month), 'NIFTY 100 PE'].iat[0]],

                ['NIFTY ALPHA LOW-VOLATILITY 30 PE', monthly.loc[(monthly.year==pe_year) & (monthly.month==pe_month), 'NIFTY ALPHA LOW-VOLATILITY 30 PE'].iat[0], rolling12m.loc[(rolling12m.year==pe_year) & (rolling12m.month==pe_month), 'NIFTY ALPHA LOW-VOLATILITY 30 PE'].iat[0], rolling36m.loc[(rolling36m.year==pe_year) & (rolling36m.month==pe_month), 'NIFTY ALPHA LOW-VOLATILITY 30 PE'].iat[0]],

                ['NIFTY50 VALUE 20 PE', monthly.loc[(monthly.year==pe_year) & (monthly.month==pe_month), 'NIFTY50 VALUE 20 PE'].iat[0], rolling12m.loc[(rolling12m.year==pe_year) & (rolling12m.month==pe_month), 'NIFTY50 VALUE 20 PE'].iat[0], rolling36m.loc[(rolling36m.year==pe_year) & (rolling36m.month==pe_month), 'NIFTY50 VALUE 20 PE'].iat[0]],
            ]
        )

        print(x)
        print("..........Finishing Valuation Analysis....\n ")
        print("...............Initialising Volatility Analysis.....\n")
        vol = self.calculate_volatility(x = 12, y = 36)
        res_vol = {'index':[], 'percentile':[]}
        for index in self.cols:
            last_val = vol[index].iat[timeframe]
            pct = len(vol[vol[index]>last_val])/len(vol)
            res_vol['index'].append(index)
            res_vol['percentile'].append(1-pct)
            
        res_vol = pd.DataFrame(res_vol)
        res_vol.sort_values(by = 'percentile', ascending=False,inplace=True)
        print(tabulate(res_vol, headers='keys', tablefmt='psql', showindex=False))

        print("..........Finishing Volatality Analysis....\n ")
    


months_back = 0
M = MarketRegimes(path1='Low_Volatility_Indices_Nov23_result.xlsx', path2= "Valuation_Multiples/all_pe.xlsx", months_back=months_back)

# M.percentile_calculator(pe_year = 2023, pe_month = 12)
# M.calculate_rsi()
vol = M.calculate_volatility()
momentum100, momentum200 = M.calculate_dma()
rolling1y, rolling5y = M.calculate_rolling(year=1, month=0), M.calculate_rolling(year=5, month=0)
# vol.to_excel('Voltility_dumpy.xlsx', index=False)
price  = M.price_data

    
with pd.ExcelWriter("signals_dump.xlsx") as writer:
    price.to_excel(writer, sheet_name="price", index=False)
    momentum100.to_excel(writer, sheet_name="100_day_ma", index=False)
    vol.to_excel(writer, sheet_name="volatility", index=False)
    rolling1y.to_excel(writer, sheet_name="1y_rolling", index=False)
    rolling5y.to_excel(writer, sheet_name="5y_rolling", index=False)



