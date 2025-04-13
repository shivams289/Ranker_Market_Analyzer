import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
# Cat_index can be 'MIDCAP', 'SMALLCAP', 'LARGECAP'


class FundRanker:
    def __init__(self, cat_index='MDCP', start_date='4/1/2010', end_date='5/11/2023'):

        cat_indices = ['NIFTY 100 - TRI',
                       'Nifty Smallcap 250 - TRI', 'Nifty Midcap 150 - TRI', 'NIFTY 500 - TRI', 'Nifty Infrastructure - TRI', 'Nifty India Consumption - TRI', 'NIFTY BANK - TRI', 'Nifty Financial Services - TRI', 'Nifty50 Value 20 - TRI']
        self.start_dt = start_date
        self.end_dt = end_date
        # load midcap data
        if cat_index == 'MDCP':
            nav = pd.read_excel('Data/all_midcap_unprocessed.xlsx')
            nav = self.data_preprocessing(nav)
            self.category_index = cat_indices[2]

        elif cat_index == 'BANKS':
            # load smallcap data
            nav = pd.read_excel('Data/all_bank_finserv_unprocessed.xlsx')
            nav = self.data_preprocessing(nav)
            self.category_index = cat_indices[7]

        elif cat_index == 'SMCP':
            # load smallcap data
            nav = pd.read_excel('Data/all_smallcap_unprocessed.xlsx')
            nav = self.data_preprocessing(nav)
            self.category_index = cat_indices[1]

        elif cat_index == 'VALUE':
            nav = pd.read_excel('Data/all_value_unprocessed.xlsx')
            nav = self.data_preprocessing(nav)
            self.category_index = cat_indices[-1]

        elif cat_index == 'RETM' or cat_index == 'ELSS':
            # load smallcap data
            if cat_index == 'ELSS':
                nav = pd.read_excel('Data/all_elss_unprocessed.xlsx')

            else:
                nav = pd.read_excel('Data/all_retirement_unprocessed.xlsx')

            nav = self.data_preprocessing(nav)
            self.category_index = cat_indices[3]

        elif cat_index == 'FOCUS' or cat_index == 'FLEXI':
            if cat_index == 'FLEXI':
                nav = pd.read_excel('Data/all_flexicap_unprocessed.xlsx')

            else:
                nav = pd.read_excel('Data/all_focussed_unprocessed.xlsx')
            nav = self.data_preprocessing(nav)
            self.category_index = cat_indices[3]

        elif cat_index == 'INFRA':
            nav = pd.read_excel('Data/all_infra_unprocessed.xlsx')
            nav = self.data_preprocessing(nav)
            self.category_index = cat_indices[4]

        else:
            # load largecap data
            if cat_index == 'ARB':
                nav = pd.read_excel(
                    'Data/all_arbitrage_funds_unprocessed.xlsx')

            if cat_index == 'LCP':
                nav = pd.read_excel('Data/all_largecap_unprocessed.xlsx')
            if cat_index == 'AGG':
                nav = pd.read_excel(
                    'Data/all_agressive_unprocessed.xlsx')
            if cat_index == 'AALOC':
                nav = pd.read_excel(
                    'Data/all_asset_allocator_unprocessed.xlsx')
            if cat_index == 'MULTIA':
                nav = pd.read_excel(
                    'Data/all_multiasset_unprocessed.xlsx')
            if cat_index == 'ESAVING':
                nav = pd.read_excel(
                    'Data/all_equity_savings_unprocessed.xlsx')
            if cat_index == 'CONSERV':  # Balanced ADV: BADV
                nav = pd.read_excel(
                    'Data/all_conservative_hybrid_unprocessed.xlsx')

            if cat_index == 'BADV':  # Balanced ADV: BADV
                nav = pd.read_excel(
                    'Data/all_dynamic_balan_adv_unprocessed.xlsx')

            if cat_index == 'MULTICAP':
                nav = pd.read_excel(
                    'Data/all_multicap_unprocessed.xlsx')
            if cat_index == 'LargeMid':
                nav = pd.read_excel(
                    'Data/all_large_mid_unprocessed.xlsx')

            nav = self.data_preprocessing(nav)
            self.category_index = cat_indices[0]

        indices = pd.read_excel('Data/indices_unprocessed.xlsx')
        self.indices = self.data_preprocessing(indices)

        self.nav = pd.merge(
            self.indices[['dates', self.category_index]], nav, on='dates', how='left')
        print("..............Data Loaded.............")
        print(f"ANALYZING {cat_index} vs {self.category_index} INDEX")

    # start, end date format id mm/dd/yy
    def preprocess_dates(self, nav):
        date_col = nav.columns[nav.columns.str.contains("ate")].values[0]
        nav.rename(columns={date_col: "dates"}, inplace=True)

        dates = pd.date_range(start=self.start_dt,
                              end=self.end_dt, freq="B").date
        nav.dates = pd.to_datetime(
            nav.dates, infer_datetime_format=True).dt.date

        merged_nav = pd.DataFrame({"dates": dates})
        merged_nav = merged_nav.merge(nav, on="dates", how="left")
        merged_nav.sort_values(by="dates", inplace=True)

        return merged_nav

    def preprocess_missing(self, nav):
        for col in nav.columns:
            if col != "dates":
                nav[col] = nav[col].ffill().add(
                    nav[col].bfill()).div(2)

        return nav

    def data_preprocessing(self, nav):
        nav = self.preprocess_dates(nav=nav)
        nav = self.preprocess_missing(nav=nav)
        print("........Data Preprocessed...........")

        return nav

    def range(self):
        dates = pd.date_range(start=self.start_dt,
                              end=self.end_dt, freq="B").date
        dates = sorted(set(dates))
        review_dates = [dt.date(2010, 4, 1)]

        for i in range(1, len(dates)):
            if dates[i-1].month == 3 and dates[i].month == 4:
                review_dates.append(dates[i])

        return review_dates

    def generate_rolling_return(self, data, dat=dt.date(2010, 4, 1), lookback_years=3, lag=22):
        rel = relativedelta(months=12*lookback_years)
        data = data[data.dates <= dat]

        if data.dates.iloc[0] <= (dat - rel):
            data = data[data.dates >= (dat - rel)]

        else:
            data = data.loc[(data.dates <= dat) & (data.dates >= (dat-rel))]

        data.reset_index(drop=True, inplace=True)
        data.dropna(axis=1, inplace=True)
        print(data)
        col_names = data.columns[~data.columns.str.contains('dates')]
        if lag <= 261:
            ret = (data[col_names].shift(-lag)/data[col_names]) - 1
        else:
            ret = (data[col_names].shift(-lag)/data[col_names])**(261/lag) - 1
        ret['dates'] = data.dates.shift(-lag)
        ret.dropna(inplace=True)
        print(
            f"...........{lag/261} Rolling Return years is generated...........")

        return ret

    def generate_returns(self, data, dat=dt.date(2010, 4, 1), lookback_years=3, lag=22):
        rel = relativedelta(months=12*lookback_years)
        data = data[data.dates <= dat]

        if data.dates.iloc[0] <= (dat - rel):
            data = data[data.dates >= (dat - rel)]

        else:
            data = data.loc[(data.dates <= dat) & (data.dates >= (dat-rel))]

        data.reset_index(drop=True, inplace=True)
        data.dropna(axis=1, inplace=True)
        print(data)
        col_names = data.columns[~data.columns.str.contains('dates')]
        if lag <= 261:
            ret = (data[col_names].shift(-lag)/data[col_names]) - 1
        else:
            ret = (data[col_names].shift(-lag)/data[col_names])**(261/lag) - 1
        ret['dates'] = data.dates.shift(-lag)
        ret.dropna(inplace=True)
        print(
            f"...........{lag/261} Rolling Return years is generated for correlation...........")

        return ret

    def Ranker(self, dat=dt.date(2010, 4, 1), lookback_years=3, rolling_lag_days=22):
        category_index = self.category_index
        data = self.generate_rolling_return(
            dat=dat, data=self.nav, lookback_years=lookback_years, lag=rolling_lag_days)
        print(data)
        col_names = data.columns[~data.columns.str.contains('dates')]
        smallcap_names = col_names

        ret_data_1d = self.generate_returns(
            dat=dat, data=self.nav, lookback_years=lookback_years, lag=1)
        ret_data_1m = self.generate_returns(
            dat=dat, data=self.nav, lookback_years=lookback_years, lag=22)
        ret_data_3m = self.generate_returns(
            dat=dat, data=self.nav, lookback_years=lookback_years, lag=66)

        def calculate_metrics(data):
            res = {"fund_name": [], "percent_rolling_avg": [], "beated_by_percent_avg": [], "lost_by_percent_avg": [], "perc_times_beated": [
            ], "beated_by_percent_max": [], "beated_by_percent_min": [], "lost_by_percent_min": [], "lost_by_percent_max": [], "Daily_Return_Corr": [], "Monthly_Return_Corr": [], "Quaterly_Return_Corr": []}
            for x in list(set(col_names)-set([category_index])):
                res['fund_name'].append(x)
                rolling_avg = data[x].mean()*100

                res['Daily_Return_Corr'].append(ret_data_1d[x].corr(
                    ret_data_1d[self.category_index]))
                res['Monthly_Return_Corr'].append(ret_data_1m[x].corr(
                    ret_data_1m[self.category_index]))
                res['Quaterly_Return_Corr'].append(ret_data_3m[x].corr(
                    ret_data_3m[self.category_index]))

                beat_perc = max(0, (data.loc[(data[x] >= data[category_index]), x] - data.loc[(
                    data[x] >= data[category_index]), category_index]).mean()*100)
                lost_perc = min(0, (data.loc[(data[x] < data[category_index]), x] - data.loc[(
                    data[x] < data[category_index]), category_index]).mean()*100)

                beat_perc_max = max(0, (data.loc[(data[x] >= data[category_index]), x] - data.loc[(
                    data[x] >= data[category_index]), category_index]).max()*100)
                beat_perc_min = max(0, (data.loc[(data[x] >= data[category_index]), x] - data.loc[(
                    data[x] >= data[category_index]), category_index]).min()*100)
                lost_perc_min = min(0, (data.loc[(data[x] < data[category_index]), x] - data.loc[(
                    data[x] < data[category_index]), category_index]).min()*100)
                lost_perc_max = min(0, (data.loc[(data[x] < data[category_index]), x] - data.loc[(
                    data[x] < data[category_index]), category_index]).max()*100)

                pct_times_beated = max(0, data.loc[(
                    data[x] >= data[category_index])].shape[0]/data.shape[0]*100)

                res['percent_rolling_avg'].append(rolling_avg)
                res['beated_by_percent_avg'].append(beat_perc)
                res['lost_by_percent_avg'].append(lost_perc)
                res['beated_by_percent_max'].append(beat_perc_max)
                res['beated_by_percent_min'].append(beat_perc_min)
                res['lost_by_percent_min'].append(lost_perc_min)
                res['lost_by_percent_max'].append(lost_perc_max)

                wtd_avg_out = pct_times_beated*beat_perc + \
                    (100-pct_times_beated)*lost_perc
                print(
                    f"for {x} beat times {pct_times_beated} & beat avg {beat_perc} lost avg {lost_perc} wtd_avg {wtd_avg_out}")

                res['perc_times_beated'].append(
                    pct_times_beated)

            res = pd.DataFrame(res)
            res['Wtd_avg_outperformance'] = res['perc_times_beated']*res['beated_by_percent_avg'] + \
                (100-res['perc_times_beated'])*res['lost_by_percent_avg']

            return res.fillna(0)

        p1 = int(0.5*len(data))  # part 1 of data: 50%
        res1 = calculate_metrics(data.iloc[:p1, :])
        res2 = calculate_metrics(data.iloc[p1:, :])
        res = 0.4*res1.set_index('fund_name')+(0.6*res2.set_index('fund_name'))
        res.sort_values(
            by=['Wtd_avg_outperformance', 'perc_times_beated'], ascending=False, inplace=True)
        res['return_rank'] = [x for x in range(1, len(res)+1)]
        rel = relativedelta(months=12*lookback_years)

        self.nav = self.nav.loc[(self.nav.dates >= (dat - rel)) & (
            self.nav.dates <= dat)]
        dd = (self.nav[smallcap_names].cummax(
        ) - self.nav[smallcap_names]).div(self.nav[smallcap_names].cummax())

        maxdd = pd.DataFrame((self.nav[smallcap_names].cummax() - self.nav[smallcap_names]).div(self.nav[smallcap_names].cummax()).max()*100, columns=[
            'MaxDrawdown']).reset_index()
        maxdd.rename(columns={'index': 'fund_name'}, inplace=True)
        resdd = {"fund_name": [], "dd_less_by_percent_avg": [], "dd_greater_by_percent_avg": [
        ], "perc_times_dd_less": [], 'dd_greater_max': [], 'dd_less_min': []}
        for x in list(set(smallcap_names)-set([category_index])):
            resdd['fund_name'].append(x)
            beat_perc = (dd.loc[(dd[x] <= dd[category_index]), x] -
                         dd.loc[(dd[x] <= dd[category_index]), category_index]).mean()*100
            lost_perc = (dd.loc[(dd[x] > dd[category_index]), x] -
                         dd.loc[(dd[x] > dd[category_index]), category_index]).mean()*100

            dd_less_min = (dd.loc[(dd[x] <= dd[category_index]), x] -
                           dd.loc[(dd[x] <= dd[category_index]), category_index]).min()*100
            dd_greater_max = (dd.loc[(dd[x] > dd[category_index]), x] -
                              dd.loc[(dd[x] > dd[category_index]), category_index]).max()*100

            resdd['dd_less_by_percent_avg'].append(beat_perc)
            resdd['dd_greater_by_percent_avg'].append(lost_perc)
            resdd['perc_times_dd_less'].append(
                dd.loc[(dd[x] <= dd[category_index])].shape[0]/dd.shape[0]*100)

            resdd['dd_greater_max'].append(dd_greater_max)
            resdd['dd_less_min'].append(dd_less_min)

        resdd = pd.DataFrame(resdd)
        print(maxdd)
        res = pd.merge(res, resdd, on='fund_name', how='left')
        res = pd.merge(res, maxdd, on='fund_name', how='left')

        # res = dict(zip(res.fund_name, res.return_rank))

        return res


cat_indices = ['MDCP', 'SMCP', 'LCP', 'LargeMid', 'VALUE', 'FOCUS', 'FLEXI',
               'AGG', 'BADV', 'MULTIA', 'MULTICAP', 'ESAVING', 'CONSERV', 'RETM', "ELSS"]


lookback_rolling = [(5, 1*261), (2, 3*22)]
# lookback_rolling = [(15, 10*261), (12, 7*261), (10, 5*261),
# (7, 3*261), (5, 1*261), (2, 3*22)]

year = 2025
for l, r in lookback_rolling:
    merged_df = pd.DataFrame()
    for cat_index in cat_indices:
        print(f"runnning for {cat_index}-------------------")
        final = FundRanker(cat_index=cat_index, start_date='4/1/2015', end_date='1/1/2025').Ranker(dat=dt.date(year, 1, 1),
                                                                                                   lookback_years=l, rolling_lag_days=r)
        print(final)
        final.to_excel('Data/'+str(cat_index)+'_FS/'+cat_index+'_' + str(round(r/261, 2)) +
                       'Y_Ranks(' + str(l) + 'Y-LB, Y-'+str(year)+')'+'.xlsx', index=False)
        merged_df = pd.concat([merged_df, final], ignore_index=True)
    merged_df.to_excel('Data/'+'_' + str(round(r/261, 2)) +
                       'Y_Ranks(' + str(l) + 'Y-LB, Y-'+str(year)+')'+'.xlsx', index=False)

"""F = FundRanker()
lst = F.range()
for dat in lst:
    print(dat)
    print(F.MidcapRanker(dat = dat ,lookback_years=1,rolling_lag_days=132))
    # print(F.SmallcapRanker(dat=dat, lookback_years=1,rolling_lag_days=132))"""
