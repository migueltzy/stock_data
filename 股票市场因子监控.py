from WindPy import *
from datetime import *
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import re
# import talib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datetime, time
from dateutil.relativedelta import relativedelta
# from sklearn import preprocessing
from tqdm import tqdm
from collections import Counter
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
w.start()

# region 日期和路径设定
path = 'data/'
start_date = '2021-12-30'
end_date = '2022-12-30'
if datetime.datetime.strptime(end_date, "%Y-%m-%d") > datetime.datetime.today():
    raise Exception(f"设定日期范围不能超出今日日期")


# endregion


# region 数据获取函数
def get_stock_satisfied(poolcode, condition=['ST'], date = end_date):
    '''
    从指数成分股列表中获取股票代码，并剔除不符合条件的股票
    :param poolcode: 指数代码，如000001.SH
    :param condition: 筛选条件，默认筛掉ST股

    '''

    pool_data = w.wset('sectorconstituent', 'date={};windcode={}'.format(date, poolcode))
    stock_name = pd.DataFrame(pool_data.Data, index=pool_data.Fields).T.set_index('wind_code')
    if 'ST' in condition:
        stock_index_nost = stock_name.apply(lambda x: 'ST' not in str(x), axis=1).replace(False, np.nan).dropna().index
        stock_name = pd.DataFrame(stock_name, index=stock_index_nost)

    pool = stock_name.index.to_list()
    return pool


def get_stock_data_fix_date(pool, indicators, start_date, end_date):
    '''根据指定日期获取所有股票池内股票的指标数据

    :param pool: 股票代码列表
    :param indicators: 指标列表

    '''
    timerange = pd.date_range(start=start_date, end=end_date, freq='D')
    frame = pd.DataFrame(index=pd.MultiIndex.from_product([timerange, pool], names=['date', 'code']))

    for i in indicators:
        tempdata = w.wsd(pool, i, startDate=start_date, endDate=end_date)
        tempdata_stack = pd.DataFrame(tempdata.Data, columns=tempdata.Times, index=tempdata.Codes).T.stack()
        tempdata_stack.name = i
        tempdata_stack = tempdata_stack.reset_index()
        tempdata_stack.columns = ['date', 'code', i]
        tempdata_stack.set_index(['date', 'code'], inplace=True)
        frame = pd.concat([frame, tempdata_stack], axis=1)
    return frame


def get_stock_industry_data(poolcode):
    # TODO 是否需要做回补？
    pool = get_stock_satisfied(poolcode)
    industry_data = w.wsd(pool, "industry_sw", end_date, end_date, 'industryType=1')
    industry_data = pd.DataFrame(index=['industry'], data=industry_data.Data, columns=industry_data.Codes).T
    return industry_data


def get_stock_data(poolcode, indicators=[], start_date=start_date, end_date=end_date):
    '''根据指定日期补充先前获取的数据

    :param poolcode: 指数代码，如000001.SH
    :param indicators: 指标列表，必须以列表格式输入，否则会报错

    '''
    if type(indicators) != list:
        print('指标应为列表格式')
        pass

    pool_satisfied = get_stock_satisfied(poolcode)

    stock_data = pd.read_excel(path + 'stock_data.xlsx', index_col=[0, 1])
    stock_data.index.names = ['date', 'code']

    stock_data_indicators = list(stock_data.columns)
    stock_data_pool = list(np.unique(stock_data.reset_index()['code'].values))
    stock_data_start_date = stock_data.reset_index().iloc[0, 0]
    stock_data_end_date = stock_data.reset_index().iloc[-1, 0]

    if pd.to_datetime(start_date) < stock_data_start_date:
        print('起始时间小于数据，开始补充')
        stock_data_append_before = get_stock_data_fix_date(stock_data_pool, stock_data_indicators,
                                                           start_date=start_date, end_date=str(stock_data_start_date))

        stock_data = pd.concat([stock_data_append_before, stock_data])

    if pd.to_datetime(end_date) > stock_data_end_date:
        print('终止时间大于数据，开始补充')
        stock_data_append_after = get_stock_data_fix_date(stock_data_pool, stock_data_indicators,
                                                          start_date=str(stock_data_end_date), end_date=end_date)
        stock_data = pd.concat([stock_data, stock_data_append_after])

    # 取新增的股票代码
    code_delta = list(set(pool_satisfied) - set(stock_data_pool))

    if len(code_delta) > 0:
        print('输入{}个股票，新增{}个股票'.format(len(pool_satisfied), len(code_delta)))
        stock_data_append_code = get_stock_data_fix_date(code_delta, stock_data_indicators, start_date, end_date)
        stock_data = pd.concat([stock_data, stock_data_append_code])

    # 取目前新增股票后所有的股票代码
    stock_data_pool_new = list(np.unique(stock_data.reset_index()['code'].values))
    # 去重
    stock_data = stock_data.loc[~stock_data.index.duplicated(), :]
    # 取新增的指标列表
    indicators_delta = list(set(indicators) - set(stock_data_indicators))

    if len(indicators_delta) > 0:
        print('新增{}个指标'.format(len(indicators_delta)))
        stock_data_append_indicators = get_stock_data_fix_date(stock_data_pool_new, indicators_delta, start_date,
                                                               end_date)
        stock_data = pd.concat([stock_data, stock_data_append_indicators], axis=1)

    stock_data.sort_index(ascending=True, inplace=True)
    stock_data = stock_data.dropna(how='all')
    stock_data.to_excel(path+'stock_data.xlsx')

    #   筛选出股票池中的股票和符合条件的数据
    stock_data.reset_index(inplace=True)
    stock_data = stock_data[stock_data['code'].isin(pool_satisfied)].set_index(['date', 'code'])[indicators]
    # print(stock_data)

    return stock_data


# endregion


# region 因子分组函数
def grouping(factor_value, number=3):
    '''
    根据股票因子值分组
    :param factor_value: 因子值
    :param number:分组的数量
    '''
    groups = []
    quantile_edge = 1 / number
    for i in range(number):
        group = factor_value.apply(lambda x:
                                   (x >= x.quantile(i * quantile_edge))
                                   &
                                   (x <= x.quantile((i + 1) * quantile_edge)),
                                   axis=1
                                   )
        groups.append(group)
    return groups


def get_factor_groups_ret(stock_ret, groups, benchmark, weighted=None):
    '''
    根据因子分组计算每组收益率
    :param stock_ret: 股票日涨跌幅数据
    :param groups: 因子分组列表
    :param weighted: 加权平均形式，默认为算术平均
    '''
    index_according_name = {'000905.SH': '中证500', '000852.SH': '中证1000', }
    ret_list = []
    number = 0
    for i in groups:
        number += 1
        group_ret = (i.replace(False, np.nan) * stock_ret).mean(axis=1)
        group_ret.name = 'G{}'.format(number)
        ret_list.append(group_ret)
    factor_ret_result = pd.concat(ret_list, axis=1)

    factor_ret_result['G{}-G1'.format(number)] = factor_ret_result['G{}'.format(number)] - factor_ret_result['G1']

    bm = get_stock_data_fix_date(benchmark, ['pct_chg'], start_date, end_date).iloc[:, 0].unstack().dropna() / 100
    for i in benchmark:
        factor_ret_result['G{}-{}'.format(number, index_according_name[i])] = factor_ret_result['G{}'.format(number)] - \
                                                                              bm[i]
        factor_ret_result['G1-{}'.format(index_according_name[i])] = factor_ret_result['G1'] - bm[i]

    return factor_ret_result


def get_factor_ret(poolcode, indicator, group_number=3, start_date=start_date, end_date=end_date,
                   benchmark=['000905.SH', '000852.SH'], weighted=None):
    '''
    总函数：获得因子分组收益率和超额收益

    :param poolcode: 指数代码
    :param indicator: 单个指标
    :param group_number: 组数，默认为3
    :param start_date: 起始日期
    :param end_date: 截止日期
    :param benchmark: benchmark，默认为中证500和中证1000
    :param weighted: 取平均方法，默认为等全平均
    '''

    factor_value = get_stock_data(poolcode, indicator, start_date=start_date, end_date=end_date).iloc[:, 0].unstack()




    stock_ret = get_stock_data(poolcode, ['pct_chg'], start_date=start_date, end_date=end_date).iloc[:,
                0].unstack() / 100

    groups = grouping(factor_value, group_number)

    factor_ret_result = get_factor_groups_ret(stock_ret, groups, benchmark, weighted=weighted)

    return factor_ret_result


# endregion

# region 获取因子类型

def get_category(factor):

    category = fl.loc[factor]['category']
    return category

# endregion
# region 行业分布函数
def industry_distribution(groups, industry_data):
    distribution_list = pd.DataFrame(index=np.unique(industry_data.dropna().values))
    number = 0

    for i in groups:
        number += 1
        group_codes = i.replace(False, np.nan).iloc[-1]
        #         group_codes = group_codes.reset_index().iloc[:,1:].set_index('code')
        industry_count = pd.concat([group_codes, industry_data], axis=1).groupby('industry').count()
        industry_count.columns = ['G{}'.format(number)]
        #         industry_ratio = industry_count / industry_count.sum()
        distribution_list['G{}'.format(number)] = industry_count

    all_member = groups[0].replace(False, True).iloc[-1]
    #     all_codes = all_member.reset_index().iloc[:,1:].set_index('code')
    all_industry_count = pd.concat([all_member, industry_data], axis=1).groupby('industry').count()
    #     all_industry_ratio = all_industry_count / all_industry_count.sum()
    distribution_list['total'] = all_industry_count

    return distribution_list


def get_industry_dist(poolcode, indicator, date=end_date, group_number=3):
    '''
    获取指定日期因子分组的行业分布(占比)
    :param poolcode: 股票池，指数代码
    :param indicator: 指标名称，即分组依据
    :param date: 指定日期，默认为end_date
    :param group_number: 分组数量，默认为三组
    '''
    factor_value = get_stock_data(poolcode, indicator).iloc[:, 0].unstack()
    industry_data = get_stock_industry_data(poolcode)

    groups = grouping(factor_value, group_number)
    distribution = industry_distribution(groups, industry_data)

    #     return groups
    return distribution


# endregion


# 时序结果输出
def output_factor_ret_ts(poolcode, factors, groups=3, start_date=start_date, end_date=end_date):
    ### 导出时间序列结果表格
    group_rets = []
    for i in factors:
        group_ret = get_factor_ret(poolcode, [i], groups, start_date, end_date)
        group_rets.append(group_ret)

    writer = pd.ExcelWriter(path + 'output_factors_ts.xlsx')
    for j in group_rets[0].columns:
        frame = pd.DataFrame()
        count = 0
        for k in factors:
            frame[k] = group_rets[count][j]
            count += 1

        #         print(frame)
        frame.to_excel(writer, sheet_name=j)
    writer.save()
    ### 导出多空净值图
    count2 = 0
    for l in group_rets:
        G1 = (l['G1'] + 1).cumprod()
        Gn = (l['G{}'.format(groups)] + 1).cumprod()

        plt.figure(figsize=(20,10))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=30)
        plt.title('{}因子收益率走势'.format(factors[count2]),fontsize=30)
        plt.plot(G1, label='G1')
        plt.plot(Gn, label='G{}'.format(groups))
        plt.legend(fontsize=20)
        plt.grid()
        plt.savefig('{}_ts.png'.format(factors[count2]), dpi=300)
        count2 += 1

    return


# 截面结果输出
def output_factor_ret_cs(poolcode, factors, groups=3, start_date=start_date, end_date=end_date):
    fl = pd.read_excel(path+'factors_list.xlsx', usecols=(0, 3))
    fl.dropna(inplace=True)
    fl.set_index('name_wd', inplace=True)

    group_rets = pd.DataFrame()
    count = 0
    for i in factors:
        group_ret = get_factor_ret(poolcode, [i], groups, start_date, end_date)
        group_ret_cs = (group_ret + 1).prod() - 1
        group_ret_cs.loc['category'] = fl.loc[i]['category']
        group_rets[factors[count]] = group_ret_cs

        count += 1

    group_rets.T.to_excel(path+'output_factors_ret_cs.xlsx')
    return


# 行业结果输出
def output_industry_bias(poolcode, factors, groups=3, date=end_date):
    writer = pd.ExcelWriter(path+'output_industry_bias.xlsx')
    for i in factors:
        # 导出行业偏离表
        frame = pd.DataFrame()
        dist = get_industry_dist(poolcode, indicator=[i], group_number=groups, date=date)
        frame['全市场股票数'] = dist['total']
        frame['全市场股票占比'] = dist['total'] / dist['total'].sum()

        frame['G1股票数'] = dist['G1']
        frame['G1股票占比'] = dist['G1'] / dist['G1'].sum()
        frame['G1超低配'] = frame['G1股票占比'] - frame['全市场股票占比']

        frame['G{}股票数'.format(groups)] = dist['G{}'.format(groups)]
        frame['G{}股票占比'.format(groups)] = dist['G{}'.format(groups)] / dist['G{}'.format(groups)].sum()
        frame['G{}超低配'.format(groups)] = frame['G{}股票占比'.format(groups)] - frame['全市场股票占比']

        frame.to_excel(writer, sheet_name=i)
        # print(frame)

        # 导出行业偏离图
        plt.figure(figsize=(50, 10))

        plt.title('{}因子行业偏离折线图'.format(i), fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=30)

        plt.plot(frame['全市场股票占比'], label='全市场股票占比')
        plt.plot(frame['G1股票占比'], label='G1股票占比')
        plt.plot(frame['G{}股票占比'.format(groups)], label='G{}股票占比'.format(groups))

        plt.grid()
        plt.legend(fontsize=20)
        # plt.show()
        plt.savefig(path+'{}_industry.png'.format(i), dpi=300)
    writer.save()
    return

def main_func(poolcode, factors, groups=3, start_date=start_date, end_date=end_date):
    count_error = []
    count_empty = []
    for i in factors:
        temp = w.wsd('600519.SH', i, startDate=end_date, endDate=end_date).Data[0][0]
        if not temp:
            count_empty.append(i)
        if temp == 'CWSDService:invalid indicators':
            count_error.append(i)

    if len(count_error) == len(count_empty) == 0:
        print('指标读取正常')
        start_time = time.perf_counter()
        output_factor_ret_ts(poolcode, factors, groups, start_date, end_date)
        print('时间序列数据已输出')
        output_factor_ret_cs(poolcode, factors, groups, start_date, end_date)
        print('截面数据已输出')
        output_industry_bias(poolcode, factors, groups, date=end_date)
        print('行业分布数据已输出')
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print('运行时间:', run_time, 's')
    else:
        print('读取异常！\n空值指标为：{}\n无效值指标为：{}'.format(count_empty, count_error))
    return


