import warnings
import os
import warnings
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from Program.Config import *
from Program.Evaluate import *
from Program.Statistics import *
import Program.Signals as Signals


def cal_equity_curve(df, slippage=1 / 1000, c_rate=5 / 10000, leverage_rate=3,
                     min_amount=0.01,
                     min_margin_ratio=1 / 100):
    """
    :param df:
    :param slippage:  滑点 ，可以用百分比，也可以用固定值。建议币圈用百分比，股票用固定值
    :param c_rate:  手续费，commission fees，默认为万分之5。不同市场手续费的收取方法不同，对结果有影响。比如和股票就不一样。
    :param leverage_rate:  杠杆倍数
    :param min_amount:  最小下单量
    :param min_margin_ratio: 最低保证金率，低于就会爆仓
    :return:
    """
    # =====下根k线开盘价
    df['next_open'] = df['open'].shift(-1)  # 下根K线的开盘价
    df['next_open'].fillna(value=df['close'], inplace=True)

    # =====找出开仓、平仓的k线
    condition1 = df['pos'] != 0  # 当前周期不为空仓
    condition2 = df['pos'] != df['pos'].shift(1)  # 当前周期和上个周期持仓方向不一样。
    open_pos_condition = condition1 & condition2

    condition1 = df['pos'] != 0  # 当前周期不为空仓
    condition2 = df['pos'] != df['pos'].shift(-1)  # 当前周期和下个周期持仓方向不一样。
    close_pos_condition = condition1 & condition2

    # =====对每次交易进行分组
    df.loc[open_pos_condition, 'start_time'] = df['candle_begin_time']
    df['start_time'].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, 'start_time'] = pd.NaT

    # =====开始计算资金曲线
    initial_cash = 10000  # 初始资金，默认为10000元
    # ===在开仓时
    # 在open_pos_condition的K线，以开盘价计算买入合约的数量。（当资金量大的时候，可以用5分钟均价）
    df.loc[open_pos_condition, 'contract_num'] = initial_cash * leverage_rate / (min_amount * df['open'])
    df['contract_num'] = np.floor(df['contract_num'])  # 对合约张数向下取整
    # 开仓价格：理论开盘价加上相应滑点
    df.loc[open_pos_condition, 'open_pos_price'] = df['open'] * (1 + slippage * df['pos'])
    # 开仓之后剩余的钱，扣除手续费
    df['cash'] = initial_cash - df['open_pos_price'] * min_amount * df['contract_num'] * c_rate  # 即保证金

    # ===开仓之后每根K线结束时
    # 买入之后cash，contract_num，open_pos_price不再发生变动
    for _ in ['contract_num', 'open_pos_price', 'cash']:
        df[_].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, ['contract_num', 'open_pos_price', 'cash']] = None

    # ===在平仓时
    # 平仓价格
    df.loc[close_pos_condition, 'close_pos_price'] = df['next_open'] * (1 - slippage * df['pos'])
    # 平仓之后剩余的钱，扣除手续费
    df.loc[close_pos_condition, 'close_pos_fee'] = df['close_pos_price'] * min_amount * df['contract_num'] * c_rate

    # ===计算利润
    # 开仓至今持仓盈亏
    df['profit'] = min_amount * df['contract_num'] * (df['close'] - df['open_pos_price']) * df['pos']
    # 平仓时理论额外处理
    df.loc[close_pos_condition, 'profit'] = min_amount * df['contract_num'] * (
            df['close_pos_price'] - df['open_pos_price']) * df['pos']
    # 账户净值
    df['net_value'] = df['cash'] + df['profit']

    # ===计算爆仓
    # 至今持仓盈亏最小值
    df.loc[df['pos'] == 1, 'price_min'] = df['low']
    df.loc[df['pos'] == -1, 'price_min'] = df['high']
    df['profit_min'] = min_amount * df['contract_num'] * (df['price_min'] - df['open_pos_price']) * df['pos']
    # 账户净值最小值
    df['net_value_min'] = df['cash'] + df['profit_min']
    # 计算保证金率
    df['margin_ratio'] = df['net_value_min'] / (min_amount * df['contract_num'] * df['price_min'])
    # 计算是否爆仓
    df.loc[df['margin_ratio'] <= (min_margin_ratio + c_rate), '是否爆仓'] = 1

    # ===平仓时扣除手续费
    df.loc[close_pos_condition, 'net_value'] -= df['close_pos_fee']
    # 应对偶然情况：下一根K线开盘价格价格突变，在平仓的时候爆仓。此处处理有省略，不够精确。
    df.loc[close_pos_condition & (df['net_value'] < 0), '是否爆仓'] = 1

    # ===对爆仓进行处理
    df['是否爆仓'] = df.groupby('start_time')['是否爆仓'].fillna(method='ffill')
    df.loc[df['是否爆仓'] == 1, 'net_value'] = 0

    # =====计算资金曲线
    df['equity_change'] = df['net_value'].pct_change()
    df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition, 'net_value'] / initial_cash - 1  # 开仓日的收益率
    df['equity_change'].fillna(value=0, inplace=True)
    df['equity_curve'] = (1 + df['equity_change']).cumprod()

    # =====删除不必要的数据，并存储
    df.drop(['next_open', 'contract_num', 'open_pos_price', 'cash', 'close_pos_price', 'close_pos_fee',
             'profit', 'net_value', 'price_min', 'profit_min', 'net_value_min', 'margin_ratio', '是否爆仓'],
            axis=1, inplace=True)

    return df


def process_symbol(symbol, rule_type):
    warnings.filterwarnings('ignore')
    print(f"Processing {symbol} for rule type {rule_type}")

    # 读取并处理数据
    df = pd.read_csv(f"{symbol_data_path}/{symbol}.csv", encoding='gbk', parse_dates=['candle_begin_time'], skiprows=1)
    df.drop_duplicates(subset=['candle_begin_time'], inplace=True)
    df.sort_values(by=['candle_begin_time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 转换数据周期
    period_df = df.resample(rule=rule_type, on='candle_begin_time', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'quote_volume': 'sum', 'trade_num': 'sum',
        'taker_buy_base_asset_volume': 'sum', 'taker_buy_quote_asset_volume': 'sum'
    })
    period_df.dropna(subset=['open'], inplace=True)
    period_df = period_df[period_df['volume'] > 0]
    period_df.reset_index(inplace=True)

    df = period_df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_num',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]

    # 时间筛选
    t = df.iloc[0]['candle_begin_time'] + timedelta(days=drop_days)
    df = df[(df['candle_begin_time'] > t) &
            (df['candle_begin_time'] >= pd.to_datetime(date_start)) &
            (df['candle_begin_time'] <= pd.to_datetime(date_end))]
    df.reset_index(drop=True, inplace=True)

    # 计算资金曲线
    df['pos'] = 1
    min_amount = min_amount_dict[symbol.replace('-', '')]
    df = cal_equity_curve(df, slippage=slippage, c_rate=c_rate, leverage_rate=leverage_rate, min_amount=min_amount,
                          min_margin_ratio=min_margin_ratio)

    # 策略评价
    original_trade = transfer_equity_curve_to_trade(df)
    original, _ = strategy_evaluate(df, original_trade)

    # 保存结果
    result = pd.DataFrame({
        '币种': [symbol],
        '累积净值': [original.loc['累积净值', 0]],
        '年化收益': [original.loc['年化收益', 0]],
        '最大回撤': [original.loc['最大回撤', 0]],
        '年化收益/回撤比': [original.loc['年化收益/回撤比', 0]]
    })
    return result




def generate_fibonacci_sequence(min_number, max_number):
    """
    生成费拨那契数列，支持小数的生成
    注意：返回的所有数据都是浮点类型(小数)的，如果需要整数需要额外处理
    :param min_number: 最小值
    :param max_number: 最大值
    :return:
    """
    sequence = []
    base = 1
    if min_number < 1:
        base = 10 ** len(str(min_number).split('.')[1])
    last_number = 0
    new_number = 1
    while True:
        last_number, new_number = new_number, last_number + new_number
        if new_number / base > min_number:
            sequence.append(new_number / base)
        if new_number / base > max_number:
            break
    return sequence[:-1]



def load_and_prepare_data(symbol, rule_type):
    """
    读取并准备数据，包括去重、排序、周期转换等。
    :param symbol: 币种名称
    :param rule_type: 数据转换周期
    :return: 处理后的数据 DataFrame
    """
    # 读取数据
    df = pd.read_csv(os.path.join(symbol_data_path, f"{symbol}.csv"), encoding='gbk', parse_dates=['candle_begin_time'],
                     skiprows=1)
    df.drop_duplicates(subset=['candle_begin_time'], inplace=True)
    df.sort_values(by=['candle_begin_time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 转换数据周期
    period_df = df.resample(rule=rule_type, on='candle_begin_time', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'quote_volume': 'sum', 'trade_num': 'sum',
        'taker_buy_base_asset_volume': 'sum', 'taker_buy_quote_asset_volume': 'sum'
    }).dropna(subset=['open'])

    period_df = period_df[period_df['volume'] > 0].reset_index()
    return period_df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_num',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]


def apply_signals(df):
    """
    计算交易信号。
    :param df: 处理后的数据 DataFrame
    :return: 含有信号的 DataFrame
    """
    return getattr(Signals, signal_name)(df, para=para, proportion=proportion, leverage_rate=leverage_rate)


def filter_time(df):
    """
    根据时间筛选数据。
    :param df: 含有信号的 DataFrame
    :return: 筛选后的 DataFrame
    """
    t = df.iloc[0]['candle_begin_time'] + timedelta(days=drop_days)
    df = df[(df['candle_begin_time'] > t) &
            (df['candle_begin_time'] >= pd.to_datetime(date_start)) &
            (df['candle_begin_time'] <= pd.to_datetime(date_end))]
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_equity_curve(df):
    """
    计算资金曲线。
    :param df: 含有信号和时间筛选的 DataFrame
    :return: 计算资金曲线后的 DataFrame
    """
    min_amount = min_amount_dict[symbol.replace('-', '')]
    return cal_equity_curve(df, slippage=slippage, c_rate=c_rate, leverage_rate=leverage_rate, min_amount=min_amount,
                            min_margin_ratio=min_margin_ratio)


def save_results(df, symbol, rule_type):
    """
    保存回测结果到 CSV 文件。
    :param df: 计算资金曲线后的 DataFrame
    :param symbol: 币种名称
    :param rule_type: 数据转换周期
    """
    df_output = df[
        ['candle_begin_time', 'open', 'high', 'low', 'close', 'signal', 'pos', 'quote_volume', 'equity_curve']]
    df_output.rename(columns={'quote_volume': 'b_bar_quote_volume', 'equity_curve': 'r_line_equity_curve'},
                     inplace=True)
    output_path = os.path.join(root_path, 'data/output/equity_curve',
                               f'{signal_name}&{symbol.split("-")[0]}&{rule_type}&{str(para)}.csv')
    df_output.to_csv(output_path, index=False, encoding='gbk')


def evaluate_strategy(df):
    """
    计算策略评价指标。
    :param df: 计算资金曲线后的 DataFrame
    :return: 策略评价结果和每月收益率
    """
    trade = transfer_equity_curve_to_trade(df)
    title = f'{symbol}_{str(para)}'
    draw_equity_curve_mat(df, trade, title)
    return strategy_evaluate(df, trade)

def num_to_pct(value):
    return '%.2f%%' % (value * 100)

def write_file(content, path):
    """
    写入文件
    :param content: 写入内容
    :param path: 文件路径
    :return:
    """
    with open(path, 'w', encoding='utf8') as f:
        f.write(content)