from Program.Function import *

import numpy as np

def process_stop_loss_close(df, stop_loss_pct, leverage_rate):
    """
    止损函数
    :param df:
    :param stop_loss_pct: 止损比例
    :param leverage_rate: 杠杆倍数
    :return:
    """

    '''
    止损函数示例
     candle_begin_time                选币                   open               close           signal        原始信号           止损价格
    2021-04-23 04:00:00            IOST-USDT...            3.69380            3.69380            -1            -1              4.06318
    2021-04-23 05:00:00            IOST-USDT...            3.75580            3.75580            nan            nan            4.06318
    2021-04-23 06:00:00            IOST-USDT...            3.70157            3.70157            nan            nan            4.06318
    2021-04-23 07:00:00            IOST-USDT...            3.59443            3.59443            nan            nan            4.06318
    2021-04-23 08:00:00            IOST-USDT...            3.78299            3.78299            nan            nan            4.06318
    2021-04-23 09:00:00            IOST-USDT...            3.73637            3.73637            -1            -1              4.06318
    2021-04-23 10:00:00            IOST-USDT...            3.92761            3.92761            nan            nan            4.06318
    2021-04-23 11:00:00            IOST-USDT...            4.02816            4.02816            nan            nan            4.06318
    2021-04-23 12:00:00            IOST-USDT...            3.85746            3.85746            nan            nan            4.06318
    2021-04-23 13:00:00            IOST-USDT...            3.84017            3.84017            nan            nan            4.06318
    2021-04-23 14:00:00            IOST-USDT...            3.94633            3.94633            nan            nan            4.06318
    2021-04-23 15:00:00            IOST-USDT...            3.96164            3.96164            nan            nan            4.06318
    2021-04-23 16:00:00            IOST-USDT...            3.95144            3.95144            nan            nan            4.06318
    2021-04-23 17:00:00            IOST-USDT...            3.91294            3.91294            nan            nan            4.06318
    2021-04-23 18:00:00            IOST-USDT...            4.02094            4.02094            nan            nan            4.06318
    2021-04-23 19:00:00            IOST-USDT...            4.04794            4.04794            nan            nan            4.06318
    2021-04-23 20:00:00            IOST-USDT...            3.99289            3.99289            nan            nan            4.06318
    2021-04-23 21:00:00            IOST-USDT...            3.96215            3.96215            nan            nan            4.06318
    2021-04-23 22:00:00            IOST-USDT...            4.01350            4.01350            nan            nan            4.06318
    2021-04-23 23:00:00            IOST-USDT...            4.14397            4.14397            0              nan            4.06318
    '''

    # ===初始化持仓方向与开仓价格
    position = 0  # 持仓方向
    open_price = np.nan  # 开仓价格

    for i in df.index:
        # 开平仓   当signal不为空的时候 并且 open_price为空 或 position与当前方向不同
        if not np.isnan(df.loc[i, 'signal']) and (np.isnan(open_price) or position != int(df.loc[i, 'signal'])):
            position = int(df.loc[i, 'signal'])
            if df.loc[i, 'signal']:  # 开仓
                # 获取开仓的价格，为了符合实盘，所以获取下一周期的开盘价
                open_price = df.loc[i + 1, 'open'] if i < df.shape[0] - 1 else df.loc[i, 'close']
            else:  # 平仓，因为在python中非0即真，所以这里直接写else即代表0
                open_price = np.nan
        # 持仓
        if position:  # 判断当天是否有持仓方向，即是否为非0的值
            # 计算止损的价格   开仓价格 * (1 - 持仓方向 * 止损比例 / 杠杆倍数)
            # 假设我们100元开仓，止损0.05，杠杆为2，那么实际上我们开仓的仓位价值是100*2=200元，那么当你的本金亏损%5的时候，实际上的亏损为 (95-100)/200 = -0.025
            # 假设我们开仓的价格：100 方向：做多    止损比例：5%     杠杆倍数：2 那么止损价格: 100 * (1 - 1 * 0.05 / 2) = 100 * (1 - 0.025) = 100 * 0.975 = 97.5
            # 即当前的价格小于95就触发止损
            stop_loss_price = open_price * (1 - position * stop_loss_pct / leverage_rate)
            # 止损条件等于 持仓方向 * (收盘价 - 止损价格) <= 0
            stop_loss_condition = position * (df.loc[i, 'close'] - stop_loss_price) <= 0  # 止损条件
            df.at[i, 'stop_loss_condition'] = stop_loss_price
            # 如果满足止损条件，并且当前的信号为空时将signal设置为0，避免覆盖其他信号
            if stop_loss_condition and np.isnan(df.loc[i, 'signal']):
                df.at[i, 'signal'] = 0
                position = 0
                open_price = np.nan

    return df

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


# === 简单布林策略 + MFI
# 简单布林策略 + MFI

def calculate_mfi_1(df, period=7):
    """
    计算MFI指标并生成看涨或看空信号。
    :param df: 包含'high', 'low', 'close', 'volume'列的DataFrame
    :param period: MFI计算的周期，默认为14天
    :return: 添加了'MFI'和'signal'列的DataFrame
    """
    # 计算典型价格 (Typical Price)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3

    # 计算原始资金流 (Raw Money Flow)
    df['RMF'] = df['TP'] * df['volume']

    # 计算正向资金流和负向资金流
    df['PMF'] = np.where(df['TP'] > df['TP'].shift(1), df['RMF'], 0)
    df['NMF'] = np.where(df['TP'] < df['TP'].shift(1), abs(df['RMF']), 0)

    # 计算平均正向资金流和平均负向资金流
    df['PMF_avg'] = df['PMF'].rolling(window=period, min_periods=3).mean()
    df['NMF_avg'] = df['NMF'].rolling(window=period, min_periods=3).mean()

    # 计算MFI
    df['MFI'] = 100 - (100 / (1 + df['PMF_avg'] / df['NMF_avg']))

    # 生成看涨或看空信号
    df['signal'] = np.where(df['MFI'] > 50, 1, 0)  # 1表示看涨，0表示看空

    # 清理临时列
    df.drop(['TP', 'RMF', 'PMF', 'NMF', 'PMF_avg', 'NMF_avg'], axis=1, inplace=True)

    return df




# def calculate_mfi_2(df, period=14):
#     """
#     计算资金流动指数 (MFI) 并生成超买和超卖信号。
#
#     :param df: 包含市场数据的 DataFrame
#     :param period: MFI 计算的时间窗口，默认为 14 天
#     :return: 包含 MFI 和超买超卖信号的 DataFrame
#     """
#     # 计算典型价格 (Typical Price)
#     df['TP'] = (df['high'] + df['low'] + df['close']) / 3
#
#     # 计算正资金流和负资金流
#     df['MF'] = df['TP'] * df['volume']
#
#     # 计算每日的资金流比率
#     df['MF_Ratio'] = np.where(df['TP'] > df['TP'].shift(1),
#                               df['MF'],
#                               np.where(df['TP'] < df['TP'].shift(1),
#                                        -df['MF'],
#                                        0))
#
#     # 计算正资金流和负资金流的滚动和
#     df['Positive_MF'] = df['MF_Ratio'].clip(lower=0).rolling(window=period).sum()
#     df['Negative_MF'] = df['MF_Ratio'].clip(upper=0).abs().rolling(window=period).sum()
#
#     # 计算资金流比率
#     df['MFR'] = df['Positive_MF'] / df['Negative_MF']
#
#     # 计算 MFI
#     df['MFI'] = 100 - (100 / (1 + df['MFR']))
#
#     # 生成超买和超卖信号
#     df['signal'] = np.where(df['MFI'] > 55, 1, 0)  # 1表示超买，0表示正常
#     df['signal'] = np.where(df['MFI'] < 56, -1, df['signal'])  # -1表示超卖
#
#     # 清理临时列
#     df.drop(['TP', 'MF', 'MF_Ratio', 'Positive_MF', 'Negative_MF', 'MFR'], axis=1, inplace=True)
#
#     return df


def signal_bollinger_squeeze_with_mfi(df, para=[200, 2, 0.05], mfi_period=14, proportion=1, leverage_rate=1):
    """
    针对原始布林策略进行修改，加入布林线频带挤压策略和MFI指标。
    bias = close / 均线 - 1
    当开仓的时候，如果bias过大，即价格离均线过远，那么就先不开仓。等价格和均线距离小于bias_pct之后，才按照原计划开仓
    :param df: 原始数据
    :param para: n,m,bias_pct siganl计算的参数
    :param mfi_period: MFI计算的周期
    :return:
    """

    # ===== 获取策略参数
    n = int(para[0])  # 获取参数n，即para第一个元素
    m = float(para[1])  # 获取参数m，即para第二个元素
    bias_pct = float(para[2])  # 获取参数bias_pct，即para第三个元素

    # ===== 计算指标
    # 计算均线
    df['median'] = df['close'].rolling(n, min_periods=1).mean()  # 计算收盘价n个周期的均线
    # 计算上轨、下轨道
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)  # 计算收盘价n日的标准差
    df['upper'] = df['median'] + m * df['std']  # 计算上轨
    df['lower'] = df['median'] - m * df['std']  # 计算下轨
    # 计算bias
    df['bias'] = df['close'] / df['median'] - 1

    # 计算布林线频带宽度
    df['bandwidth'] = (df['upper'] - df['lower']) / df['median']

    # 计算MFI
    df = calculate_mfi_1(df, period=mfi_period)

    # ===== 找出交易信号
    # === 找出布林线频带挤压信号
    df['squeeze_on'] = (df['upper'] - df['lower']).rolling(n, min_periods=1).mean() < (2 * df['std'])

    # === 找出做多信号
    condition1 = df['squeeze_on']  # 布林线频带处于挤压状态
    condition2 = df['close'] > df['upper']  # 当前K线的收盘价 > 上轨
    condition3 = df['close'].shift(1) <= df['upper'].shift(1)  # 之前K线的收盘价 <= 上轨
    condition4 = df['MFI'] < 20  # MFI指标低于20，表示超卖
    df.loc[condition1 & condition2 & condition3 & condition4, 'signal_long'] = 1  # 将产生做多信号的那根K线的signal设置为1

    # === 找出做多平仓信号
    condition1 = df['close'] < df['median']  # 当前K线的收盘价 < 中轨
    condition2 = df['close'].shift(1) >= df['median'].shift(1)  # 之前K线的收盘价 >= 中轨
    df.loc[condition1 & condition2, 'signal_long'] = 0  # 将产生平仓信号当天的signal设置为0

    # === 找出做空信号
    condition1 = df['squeeze_on']  # 布林线频带处于挤压状态
    condition2 = df['close'] < df['lower']  # 当前K线的收盘价 < 下轨
    condition3 = df['close'].shift(1) >= df['lower'].shift(1)  # 之前K线的收盘价 >= 下轨
    condition4 = df['MFI'] > 80  # MFI指标高于80，表示超买
    df.loc[condition1 & condition2 & condition3 & condition4, 'signal_short'] = -1  # 将产生做空信号的那根K线的signal设置为-1

    # === 找出做空平仓信号
    condition1 = df['close'] > df['median']  # 当前K线的收盘价 > 中轨
    condition2 = df['close'].shift(1) <= df['median'].shift(1)  # 之前K线的收盘价 <= 中轨
    df.loc[condition1 & condition2, 'signal_short'] = 0  # 将产生平仓信号当天的signal设置为0

    # ===== 合并做多做空信号
    df['signal'] = df[['signal_long', 'signal_short']].sum(axis=1, min_count=1,
                                                           skipna=True)  # 合并多空信号

    # ===== 根据bias，修改开仓时间
    df['temp'] = df['signal']
    # === 将原始信号做多时，当bias大于阀值，设置为空
    condition1 = (df['signal'] == 1)  # signal为1
    condition2 = (df['bias'] > bias_pct)  # bias大于bias_pct
    df.loc[condition1 & condition2, 'temp'] = None  # 将signal设置为空

    # === 将原始信号做空时，当bias大于阀值，设置为空
    condition1 = (df['signal'] == -1)  # signal为-1
    condition2 = (df['bias'] < -1 * bias_pct)  # bias小于 (-1 * bias_pct)
    df.loc[condition1 & condition2, 'temp'] = None  # 将signal设置为空

    # 原始信号刚开仓，并且大于阀值，将信号设置为0
    condition1 = (df['signal'] != df['signal'].shift(1))
    condition2 = (df['temp'].isnull())
    df.loc[condition1 & condition2, 'temp'] = 0

    # ===== 合去除重复信号
    # === 去除重复信号
    df['signal'] = df['temp']
    temp = df[df['signal'].notnull()][['signal']]  # 筛选siganla不为空的数据，并另存一个变量
    temp = temp[temp['signal'] != temp['signal'].shift(1)]  # 筛选出当前周期与上个周期持仓信号不一致的，即去除重复信号
    df['signal'] = temp['signal']  # 将处理后的signal覆盖到原始数据的signal列

    # ===== 删除无关变量
    columns_to_drop = ['median', 'std', 'upper', 'lower', 'bias', 'bandwidth', 'squeeze_on', 'signal_long', 'signal_short', 'temp']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # ===== 止盈止损
    # 校验当前的交易是否需要进行止盈止损
    df = process_stop_loss_close(df, proportion, leverage_rate=leverage_rate)  # 调用函数，判断是否需要止盈止损，df需包含signal列

    return df

def signal_bollinger_squeeze_with_mfi_2(df, para=[200, 2], mfi_period=14, proportion=1, leverage_rate=1):
    """
    针对原始布林策略进行修改，加入布林线频带挤压策略和MFI指标。
    :param df: 原始数据
    :param para: n,m siganl计算的参数
    :param mfi_period: MFI计算的周期
    :return:
    """

    # ===== 获取策略参数
    n = int(para[0])  # 获取参数n，即para第一个元素
    m = float(para[1])  # 获取参数m，即para第二个元素

    # ===== 计算指标
    # 计算均线
    df['median'] = df['close'].rolling(n, min_periods=1).mean()  # 计算收盘价n个周期的均线
    # 计算上轨、下轨道
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)  # 计算收盘价n日的标准差
    df['upper'] = df['median'] + m * df['std']  # 计算上轨
    df['lower'] = df['median'] - m * df['std']  # 计算下轨
    # 计算bias
    df['bias'] = df['close'] / df['median'] - 1

    # 计算布林线频带宽度
    df['bandwidth'] = (df['upper'] - df['lower']) / df['median']

    # 计算MFI
    df = calculate_mfi_1(df, period=mfi_period)

    # ===== 找出交易信号
    # === 找出布林线频带挤压信号
    df['squeeze_on'] = (df['upper'] - df['lower']).rolling(n, min_periods=1).mean() < (2 * df['std'])

    # === 找出做多信号
    condition1 = df['squeeze_on']  # 布林线频带处于挤压状态
    condition2 = df['close'] > df['upper']  # 当前K线的收盘价 > 上轨
    condition3 = df['close'].shift(1) <= df['upper'].shift(1)  # 之前K线的收盘价 <= 上轨
    condition4 = df['MFI'] < 20  # MFI指标低于20，表示超卖
    df.loc[condition1 & condition2 & condition3 & condition4, 'signal_long'] = 1  # 将产生做多信号的那根K线的signal设置为1

    # === 找出做多平仓信号
    condition1 = df['close'] < df['median']  # 当前K线的收盘价 < 中轨
    condition2 = df['close'].shift(1) >= df['median'].shift(1)  # 之前K线的收盘价 >= 中轨
    df.loc[condition1 & condition2, 'signal_long'] = 0  # 将产生平仓信号当天的signal设置为0

    # === 找出做空信号
    condition1 = df['squeeze_on']  # 布林线频带处于挤压状态
    condition2 = df['close'] < df['lower']  # 当前K线的收盘价 < 下轨
    condition3 = df['close'].shift(1) >= df['lower'].shift(1)  # 之前K线的收盘价 >= 下轨
    condition4 = df['MFI'] > 80  # MFI指标高于80，表示超买
    df.loc[condition1 & condition2 & condition3 & condition4, 'signal_short'] = -1  # 将产生做空信号的那根K线的signal设置为-1

    # === 找出做空平仓信号
    condition1 = df['close'] > df['median']  # 当前K线的收盘价 > 中轨
    condition2 = df['close'].shift(1) <= df['median'].shift(1)  # 之前K线的收盘价 <= 中轨
    df.loc[condition1 & condition2, 'signal_short'] = 0  # 将产生平仓信号当天的signal设置为0

    # ===== 合并做多做空信号
    df['signal'] = df[['signal_long', 'signal_short']].sum(axis=1, min_count=1,
                                                           skipna=True)  # 合并多空信号

    # ===== 去除重复信号
    # === 去除重复信号
    temp = df[df['signal'].notnull()][['signal']]  # 筛选siganla不为空的数据，并另存一个变量
    temp = temp[temp['signal'] != temp['signal'].shift(1)]  # 筛选出当前周期与上个周期持仓信号不一致的，即去除重复信号
    df['signal'] = temp['signal']  # 将处理后的signal覆盖到原始数据的signal列

    # # ===== 删除无关变量
    # columns_to_drop = ['median', 'std', 'upper', 'lower', 'bias', 'bandwidth', 'squeeze_on', 'signal_long', 'signal_short']
    # df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # ===== 止盈止损
    # 校验当前的交易是否需要进行止盈止损
    df = process_stop_loss_close(df, proportion, leverage_rate=leverage_rate)  # 调用函数，判断是否需要止盈止损，df需包含signal列

    return df
