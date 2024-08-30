# 由交易信号产生实际持仓
def position_for_future(df):
    """
    根据signal产生实际持仓。考虑各种不能买入卖出的情况。
    所有的交易都是发生在产生信号的K线的结束时
    :param df:
    :return:
    """

    # ===由signal计算出实际的每天持有仓位
    df['signal_'] = df['signal']
    # 在产生signal的k线结束的时候，进行买入
    df['signal_'].fillna(method='ffill', inplace=True)
    df['signal_'].fillna(value=0, inplace=True)  # 将初始行数的signal补全为0
    df['pos'] = df['signal_'].shift()
    df['pos'].fillna(value=0, inplace=True)  # 将初始行数的pos补全为0

    # pos为空的时，不能买卖，只能和前一周期保持一致。
    df['pos'].fillna(method='ffill', inplace=True)

    # 在实际操作中，不一定会直接跳过4点这个周期，而是会停止N分钟下单。此时可以注释掉上面的代码。

    # ===将数据存入hdf文件中
    # 删除无关中间变量
    # df.drop(['signal'], axis=1, inplace=True)

    return df
